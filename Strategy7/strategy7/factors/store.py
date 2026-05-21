"""Persistent factor value store for cache-first factor loading."""

from __future__ import annotations

import gc
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.constants import INTRADAY_FREQS
from ..core.utils import ensure_dir, log_progress
from .base import FactorLibrary, compute_factor_panel
from .reporting import factor_group_key, normalize_factor_package_alias

FACTOR_SPAN_SUMMARY_COLUMNS: List[str] = [
    "factor",
    "factor_freq",
    "factor_group",
    "factor_package",
    "start_time",
    "end_time",
    "obs_count",
    "code_count",
    "updated_at",
]


@dataclass
class FactorStoreOptions:
    enabled: bool = False
    root: str = "auto"
    file_format: str = "parquet"  # parquet/csv
    build_all: bool = False
    build_only: bool = False
    chunk_size: int = 64
    io_workers: int = 0


def _effective_factor_store_workers(requested: int, item_count: int) -> int:
    if item_count <= 1:
        return 1
    requested = int(requested or 0)
    if requested == 1:
        return 1
    if requested > 1:
        return min(requested, int(item_count))
    cpu = os.cpu_count() or 2
    return max(1, min(8, max(1, cpu // 2), int(item_count)))


def _infer_data_baostock_root(data_root: str) -> Path:
    p = Path(str(data_root)).expanduser()
    for cand in [p, *p.parents]:
        if cand.name.lower() == "data_baostock":
            return cand
    if p.name.lower() in {"hs300", "sz50", "zz500", "all"} and p.parent.name.lower() == "stock_hist":
        return p.parent.parent
    if p.name.lower() == "stock_hist":
        return p.parent
    return p


def resolve_factor_store_root(*, data_root: str, store_root_arg: str | None) -> Path:
    raw = str(store_root_arg or "").strip()
    if raw and raw.lower() not in {"auto", "default"}:
        return Path(raw).expanduser()
    return _infer_data_baostock_root(data_root) / "factor_value_store"


def _safe_token(text: object) -> str:
    s = str(text or "").strip()
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    t = "".join(out).strip("_")
    return t or "unknown"


def _normalize_time_cols(df: pd.DataFrame, *, factor_freq: str) -> pd.DataFrame:
    out = df.copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    if str(factor_freq).lower() in INTRADAY_FREQS and "datetime" in out.columns and "date" not in out.columns:
        out["date"] = pd.to_datetime(out["datetime"], errors="coerce").dt.normalize()
    if "code" in out.columns:
        out["code"] = out["code"].astype(str).str.strip()
    return out


def _time_key_cols(base_df: pd.DataFrame, *, factor_freq: str) -> List[str]:
    if str(factor_freq).lower() in INTRADAY_FREQS and "datetime" in base_df.columns:
        return ["code", "datetime"]
    return ["code", "date"]


def _time_extra_cols(base_df: pd.DataFrame, *, factor_freq: str) -> List[str]:
    if str(factor_freq).lower() in INTRADAY_FREQS and "datetime" in base_df.columns:
        cols = ["date"] if "date" in base_df.columns else []
        return cols
    return []


def _choose_format(fmt: str) -> str:
    f = str(fmt or "parquet").strip().lower()
    return "csv" if f == "csv" else "parquet"


def _package_dir(
    *,
    store_root: Path,
    factor_freq: str,
    factor_group: str,
    factor_package: str,
) -> Path:
    return (
        store_root
        / str(factor_freq)
        / "by_group"
        / _safe_token(factor_group)
        / _safe_token(factor_package)
    )


def _code_file_path(pkg_dir: Path, code: str, file_format: str) -> Path:
    ext = ".csv" if _choose_format(file_format) == "csv" else ".parquet"
    return pkg_dir / f"{_safe_token(code)}{ext}"


def _read_table(path: Path, file_format: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if _choose_format(file_format) == "csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _write_table(df: pd.DataFrame, path: Path, file_format: str) -> None:
    ensure_dir(path.parent)
    if _choose_format(file_format) == "csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        df.to_parquet(path, index=False)


def _dedup_keep_order(cols: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for c in cols:
        cs = str(c)
        if cs in seen:
            continue
        seen.add(cs)
        out.append(cs)
    return out


def _numeric_col_or_nan(df: pd.DataFrame, col: str, index: pd.Index) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").reindex(index)


def _concat_non_empty(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    valid = [f for f in frames if f is not None and not f.empty]
    if not valid:
        return pd.DataFrame()
    if len(valid) == 1:
        return valid[0].copy()
    return pd.concat(valid, ignore_index=True)


def _factor_package_maps(
    *,
    factors: Sequence[str],
    factor_package_map: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    pkg_map: Dict[str, str] = {}
    grp_map: Dict[str, str] = {}
    for fac in factors:
        pkg = normalize_factor_package_alias(factor_package_map.get(str(fac), "unknown"))
        pkg_map[str(fac)] = pkg
        grp_map[str(fac)] = factor_group_key(pkg)
    return pkg_map, grp_map


def _group_factors_by_package(
    *,
    factors: Sequence[str],
    pkg_map: Dict[str, str],
    grp_map: Dict[str, str],
) -> Dict[Tuple[str, str], List[str]]:
    grouped: Dict[Tuple[str, str], List[str]] = {}
    for fac in factors:
        k = (str(grp_map.get(str(fac), "price_volume")), str(pkg_map.get(str(fac), "unknown")))
        grouped.setdefault(k, []).append(str(fac))
    return grouped


def load_factors_from_store(
    *,
    base_df: pd.DataFrame,
    factors: Sequence[str],
    factor_freq: str,
    store_root: Path,
    file_format: str,
    factor_package_map: Dict[str, str],
    io_workers: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if base_df.empty or not factors:
        return pd.DataFrame(index=base_df.index), {
            "loaded_factor_count": 0,
            "loaded_full_count": 0,
            "loaded_partial_count": 0,
            "store_io_workers": 1,
            "factor_present_coverage": {},
            "factor_value_coverage": {},
        }

    fmt = _choose_format(file_format)
    key_cols = _time_key_cols(base_df, factor_freq=factor_freq)
    extra_cols = _time_extra_cols(base_df, factor_freq=factor_freq)
    base_keys = _normalize_time_cols(base_df[key_cols + extra_cols].copy(), factor_freq=factor_freq)

    pkg_map, grp_map = _factor_package_maps(factors=factors, factor_package_map=factor_package_map)
    grouped = _group_factors_by_package(factors=factors, pkg_map=pkg_map, grp_map=grp_map)

    merged = base_keys.copy()
    loaded_cols: List[str] = []
    factor_present_cov: Dict[str, float] = {str(f): 0.0 for f in factors}
    factor_value_cov: Dict[str, float] = {str(f): 0.0 for f in factors}
    base_row_count = float(len(base_keys))
    codes = sorted(base_keys["code"].astype(str).drop_duplicates().tolist())
    workers = _effective_factor_store_workers(int(io_workers or 0), len(codes))
    for (grp, pkg), facs in grouped.items():
        pkg_dir = _package_dir(
            store_root=store_root,
            factor_freq=factor_freq,
            factor_group=grp,
            factor_package=pkg,
        )
        if not pkg_dir.exists():
            continue

        def _read_one_code(code: str) -> pd.DataFrame:
            fp = _code_file_path(pkg_dir, code, fmt)
            if not fp.exists():
                return pd.DataFrame()
            tbl = _read_table(fp, fmt)
            if tbl.empty:
                return pd.DataFrame()
            tbl = _normalize_time_cols(tbl, factor_freq=factor_freq)
            keep = [c for c in key_cols + extra_cols + facs if c in tbl.columns]
            if len(keep) <= len(key_cols):
                return pd.DataFrame()
            tbl = tbl[keep].copy()
            tbl = tbl[tbl["code"].astype(str) == str(code)]
            return tbl.drop_duplicates(key_cols, keep="last")

        parts: List[pd.DataFrame] = []
        if workers <= 1:
            for code in codes:
                part = _read_one_code(code)
                if not part.empty:
                    parts.append(part)
        else:
            batch_size = max(8, workers * 2)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for i in range(0, len(codes), batch_size):
                    for part in executor.map(_read_one_code, codes[i : i + batch_size]):
                        if not part.empty:
                            parts.append(part)
        if not parts:
            continue
        pkg_df = _concat_non_empty(parts)
        if pkg_df.empty:
            continue
        pkg_df = pkg_df.drop_duplicates(key_cols, keep="last")
        pkg_present_cov = (float(len(pkg_df)) / base_row_count) if base_row_count > 0.0 else 0.0
        merged = merged.merge(pkg_df, on=key_cols + extra_cols, how="left")
        present_cols = [c for c in facs if c in merged.columns]
        loaded_cols.extend(present_cols)
        for fac in present_cols:
            factor_present_cov[str(fac)] = max(float(factor_present_cov.get(str(fac), 0.0)), float(pkg_present_cov))

    out = pd.DataFrame(index=base_df.index)
    full_cnt = 0
    partial_cnt = 0
    for fac in factors:
        if fac not in merged.columns:
            out[str(fac)] = np.nan
            continue
        s = pd.to_numeric(merged[fac], errors="coerce")
        out[str(fac)] = s.to_numpy()
        val_cov = float(s.notna().mean()) if len(s) else 0.0
        factor_value_cov[str(fac)] = float(val_cov)
        present_cov = float(factor_present_cov.get(str(fac), 0.0))
        if present_cov >= 0.999999:
            full_cnt += 1
        elif present_cov > 0.0:
            partial_cnt += 1
    return out, {
        "loaded_factor_count": int(len(set(loaded_cols))),
        "loaded_full_count": int(full_cnt),
        "loaded_partial_count": int(partial_cnt),
        "store_io_workers": int(workers),
        "factor_present_coverage": factor_present_cov,
        "factor_value_coverage": factor_value_cov,
    }


def _merge_existing_and_new(
    *,
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    key_cols: Sequence[str],
    value_cols: Sequence[str],
) -> pd.DataFrame:
    if old_df.empty:
        out = new_df.copy()
        return out.drop_duplicates(list(key_cols), keep="last")
    out = old_df.copy()
    if out.empty:
        out = new_df.copy()
        return out.drop_duplicates(list(key_cols), keep="last")

    merged = out.merge(
        new_df,
        on=list(key_cols),
        how="outer",
        suffixes=("_old", "_new"),
    )
    for c in value_cols:
        old_c = f"{c}_old" if f"{c}_old" in merged.columns else c
        new_c = f"{c}_new" if f"{c}_new" in merged.columns else c
        new_s = _numeric_col_or_nan(merged, new_c, merged.index)
        old_s = _numeric_col_or_nan(merged, old_c, merged.index)
        # New values win, old cached values fill gaps. Avoid Series.combine_first:
        # pandas 2.2+ warns when its internal concat sees empty/all-NA blocks.
        merged[c] = new_s.where(new_s.notna(), old_s).astype("float32")
    keep_cols = _dedup_keep_order(
        list(key_cols)
        + [c for c in out.columns if c not in key_cols]
        + [c for c in value_cols if c not in out.columns]
    )
    keep_cols = [c for c in keep_cols if c in merged.columns]
    merged = merged[keep_cols]
    merged = merged.drop_duplicates(list(key_cols), keep="last")
    return merged


def _upsert_span_summary(
    *,
    store_root: Path,
    factor_freq: str,
    factors: Sequence[str],
    panel_df: pd.DataFrame,
    factor_package_map: Dict[str, str],
) -> Path:
    if panel_df.empty or not factors:
        return ensure_dir(store_root / str(factor_freq)) / "factor_span_summary.csv"

    key_cols = _time_key_cols(panel_df, factor_freq=factor_freq)
    time_col = "datetime" if "datetime" in key_cols else "date"
    time_s = pd.to_datetime(panel_df[time_col], errors="coerce")
    code_s = panel_df["code"].astype(str).str.strip()

    rows: List[Dict[str, object]] = []
    for fac in factors:
        if fac not in panel_df.columns:
            continue
        v = pd.to_numeric(panel_df[fac], errors="coerce")
        mask = v.notna() & time_s.notna() & code_s.ne("")
        if int(mask.sum()) <= 0:
            continue
        pkg = normalize_factor_package_alias(factor_package_map.get(str(fac), "unknown"))
        rows.append(
            {
                "factor": str(fac),
                "factor_freq": str(factor_freq),
                "factor_group": factor_group_key(pkg),
                "factor_package": pkg,
                "start_time": str(time_s.loc[mask].min()),
                "end_time": str(time_s.loc[mask].max()),
                "obs_count": int(mask.sum()),
                "code_count": int(code_s.loc[mask].nunique()),
                "updated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            }
        )
    summary_path = ensure_dir(store_root / str(factor_freq)) / "factor_span_summary.csv"
    now_df = pd.DataFrame(rows, columns=FACTOR_SPAN_SUMMARY_COLUMNS)
    if now_df.empty:
        if not summary_path.exists():
            now_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        return summary_path
    if summary_path.exists():
        try:
            old = pd.read_csv(summary_path)
        except pd.errors.EmptyDataError:
            old = pd.DataFrame(columns=FACTOR_SPAN_SUMMARY_COLUMNS)
        merged = _concat_non_empty([old, now_df])
        if merged.empty:
            merged = now_df.copy()
        merged = merged.drop_duplicates(subset=["factor", "factor_freq"], keep="last")
    else:
        merged = now_df
    for c in FACTOR_SPAN_SUMMARY_COLUMNS:
        if c not in merged.columns:
            merged[c] = np.nan
    merged = merged[FACTOR_SPAN_SUMMARY_COLUMNS]
    merged = merged.sort_values(["factor_package", "factor"]).reset_index(drop=True)
    merged.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_path


def save_factors_to_store(
    *,
    panel_df: pd.DataFrame,
    factors: Sequence[str],
    factor_freq: str,
    store_root: Path,
    file_format: str,
    factor_package_map: Dict[str, str],
    io_workers: int = 0,
) -> Dict[str, object]:
    if panel_df.empty or not factors:
        return {"saved_factor_count": 0, "saved_code_files": 0, "span_summary_path": "", "store_io_workers": 1}
    fmt = _choose_format(file_format)
    key_cols = _time_key_cols(panel_df, factor_freq=factor_freq)
    extra_cols = _time_extra_cols(panel_df, factor_freq=factor_freq)

    needed_cols = _dedup_keep_order(key_cols + extra_cols + [str(c) for c in factors if str(c) in panel_df.columns])
    panel = _normalize_time_cols(panel_df[needed_cols].copy(), factor_freq=factor_freq)
    panel = panel.dropna(subset=key_cols)
    if panel.empty:
        return {"saved_factor_count": 0, "saved_code_files": 0, "span_summary_path": "", "store_io_workers": 1}

    pkg_map, grp_map = _factor_package_maps(factors=factors, factor_package_map=factor_package_map)
    grouped = _group_factors_by_package(factors=factors, pkg_map=pkg_map, grp_map=grp_map)
    code_count = int(panel["code"].astype(str).nunique()) if "code" in panel.columns else 0
    workers = _effective_factor_store_workers(int(io_workers or 0), code_count)

    saved_files = 0
    for (grp, pkg), facs in grouped.items():
        pkg_dir = ensure_dir(
            _package_dir(
                store_root=store_root,
                factor_freq=factor_freq,
                factor_group=grp,
                factor_package=pkg,
            )
        )
        keep_cols = key_cols + extra_cols + [c for c in facs if c in panel.columns]
        if len(keep_cols) <= len(key_cols):
            continue
        pkg_panel = panel[keep_cols].copy()
        pkg_panel = pkg_panel.drop_duplicates(key_cols, keep="last")

        def _write_one_group(item: Tuple[str, pd.DataFrame]) -> int:
            code, g = item
            fp = _code_file_path(pkg_dir, str(code), fmt)
            old = _read_table(fp, fmt)
            if not old.empty:
                old = _normalize_time_cols(old, factor_freq=factor_freq)
            merged = _merge_existing_and_new(
                old_df=old,
                new_df=g,
                key_cols=key_cols + extra_cols,
                value_cols=[c for c in facs if c in g.columns],
            )
            merged = merged.sort_values(key_cols + extra_cols).reset_index(drop=True)
            _write_table(merged, fp, fmt)
            return 1

        if workers <= 1:
            for code, g in pkg_panel.groupby("code", sort=False):
                saved_files += _write_one_group((str(code), g.copy()))
        else:
            batch: List[Tuple[str, pd.DataFrame]] = []
            batch_size = max(8, workers * 2)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for code, g in pkg_panel.groupby("code", sort=False):
                    batch.append((str(code), g.copy()))
                    if len(batch) >= batch_size:
                        saved_files += sum(executor.map(_write_one_group, batch))
                        batch.clear()
                if batch:
                    saved_files += sum(executor.map(_write_one_group, batch))

    span_path = _upsert_span_summary(
        store_root=store_root,
        factor_freq=factor_freq,
        factors=factors,
        panel_df=panel,
        factor_package_map=factor_package_map,
    )
    return {
        "saved_factor_count": int(len(set([str(x) for x in factors]))),
        "saved_code_files": int(saved_files),
        "span_summary_path": str(span_path),
        "store_io_workers": int(workers),
    }


def hydrate_factor_panel_with_store(
    *,
    base_df: pd.DataFrame,
    library: FactorLibrary,
    freq: str,
    selected_factors: Sequence[str],
    store_root: Path,
    file_format: str,
    factor_package_map: Dict[str, str],
    coverage_threshold: float = 0.999999,
    write_back: bool = True,
    io_workers: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    # Factor columns are appended/replaced below; a shallow shell avoids
    # duplicating the whole base feature matrix before cache hydration.
    panel = base_df.copy(deep=False)
    if not selected_factors:
        return panel, {"cache_enabled": True, "loaded_factor_count": 0, "computed_factor_count": 0, "saved_factor_count": 0}

    loaded_df, load_stats = load_factors_from_store(
        base_df=base_df,
        factors=selected_factors,
        factor_freq=freq,
        store_root=store_root,
        file_format=file_format,
        factor_package_map=factor_package_map,
        io_workers=io_workers,
    )
    need_compute: List[str] = []
    present_cov_map: Dict[str, float] = {
        str(k): float(v)
        for k, v in dict(load_stats.get("factor_present_coverage", {})).items()
    }
    for fac in selected_factors:
        if fac not in loaded_df.columns:
            need_compute.append(str(fac))
            continue
        s = pd.to_numeric(loaded_df[fac], errors="coerce")
        panel[fac] = s.to_numpy()
        present_cov = float(present_cov_map.get(str(fac), 0.0))
        if present_cov < float(coverage_threshold):
            need_compute.append(str(fac))

    computed_count = 0
    if need_compute:
        comp_panel = compute_factor_panel(
            base_df=base_df,
            library=library,
            freq=freq,
            selected_factors=need_compute,
        )
        for fac in need_compute:
            computed_count += 1
            new_s = pd.to_numeric(comp_panel[fac], errors="coerce")
            if fac in panel.columns:
                old_s = pd.to_numeric(panel[fac], errors="coerce")
                panel[fac] = old_s.where(old_s.notna(), new_s)
            else:
                panel[fac] = new_s

    save_stats = {"saved_factor_count": 0, "saved_code_files": 0, "span_summary_path": ""}
    if write_back:
        save_stats = save_factors_to_store(
            panel_df=panel,
            factors=list(selected_factors),
            factor_freq=freq,
            store_root=store_root,
            file_format=file_format,
            factor_package_map=factor_package_map,
            io_workers=io_workers,
        )

    return panel, {
        "cache_enabled": True,
        **load_stats,
        "computed_factor_count": int(computed_count),
        **save_stats,
    }


def build_factor_store_for_full_list(
    *,
    base_df: pd.DataFrame,
    library: FactorLibrary,
    freq: str,
    all_factors: Sequence[str],
    store_root: Path,
    file_format: str,
    factor_package_map: Dict[str, str],
    chunk_size: int = 64,
    io_workers: int = 0,
) -> Dict[str, object]:
    if not all_factors:
        return {
            "full_build_factor_count": 0,
            "chunk_count": 0,
            "chunk_size": int(chunk_size),
            "saved_factor_count": 0,
            "saved_code_files": 0,
            "span_summary_path": "",
            "store_io_workers": 1,
        }
    n = max(int(chunk_size), 1)
    total_saved_files = 0
    span_path = ""
    effective_workers = 1
    total_chunks = int((len(all_factors) + n - 1) // n)
    log_progress(
        f"factor store full build start: freq={freq}, factors={len(all_factors)}, "
        f"chunk_size={n}, chunks={total_chunks}, io_workers={int(io_workers or 0)}",
        module="factor_store",
    )
    for i in range(0, len(all_factors), n):
        chunk = [str(x) for x in all_factors[i : i + n]]
        chunk_idx = int(i // n + 1)
        log_progress(
            f"factor store chunk {chunk_idx}/{total_chunks}: size={len(chunk)}, range={i + 1}-{i + len(chunk)}",
            module="factor_store",
            level="debug",
        )
        try:
            chunk_panel = compute_factor_panel(
                base_df=base_df,
                library=library,
                freq=freq,
                selected_factors=chunk,
            )
        except Exception as exc:
            raise RuntimeError(
                f"factor store build failed while computing chunk {chunk_idx}/{total_chunks} "
                f"(size={len(chunk)}, first_factors={chunk[:5]}): {exc}"
            ) from exc
        try:
            save_stats = save_factors_to_store(
                panel_df=chunk_panel,
                factors=chunk,
                factor_freq=freq,
                store_root=store_root,
                file_format=file_format,
                factor_package_map=factor_package_map,
                io_workers=io_workers,
            )
        except Exception as exc:
            raise RuntimeError(
                f"factor store build failed while saving chunk {chunk_idx}/{total_chunks} "
                f"(size={len(chunk)}, first_factors={chunk[:5]}): {exc}"
            ) from exc
        total_saved_files += int(save_stats.get("saved_code_files", 0))
        effective_workers = int(save_stats.get("store_io_workers", effective_workers))
        span_path = str(save_stats.get("span_summary_path", span_path))
        del chunk_panel
        gc.collect()
    log_progress(
        f"factor store full build done: factors={len(all_factors)}, saved_code_files={total_saved_files}",
        module="factor_store",
    )

    return {
        "full_build_factor_count": int(len(all_factors)),
        "chunk_count": int(total_chunks),
        "chunk_size": int(n),
        "saved_factor_count": int(len(all_factors)),
        "saved_code_files": int(total_saved_files),
        "span_summary_path": str(span_path),
        "store_io_workers": int(effective_workers),
    }
