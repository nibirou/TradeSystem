"""Market data loader and base feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import DataConfig, DateConfig
from ..core.constants import EPS
from ..core.time_utils import compute_load_window
from ..core.types import FeatureBundle, MarketBundle
from ..core.utils import (
    log_progress,
    infer_board_type,
    infer_industry_bucket,
    is_main_board_symbol,
    split_exchange_code,
    symbol_key_from_filename,
)
from .base import MarketDataLoader
from .frequency import add_generic_micro_structure_features, add_multifreq_bridge_features, build_frequency_views
from .text_nlp import (
    TEXT_CONTEXT_COLUMNS,
    add_text_rolling_and_fusion_features,
    build_text_daily_features,
    load_text_source_events_from_file,
    pick_symbol_text_file,
    select_text_universe_dirs,
)

FUND_CATEGORIES: List[str] = [
    "growth",
    "valuation",
    "profitability",
    "quality",
    "leverage",
    "cashflow",
    "efficiency",
    "expectation",
]

FUND_CATEGORY_KEYWORDS: Dict[str, Sequence[str]] = {
    "growth": (
        "growth",
        "yoy",
        "chg",
        "gr",
        "增",
        "增长",
    ),
    "valuation": (
        "pe",
        "pb",
        "ps",
        "pcf",
        "ev",
        "mv",
        "估值",
        "市盈",
        "市净",
        "股本",
        "share",
    ),
    "profitability": (
        "profit",
        "margin",
        "roe",
        "roa",
        "eps",
        "ebit",
        "netprofit",
        "净利",
        "利润",
        "收益",
    ),
    "quality": (
        "dupont",
        "taxburden",
        "intburden",
        "扣非",
        "quality",
        "accrual",
        "净资产收益率",
    ),
    "leverage": (
        "liability",
        "debt",
        "equity",
        "assettoequity",
        "assetstoequity",
        "quickratio",
        "currentratio",
        "cashratio",
        "负债",
        "杠杆",
    ),
    "cashflow": (
        "cash",
        "cfo",
        "cashflow",
        "ebittointerest",
        "经营现金",
        "现金流",
    ),
    "efficiency": (
        "turn",
        "turnover",
        "days",
        "assetturn",
        "invturn",
        "nrturn",
        "周转",
        "效率",
    ),
    "expectation": (
        "forecast",
        "forcast",
        "performanceexp",
        "express",
        "notice",
        "预告",
        "快报",
    ),
}

FUND_CANONICAL_PER_CATEGORY = 12
FUND_INTRADAY_CONTEXT_RAW_COUNT = 5


def list_symbol_keys(daily_dir: Path, file_format: str = "auto") -> List[str]:
    keys: set[str] = set()
    if file_format in {"auto", "parquet"}:
        for p in daily_dir.glob("*_d.parquet"):
            k = symbol_key_from_filename(p.name)
            if k:
                keys.add(k)
    if file_format in {"auto", "csv"}:
        for p in daily_dir.glob("*_d.csv"):
            k = symbol_key_from_filename(p.name)
            if k:
                keys.add(k)
    return sorted(keys)


def pick_existing_file(folder: Path, symbol_key: str, freq_tag: str, file_format: str = "auto") -> Optional[Path]:
    if file_format == "parquet":
        cands = [folder / f"{symbol_key}_{freq_tag}.parquet"]
    elif file_format == "csv":
        cands = [folder / f"{symbol_key}_{freq_tag}.csv"]
    else:
        cands = [folder / f"{symbol_key}_{freq_tag}.parquet", folder / f"{symbol_key}_{freq_tag}.csv"]
    for p in cands:
        if p.exists():
            return p
    return None


def read_data_file(path: Path, usecols: List[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    date_start = start_date.strftime("%Y-%m-%d")
    date_end = end_date.strftime("%Y-%m-%d")
    if path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(path, columns=usecols, filters=[("date", ">=", date_start), ("date", "<=", date_end)])
        except Exception:
            csv_path = path.with_suffix(".csv")
            if not csv_path.exists():
                raise
            df = pd.read_csv(csv_path, usecols=lambda c: c in usecols)
    else:
        df = pd.read_csv(path, usecols=lambda c: c in usecols)
    if "date" not in df.columns:
        return pd.DataFrame(columns=usecols)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    return df


def load_stock_list_keys(stock_list_path: Path) -> List[str]:
    if not stock_list_path.exists():
        raise FileNotFoundError(f"stock list file not found: {stock_list_path}")
    df = pd.read_csv(stock_list_path)
    if "code" in df.columns:
        codes = df["code"].astype(str)
    else:
        codes = df.iloc[:, 0].astype(str)
    keys = codes.str.strip().str.lower().str.replace(".", "_", regex=False)
    keys = keys[keys.str.match(r"^[a-z]{2}_\d{6}$", na=False)]
    return sorted(keys.drop_duplicates().tolist())


def load_hs300_constituent_keys(hs300_list_path: Path) -> List[str]:
    """Backward-compatible alias. Prefer `load_stock_list_keys`."""
    return load_stock_list_keys(hs300_list_path)


def _normalize_code(code_or_key: object, *, dotted: bool = True) -> str:
    ex, code = split_exchange_code(str(code_or_key))
    if ex not in {"sh", "sz"} or not code:
        return ""
    return f"{ex}.{code}" if dotted else f"{ex}_{code}"


def _safe_feature_name(name: object) -> str:
    raw = str(name).strip()
    txt = (
        raw.lower()
        .replace("%", "pct")
        .replace("\u2030", "permille")
        .replace("\uffe5", "cny")
        .replace("$", "usd")
    )
    txt = re.sub(r"[^0-9a-zA-Z]+", "_", txt).strip("_").lower()
    if not txt:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        txt = f"c_{digest}"
    if txt[0].isdigit():
        txt = f"f_{txt}"
    return txt[:96]


def _read_table_file(path: Path, file_format: str = "auto") -> pd.DataFrame:
    fmt = str(file_format).strip().lower()
    ext = path.suffix.lower()
    use_parquet = (fmt == "parquet") or (fmt == "auto" and ext == ".parquet")
    if use_parquet:
        try:
            return pd.read_parquet(path)
        except Exception:
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path, low_memory=False)
            raise
    return pd.read_csv(path, low_memory=False)


def _pick_stem_file(folder: Path, stem: str, file_format: str = "auto") -> Optional[Path]:
    if file_format == "parquet":
        cands = [folder / f"{stem}.parquet"]
    elif file_format == "csv":
        cands = [folder / f"{stem}.csv"]
    else:
        cands = [folder / f"{stem}.parquet", folder / f"{stem}.csv"]
    for p in cands:
        if p.exists():
            return p
    return None


def _select_fundamental_universe_dir(base_dir: Path, universe: str) -> Optional[Path]:
    if not base_dir.exists():
        return None
    wanted = [str(universe).strip().lower(), "all", "hs300", "zz500", "sz50"]
    seen: set[str] = set()
    for u in wanted:
        if not u or u in seen:
            continue
        seen.add(u)
        cand = base_dir / u
        if cand.exists() and cand.is_dir():
            return cand
    return None


def _choose_date_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = {str(c): str(c).lower() for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
    cand_l = [str(c).lower() for c in candidates]
    for c, cl in cols.items():
        if cl in cand_l:
            return c
    # fallback: choose first date-like column name.
    for c, cl in cols.items():
        if "date" in cl or "日期" in str(c):
            return c
    return None


def _extract_numeric_columns(
    df: pd.DataFrame,
    *,
    exclude_cols: Iterable[str],
    prefix: str,
) -> pd.DataFrame:
    excluded = {str(c) for c in exclude_cols}
    out: Dict[str, pd.Series] = {}
    used: set[str] = set()
    for col in df.columns:
        if col in excluded:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() <= 0:
            continue
        base = _safe_feature_name(col)
        name = f"{prefix}{base}"
        idx = 2
        while name in used:
            name = f"{prefix}{base}_{idx:02d}"
            idx += 1
        used.add(name)
        out[name] = s.astype("float32")
    if not out:
        return pd.DataFrame(index=df.index)
    return pd.DataFrame(out, index=df.index)


def _load_ak_indicator_em_file(
    path: Path,
    symbol_key: str,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    file_format: str,
) -> pd.DataFrame:
    df = _read_table_file(path, file_format=file_format)
    if df.empty:
        return pd.DataFrame()
    date_col = _choose_date_column(df, ["NOTICE_DATE", "UPDATE_DATE", "REPORT_DATE"])
    if date_col is None:
        return pd.DataFrame()
    code_col = next((c for c in ["SECUCODE", "SECURITY_CODE", "code", "CODE"] if c in df.columns), None)
    if code_col is not None:
        code = df[code_col].astype(str).map(_normalize_code)
    else:
        code = pd.Series([_normalize_code(symbol_key)] * len(df), index=df.index)
    date_s = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    valid = date_s.notna() & code.ne("")
    if not valid.any():
        return pd.DataFrame()
    out = df.loc[valid].copy()
    out["date"] = date_s.loc[valid]
    out["code"] = code.loc[valid]
    out = out[(out["date"] >= start_date) & (out["date"] <= end_date)]
    if out.empty:
        return out
    num = _extract_numeric_columns(
        out,
        exclude_cols={
            "date",
            "code",
            date_col,
            code_col or "",
            "REPORT_DATE",
            "NOTICE_DATE",
            "UPDATE_DATE",
            "SECUCODE",
            "SECURITY_CODE",
            "SECURITY_NAME_ABBR",
            "ORG_CODE",
            "ORG_TYPE",
            "REPORT_TYPE",
            "REPORT_DATE_NAME",
            "SECURITY_TYPE_CODE",
            "CURRENCY",
            "TRADE_MARKET_CODE",
            "TRADE_MARKET",
        },
        prefix="fdsrc_ak_em__",
    )
    out = pd.concat([out[["date", "code"]], num], axis=1)
    if "REPORT_DATE" in df.columns:
        rep = pd.to_datetime(df.loc[valid, "REPORT_DATE"], errors="coerce").dt.normalize().reindex(out.index)
        out["fdsrc_ak_em__report_lag_days"] = (out["date"] - rep).dt.days.astype("float32")
    return out.drop_duplicates(["date", "code"], keep="last").reset_index(drop=True)


def _load_ak_indicator_sina_file(
    path: Path,
    symbol_key: str,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    file_format: str,
) -> pd.DataFrame:
    df = _read_table_file(path, file_format=file_format)
    if df.empty:
        return pd.DataFrame()
    date_col = _choose_date_column(df, ["日期", "date", "REPORT_DATE", "report_date"])
    if date_col is None:
        return pd.DataFrame()
    out = df.copy()
    out["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    out["code"] = _normalize_code(symbol_key)
    out = out[(out["date"] >= start_date) & (out["date"] <= end_date)]
    out = out.dropna(subset=["date", "code"])
    if out.empty:
        return out
    num = _extract_numeric_columns(
        out,
        exclude_cols={"date", "code", date_col},
        prefix="fdsrc_ak_sina__",
    )
    out = pd.concat([out[["date", "code"]], num], axis=1)
    return out.drop_duplicates(["date", "code"], keep="last").reset_index(drop=True)


def _load_ak_abstract_sina_file(
    path: Path,
    symbol_key: str,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    file_format: str,
) -> pd.DataFrame:
    df = _read_table_file(path, file_format=file_format)
    if df.empty:
        return pd.DataFrame()
    date_cols = [c for c in df.columns if re.fullmatch(r"\d{8}", str(c))]
    if not date_cols:
        return pd.DataFrame()
    text_cols = [c for c in ["选项", "指标"] if c in df.columns]
    if not text_cols:
        text_cols = list(df.columns[:2])
    if not text_cols:
        return pd.DataFrame()
    long_df = df.melt(id_vars=text_cols, value_vars=date_cols, var_name="report_date", value_name="value")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["value"])
    if long_df.empty:
        return pd.DataFrame()
    metric = long_df[text_cols[0]].astype(str)
    if len(text_cols) > 1:
        metric = metric + "_" + long_df[text_cols[1]].astype(str)
    long_df["feature"] = metric.map(_safe_feature_name)
    pivot = long_df.pivot_table(index="report_date", columns="feature", values="value", aggfunc="last")
    if pivot.empty:
        return pd.DataFrame()
    out = pivot.reset_index()
    out["date"] = pd.to_datetime(out["report_date"], format="%Y%m%d", errors="coerce").dt.normalize()
    out["code"] = _normalize_code(symbol_key)
    out = out[(out["date"] >= start_date) & (out["date"] <= end_date)]
    out = out.dropna(subset=["date", "code"])
    if out.empty:
        return out
    rename = {c: f"fdsrc_ak_abs__{_safe_feature_name(c)}" for c in out.columns if c not in {"date", "code", "report_date"}}
    out = out.rename(columns=rename)
    keep_cols = ["date", "code", *sorted(rename.values())]
    out = out[keep_cols].copy()
    for c in keep_cols[2:]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    return out.drop_duplicates(["date", "code"], keep="last").reset_index(drop=True)


def _load_bsq_category_file(
    path: Path,
    symbol_key: str,
    category: str,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    file_format: str,
) -> pd.DataFrame:
    df = _read_table_file(path, file_format=file_format)
    if df.empty:
        return pd.DataFrame()
    date_candidates = ["pubDate", "statDate"]
    if str(category) == "forecast":
        date_candidates = ["profitForcastExpPubDate", "profitForcastExpStatDate", *date_candidates]
    if str(category) == "perf_express":
        date_candidates = ["performanceExpPubDate", "performanceExpUpdateDate", "performanceExpStatDate", *date_candidates]
    date_col = _choose_date_column(df, date_candidates)
    if date_col is None:
        return pd.DataFrame()
    out = df.copy()
    out["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    if "code" in out.columns:
        out["code"] = out["code"].astype(str).map(_normalize_code)
    else:
        out["code"] = _normalize_code(symbol_key)
    out = out[(out["date"] >= start_date) & (out["date"] <= end_date)]
    out = out.dropna(subset=["date", "code"])
    out = out[out["code"].ne("")]
    if out.empty:
        return out
    num = _extract_numeric_columns(
        out,
        exclude_cols={"date", "code", date_col, "pubDate", "statDate", "performanceExpPubDate", "performanceExpStatDate", "performanceExpUpdateDate", "profitForcastExpPubDate", "profitForcastExpStatDate"},
        prefix=f"fdsrc_bsq_{_safe_feature_name(category)}__",
    )
    out = pd.concat([out[["date", "code"]], num], axis=1)
    return out.drop_duplicates(["date", "code"], keep="last").reset_index(drop=True)


def _limited_append(samples: List[str], value: object, *, limit: int = 12) -> None:
    if len(samples) < max(int(limit), 1):
        samples.append(str(value))


def _format_market_load_diagnostics(diag: Dict[str, object]) -> str:
    parts: List[str] = []
    simple_keys = [
        "candidate_symbol_count",
        "filtered_symbol_count",
        "loaded_symbol_count",
        "skipped_symbol_count",
        "broken_symbol_count",
        "missing_daily_file_count",
        "missing_minute_file_count",
        "empty_daily_count",
        "empty_minute_count",
        "empty_after_tradestatus_filter_count",
        "empty_after_minute_clean_count",
        "read_error_count",
    ]
    for k in simple_keys:
        if k in diag:
            parts.append(f"{k}={diag[k]}")
    sample_keys = [
        "sample_missing_daily_symbols",
        "sample_missing_minute_symbols",
        "sample_empty_daily_symbols",
        "sample_empty_minute_symbols",
        "sample_read_errors",
    ]
    for k in sample_keys:
        vals = diag.get(k, [])
        if isinstance(vals, list) and vals:
            parts.append(f"{k}={vals}")
    if not parts:
        return "diagnostics unavailable"
    return "; ".join(parts)


def _merge_fundamental_asof(daily_df: pd.DataFrame, fundamental_events: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty or fundamental_events.empty:
        return daily_df.copy()
    left = daily_df.copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize()
    left["code"] = left["code"].astype(str).map(_normalize_code)
    left = left.dropna(subset=["date", "code"])
    left = left[left["code"].ne("")]
    if left.empty:
        return left
    left["__ord"] = np.arange(len(left), dtype=np.int64)
    left = left.sort_values(["code", "date", "__ord"]).reset_index(drop=True)

    right = fundamental_events.copy()
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.normalize()
    right["code"] = right["code"].astype(str).map(_normalize_code)
    right = right.dropna(subset=["date", "code"])
    right = right[right["code"].ne("")]
    right = right.sort_values(["code", "date"]).drop_duplicates(["code", "date"], keep="last").reset_index(drop=True)
    if right.empty:
        return daily_df.copy()

    value_cols = [c for c in right.columns if c not in {"date", "code"}]
    right_map: Dict[str, pd.DataFrame] = {
        str(code): grp.sort_values("date").reset_index(drop=True)
        for code, grp in right.groupby("code", sort=False)
    }

    parts: List[pd.DataFrame] = []
    for code, lgrp in left.groupby("code", sort=False):
        lg = lgrp.sort_values("date").copy()
        rg = right_map.get(str(code))
        if rg is None or rg.empty:
            if value_cols:
                na_block = pd.DataFrame(np.nan, index=lg.index, columns=value_cols)
                lg = pd.concat([lg, na_block], axis=1)
            parts.append(lg)
            continue
        merged = pd.merge_asof(
            left=lg,
            right=rg.drop(columns=["code"], errors="ignore"),
            on="date",
            direction="backward",
            allow_exact_matches=True,
        )
        merged["code"] = str(code)
        parts.append(merged)

    if not parts:
        return left.sort_values("__ord").drop(columns=["__ord"]).reset_index(drop=True)
    base_cols = [c for c in left.columns if c != "code"] + ["code"] + [c for c in value_cols if c not in left.columns]
    normalized_parts: List[pd.DataFrame] = []
    for p in parts:
        if p.empty:
            continue
        missing_cols = [c for c in base_cols if c not in p.columns]
        if missing_cols:
            p = pd.concat([p, pd.DataFrame(np.nan, index=p.index, columns=missing_cols)], axis=1)
        normalized_parts.append(p[base_cols])
    if not normalized_parts:
        return left.sort_values("__ord").drop(columns=["__ord"]).reset_index(drop=True)

    out = (
        pd.concat(normalized_parts, ignore_index=True)
        .sort_values("__ord")
        .drop(columns=["__ord"])
        .reset_index(drop=True)
    )
    return out


def _infer_fundamental_category(col_name: str) -> str:
    c = str(col_name).lower()
    for cat in FUND_CATEGORIES:
        if any(k in c for k in FUND_CATEGORY_KEYWORDS.get(cat, ())):
            return cat
    return ""


def _attach_fundamental_canonical_features(
    daily_df: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, str]]:
    out = daily_df.copy()
    source_cols = [c for c in out.columns if c.startswith("fdsrc_")]
    if not source_cols:
        return out, {}

    coverage = {c: int(pd.to_numeric(out[c], errors="coerce").notna().sum()) for c in source_cols}
    global_sorted = sorted(source_cols, key=lambda c: (-coverage.get(c, 0), c))
    cat_pool: Dict[str, List[str]] = {k: [] for k in FUND_CATEGORIES}
    for c in source_cols:
        cat = _infer_fundamental_category(c)
        if cat:
            cat_pool[cat].append(c)
    # fallback: ensure every category has enough material.
    for cat in FUND_CATEGORIES:
        ranked = sorted(cat_pool.get(cat, []), key=lambda c: (-coverage.get(c, 0), c))
        used = list(ranked[:FUND_CANONICAL_PER_CATEGORY])
        if len(used) < FUND_CANONICAL_PER_CATEGORY:
            for c in global_sorted:
                if c in used:
                    continue
                used.append(c)
                if len(used) >= FUND_CANONICAL_PER_CATEGORY:
                    break
        cat_pool[cat] = used

    out = out.sort_values(["code", "date"]).reset_index(drop=True)
    notes: Dict[str, str] = {}
    added_cols: Dict[str, pd.Series] = {}
    for cat in FUND_CATEGORIES:
        selected = cat_pool.get(cat, [])
        raw_cols: List[str] = []
        for i, src_col in enumerate(selected, start=1):
            c_new = f"fd_{cat}_raw_{i:02d}"
            added_cols[c_new] = pd.to_numeric(out[src_col], errors="coerce").astype("float32")
            raw_cols.append(c_new)
        if not raw_cols:
            continue
        raw_mat = pd.DataFrame({c: added_cols[c] for c in raw_cols}, index=out.index)
        score_col = f"fd_{cat}_score"
        trend_col = f"fd_{cat}_trend"
        disp_col = f"fd_{cat}_disp"
        score = raw_mat.mean(axis=1, skipna=True).astype("float32")
        added_cols[score_col] = score
        added_cols[disp_col] = raw_mat.std(axis=1, skipna=True).astype("float32")
        trend = (
            pd.DataFrame({"code": out["code"], "__score": score})
            .groupby("code", observed=True)["__score"]
            .pct_change(63, fill_method=None)
        )
        added_cols[trend_col] = trend.astype("float32")
        notes[f"canonical_{cat}"] = ",".join(selected[:5])  # keep notes concise
    if added_cols:
        out = pd.concat([out, pd.DataFrame(added_cols, index=out.index)], axis=1)
    return out, notes

@dataclass
class MarketUniverseDataLoader(MarketDataLoader):
    """Generic loader for stock-universe daily + 5min files."""

    data_cfg: DataConfig
    date_cfg: DateConfig
    lookback_days: int
    horizon: int
    factor_freq: str = "D"

    def _load_market_frames(
        self,
        data_root: Path,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        file_format: str = "auto",
        max_files: Optional[int] = None,
        stock_list_path: Optional[Path] = None,
        main_board_only: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
        daily_dir = data_root / "d"
        minute_dir = data_root / "5"
        if not daily_dir.exists() or not minute_dir.exists():
            raise FileNotFoundError(f"data directories missing: {daily_dir} or {minute_dir}")

        keys_all = list_symbol_keys(daily_dir, file_format=file_format)
        keys = keys_all
        log_progress(
            f"scanned market files: daily_dir={daily_dir}, minute_dir={minute_dir}, total_symbols={len(keys_all)}",
            module="loader",
            level="debug",
        )
        if stock_list_path is not None:
            listed_keys = set(load_stock_list_keys(stock_list_path))
            keys = [k for k in keys if k in listed_keys]
            log_progress(f"after stock-list filter: symbols={len(keys)}", module="loader", level="debug")
        if main_board_only:
            keys = [k for k in keys if is_main_board_symbol(k)]
            log_progress(f"after main-board filter: symbols={len(keys)}", module="loader", level="debug")
        target_loaded: Optional[int] = None
        if max_files is not None:
            target_loaded = int(max_files)
            log_progress(
                f"max-files target (effective loaded symbols): {target_loaded}",
                module="loader",
                level="debug",
            )

        daily_cols = [
            "date",
            "code",
            "open",
            "high",
            "low",
            "close",
            "preclose",
            "volume",
            "amount",
            "turn",
            "tradestatus",
        ]
        minute_cols = ["date", "time", "code", "open", "high", "low", "close", "volume", "amount"]

        daily_frames: List[pd.DataFrame] = []
        minute_frames: List[pd.DataFrame] = []
        loaded = 0
        skipped = 0
        broken = 0
        missing_daily = 0
        missing_minute = 0
        empty_daily = 0
        empty_minute = 0
        empty_after_tradestatus = 0
        empty_after_minute_clean = 0
        read_errors = 0
        sample_missing_daily: List[str] = []
        sample_missing_minute: List[str] = []
        sample_empty_daily: List[str] = []
        sample_empty_minute: List[str] = []
        sample_read_errors: List[str] = []
        for idx, key in enumerate(keys, start=1):
            if target_loaded is not None and loaded >= target_loaded:
                break
            daily_path = pick_existing_file(daily_dir, key, "d", file_format=file_format)
            minute_path = pick_existing_file(minute_dir, key, "5", file_format=file_format)
            if daily_path is None:
                missing_daily += 1
                _limited_append(sample_missing_daily, key)
            if minute_path is None:
                missing_minute += 1
                _limited_append(sample_missing_minute, key)
            if daily_path is None or minute_path is None:
                skipped += 1
                continue

            try:
                ddf = read_data_file(daily_path, daily_cols, start_date, end_date)
                mdf = read_data_file(minute_path, minute_cols, start_date, end_date)
            except Exception as exc:
                broken += 1
                read_errors += 1
                _limited_append(sample_read_errors, f"{key}::{type(exc).__name__}: {exc}")
                continue
            if ddf.empty:
                empty_daily += 1
                _limited_append(sample_empty_daily, key)
                skipped += 1
                continue
            if mdf.empty:
                empty_minute += 1
                _limited_append(sample_empty_minute, key)
                skipped += 1
                continue

            ddf["code"] = ddf["code"].astype(str).str.strip()
            mdf["code"] = mdf["code"].astype(str).str.strip()
            for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "tradestatus"]:
                if col in ddf.columns:
                    ddf[col] = pd.to_numeric(ddf[col], errors="coerce")
            for col in ["time", "open", "high", "low", "close", "volume", "amount"]:
                if col in mdf.columns:
                    mdf[col] = pd.to_numeric(mdf[col], errors="coerce")

            if "tradestatus" in ddf.columns:
                ddf = ddf[ddf["tradestatus"].fillna(1) == 1].copy()
            if ddf.empty:
                empty_after_tradestatus += 1
                _limited_append(sample_empty_daily, f"{key}::tradestatus_filter")
                skipped += 1
                continue

            if "time" in mdf.columns:
                tstr = mdf["time"].astype("Int64").astype(str).str.zfill(17).str.slice(0, 14)
                mdf["datetime"] = pd.to_datetime(tstr, format="%Y%m%d%H%M%S", errors="coerce")
                mdf["datetime"] = mdf["datetime"].fillna(pd.to_datetime(mdf["date"], errors="coerce"))
            else:
                mdf["datetime"] = pd.to_datetime(mdf["date"], errors="coerce")
            mdf["date"] = pd.to_datetime(mdf["date"], errors="coerce").dt.normalize()
            mdf = mdf.dropna(subset=["date", "datetime", "code"]).copy()
            if mdf.empty:
                empty_after_minute_clean += 1
                _limited_append(sample_empty_minute, f"{key}::minute_datetime_clean")
                skipped += 1
                continue

            daily_frames.append(ddf)
            minute_frames.append(mdf)
            loaded += 1
            if loaded % 50 == 0:
                target_desc = str(target_loaded) if target_loaded is not None else str(len(keys))
                log_progress(
                    f"market file progress: loaded={loaded}/{target_desc}, "
                    f"scanned={idx}/{len(keys)}, skipped={skipped}, broken={broken}",
                    module="loader",
                    level="debug",
                )

        diagnostics: Dict[str, object] = {
            "data_root": str(data_root),
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
            "file_format": str(file_format),
            "candidate_symbol_count": int(len(keys_all)),
            "filtered_symbol_count": int(len(keys)),
            "loaded_symbol_count": int(loaded),
            "skipped_symbol_count": int(skipped),
            "broken_symbol_count": int(broken),
            "missing_daily_file_count": int(missing_daily),
            "missing_minute_file_count": int(missing_minute),
            "empty_daily_count": int(empty_daily),
            "empty_minute_count": int(empty_minute),
            "empty_after_tradestatus_filter_count": int(empty_after_tradestatus),
            "empty_after_minute_clean_count": int(empty_after_minute_clean),
            "read_error_count": int(read_errors),
            "sample_missing_daily_symbols": sample_missing_daily,
            "sample_missing_minute_symbols": sample_missing_minute,
            "sample_empty_daily_symbols": sample_empty_daily,
            "sample_empty_minute_symbols": sample_empty_minute,
            "sample_read_errors": sample_read_errors,
        }
        if not daily_frames:
            raise RuntimeError("no daily market data loaded. " + _format_market_load_diagnostics(diagnostics))
        if not minute_frames:
            raise RuntimeError("no minute market data loaded. " + _format_market_load_diagnostics(diagnostics))
        log_progress(
            f"market file loading done: loaded={loaded}, skipped={skipped}, broken={broken}, "
            f"daily_parts={len(daily_frames)}, minute_parts={len(minute_frames)}",
            module="loader",
        )

        daily_df = pd.concat(daily_frames, ignore_index=True).sort_values(["code", "date"]).reset_index(drop=True)
        minute_df = pd.concat(minute_frames, ignore_index=True).sort_values(["code", "datetime"]).reset_index(drop=True)
        daily_df = daily_df.drop_duplicates(["code", "date"], keep="last").reset_index(drop=True)
        minute_df = minute_df.drop_duplicates(["code", "datetime"], keep="last").reset_index(drop=True)
        diagnostics["daily_rows"] = int(len(daily_df))
        diagnostics["minute_rows"] = int(len(minute_df))
        diagnostics["daily_code_count"] = int(daily_df["code"].astype(str).nunique()) if "code" in daily_df.columns else 0
        diagnostics["minute_code_count"] = int(minute_df["code"].astype(str).nunique()) if "code" in minute_df.columns else 0
        return daily_df, minute_df, diagnostics

    def _load_fundamental_events(
        self,
        *,
        code_universe: Sequence[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> tuple[pd.DataFrame, Dict[str, str]]:
        if not bool(self.data_cfg.enable_fundamental_data):
            return pd.DataFrame(), {"fundamental_enabled": "false"}

        keys = sorted({_normalize_code(c, dotted=False) for c in code_universe if _normalize_code(c, dotted=False)})
        notes: Dict[str, str] = {
            "fundamental_enabled": "true",
            "fundamental_keys": str(len(keys)),
            "fundamental_root_ak": str(self.data_cfg.fundamental_root_ak),
            "fundamental_root_bsq": str(self.data_cfg.fundamental_root_bsq),
        }
        if not keys:
            return pd.DataFrame(), notes

        file_format = str(self.data_cfg.fundamental_file_format or "auto")
        frames: List[pd.DataFrame] = []
        loaded_files = 0
        skipped_files = 0

        ak_root = Path(self.data_cfg.fundamental_root_ak).expanduser()
        ak_dataset_cfg: List[tuple[str, str]] = [
            ("financial_indicator_em", "em"),
            ("financial_indicator_sina", "sina"),
            ("financial_abstract_sina", "abs"),
        ]
        for ds_name, ds_tag in ak_dataset_cfg:
            ds_root = ak_root / ds_name
            uni_dir = _select_fundamental_universe_dir(ds_root, self.data_cfg.universe)
            if uni_dir is None:
                notes[f"ak_{ds_tag}_status"] = "missing_dir"
                continue
            local_loaded = 0
            local_rows = 0
            for key in keys:
                stem = f"{key}_report" if ds_name == "financial_indicator_em" else key
                fp = _pick_stem_file(uni_dir, stem, file_format=file_format)
                if fp is None:
                    skipped_files += 1
                    continue
                try:
                    if ds_name == "financial_indicator_em":
                        part = _load_ak_indicator_em_file(
                            fp,
                            key,
                            start_date=start_date,
                            end_date=end_date,
                            file_format=file_format,
                        )
                    elif ds_name == "financial_indicator_sina":
                        part = _load_ak_indicator_sina_file(
                            fp,
                            key,
                            start_date=start_date,
                            end_date=end_date,
                            file_format=file_format,
                        )
                    else:
                        part = _load_ak_abstract_sina_file(
                            fp,
                            key,
                            start_date=start_date,
                            end_date=end_date,
                            file_format=file_format,
                        )
                except Exception:
                    skipped_files += 1
                    continue
                if part.empty:
                    skipped_files += 1
                    continue
                frames.append(part)
                local_loaded += 1
                local_rows += int(len(part))
            loaded_files += local_loaded
            notes[f"ak_{ds_tag}_loaded_files"] = str(local_loaded)
            notes[f"ak_{ds_tag}_rows"] = str(local_rows)

        bsq_root = Path(self.data_cfg.fundamental_root_bsq).expanduser()
        bsq_uni_dir = _select_fundamental_universe_dir(bsq_root, self.data_cfg.universe)
        if bsq_uni_dir is not None:
            cat_dirs = sorted([p for p in bsq_uni_dir.iterdir() if p.is_dir()])
            for cat_dir in cat_dirs:
                cat = cat_dir.name
                local_loaded = 0
                local_rows = 0
                for key in keys:
                    fp = _pick_stem_file(cat_dir, key, file_format=file_format)
                    if fp is None:
                        skipped_files += 1
                        continue
                    try:
                        part = _load_bsq_category_file(
                            fp,
                            key,
                            category=cat,
                            start_date=start_date,
                            end_date=end_date,
                            file_format=file_format,
                        )
                    except Exception:
                        skipped_files += 1
                        continue
                    if part.empty:
                        skipped_files += 1
                        continue
                    frames.append(part)
                    local_loaded += 1
                    local_rows += int(len(part))
                loaded_files += local_loaded
                notes[f"bsq_{cat}_loaded_files"] = str(local_loaded)
                notes[f"bsq_{cat}_rows"] = str(local_rows)
        else:
            notes["bsq_status"] = "missing_dir"

        if not frames:
            notes["fundamental_loaded_files"] = str(loaded_files)
            notes["fundamental_skipped_files"] = str(skipped_files)
            notes["fundamental_event_rows"] = "0"
            return pd.DataFrame(), notes

        fund = pd.concat(frames, ignore_index=True)
        fund["date"] = pd.to_datetime(fund["date"], errors="coerce").dt.normalize()
        fund["code"] = fund["code"].astype(str).map(_normalize_code)
        fund = fund.dropna(subset=["date", "code"])
        fund = fund[fund["code"].ne("")]
        fund = fund[(fund["date"] >= start_date) & (fund["date"] <= end_date)]
        if fund.empty:
            notes["fundamental_loaded_files"] = str(loaded_files)
            notes["fundamental_skipped_files"] = str(skipped_files)
            notes["fundamental_event_rows"] = "0"
            return pd.DataFrame(), notes

        fund = (
            fund.sort_values(["code", "date"])
            .groupby(["code", "date"], as_index=False)
            .last()
            .reset_index(drop=True)
        )
        notes["fundamental_loaded_files"] = str(loaded_files)
        notes["fundamental_skipped_files"] = str(skipped_files)
        notes["fundamental_event_rows"] = str(len(fund))
        notes["fundamental_event_cols"] = str(len(fund.columns))
        return fund, notes

    def _attach_fundamental_to_daily(
        self,
        daily_df: pd.DataFrame,
        *,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> tuple[pd.DataFrame, Dict[str, str]]:
        fund_events, notes = self._load_fundamental_events(
            code_universe=daily_df["code"].astype(str).unique().tolist(),
            start_date=start_date,
            end_date=end_date,
        )
        if fund_events.empty:
            return daily_df.copy(), notes
        merged = _merge_fundamental_asof(daily_df, fund_events)
        merged, canonical_notes = _attach_fundamental_canonical_features(merged)
        notes["fundamental_merged_cols"] = str(
            len([c for c in merged.columns if c.startswith("fdsrc_") or c.startswith("fd_")])
        )
        notes.update(canonical_notes)
        return merged, notes

    def _load_text_daily_panel(
        self,
        *,
        code_universe: Sequence[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> tuple[pd.DataFrame, Dict[str, str]]:
        if not bool(getattr(self.data_cfg, "enable_text_data", False)):
            return pd.DataFrame(), {"text_enabled": "false"}

        keys = sorted({_normalize_code(c, dotted=False) for c in code_universe if _normalize_code(c, dotted=False)})
        notes: Dict[str, str] = {
            "text_enabled": "true",
            "text_keys": str(len(keys)),
            "text_root_news": str(getattr(self.data_cfg, "text_root_news", "")),
            "text_root_notice": str(getattr(self.data_cfg, "text_root_notice", "")),
            "text_root_report_em": str(getattr(self.data_cfg, "text_root_report_em", "")),
            "text_root_report_iwencai": str(getattr(self.data_cfg, "text_root_report_iwencai", "")),
        }
        if not keys:
            return pd.DataFrame(), notes

        source_roots: List[tuple[str, Path]] = [
            ("news", Path(str(getattr(self.data_cfg, "text_root_news", ""))).expanduser()),
            ("notice", Path(str(getattr(self.data_cfg, "text_root_notice", ""))).expanduser()),
            ("em_report", Path(str(getattr(self.data_cfg, "text_root_report_em", ""))).expanduser()),
            ("iwencai", Path(str(getattr(self.data_cfg, "text_root_report_iwencai", ""))).expanduser()),
        ]
        file_format = str(getattr(self.data_cfg, "text_file_format", "auto") or "auto")

        loaded_files = 0
        skipped_files = 0
        events_frames: List[pd.DataFrame] = []
        for source, root in source_roots:
            if not str(root).strip():
                notes[f"text_{source}_status"] = "empty_root"
                continue
            uni_dirs = select_text_universe_dirs(root, self.data_cfg.universe)
            if not uni_dirs:
                notes[f"text_{source}_status"] = "missing_dir"
                continue

            local_loaded = 0
            local_rows = 0
            for key in keys:
                fp = pick_symbol_text_file(uni_dirs, key, file_format=file_format)
                if fp is None:
                    skipped_files += 1
                    continue
                try:
                    part = load_text_source_events_from_file(
                        fp,
                        symbol_key=key,
                        source=source,
                        start_date=start_date,
                        end_date=end_date,
                        file_format=file_format,
                    )
                except Exception:
                    skipped_files += 1
                    continue
                if part.empty:
                    skipped_files += 1
                    continue
                events_frames.append(part)
                local_loaded += 1
                local_rows += int(len(part))
            loaded_files += local_loaded
            notes[f"text_{source}_loaded_files"] = str(local_loaded)
            notes[f"text_{source}_event_rows"] = str(local_rows)

        if not events_frames:
            notes["text_loaded_files"] = str(loaded_files)
            notes["text_skipped_files"] = str(skipped_files)
            notes["text_event_rows"] = "0"
            notes["text_daily_rows"] = "0"
            return pd.DataFrame(), notes

        events = pd.concat(events_frames, ignore_index=True)
        events["date"] = pd.to_datetime(events["date"], errors="coerce").dt.normalize()
        events["code"] = events["code"].astype(str).map(_normalize_code)
        events = events.dropna(subset=["date", "code"])
        events = events[events["code"].ne("")]
        events = events[(events["date"] >= start_date) & (events["date"] <= end_date)]
        if events.empty:
            notes["text_loaded_files"] = str(loaded_files)
            notes["text_skipped_files"] = str(skipped_files)
            notes["text_event_rows"] = "0"
            notes["text_daily_rows"] = "0"
            return pd.DataFrame(), notes

        daily_text = build_text_daily_features(events)
        notes["text_loaded_files"] = str(loaded_files)
        notes["text_skipped_files"] = str(skipped_files)
        notes["text_event_rows"] = str(len(events))
        notes["text_daily_rows"] = str(len(daily_text))
        notes["text_daily_cols"] = str(len(daily_text.columns))
        return daily_text, notes

    def _attach_text_to_daily(
        self,
        daily_df: pd.DataFrame,
        *,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> tuple[pd.DataFrame, Dict[str, str]]:
        text_daily, notes = self._load_text_daily_panel(
            code_universe=daily_df["code"].astype(str).unique().tolist(),
            start_date=start_date,
            end_date=end_date,
        )
        if text_daily.empty:
            return daily_df.copy(), notes

        left = daily_df.copy()
        left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize()
        left["code"] = left["code"].astype(str).map(_normalize_code)
        left = left.dropna(subset=["date", "code"])
        left = left[left["code"].ne("")]

        text_daily = text_daily.copy()
        text_daily["date"] = pd.to_datetime(text_daily["date"], errors="coerce").dt.normalize()
        text_daily["code"] = text_daily["code"].astype(str).map(_normalize_code)
        text_daily = text_daily.dropna(subset=["date", "code"])
        text_daily = text_daily[text_daily["code"].ne("")]
        text_daily = (
            text_daily.sort_values(["code", "date"])
            .groupby(["code", "date"], as_index=False)
            .last()
            .reset_index(drop=True)
        )
        if text_daily.empty:
            return daily_df.copy(), notes

        merged = left.merge(text_daily, on=["date", "code"], how="left")
        notes["text_merged_cols"] = str(len([c for c in merged.columns if str(c).startswith("txt_")]))
        return merged, notes

    def load(self) -> MarketBundle:
        load_start, load_end = compute_load_window(
            train_start=self.date_cfg.train_start,
            test_end=self.date_cfg.test_end,
            lookback_days=int(self.lookback_days),
            horizon=int(self.horizon),
            factor_freq=str(self.factor_freq),
        )
        log_progress(
            f"computed load window: start={load_start.date()}, end={load_end.date()}, "
            f"lookback_days={self.lookback_days}, horizon={self.horizon}, factor_freq={self.factor_freq}",
            module="loader",
        )
        daily_df, minute_df, market_diag = self._load_market_frames(
            data_root=Path(self.data_cfg.data_root),
            start_date=load_start,
            end_date=load_end,
            file_format=self.data_cfg.file_format,
            max_files=self.data_cfg.max_files,
            stock_list_path=Path(self.data_cfg.stock_list_path) if self.data_cfg.stock_list_path else None,
            main_board_only=self.data_cfg.main_board_only,
        )
        daily_df, fund_notes = self._attach_fundamental_to_daily(
            daily_df,
            start_date=load_start,
            end_date=load_end,
        )
        daily_df, text_notes = self._attach_text_to_daily(
            daily_df,
            start_date=load_start,
            end_date=load_end,
        )
        codes = sorted(daily_df["code"].astype(str).drop_duplicates().tolist())
        notes = {
            "data_root": self.data_cfg.data_root,
            "universe": self.data_cfg.universe,
            "stock_list_path": str(self.data_cfg.stock_list_path or ""),
            "main_board_only": str(self.data_cfg.main_board_only),
            "file_format": self.data_cfg.file_format,
        }
        for k, v in market_diag.items():
            if isinstance(v, list):
                notes[f"market_{k}"] = "|".join([str(x) for x in v[:12]])
            else:
                notes[f"market_{k}"] = str(v)
        notes.update({k: str(v) for k, v in fund_notes.items()})
        notes.update({k: str(v) for k, v in text_notes.items()})
        log_progress(
            f"market bundle ready: daily_rows={len(daily_df)}, minute_rows={len(minute_df)}, codes={len(codes)}",
            module="loader",
        )
        return MarketBundle(
            daily=daily_df,
            minute5=minute_df,
            start_date=load_start,
            end_date=load_end,
            codes=codes,
            source_notes=notes,
        )


HS300MarketDataLoader = MarketUniverseDataLoader


def build_minute_daily_features(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate minute bars to daily micro-structure and executable prices."""
    m = minute_df.copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.normalize()
    m["datetime"] = pd.to_datetime(m["datetime"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors="coerce")
    m = m.dropna(subset=["date", "datetime", "code", "open", "close", "volume"]).copy()
    m = m.sort_values(["code", "date", "datetime"])
    grp_cols = ["code", "date"]

    m["bar_idx"] = m.groupby(grp_cols).cumcount()
    m["bar_rev_idx"] = m.groupby(grp_cols).cumcount(ascending=False)
    m["ret_5m"] = m.groupby(grp_cols)["close"].pct_change()
    m["pv"] = m["close"] * m["volume"]
    m["signed_vol"] = np.sign(m["ret_5m"].fillna(0.0)) * m["volume"]
    m["abs_ret_5m"] = m["ret_5m"].abs()

    base = m.groupby(grp_cols, as_index=False).agg(
        first_open_5m=("open", "first"),
        first_close_5m=("close", "first"),
        last_close_5m=("close", "last"),
        day_vol=("volume", "sum"),
        day_pv=("pv", "sum"),
        minute_realized_vol_5m=("ret_5m", "std"),
        minute_up_ratio_5m=("ret_5m", lambda x: float((x > 0).mean()) if x.notna().any() else np.nan),
        minute_ret_skew_5m=("ret_5m", lambda x: float(x.skew()) if x.notna().sum() > 2 else np.nan),
        minute_ret_kurt_5m=("ret_5m", lambda x: float(x.kurt()) if x.notna().sum() > 3 else np.nan),
        signed_vol_sum=("signed_vol", "sum"),
        abs_ret_max=("abs_ret_5m", "max"),
    )
    first30 = m[m["bar_idx"] < 6].groupby(grp_cols, as_index=False).agg(pv_first30=("pv", "sum"), vol_first30=("volume", "sum"), close_30m=("close", "last"))
    first60 = m[m["bar_idx"] < 12].groupby(grp_cols, as_index=False).agg(pv_first60=("pv", "sum"), vol_first60=("volume", "sum"))
    last30 = m[m["bar_rev_idx"] < 6].groupby(grp_cols, as_index=False).agg(twap_last30=("close", "mean"), close_last30_start=("close", "first"), close_last30_end=("close", "last"))

    feat = base.merge(first30, on=grp_cols, how="left").merge(first60, on=grp_cols, how="left").merge(last30, on=grp_cols, how="left")
    feat["vwap_day"] = feat["day_pv"] / (feat["day_vol"] + EPS)
    feat["vwap_first30"] = feat["pv_first30"] / (feat["vol_first30"] + EPS)
    feat["vwap_first60"] = feat["pv_first60"] / (feat["vol_first60"] + EPS)
    feat["signed_vol_imbalance_5m"] = feat["signed_vol_sum"] / (feat["day_vol"] + EPS)
    feat["jump_ratio_5m"] = feat["abs_ret_max"] / (feat["minute_realized_vol_5m"] + EPS)
    feat["open_to_close_intraday"] = feat["last_close_5m"] / (feat["first_open_5m"] + EPS) - 1.0
    feat["morning_momentum_30m"] = feat["close_30m"] / (feat["first_open_5m"] + EPS) - 1.0
    feat["last30_momentum"] = feat["close_last30_end"] / (feat["close_last30_start"] + EPS) - 1.0
    feat["vwap30_vs_day"] = feat["vwap_first30"] / (feat["vwap_day"] + EPS) - 1.0

    feat["px_open5"] = feat["first_open_5m"]
    feat["px_vwap30"] = feat["vwap_first30"]
    feat["px_twap_last30"] = feat["twap_last30"]

    keep_cols = [
        "code",
        "date",
        "vwap_day",
        "vwap_first30",
        "vwap_first60",
        "minute_realized_vol_5m",
        "minute_up_ratio_5m",
        "minute_ret_skew_5m",
        "minute_ret_kurt_5m",
        "signed_vol_imbalance_5m",
        "jump_ratio_5m",
        "morning_momentum_30m",
        "last30_momentum",
        "open_to_close_intraday",
        "vwap30_vs_day",
        "px_open5",
        "px_vwap30",
        "px_twap_last30",
    ]
    return feat[keep_cols]


def _rolling_mean(g: pd.core.groupby.generic.SeriesGroupBy, window: int) -> pd.Series:
    return g.transform(lambda s: s.rolling(window, min_periods=window).mean())


def _rolling_std(g: pd.core.groupby.generic.SeriesGroupBy, window: int) -> pd.Series:
    return g.transform(lambda s: s.rolling(window, min_periods=window).std())


def _get_numeric_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _add_fundamental_hf_fusion_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for cat in FUND_CATEGORIES:
        score_col = f"fd_{cat}_score"
        trend_col = f"fd_{cat}_trend"
        disp_col = f"fd_{cat}_disp"
        if score_col not in out.columns:
            continue
        score = _get_numeric_or_nan(out, score_col)
        trend = _get_numeric_or_nan(out, trend_col)
        disp = _get_numeric_or_nan(out, disp_col)
        signed_flow = _get_numeric_or_nan(out, "signed_vol_imbalance_5m")
        jump = _get_numeric_or_nan(out, "jump_ratio_5m")
        intraday_trend = _get_numeric_or_nan(out, "open_to_close_intraday")
        vwap_bias = _get_numeric_or_nan(out, "close_to_vwap_day")
        minute_up = _get_numeric_or_nan(out, "minute_up_ratio_5m")
        minute_skew = _get_numeric_or_nan(out, "minute_ret_skew_5m")

        out[f"fd_hf_{cat}_flow"] = (score * signed_flow).astype("float32")
        out[f"fd_hf_{cat}_jump"] = (-score * jump).astype("float32")
        out[f"fd_hf_{cat}_intraday"] = (trend * intraday_trend).astype("float32")
        out[f"fd_hf_{cat}_vwap"] = (score * vwap_bias).astype("float32")
        out[f"fd_hf_{cat}_sentiment"] = ((score - disp) * (minute_up + minute_skew)).astype("float32")

    fusion_cols = [c for c in out.columns if c.startswith("fd_hf_")]
    if fusion_cols:
        out["fd_hf_fusion_score"] = out[fusion_cols].mean(axis=1, skipna=True).astype("float32")
        out["fd_hf_fusion_disp"] = out[fusion_cols].std(axis=1, skipna=True).astype("float32")
    return out


def build_daily_feature_base(daily_df: pd.DataFrame, minute_daily_feat: pd.DataFrame) -> pd.DataFrame:
    """Build robust daily base features for factor library and portfolio models.

    The output acts as the shared feature substrate for:
    - factor computation (default/custom/catalog factors)
    - timing and portfolio dynamic features
    - execution-related liquidity proxies
    """
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "code", "open", "high", "low", "close", "volume"]).sort_values(["code", "date"])
    for col in [c for c in df.columns if c.startswith("fdsrc_") or c.startswith("fd_")]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    if "preclose" not in df.columns:
        df["preclose"] = np.nan
    df["preclose"] = df["preclose"].where(df["preclose"].notna(), df.groupby("code")["close"].shift(1))
    if "turn" not in df.columns:
        df["turn"] = np.nan

    # Merge minute-derived daily micro-structure features/prices.
    df = df.merge(minute_daily_feat, on=["code", "date"], how="left")
    g = df.groupby("code")
    df["ret_1d"] = g["close"].pct_change(1)
    df["ret_3d"] = g["close"].pct_change(3)
    df["ret_5d"] = g["close"].pct_change(5)
    df["ret_10d"] = g["close"].pct_change(10)
    df["ret_20d"] = g["close"].pct_change(20)
    df["vol_chg_1d"] = g["volume"].pct_change(1)

    df["ma5"] = _rolling_mean(g["close"], 5)
    df["ma10"] = _rolling_mean(g["close"], 10)
    df["ma20"] = _rolling_mean(g["close"], 20)
    df["roll_high_20"] = g["high"].transform(lambda s: s.rolling(20, min_periods=20).max())
    df["roll_low_20"] = g["low"].transform(lambda s: s.rolling(20, min_periods=20).min())
    df["vol_ma5"] = _rolling_mean(g["volume"], 5)
    df["vol_ma20"] = _rolling_mean(g["volume"], 20)
    df["amount_ma20"] = _rolling_mean(g["amount"], 20)
    df["turn_ma5"] = _rolling_mean(g["turn"], 5)

    df["ma_gap_5"] = df["close"] / (df["ma5"] + EPS) - 1.0
    df["ma_gap_10"] = df["close"] / (df["ma10"] + EPS) - 1.0
    df["ma_gap_20"] = df["close"] / (df["ma20"] + EPS) - 1.0
    df["ma_cross_5_20"] = df["ma5"] / (df["ma20"] + EPS) - 1.0
    df["breakout_20"] = df["close"] / (df["roll_high_20"] + EPS) - 1.0
    df["vol_ratio_5"] = df["volume"] / (df["vol_ma5"] + EPS)
    df["vol_ratio_20"] = df["volume"] / (df["vol_ma20"] + EPS)
    df["amount_ratio_20"] = df["amount"] / (df["amount_ma20"] + EPS)
    df["turn_ratio_5"] = df["turn"] / (df["turn_ma5"] + EPS)
    df["intraday_range"] = (df["high"] - df["low"]) / (df["preclose"] + EPS)

    hl = df["high"] - df["low"]
    df["body_ratio"] = (df["close"] - df["open"]) / (hl + EPS)
    df["close_pos"] = (df["close"] - df["low"]) / (hl + EPS)
    prev_close = g["close"].shift(1)
    tr = np.maximum.reduce([(df["high"] - df["low"]).to_numpy(), (df["high"] - prev_close).abs().to_numpy(), (df["low"] - prev_close).abs().to_numpy()])
    df["tr"] = tr
    df["atr14"] = g["tr"].transform(lambda s: s.rolling(14, min_periods=14).mean())
    df["atr_norm_14"] = df["atr14"] / (df["close"] + EPS)
    df["realized_vol_20"] = _rolling_std(g["ret_1d"], 20)
    df["ret_neg_1d"] = df["ret_1d"].where(df["ret_1d"] < 0.0, 0.0)
    df["downside_vol_20"] = df.groupby("code")["ret_neg_1d"].transform(lambda s: s.rolling(20, min_periods=20).std())
    df["downside_vol_ratio_20"] = df["downside_vol_20"] / (df["realized_vol_20"] + EPS)

    delta = g["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.groupby(df["code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    avg_loss = loss.groupby(df["code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    rs = avg_gain / (avg_loss + EPS)
    df["rsi14"] = 100.0 - (100.0 / (1.0 + rs))

    df["amihud_1d"] = np.abs(df["ret_1d"]) / (df["amount"] + EPS)
    df["amihud_20"] = df.groupby("code")["amihud_1d"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    # Rolling correlation via rolling moments.
    # This avoids pandas rolling-corr pairwise/index edge cases and always returns 1D aligned output.
    by_code = df["code"]
    mean_r = df["ret_1d"].groupby(by_code).transform(lambda s: s.rolling(20, min_periods=20).mean())
    mean_v = df["vol_chg_1d"].groupby(by_code).transform(lambda s: s.rolling(20, min_periods=20).mean())
    mean_rv = (df["ret_1d"] * df["vol_chg_1d"]).groupby(by_code).transform(lambda s: s.rolling(20, min_periods=20).mean())
    mean_r2 = (df["ret_1d"] * df["ret_1d"]).groupby(by_code).transform(lambda s: s.rolling(20, min_periods=20).mean())
    mean_v2 = (df["vol_chg_1d"] * df["vol_chg_1d"]).groupby(by_code).transform(lambda s: s.rolling(20, min_periods=20).mean())
    cov_rv = mean_rv - mean_r * mean_v
    std_r = np.sqrt(np.clip(mean_r2 - mean_r * mean_r, a_min=0.0, a_max=None))
    std_v = np.sqrt(np.clip(mean_v2 - mean_v * mean_v, a_min=0.0, a_max=None))
    df["ret_vol_corr_20"] = cov_rv / (std_r * std_v + EPS)
    df["close_to_vwap_day"] = df["close"] / (df["vwap_day"] + EPS) - 1.0
    df["overnight_gap"] = df["open"] / (df["preclose"] + EPS) - 1.0
    df["px_daily_close"] = df["close"]

    # Portfolio/timing state features (style + crowding proxies).
    df["barra_size_proxy"] = np.log(df["amount_ma20"].clip(lower=0.0) + 1.0)
    df["barra_momentum_proxy"] = df["ret_20d"]
    df["barra_volatility_proxy"] = df["realized_vol_20"]
    df["barra_liquidity_proxy"] = -df["amihud_20"]
    df["barra_beta_proxy"] = df["ret_vol_corr_20"]
    df["crowding_proxy_raw"] = 0.45 * df["vol_ratio_20"].abs() + 0.35 * df["turn_ratio_5"].abs() + 0.20 * df["ret_vol_corr_20"].abs()
    df["board_type"] = df["code"].astype(str).map(infer_board_type)
    df["industry_bucket"] = df["code"].astype(str).map(infer_industry_bucket)
    df = _add_fundamental_hf_fusion_features(df)
    df = add_text_rolling_and_fusion_features(df)
    return df


def _merge_daily_context_into_panel(panel: pd.DataFrame, daily_base: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Attach selected daily-context fields onto non-daily panels by [date, code]."""
    if panel.empty:
        return panel.copy()

    out = panel.copy()
    if "code" not in out.columns:
        return out
    out["code"] = out["code"].astype(str).str.strip()

    if "date" not in out.columns:
        if time_col == "datetime" and "datetime" in out.columns:
            out["date"] = pd.to_datetime(out["datetime"], errors="coerce").dt.normalize()
        else:
            return out
    else:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    ctx_candidates = [
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "vol_chg_1d",
        "ma_gap_5",
        "ma_gap_10",
        "ma_gap_20",
        "ma_cross_5_20",
        "breakout_20",
        "vol_ratio_5",
        "vol_ratio_20",
        "amount_ratio_20",
        "turn_ratio_5",
        "amount_ma20",
        "turn_ma5",
        "atr_norm_14",
        "realized_vol_20",
        "downside_vol_ratio_20",
        "amihud_20",
        "ret_vol_corr_20",
        "close_to_vwap_day",
        "morning_momentum_30m",
        "last30_momentum",
        "open_to_close_intraday",
        "vwap30_vs_day",
        "minute_up_ratio_5m",
        "minute_ret_skew_5m",
        "minute_ret_kurt_5m",
        "signed_vol_imbalance_5m",
        "jump_ratio_5m",
        "px_open5",
        "px_vwap30",
        "px_twap_last30",
        "px_daily_close",
        "barra_size_proxy",
        "barra_momentum_proxy",
        "barra_volatility_proxy",
        "barra_liquidity_proxy",
        "barra_beta_proxy",
        "crowding_proxy_raw",
        "industry_bucket",
        "board_type",
    ]
    fund_ctx: List[str] = []
    for cat in FUND_CATEGORIES:
        for i in range(1, FUND_INTRADAY_CONTEXT_RAW_COUNT + 1):
            fund_ctx.append(f"fd_{cat}_raw_{i:02d}")
        fund_ctx.extend([f"fd_{cat}_score", f"fd_{cat}_trend", f"fd_{cat}_disp"])
        fund_ctx.extend([f"fd_hf_{cat}_flow", f"fd_hf_{cat}_jump", f"fd_hf_{cat}_intraday", f"fd_hf_{cat}_vwap", f"fd_hf_{cat}_sentiment"])
    fund_ctx.extend(["fd_hf_fusion_score", "fd_hf_fusion_disp"])
    text_ctx = [c for c in TEXT_CONTEXT_COLUMNS if c in daily_base.columns]
    text_ctx_dynamic = [c for c in daily_base.columns if str(c).startswith("txt_")]
    ctx_candidates = [*ctx_candidates, *fund_ctx, *text_ctx, *text_ctx_dynamic]
    # de-duplicate while preserving order
    seen_ctx: set[str] = set()
    dedup_ctx: List[str] = []
    for c in ctx_candidates:
        if c in seen_ctx:
            continue
        seen_ctx.add(c)
        dedup_ctx.append(c)
    ctx_candidates = dedup_ctx
    ctx_cols = [c for c in ctx_candidates if c in daily_base.columns and c not in out.columns]
    if not ctx_cols:
        return out

    ctx = daily_base[["date", "code", *ctx_cols]].copy()
    ctx["date"] = pd.to_datetime(ctx["date"], errors="coerce").dt.normalize()
    ctx["code"] = ctx["code"].astype(str).str.strip()
    ctx = ctx.dropna(subset=["date", "code"]).drop_duplicates(["date", "code"], keep="last")

    out = out.merge(ctx, on=["date", "code"], how="left")
    return out


def build_feature_bundle(bundle: MarketBundle) -> FeatureBundle:
    """Generate daily and multi-frequency feature bases."""
    log_progress("开始聚合分钟级日特征。", module="loader")
    minute_daily_feat = build_minute_daily_features(bundle.minute5)
    log_progress(f"分钟级日特征完成：rows={len(minute_daily_feat)}。", module="loader")
    log_progress("开始构建日频基础特征。", module="loader")
    daily_base = build_daily_feature_base(bundle.daily, minute_daily_feat)
    log_progress(f"日频基础特征完成：rows={len(daily_base)}, cols={len(daily_base.columns)}。", module="loader")

    log_progress("开始构建多频视图。", module="loader")
    views = build_frequency_views(daily_base, bundle.minute5)
    # convert non-daily views to generic features if needed
    for freq in ["5min", "15min", "30min", "60min", "120min"]:
        if freq in views and not views[freq].empty:
            v = add_generic_micro_structure_features(views[freq], time_col="datetime")
            views[freq] = _merge_daily_context_into_panel(v, daily_base=daily_base, time_col="datetime")
    if "W" in views and not views["W"].empty:
        v = add_generic_micro_structure_features(views["W"], time_col="date")
        views["W"] = _merge_daily_context_into_panel(v, daily_base=daily_base, time_col="date")
    if "M" in views and not views["M"].empty:
        v = add_generic_micro_structure_features(views["M"], time_col="date")
        views["M"] = _merge_daily_context_into_panel(v, daily_base=daily_base, time_col="date")

    # Add explicit finer->coarser bridge features, so factors on target frequency can
    # directly consume transformed information from higher-frequency source views.
    views = add_multifreq_bridge_features(views)

    price_cols = ["code", "date", "px_open5", "px_vwap30", "px_twap_last30", "px_daily_close"]
    price_table = daily_base[price_cols].copy()
    log_progress(f"特征视图构建完成：freqs={sorted(views.keys())}。", module="loader")
    return FeatureBundle(
        by_freq=views,
        price_table_daily=price_table,
        meta={"daily_rows": len(bundle.daily), "minute_rows": len(bundle.minute5)},
    )


def pick_named_file(folder: Path, stem: str, file_format: str = "auto") -> Optional[Path]:
    if file_format == "parquet":
        cands = [folder / f"{stem}.parquet"]
    elif file_format == "csv":
        cands = [folder / f"{stem}.csv"]
    else:
        cands = [folder / f"{stem}.parquet", folder / f"{stem}.csv"]
    for p in cands:
        if p.exists():
            return p
    return None


def load_index_benchmark_data(
    index_root: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    file_format: str = "auto",
) -> Dict[str, pd.DataFrame]:
    mapping = {"hs300": "hs300_price", "zz500": "zz500_price", "zz1000": "zz1000_price"}
    idx_data: Dict[str, pd.DataFrame] = {}
    for name, stem in mapping.items():
        fp = pick_named_file(index_root, stem, file_format=file_format)
        if fp is None:
            idx_data[name] = pd.DataFrame(columns=["date", "close"])
            continue
        df = read_data_file(fp, usecols=["date", "close"], start_date=start_date, end_date=end_date)
        if df.empty:
            idx_data[name] = pd.DataFrame(columns=["date", "close"])
            continue
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date")
        idx_data[name] = df[["date", "close"]].copy()
    return idx_data


def lookup_index_period_return(index_df: pd.DataFrame, entry_date: pd.Timestamp, exit_date: pd.Timestamp) -> float:
    if index_df.empty or pd.isna(entry_date) or pd.isna(exit_date):
        return float("nan")
    s = index_df.sort_values("date")
    s_entry = s[s["date"] >= pd.Timestamp(entry_date)]
    s_exit = s[s["date"] >= pd.Timestamp(exit_date)]
    if s_entry.empty or s_exit.empty:
        return float("nan")
    entry_px = float(s_entry.iloc[0]["close"])
    exit_px = float(s_exit.iloc[0]["close"])
    return exit_px / (entry_px + EPS) - 1.0

