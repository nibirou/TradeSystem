"""Factor snapshot reporting helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd

from ..core.utils import dump_json, ensure_dir


_GROUP_LABELS: Dict[str, str] = {
    "price_volume": "Price-Volume / Non-Fundamental Packages",
    "fundamental": "Fundamental Packages",
    "text": "Text Packages",
    "mined": "Mined Packages",
}


def normalize_factor_package_alias(package: object) -> str:
    """Normalize historical/legacy package names to current package taxonomy."""
    p = str(package or "").strip()
    if not p:
        return "unknown"
    if p in {"catalog_custom", "custom_factor", "mined_factor"}:
        return "mined_custom"
    if p.startswith("catalog_"):
        return "mined_custom"
    return p


def _normalize_factor_packages_expr(expr: object, fallback_package: str) -> str:
    toks = [normalize_factor_package_alias(x.strip()) for x in str(expr or "").split(",") if x.strip()]
    toks = [x for x in toks if x and x != "unknown"]
    if fallback_package and fallback_package != "unknown":
        toks.append(fallback_package)
    deduped: list[str] = []
    seen: set[str] = set()
    for x in toks:
        if x in seen:
            continue
        seen.add(x)
        deduped.append(x)
    if not deduped:
        return fallback_package or "unknown"
    return ",".join(deduped)


def factor_group_key(package: object) -> str:
    p = normalize_factor_package_alias(package)
    if p.startswith("fund_") or p.startswith("fundamental_"):
        return "fundamental"
    if p.startswith("text_"):
        return "text"
    if p.startswith("mined_"):
        return "mined"
    return "price_volume"


def factor_group_label(package: object) -> str:
    return _GROUP_LABELS.get(factor_group_key(package), _GROUP_LABELS["price_volume"])


def normalize_factor_metadata_view(meta_df_view: pd.DataFrame) -> pd.DataFrame:
    out = meta_df_view.copy()
    if out.empty:
        for c in ["factor_package", "factor_packages", "factor_group_key", "factor_group_label"]:
            out[c] = pd.Series(dtype=str)
        return out

    if "factor_package" not in out.columns:
        if "category" in out.columns:
            out["factor_package"] = out["category"].astype(str)
        else:
            out["factor_package"] = pd.Series(["unknown"] * len(out), index=out.index, dtype=str)
    out["factor_package"] = out["factor_package"].map(normalize_factor_package_alias)

    if "factor_packages" not in out.columns:
        out["factor_packages"] = out["factor_package"]
    out["factor_packages"] = [
        _normalize_factor_packages_expr(v, fallback_package=str(p))
        for v, p in zip(out["factor_packages"].tolist(), out["factor_package"].tolist())
    ]

    out["factor_group_key"] = out["factor_package"].map(factor_group_key)
    out["factor_group_label"] = out["factor_package"].map(factor_group_label)
    return out


def _safe_name(name: str) -> str:
    txt = re.sub(r"[^0-9a-zA-Z_\-]+", "_", str(name).strip()).strip("_").lower()
    return txt or "unknown"


def _preferred_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "factor",
        "freq",
        "factor_group_label",
        "factor_package",
        "factor_packages",
        "name_cn",
        "meaning_cn",
        "formula_cn",
        "description",
        "used_in_run",
    ]
    return [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]


def build_factor_snapshot_statistics(snapshot_df: pd.DataFrame) -> Dict[str, object]:
    if snapshot_df.empty:
        return {
            "total_factor_count": 0,
            "used_factor_count": 0,
            "group_counts": {},
            "group_used_counts": {},
            "package_counts": {},
            "package_used_counts": {},
        }

    group_counts = snapshot_df.groupby("factor_group_key").size().astype(int).to_dict()
    group_used_counts = (
        snapshot_df[snapshot_df["used_in_run"]]
        .groupby("factor_group_key")
        .size()
        .astype(int)
        .to_dict()
    )
    package_counts = snapshot_df.groupby("factor_package").size().astype(int).to_dict()
    package_used_counts = (
        snapshot_df[snapshot_df["used_in_run"]]
        .groupby("factor_package")
        .size()
        .astype(int)
        .to_dict()
    )
    return {
        "total_factor_count": int(len(snapshot_df)),
        "used_factor_count": int(snapshot_df["used_in_run"].sum()),
        "group_counts": {str(k): int(v) for k, v in group_counts.items()},
        "group_used_counts": {str(k): int(v) for k, v in group_used_counts.items()},
        "package_counts": {str(k): int(v) for k, v in package_counts.items()},
        "package_used_counts": {str(k): int(v) for k, v in package_used_counts.items()},
    }


def export_factor_snapshot(
    *,
    meta_df_view: pd.DataFrame,
    used_factors: Sequence[str],
    entrypoint: str,
    factor_freq: str,
    output_root: Path,
    run_tag: str = "",
    extra_summary: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """Export per-run factor snapshot (all factors vs used factors) with grouped folders."""
    normalized = normalize_factor_metadata_view(meta_df_view)
    used_set = {str(x).strip() for x in (used_factors or []) if str(x).strip()}
    if used_set and "factor" in normalized.columns:
        known = set(normalized["factor"].astype(str).tolist())
        missing = sorted([x for x in used_set if x not in known])
        if missing:
            fill_rows = pd.DataFrame(
                {
                    "factor": missing,
                    "freq": [str(factor_freq)] * len(missing),
                    "factor_package": ["unknown"] * len(missing),
                    "factor_packages": ["unknown"] * len(missing),
                    "name_cn": ["feature_engineering_derived_factor"] * len(missing),
                    "meaning_cn": ["derived by feature engineering (for example, PCA projection)"] * len(missing),
                    "formula_cn": ["fitted on train split and transformed on test split"] * len(missing),
                    "description": ["derived_feature_engineering_factor"] * len(missing),
                }
            )
            normalized = pd.concat([normalized, fill_rows], ignore_index=True)
            normalized = normalize_factor_metadata_view(normalized)
    if "factor" in normalized.columns:
        normalized["used_in_run"] = normalized["factor"].astype(str).isin(used_set)
    else:
        normalized["used_in_run"] = False

    ts = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d_%H%M%S")
    tag = _safe_name(run_tag) if str(run_tag).strip() else "default"
    folder = ensure_dir(output_root / _safe_name(entrypoint) / f"{ts}_{_safe_name(factor_freq)}_{tag}")

    normalized = normalized[_preferred_columns(normalized)]
    used_df = normalized[normalized["used_in_run"]].copy()

    all_path = folder / "all_factors.csv"
    used_path = folder / "used_factors.csv"
    normalized.to_csv(all_path, index=False, encoding="utf-8-sig")
    used_df.to_csv(used_path, index=False, encoding="utf-8-sig")

    by_group_root = ensure_dir(folder / "by_group")
    for group_key, group_df in normalized.groupby("factor_group_key", dropna=False):
        group_dir = ensure_dir(by_group_root / _safe_name(str(group_key)))
        group_df = group_df.copy()
        group_df.to_csv(group_dir / "all.csv", index=False, encoding="utf-8-sig")
        group_df[group_df["used_in_run"]].to_csv(group_dir / "used.csv", index=False, encoding="utf-8-sig")
        for pkg, pkg_df in group_df.groupby("factor_package", dropna=False):
            pkg_dir = ensure_dir(group_dir / _safe_name(str(pkg)))
            pkg_df.to_csv(pkg_dir / "all.csv", index=False, encoding="utf-8-sig")
            pkg_df[pkg_df["used_in_run"]].to_csv(pkg_dir / "used.csv", index=False, encoding="utf-8-sig")

    stats = build_factor_snapshot_statistics(normalized)
    summary = {
        "entrypoint": str(entrypoint),
        "factor_freq": str(factor_freq),
        "run_tag": str(run_tag),
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        **stats,
    }
    if extra_summary:
        summary.update(extra_summary)
    summary_path = folder / "snapshot_summary.json"
    dump_json(summary_path, summary)

    md_lines = [
        f"# Factor Snapshot: {entrypoint} ({factor_freq})",
        "",
        f"- generated_at: {summary['generated_at']}",
        f"- total_factor_count: {summary['total_factor_count']}",
        f"- used_factor_count: {summary['used_factor_count']}",
        "",
        "## Group Summary",
        "",
        "| factor_group | total | used |",
        "| --- | ---: | ---: |",
    ]
    for gk in ["price_volume", "fundamental", "text", "mined"]:
        md_lines.append(
            f"| {_GROUP_LABELS[gk]} | {int(summary['group_counts'].get(gk, 0))} | {int(summary['group_used_counts'].get(gk, 0))} |"
        )
    md_lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- all factors: `{all_path}`",
            f"- used factors: `{used_path}`",
            f"- grouped folders: `{by_group_root}`",
            f"- summary: `{summary_path}`",
        ]
    )
    (folder / "snapshot_overview.md").write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "snapshot_dir": str(folder),
        "all_factors_path": str(all_path),
        "used_factors_path": str(used_path),
        "summary_path": str(summary_path),
        "grouped_root": str(by_group_root),
    }

