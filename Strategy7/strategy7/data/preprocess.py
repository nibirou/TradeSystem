"""Data preprocessing and feature sanitation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd

from ..core.constants import EPS


@dataclass
class PreprocessOptions:
    winsorize_limit: float = 0.01
    do_zscore: bool = True
    neutralize: bool = False
    neutralize_industry_col: str = "industry_bucket"
    neutralize_size_col: str = "barra_size_proxy"
    fill_method: str = "median"


def clean_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def dedup_frame(df: pd.DataFrame, keys: List[str], keep: str = "last") -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(keys).drop_duplicates(keys, keep=keep).reset_index(drop=True)


def winsorize_series(s: pd.Series, limit: float = 0.01) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() < 5:
        return x
    lo = x.quantile(limit)
    hi = x.quantile(1.0 - limit)
    return x.clip(lower=lo, upper=hi)


def zscore_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    std = float(x.std(ddof=0)) if x.notna().sum() > 1 else 0.0
    if std <= EPS:
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    return (x - float(x.mean())) / (std + EPS)


def neutralize_cross_section(
    df: pd.DataFrame,
    value_col: str,
    industry_col: str,
    size_col: str,
) -> pd.Series:
    """Cross-sectional neutralization by industry dummies + size proxy."""
    y = pd.to_numeric(df[value_col], errors="coerce")
    valid = y.notna()
    if valid.sum() < 10:
        return y

    x_parts: List[np.ndarray] = []
    # intercept
    x_parts.append(np.ones(valid.sum(), dtype=float).reshape(-1, 1))

    if industry_col in df.columns:
        ind = df.loc[valid, industry_col].astype(str).fillna("unknown")
        dummies = pd.get_dummies(ind, drop_first=True)
        if dummies.shape[1] > 0:
            x_parts.append(dummies.to_numpy(dtype=float))

    if size_col in df.columns:
        size_x = pd.to_numeric(df.loc[valid, size_col], errors="coerce").fillna(0.0).to_numpy(dtype=float).reshape(-1, 1)
        x_parts.append(size_x)

    x = np.concatenate(x_parts, axis=1)
    yy = y.loc[valid].to_numpy(dtype=float)
    try:
        beta, *_ = np.linalg.lstsq(x, yy, rcond=None)
        resid = yy - x @ beta
        out = y.copy()
        out.loc[valid] = resid
        return out
    except Exception:
        return y


def _stat_rows_by_group(stat_df: pd.DataFrame, groups: pd.Series, index: pd.Index) -> pd.DataFrame:
    rows = stat_df.reindex(pd.Index(groups.to_numpy(), name=stat_df.index.name))
    rows.index = index
    return rows


def _apply_fast_winsor_zscore(
    panel: pd.DataFrame,
    factor_cols: List[str],
    options: PreprocessOptions,
    group_col: str,
) -> pd.DataFrame:
    """Batch cross-section winsorize/zscore without per-factor groupby loops."""
    valid_cols = [c for c in factor_cols if c in panel.columns]
    if not valid_cols:
        return panel

    out = panel.copy()
    groups = out[group_col]
    values = out[valid_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    grouped = values.groupby(groups, sort=False)

    if float(options.winsorize_limit) > 0.0:
        counts = grouped.count()
        q_lo = grouped.quantile(float(options.winsorize_limit))
        q_hi = grouped.quantile(1.0 - float(options.winsorize_limit))
        lo_rows = _stat_rows_by_group(q_lo, groups, values.index)
        hi_rows = _stat_rows_by_group(q_hi, groups, values.index)
        count_rows = _stat_rows_by_group(counts, groups, values.index)
        clipped = values.clip(lower=lo_rows, upper=hi_rows, axis=1)
        values = clipped.where(count_rows >= 5, values)

    if bool(options.do_zscore):
        grouped = values.groupby(groups, sort=False)
        mean = grouped.mean()
        std = grouped.std(ddof=0)
        mean_rows = _stat_rows_by_group(mean, groups, values.index)
        std_rows = _stat_rows_by_group(std, groups, values.index)
        z = (values - mean_rows) / (std_rows + EPS)
        zero_std = (~np.isfinite(std_rows)) | (std_rows <= EPS)
        values = z.mask(zero_std, 0.0)

    out[valid_cols] = values
    return out


def fill_feature_na(df: pd.DataFrame, cols: List[str], method: str = "median") -> pd.DataFrame:
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return df.copy()
    if not bool(df[valid_cols].isna().to_numpy().any()):
        return df.copy()
    out = df.copy()
    if method == "zero":
        out[valid_cols] = out[valid_cols].fillna(0.0)
        return out

    if method == "ffill_by_code" and "code" in out.columns:
        out[valid_cols] = out.groupby("code")[valid_cols].ffill()
        out[valid_cols] = out[valid_cols].fillna(out[valid_cols].median(numeric_only=True))
        return out

    # median default
    med = out[valid_cols].median(numeric_only=True)
    out[valid_cols] = out[valid_cols].fillna(med)
    return out


def fit_feature_fill_values(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return pd.Series(dtype=float)
    return df[valid_cols].median(numeric_only=True)


def fill_feature_na_with_reference(
    df: pd.DataFrame,
    cols: List[str],
    method: str = "median",
    reference_fill_values: pd.Series | None = None,
) -> pd.DataFrame:
    """Fill NA by train-fitted statistics to avoid test-period leakage."""
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return df.copy()
    if not bool(df[valid_cols].isna().to_numpy().any()):
        return df.copy()

    out = df.copy()
    if method == "zero":
        out[valid_cols] = out[valid_cols].fillna(0.0)
        return out

    fill_values = reference_fill_values if reference_fill_values is not None else fit_feature_fill_values(df, valid_cols)
    fill_values = pd.Series(fill_values, dtype=float).reindex(valid_cols)

    if method == "ffill_by_code" and "code" in out.columns:
        out[valid_cols] = out.groupby("code")[valid_cols].ffill()
        out[valid_cols] = out[valid_cols].fillna(fill_values)
        out[valid_cols] = out[valid_cols].fillna(0.0)
        return out

    out[valid_cols] = out[valid_cols].fillna(fill_values)
    out[valid_cols] = out[valid_cols].fillna(0.0)
    return out


def apply_cross_section_pipeline(
    panel: pd.DataFrame,
    factor_cols: List[str],
    options: PreprocessOptions,
    group_col: str = "date",
) -> pd.DataFrame:
    """Winsorize/z-score/neutralize by cross section."""
    if panel.empty or not factor_cols:
        return panel
    if group_col in panel.columns and not bool(options.neutralize):
        return _apply_fast_winsor_zscore(panel, factor_cols, options, group_col)
    out = panel.copy()
    for fac in factor_cols:
        if fac not in out.columns:
            continue
        grouped = out.groupby(group_col)[fac]
        out[fac] = grouped.transform(lambda s: winsorize_series(s, options.winsorize_limit))
        if options.neutralize:
            out[fac] = out.groupby(group_col, group_keys=False).apply(
                lambda g: neutralize_cross_section(
                    g,
                    value_col=fac,
                    industry_col=options.neutralize_industry_col,
                    size_col=options.neutralize_size_col,
                )
            ).reset_index(level=0, drop=True)
        if options.do_zscore:
            out[fac] = out.groupby(group_col)[fac].transform(zscore_series)
    return out
