"""Factor feature-engineering utilities (filtering, de-correlation, orthogonalization)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.constants import EPS


@dataclass
class FactorEngineeringOptions:
    enabled: bool = False
    min_coverage: float = 0.70
    min_std: float = 1e-8
    corr_threshold: float = 0.92
    preselect_top_n: int = 2000
    min_factors: int = 20
    max_factors: int = 600
    orth_method: str = "none"  # none / pca
    pca_variance_ratio: float = 0.95
    pca_max_components: int = 128


def _normalize_options(options: FactorEngineeringOptions) -> FactorEngineeringOptions:
    out = FactorEngineeringOptions(**vars(options))
    out.min_coverage = float(np.clip(out.min_coverage, 0.0, 1.0))
    out.min_std = float(max(out.min_std, 0.0))
    out.corr_threshold = float(np.clip(out.corr_threshold, 0.0, 0.999999))
    out.preselect_top_n = int(max(out.preselect_top_n, 0))
    out.min_factors = int(max(out.min_factors, 1))
    out.max_factors = int(max(out.max_factors, 0))
    out.orth_method = str(out.orth_method).strip().lower() or "none"
    if out.orth_method not in {"none", "pca"}:
        out.orth_method = "none"
    out.pca_variance_ratio = float(np.clip(out.pca_variance_ratio, 0.0, 1.0))
    out.pca_max_components = int(max(out.pca_max_components, 1))
    return out


def _safe_numeric_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    use_cols = [str(c) for c in cols if str(c) in df.columns]
    if not use_cols:
        return pd.DataFrame(index=df.index)
    out = df[use_cols].copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out.columns = use_cols
    return out


def _rank_quality_scores(
    *,
    train_filled: pd.DataFrame,
    raw_train: pd.DataFrame,
    cols: Sequence[str],
    target_col: str | None,
) -> pd.Series:
    if not cols:
        return pd.Series(dtype=float)
    idx = [str(c) for c in cols]
    coverage = raw_train[idx].notna().mean().fillna(0.0)
    std = train_filled[idx].std(ddof=0).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    score = 0.55 * coverage + 0.45 * np.log1p(std)

    # Optional supervised tie-breaker: only uses train split, so no leakage.
    if target_col and target_col in train_filled.columns and len(idx) <= 3000:
        y = pd.to_numeric(train_filled[target_col], errors="coerce")
        ic_abs = pd.Series(0.0, index=idx, dtype=float)
        for c in idx:
            x = pd.to_numeric(train_filled[c], errors="coerce")
            valid = x.notna() & y.notna()
            if int(valid.sum()) < 25:
                continue
            ic = x.loc[valid].corr(y.loc[valid], method="spearman")
            if np.isfinite(ic):
                ic_abs.loc[c] = float(abs(ic))
        score = score + 0.35 * ic_abs
    return score.sort_values(ascending=False)


def _greedy_corr_prune(train_filled: pd.DataFrame, ranked_cols: Sequence[str], corr_threshold: float) -> Tuple[List[str], List[str]]:
    if not ranked_cols:
        return [], []
    if corr_threshold >= 0.999999:
        return list(ranked_cols), []

    rank_df = train_filled[list(ranked_cols)].rank(method="average", pct=True)
    corr = rank_df.corr(method="pearson").abs().fillna(0.0)
    kept: List[str] = []
    dropped: List[str] = []
    for c in ranked_cols:
        if not kept:
            kept.append(c)
            continue
        corr_to_kept = corr.loc[c, kept]
        if bool((corr_to_kept >= corr_threshold).any()):
            dropped.append(c)
            continue
        kept.append(c)
    return kept, dropped


def _apply_pca_projection(
    *,
    train_filled: pd.DataFrame,
    test_filled: pd.DataFrame,
    cols: Sequence[str],
    variance_ratio: float,
    pca_max_components: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, object]]:
    if not cols:
        return train_filled, test_filled, [], {"pca_components": 0, "pca_explained_ratio": 0.0}

    x_train = _safe_numeric_frame(train_filled, cols).fillna(0.0).to_numpy(dtype=float)
    x_test = _safe_numeric_frame(test_filled, cols).fillna(0.0).to_numpy(dtype=float)
    mu = np.nanmean(x_train, axis=0)
    sigma = np.nanstd(x_train, axis=0)
    sigma = np.where(np.isfinite(sigma) & (sigma > EPS), sigma, 1.0)

    z_train = (x_train - mu) / sigma
    z_test = (x_test - mu) / sigma
    z_train = np.nan_to_num(z_train, nan=0.0, posinf=0.0, neginf=0.0)
    z_test = np.nan_to_num(z_test, nan=0.0, posinf=0.0, neginf=0.0)

    u, s, vt = np.linalg.svd(z_train, full_matrices=False)
    if len(s) == 0:
        return train_filled, test_filled, list(cols), {"pca_components": len(cols), "pca_explained_ratio": 1.0}

    var = (s**2) / max(int(z_train.shape[0]) - 1, 1)
    total_var = float(np.sum(var))
    if total_var <= EPS:
        return train_filled, test_filled, list(cols), {"pca_components": len(cols), "pca_explained_ratio": 1.0}

    ratio = np.cumsum(var / total_var)
    k = int(np.searchsorted(ratio, float(variance_ratio), side="left") + 1)
    k = int(max(1, min(k, int(pca_max_components), int(vt.shape[0]))))

    comps = vt[:k].T
    train_proj = z_train @ comps
    test_proj = z_test @ comps
    pca_cols = [f"fe_pca_{i + 1:03d}" for i in range(k)]

    train_out = train_filled.copy()
    test_out = test_filled.copy()
    train_out[pca_cols] = pd.DataFrame(train_proj, index=train_out.index, columns=pca_cols)
    test_out[pca_cols] = pd.DataFrame(test_proj, index=test_out.index, columns=pca_cols)
    return train_out, test_out, pca_cols, {"pca_components": k, "pca_explained_ratio": float(ratio[k - 1])}


def apply_factor_engineering(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    factor_cols: Sequence[str],
    options: FactorEngineeringOptions,
    target_col: str | None = None,
    raw_train_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, object]]:
    """Filter/reduce factor columns using train-only statistics.

    Returns:
    1. transformed train dataframe
    2. transformed test dataframe
    3. final factor column names used for model training
    4. engineering report dict
    """
    opts = _normalize_options(options)
    all_cols = [str(c) for c in factor_cols if str(c) in train_df.columns and str(c) in test_df.columns]
    report: Dict[str, object] = {
        "enabled": bool(opts.enabled),
        "input_factor_count": int(len(all_cols)),
        "coverage_kept_count": 0,
        "variance_kept_count": 0,
        "corr_kept_count": 0,
        "final_factor_count": int(len(all_cols)),
        "orth_method": str(opts.orth_method),
        "removed_by_coverage": [],
        "removed_by_variance": [],
        "removed_by_correlation": [],
        "pca_components": 0,
        "pca_explained_ratio": 0.0,
    }
    if (not opts.enabled) or (not all_cols):
        return train_df, test_df, all_cols, report

    train_num = _safe_numeric_frame(train_df, all_cols)
    test_num = _safe_numeric_frame(test_df, all_cols)
    raw_train = _safe_numeric_frame(raw_train_df, all_cols) if raw_train_df is not None else train_num

    coverage = raw_train.notna().mean().fillna(0.0)
    cov_kept = coverage[coverage >= opts.min_coverage].index.tolist()
    if not cov_kept:
        cov_kept = coverage.sort_values(ascending=False).head(max(opts.min_factors, 1)).index.tolist()
    dropped_cov = sorted(set(all_cols) - set(cov_kept))
    report["removed_by_coverage"] = dropped_cov
    report["coverage_kept_count"] = int(len(cov_kept))

    train_med = train_num[cov_kept].median(numeric_only=True) if cov_kept else pd.Series(dtype=float)
    train_fill = train_num[cov_kept].fillna(train_med).fillna(0.0) if cov_kept else train_num[cov_kept]
    test_fill = test_num[cov_kept].fillna(train_med).fillna(0.0) if cov_kept else test_num[cov_kept]

    std = train_fill.std(ddof=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    var_kept = std[std >= opts.min_std].index.tolist()
    if not var_kept:
        var_kept = std.sort_values(ascending=False).head(max(opts.min_factors, 1)).index.tolist()
    dropped_var = sorted(set(cov_kept) - set(var_kept))
    report["removed_by_variance"] = dropped_var
    report["variance_kept_count"] = int(len(var_kept))

    score = _rank_quality_scores(
        train_filled=train_fill,
        raw_train=raw_train,
        cols=var_kept,
        target_col=target_col,
    )
    ranked_cols = score.index.tolist()
    if opts.preselect_top_n > 0 and len(ranked_cols) > opts.preselect_top_n:
        ranked_cols = ranked_cols[: opts.preselect_top_n]

    corr_kept, dropped_corr = _greedy_corr_prune(train_fill, ranked_cols, opts.corr_threshold)
    if len(corr_kept) < opts.min_factors:
        refill = [c for c in ranked_cols if c not in set(corr_kept)]
        corr_kept.extend(refill[: max(0, opts.min_factors - len(corr_kept))])
    if opts.max_factors > 0 and len(corr_kept) > opts.max_factors:
        corr_kept = corr_kept[: opts.max_factors]
    report["removed_by_correlation"] = dropped_corr
    report["corr_kept_count"] = int(len(corr_kept))

    final_cols = [c for c in corr_kept if c in train_fill.columns]
    if not final_cols:
        final_cols = ranked_cols[: max(opts.min_factors, 1)]

    if opts.orth_method == "pca" and len(final_cols) >= 2:
        train_out, test_out, pca_cols, pca_report = _apply_pca_projection(
            train_filled=train_df,
            test_filled=test_df,
            cols=final_cols,
            variance_ratio=opts.pca_variance_ratio,
            pca_max_components=opts.pca_max_components,
        )
        report.update(pca_report)
        report["source_factor_count_for_orth"] = int(len(final_cols))
        report["final_factor_count"] = int(len(pca_cols))
        report["selected_source_factors"] = list(final_cols)
        report["selected_factors"] = list(pca_cols)
        return train_out, test_out, pca_cols, report

    report["final_factor_count"] = int(len(final_cols))
    report["selected_factors"] = list(final_cols)
    return train_df, test_df, final_cols, report
