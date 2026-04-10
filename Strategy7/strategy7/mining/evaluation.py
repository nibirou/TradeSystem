"""Factor evaluation and admission standards for mining framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..backtest.metrics import compute_return_stats
from ..core.constants import EPS
from ..core.time_utils import infer_periods_per_year


@dataclass
class FactorAdmissionStandard:
    profile: str
    min_abs_ic_mean: float = 0.0
    min_ic_win_rate: float = 0.0
    min_ic_ir: float = 0.0
    min_long_excess_annualized: float = -1.0
    min_long_sharpe: float = -99.0
    min_long_win_rate: float = 0.0
    min_coverage: float = 0.0


def periods_per_year_from_freq(freq: str, horizon: int) -> float:
    # Mining evaluates horizon-ahead return labels per signal, so stride=horizon.
    return infer_periods_per_year(factor_freq=str(freq), stride=max(int(horizon), 1))


def _ndcg(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    n = len(scores)
    if n == 0:
        return float("nan")
    kk = max(1, min(int(k), n))
    order_pred = np.argsort(-scores)[:kk]
    order_true = np.argsort(-labels)[:kk]

    min_label = min(float(np.min(labels)), 0.0)
    gain = labels - min_label + 1e-12

    denom = np.log2(np.arange(2, kk + 2))
    dcg = float(np.sum(gain[order_pred] / denom))
    idcg = float(np.sum(gain[order_true] / denom))
    if idcg <= EPS:
        return float("nan")
    return dcg / idcg


def evaluate_factor_panel(
    panel: pd.DataFrame,
    factor_col: str,
    ret_col: str = "future_ret_n",
    group_col: str = "date",
    top_frac: float = 0.1,
    min_cross_section: int = 30,
    periods_per_year: float = 252.0,
) -> Dict[str, float]:
    def _s(v: float) -> float:
        return float(v) if np.isfinite(v) else 0.0

    base = panel[[group_col, factor_col, ret_col]].copy()
    base = base.replace([np.inf, -np.inf], np.nan)

    total_rows = int(len(base))
    covered_rows = int(base[factor_col].notna().sum())
    coverage = float(covered_rows / total_rows) if total_rows > 0 else 0.0

    ic_list: List[float] = []
    rank_ic_list: List[float] = []
    ndcg_list: List[float] = []
    long_ret_list: List[float] = []
    pool_ret_list: List[float] = []
    long_excess_list: List[float] = []
    long_win_flags: List[float] = []

    for _t, g in base.groupby(group_col):
        sub = g[[factor_col, ret_col]].dropna()
        if len(sub) < int(min_cross_section):
            continue
        if sub[factor_col].nunique() < 2 or sub[ret_col].nunique() < 2:
            continue

        x = sub[factor_col].to_numpy(dtype=float)
        y = sub[ret_col].to_numpy(dtype=float)

        ic = pd.Series(x).corr(pd.Series(y), method="pearson")
        rank_ic = pd.Series(x).corr(pd.Series(y), method="spearman")
        if pd.notna(ic):
            ic_list.append(float(ic))
        if pd.notna(rank_ic):
            rank_ic_list.append(float(rank_ic))

        k = max(1, int(len(sub) * float(top_frac)))
        ord_desc = np.argsort(-x)
        top_idx = ord_desc[:k]
        top_ret = float(np.mean(y[top_idx]))
        pool_ret = float(np.mean(y))
        long_ret_list.append(top_ret)
        pool_ret_list.append(pool_ret)
        long_excess_list.append(top_ret - pool_ret)
        long_win_flags.append(float(top_ret > 0.0))

        nd = _ndcg(scores=x, labels=y, k=k)
        if np.isfinite(nd):
            ndcg_list.append(float(nd))

    ic_ts = pd.Series(ic_list, dtype=float)
    rank_ic_ts = pd.Series(rank_ic_list, dtype=float)
    long_ret_ts = pd.Series(long_ret_list, dtype=float)
    long_excess_ts = pd.Series(long_excess_list, dtype=float)

    rank_ic_mean = float(rank_ic_ts.mean()) if not rank_ic_ts.empty else 0.0
    sign = 1.0 if not np.isfinite(rank_ic_mean) or rank_ic_mean >= 0.0 else -1.0
    oriented_rank_ic = rank_ic_ts * sign

    ic_std = float(oriented_rank_ic.std(ddof=1)) if len(oriented_rank_ic) > 1 else float("nan")
    ic_ir = float(oriented_rank_ic.mean() / ic_std * np.sqrt(periods_per_year)) if np.isfinite(ic_std) and ic_std > 0 else 0.0

    long_stats = compute_return_stats(long_ret_ts, horizon=1, periods_per_year=periods_per_year)
    excess_stats = compute_return_stats(long_excess_ts, horizon=1, periods_per_year=periods_per_year)

    out = {
        "obs": float(len(rank_ic_ts)),
        "coverage": _s(coverage),
        "ic_mean": _s(float(ic_ts.mean()) if not ic_ts.empty else 0.0),
        "rank_ic_mean": _s(rank_ic_mean),
        "abs_ic_mean": _s(float(abs(rank_ic_mean))),
        "ic_win_rate": _s(float((oriented_rank_ic > 0).mean()) if not oriented_rank_ic.empty else 0.0),
        "ic_ir": _s(ic_ir),
        "ndcg_k": _s(float(np.mean(ndcg_list)) if ndcg_list else 0.0),
        "long_ret_mean": _s(float(long_ret_ts.mean()) if not long_ret_ts.empty else 0.0),
        "long_ret_annualized": _s(float(long_stats.get("annualized_return", 0.0))),
        "long_sharpe": _s(float(long_stats.get("sharpe", 0.0))),
        "long_win_rate": _s(float(np.mean(long_win_flags)) if long_win_flags else 0.0),
        "long_excess_mean": _s(float(long_excess_ts.mean()) if not long_excess_ts.empty else 0.0),
        "long_excess_annualized": _s(float(excess_stats.get("annualized_return", 0.0))),
    }
    return out


def objectives_from_metrics(metrics: Dict[str, float], framework: str) -> List[float]:
    fw = str(framework).strip().lower()
    if "gplearn_symbolic_alpha" in fw:
        return [
            float(metrics.get("abs_ic_mean", float("nan"))),
            float(metrics.get("ic_win_rate", float("nan"))),
            float(metrics.get("ndcg_k", float("nan"))),
            float(metrics.get("long_excess_annualized", float("nan"))),
            float(metrics.get("long_sharpe", float("nan"))),
        ]
    if "ml_ensemble" in fw:
        return [
            float(metrics.get("abs_ic_mean", float("nan"))),
            float(metrics.get("ic_win_rate", float("nan"))),
            float(metrics.get("ndcg_k", float("nan"))),
            float(metrics.get("long_excess_annualized", float("nan"))),
            float(metrics.get("long_sharpe", float("nan"))),
        ]

    if "minute" in fw:
        return [
            float(metrics.get("abs_ic_mean", float("nan"))),
            float(metrics.get("ic_win_rate", float("nan"))),
            float(metrics.get("long_ret_annualized", float("nan"))),
            float(metrics.get("long_sharpe", float("nan"))),
            float(metrics.get("long_win_rate", float("nan"))),
        ]

    # default: fundamental/custom daily style
    return [
        float(metrics.get("abs_ic_mean", float("nan"))),
        float(metrics.get("ic_win_rate", float("nan"))),
        float(metrics.get("ndcg_k", float("nan"))),
    ]


def resolve_admission_standard(freq: str, framework: str) -> FactorAdmissionStandard:
    f = str(freq).strip().upper()
    fw = str(framework).strip().lower()

    if fw == "minute_parametric_plus":
        return FactorAdmissionStandard(
            profile=f"{f}_minute_nsga3_plus_v1",
            min_abs_ic_mean=0.018,
            min_ic_win_rate=0.54,
            min_ic_ir=0.18,
            min_long_excess_annualized=0.03,
            min_long_sharpe=0.90,
            min_long_win_rate=0.53,
            min_coverage=0.88,
        )

    if "minute" in fw:
        return FactorAdmissionStandard(
            profile=f"{f}_minute_nsga3_v1",
            min_abs_ic_mean=0.015,
            min_ic_win_rate=0.53,
            min_ic_ir=0.15,
            min_long_excess_annualized=0.02,
            min_long_sharpe=0.80,
            min_long_win_rate=0.52,
            min_coverage=0.85,
        )

    if "fundamental" in fw:
        return FactorAdmissionStandard(
            profile=f"{f}_fundamental_nsga2_v1",
            min_abs_ic_mean=0.020,
            min_ic_win_rate=0.55,
            min_ic_ir=0.20,
            min_long_excess_annualized=0.02,
            min_long_sharpe=0.60,
            min_long_win_rate=0.50,
            min_coverage=0.90,
        )

    if "ml_ensemble" in fw:
        return FactorAdmissionStandard(
            profile=f"{f}_ml_ensemble_v1",
            min_abs_ic_mean=0.018,
            min_ic_win_rate=0.54,
            min_ic_ir=0.16,
            min_long_excess_annualized=0.02,
            min_long_sharpe=0.70,
            min_long_win_rate=0.51,
            min_coverage=0.90,
        )

    if "gplearn_symbolic_alpha" in fw:
        return FactorAdmissionStandard(
            profile=f"{f}_gplearn_symbolic_v1",
            min_abs_ic_mean=0.020,
            min_ic_win_rate=0.54,
            min_ic_ir=0.18,
            min_long_excess_annualized=0.02,
            min_long_sharpe=0.80,
            min_long_win_rate=0.51,
            min_coverage=0.90,
        )

    return FactorAdmissionStandard(
        profile=f"{f}_custom_default_v1",
        min_abs_ic_mean=0.015,
        min_ic_win_rate=0.53,
        min_ic_ir=0.10,
        min_long_excess_annualized=0.01,
        min_long_sharpe=0.30,
        min_long_win_rate=0.50,
        min_coverage=0.80,
    )


def check_admission(metrics: Dict[str, float], standard: FactorAdmissionStandard) -> Tuple[bool, Dict[str, object]]:
    checks = {
        "abs_ic_mean": (float(metrics.get("abs_ic_mean", float("nan"))), float(standard.min_abs_ic_mean)),
        "ic_win_rate": (float(metrics.get("ic_win_rate", float("nan"))), float(standard.min_ic_win_rate)),
        "ic_ir": (float(metrics.get("ic_ir", float("nan"))), float(standard.min_ic_ir)),
        "long_excess_annualized": (
            float(metrics.get("long_excess_annualized", float("nan"))),
            float(standard.min_long_excess_annualized),
        ),
        "long_sharpe": (float(metrics.get("long_sharpe", float("nan"))), float(standard.min_long_sharpe)),
        "long_win_rate": (float(metrics.get("long_win_rate", float("nan"))), float(standard.min_long_win_rate)),
        "coverage": (float(metrics.get("coverage", float("nan"))), float(standard.min_coverage)),
    }

    failures: Dict[str, Dict[str, float]] = {}
    for k, (v, thr) in checks.items():
        if not np.isfinite(v) or v < thr:
            failures[k] = {"value": v, "threshold": thr}

    return len(failures) == 0, {
        "profile": standard.profile,
        "passed": len(failures) == 0,
        "failures": failures,
    }
