"""Metrics for model and backtest evaluation."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from ..core.constants import EPS, TRADING_DAYS_PER_YEAR


def calc_trade_return(entry_price: pd.Series, exit_price: pd.Series, fee_bps: float, slippage_bps: float) -> pd.Series:
    cost = (fee_bps + slippage_bps) / 10000.0
    net_ret = (exit_price * (1.0 - cost)) / (entry_price * (1.0 + cost) + EPS) - 1.0
    return net_ret


def max_drawdown(net_values: pd.Series) -> float:
    if net_values.empty:
        return 0.0
    running_max = net_values.cummax()
    dd = net_values / (running_max + EPS) - 1.0
    mdd = float(dd.min())
    if not np.isfinite(mdd) or abs(mdd) < 1e-10:
        return 0.0
    return mdd


def compute_return_stats(
    returns: pd.Series,
    horizon: int,
    periods_per_year: float | None = None,
) -> Dict[str, float]:
    def _safe(v: float, default: float = 0.0) -> float:
        return float(v) if np.isfinite(v) else float(default)

    ppy = float(periods_per_year) if periods_per_year is not None else float(TRADING_DAYS_PER_YEAR / max(horizon, 1))
    if not np.isfinite(ppy) or ppy <= 0:
        ppy = float(TRADING_DAYS_PER_YEAR / max(horizon, 1))

    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return {
            "periods": 0.0,
            "periods_per_year": ppy,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "cum_return": 0.0,
            "annualized_return": 0.0,
            "annualized_vol": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    periods = len(r)
    cum_return = float((1.0 + r).prod() - 1.0)
    terminal_wealth = 1.0 + cum_return
    if terminal_wealth > 0:
        annualized_return = float(terminal_wealth ** (ppy / periods) - 1.0)
    else:
        # If compounded wealth is non-positive (e.g. spread/excess return path),
        # geometric annualization is undefined in real numbers. Fall back to
        # arithmetic annualization to keep metrics stable and non-crashing.
        annualized_return = float(r.mean() * ppy)
    std = float(r.std(ddof=1)) if periods > 1 else float("nan")
    annualized_vol = float(std * np.sqrt(ppy)) if periods > 1 and np.isfinite(std) else 0.0
    sharpe = float((r.mean() / std) * np.sqrt(ppy)) if periods > 1 and np.isfinite(std) and std > 0 else 0.0

    downside = r[r < 0]
    down_std = float(downside.std(ddof=1)) if len(downside) > 1 else float("nan")
    sortino = float((r.mean() / down_std) * np.sqrt(ppy)) if periods > 1 and np.isfinite(down_std) and down_std > 0 else 0.0

    net = (1.0 + r).cumprod()
    mdd = max_drawdown(net)
    calmar = float(annualized_return / abs(mdd)) if pd.notna(mdd) and np.isfinite(mdd) and mdd < 0 else 0.0
    pos = r[r > 0]
    neg = r[r < 0]
    loss_sum = abs(float(neg.sum()))
    profit_factor = float(float(pos.sum()) / max(loss_sum, EPS))

    return {
        "periods": float(periods),
        "periods_per_year": ppy,
        "win_rate": _safe(float((r > 0).mean())),
        "avg_return": _safe(float(r.mean())),
        "cum_return": _safe(cum_return),
        "annualized_return": _safe(annualized_return),
        "annualized_vol": _safe(annualized_vol),
        "sharpe": _safe(sharpe),
        "sortino": _safe(sortino),
        "max_drawdown": _safe(mdd),
        "calmar": _safe(calmar),
        "profit_factor": _safe(profit_factor),
        "avg_win": _safe(float(pos.mean()) if not pos.empty else 0.0),
        "avg_loss": _safe(float(neg.mean()) if not neg.empty else 0.0),
    }


def evaluate_selection_model(target: pd.Series, pred_score: pd.Series, threshold: float = 0.5) -> Dict[str, object]:
    y_true = pd.to_numeric(target, errors="coerce")
    y_pred_score = pd.to_numeric(pred_score, errors="coerce")
    valid = y_true.notna() & y_pred_score.notna()
    y_true = y_true.loc[valid]
    y_pred_score = y_pred_score.loc[valid]

    def _regression_metrics(y_t: pd.Series, y_p: pd.Series) -> Dict[str, float]:
        def _s(v: float) -> float:
            return float(v) if np.isfinite(v) else 0.0

        if y_t.empty:
            return {
                "reg_mae": 0.0,
                "reg_rmse": 0.0,
                "reg_r2": 0.0,
                "reg_pearson_corr": 0.0,
                "reg_target_std": 0.0,
                "reg_pred_std": 0.0,
            }
        yt = y_t.astype(float)
        yp = y_p.astype(float)
        err = yt - yp
        mse = float(np.mean(np.square(err)))
        ss_res = float(np.sum(np.square(err)))
        ss_tot = float(np.sum(np.square(yt - float(yt.mean()))))
        if yt.nunique() < 2 or yp.nunique() < 2:
            pearson = 0.0
        else:
            pearson = yt.corr(yp, method="pearson")
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > EPS else 0.0
        return {
            "reg_mae": _s(float(np.mean(np.abs(err)))),
            "reg_rmse": _s(float(np.sqrt(max(mse, 0.0)))),
            "reg_r2": _s(r2),
            "reg_pearson_corr": _s(float(pearson) if pd.notna(pearson) else 0.0),
            "reg_target_std": _s(float(yt.std(ddof=1)) if len(yt) > 1 else 0.0),
            "reg_pred_std": _s(float(yp.std(ddof=1)) if len(yp) > 1 else 0.0),
        }

    reg_metrics = _regression_metrics(y_true, y_pred_score)
    if y_true.empty:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "auc": 0.5,
            "rank_corr": 0.0,
            "target_mode": "empty",
            "direction_labeling": "empty",
            "sample_count": 0.0,
            **reg_metrics,
        }

    uniq = sorted(pd.Series(y_true).dropna().unique().tolist())
    is_binary_target = set(uniq).issubset({0, 1}) and len(uniq) <= 2
    if is_binary_target:
        y_eval = y_true.astype(int)
        target_mode = "binary"
        direction_labeling = "native_binary"
    else:
        # For continuous labels (e.g. return/volatility), derive a direction label
        # so framework-level classification metrics stay comparable and non-NaN.
        if (y_true > 0).any() and (y_true <= 0).any():
            y_eval = (y_true > 0).astype(int)
            direction_labeling = "gt_zero"
        else:
            y_eval = (y_true >= float(y_true.median())).astype(int)
            if y_eval.nunique() < 2:
                y_eval = (y_true.rank(method="first", pct=True) >= 0.5).astype(int)
                direction_labeling = "rank50"
            else:
                direction_labeling = "median"
        target_mode = "continuous"

    pred = (y_pred_score >= threshold).astype(int)
    auc = 0.5
    if y_eval.nunique() == 2:
        try:
            auc = float(roc_auc_score(y_eval, y_pred_score))
        except Exception:
            auc = 0.5
    if y_true.nunique() < 2 or y_pred_score.nunique() < 2:
        corr = 0.0
    else:
        corr = y_true.corr(y_pred_score, method="spearman")
    return {
        "accuracy": float(accuracy_score(y_eval, pred)),
        "precision": float(precision_score(y_eval, pred, zero_division=0)),
        "recall": float(recall_score(y_eval, pred, zero_division=0)),
        "auc": auc,
        "rank_corr": float(corr) if pd.notna(corr) else 0.0,
        "target_mode": target_mode,
        "direction_labeling": direction_labeling,
        "sample_count": float(len(y_true)),
        **reg_metrics,
    }


def _filter_by_group_stride(df: pd.DataFrame, group_col: str, eval_stride: int) -> pd.DataFrame:
    stride = max(int(eval_stride), 1)
    if stride <= 1 or df.empty or group_col not in df.columns:
        return df
    work = df.copy()
    raw = work[group_col]
    ts = pd.to_datetime(raw, errors="coerce")
    if ts.notna().sum() >= int(raw.notna().sum()) and ts.notna().sum() > 0:
        uniq = pd.DatetimeIndex(ts.dropna().unique()).sort_values()
        keep = pd.DatetimeIndex(uniq[::stride])
        return work.loc[ts.isin(keep)].copy()
    uniq = pd.Index(raw.dropna().unique()).sort_values()
    keep = set(uniq[::stride].tolist())
    return work.loc[raw.isin(keep)].copy()


def calc_ic_for_column(
    df: pd.DataFrame,
    score_col: str,
    ret_col: str,
    min_cross_section: int,
    group_col: str = "date",
    eval_stride: int = 1,
    constant_as_zero: bool = True,
) -> pd.DataFrame:
    work = _filter_by_group_stride(df, group_col=group_col, eval_stride=eval_stride)
    records: List[Dict[str, object]] = []
    for dt, g in work.groupby(group_col):
        sub = g[[score_col, ret_col]].replace([np.inf, -np.inf], np.nan).dropna()
        n = len(sub)
        if n < min_cross_section:
            continue
        if sub[score_col].nunique() < 2 or sub[ret_col].nunique() < 2:
            if not constant_as_zero:
                continue
            ic = 0.0
            rank_ic = 0.0
        else:
            ic = sub[score_col].corr(sub[ret_col], method="pearson")
            rank_ic = sub[score_col].corr(sub[ret_col], method="spearman")
            ic = float(ic) if pd.notna(ic) else (0.0 if constant_as_zero else float("nan"))
            rank_ic = float(rank_ic) if pd.notna(rank_ic) else (0.0 if constant_as_zero else float("nan"))
        records.append({group_col: pd.Timestamp(dt), "ic": float(ic), "rank_ic": float(rank_ic), "n": int(n)})
    return pd.DataFrame(records).sort_values(group_col) if records else pd.DataFrame(columns=[group_col, "ic", "rank_ic", "n"])


def summarize_ic(ic_ts: pd.DataFrame, periods_per_year: float = TRADING_DAYS_PER_YEAR) -> Dict[str, float]:
    def _s(v: float) -> float:
        return float(v) if np.isfinite(v) else 0.0

    ppy = float(periods_per_year) if np.isfinite(float(periods_per_year)) and float(periods_per_year) > 0 else float(TRADING_DAYS_PER_YEAR)
    if ic_ts.empty:
        return {
            "obs": 0.0,
            "ic_mean": 0.0,
            "ic_std": 0.0,
            "ic_ir": 0.0,
            "ic_positive_ratio": 0.0,
            "ic_tstat": 0.0,
            "rank_ic_mean": 0.0,
            "rank_ic_std": 0.0,
            "rank_ic_ir": 0.0,
            "rank_ic_positive_ratio": 0.0,
            "rank_ic_tstat": 0.0,
        }
    n = len(ic_ts)
    ic_mean = float(ic_ts["ic"].mean()) if n > 0 else 0.0
    ic_std = float(ic_ts["ic"].std(ddof=1)) if n > 1 else float("nan")
    rank_mean = float(ic_ts["rank_ic"].mean()) if n > 0 else 0.0
    rank_std = float(ic_ts["rank_ic"].std(ddof=1)) if n > 1 else float("nan")
    ic_ir = float(ic_mean / ic_std * np.sqrt(ppy)) if pd.notna(ic_std) and np.isfinite(ic_std) and ic_std > 0 else 0.0
    rank_ir = float(rank_mean / rank_std * np.sqrt(ppy)) if pd.notna(rank_std) and np.isfinite(rank_std) and rank_std > 0 else 0.0
    ic_tstat = float(ic_mean / (ic_std / np.sqrt(n))) if pd.notna(ic_std) and np.isfinite(ic_std) and ic_std > 0 else 0.0
    rank_tstat = float(rank_mean / (rank_std / np.sqrt(n))) if pd.notna(rank_std) and np.isfinite(rank_std) and rank_std > 0 else 0.0
    return {
        "obs": float(n),
        "ic_mean": _s(ic_mean),
        "ic_std": _s(ic_std),
        "ic_ir": _s(ic_ir),
        "ic_positive_ratio": _s(float((ic_ts["ic"] > 0).mean())),
        "ic_tstat": _s(ic_tstat),
        "rank_ic_mean": _s(rank_mean),
        "rank_ic_std": _s(rank_std),
        "rank_ic_ir": _s(rank_ir),
        "rank_ic_positive_ratio": _s(float((ic_ts["rank_ic"] > 0).mean())),
        "rank_ic_tstat": _s(rank_tstat),
    }


def compute_factor_ic_statistics(
    pred_df: pd.DataFrame,
    factor_cols: List[str],
    ret_col: str,
    min_cross_section: int,
    group_col: str = "date",
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
    eval_stride: int = 1,
    constant_as_zero: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, object]] = []
    ts_rows: List[pd.DataFrame] = []
    for fac in factor_cols:
        if fac not in pred_df.columns:
            continue
        ic_tmp = calc_ic_for_column(
            pred_df,
            score_col=fac,
            ret_col=ret_col,
            min_cross_section=min_cross_section,
            group_col=group_col,
            eval_stride=eval_stride,
            constant_as_zero=constant_as_zero,
        )
        summary = summarize_ic(ic_tmp, periods_per_year=periods_per_year)
        summary["factor"] = fac
        summary_rows.append(summary)
        if not ic_tmp.empty:
            ic_tmp = ic_tmp.copy()
            ic_tmp["factor"] = fac
            ts_rows.append(ic_tmp)
    summary_df = pd.DataFrame(summary_rows).sort_values("ic_ir", ascending=False) if summary_rows else pd.DataFrame(columns=["factor"])
    ts_df = pd.concat(ts_rows, ignore_index=True) if ts_rows else pd.DataFrame(columns=[group_col, "ic", "rank_ic", "n", "factor"])
    return summary_df, ts_df


def compute_score_spread(
    df: pd.DataFrame,
    score_col: str,
    ret_col: str,
    quantiles: int = 5,
    group_col: str = "date",
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
    eval_stride: int = 1,
) -> Dict[str, float]:
    work = _filter_by_group_stride(df, group_col=group_col, eval_stride=eval_stride)
    spreads: List[float] = []
    qn = max(int(quantiles), 2)
    for _, g in work.groupby(group_col):
        sub = g[[score_col, ret_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < qn:
            continue
        if sub[score_col].nunique() < 2 or sub[ret_col].nunique() < 2:
            spreads.append(0.0)
            continue
        try:
            q = pd.qcut(sub[score_col], q=qn, labels=False, duplicates="drop")
        except Exception:
            rank_pct = sub[score_col].rank(method="first", pct=True)
            q = (rank_pct * qn).astype(int).clip(lower=0, upper=qn - 1)
        if q.nunique() < 2:
            spreads.append(0.0)
            continue
        top = sub.loc[q == q.max(), ret_col].mean()
        bot = sub.loc[q == q.min(), ret_col].mean()
        spreads.append(float(top - bot))

    if not spreads:
        return {"obs": 0.0, "spread_mean": 0.0, "spread_std": 0.0, "spread_ir": 0.0}
    s = pd.Series(spreads, dtype=float)
    std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    ppy = float(periods_per_year) if np.isfinite(float(periods_per_year)) and float(periods_per_year) > 0 else float(TRADING_DAYS_PER_YEAR)
    ir = float(s.mean() / std * np.sqrt(ppy)) if pd.notna(std) and np.isfinite(std) and std > 0 else 0.0
    mean_v = float(s.mean()) if not s.empty else 0.0
    if not np.isfinite(mean_v):
        mean_v = 0.0
    if not np.isfinite(std):
        std = 0.0
    if not np.isfinite(ir):
        ir = 0.0
    return {"obs": float(len(s)), "spread_mean": mean_v, "spread_std": std, "spread_ir": ir}
