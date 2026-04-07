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
        return float("nan")
    running_max = net_values.cummax()
    dd = net_values / (running_max + EPS) - 1.0
    return float(dd.min())


def compute_return_stats(
    returns: pd.Series,
    horizon: int,
    periods_per_year: float | None = None,
) -> Dict[str, float]:
    ppy = float(periods_per_year) if periods_per_year is not None else float(TRADING_DAYS_PER_YEAR / max(horizon, 1))
    if not np.isfinite(ppy) or ppy <= 0:
        ppy = float(TRADING_DAYS_PER_YEAR / max(horizon, 1))

    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return {
            "periods": 0.0,
            "periods_per_year": ppy,
            "win_rate": float("nan"),
            "avg_return": float("nan"),
            "cum_return": float("nan"),
            "annualized_return": float("nan"),
            "annualized_vol": float("nan"),
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "max_drawdown": float("nan"),
            "calmar": float("nan"),
            "profit_factor": float("nan"),
            "avg_win": float("nan"),
            "avg_loss": float("nan"),
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
    annualized_vol = float(std * np.sqrt(ppy)) if periods > 1 else float("nan")
    sharpe = float((r.mean() / std) * np.sqrt(ppy)) if periods > 1 and std > 0 else float("nan")

    downside = r[r < 0]
    down_std = float(downside.std(ddof=1)) if len(downside) > 1 else float("nan")
    sortino = float((r.mean() / down_std) * np.sqrt(ppy)) if periods > 1 and down_std and down_std > 0 else float("nan")

    net = (1.0 + r).cumprod()
    mdd = max_drawdown(net)
    calmar = float(annualized_return / abs(mdd)) if pd.notna(mdd) and mdd < 0 else float("nan")
    pos = r[r > 0]
    neg = r[r < 0]
    profit_factor = float(pos.sum() / abs(neg.sum())) if abs(neg.sum()) > EPS else float("nan")

    return {
        "periods": float(periods),
        "periods_per_year": ppy,
        "win_rate": float((r > 0).mean()),
        "avg_return": float(r.mean()),
        "cum_return": cum_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "profit_factor": profit_factor,
        "avg_win": float(pos.mean()) if not pos.empty else float("nan"),
        "avg_loss": float(neg.mean()) if not neg.empty else float("nan"),
    }


def evaluate_selection_model(target: pd.Series, pred_score: pd.Series, threshold: float = 0.5) -> Dict[str, object]:
    y_true = pd.to_numeric(target, errors="coerce")
    y_pred_score = pd.to_numeric(pred_score, errors="coerce")
    valid = y_true.notna() & y_pred_score.notna()
    y_true = y_true.loc[valid]
    y_pred_score = y_pred_score.loc[valid]

    def _regression_metrics(y_t: pd.Series, y_p: pd.Series) -> Dict[str, float]:
        if y_t.empty:
            return {
                "reg_mae": float("nan"),
                "reg_rmse": float("nan"),
                "reg_r2": float("nan"),
                "reg_pearson_corr": float("nan"),
                "reg_target_std": float("nan"),
                "reg_pred_std": float("nan"),
            }
        yt = y_t.astype(float)
        yp = y_p.astype(float)
        err = yt - yp
        mse = float(np.mean(np.square(err)))
        ss_res = float(np.sum(np.square(err)))
        ss_tot = float(np.sum(np.square(yt - float(yt.mean()))))
        pearson = yt.corr(yp, method="pearson")
        return {
            "reg_mae": float(np.mean(np.abs(err))),
            "reg_rmse": float(np.sqrt(max(mse, 0.0))),
            "reg_r2": float(1.0 - ss_res / ss_tot) if ss_tot > EPS else float("nan"),
            "reg_pearson_corr": float(pearson) if pd.notna(pearson) else float("nan"),
            "reg_target_std": float(yt.std(ddof=1)) if len(yt) > 1 else float("nan"),
            "reg_pred_std": float(yp.std(ddof=1)) if len(yp) > 1 else float("nan"),
        }

    reg_metrics = _regression_metrics(y_true, y_pred_score)
    if y_true.empty:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "auc": float("nan"),
            "rank_corr": float("nan"),
            "target_mode": "empty",
            "direction_labeling": "empty",
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
    auc = float("nan")
    if y_eval.nunique() == 2:
        try:
            auc = float(roc_auc_score(y_eval, y_pred_score))
        except Exception:
            auc = float("nan")
    corr = y_true.corr(y_pred_score, method="spearman")
    return {
        "accuracy": float(accuracy_score(y_eval, pred)),
        "precision": float(precision_score(y_eval, pred, zero_division=0)),
        "recall": float(recall_score(y_eval, pred, zero_division=0)),
        "auc": auc,
        "rank_corr": float(corr) if pd.notna(corr) else float("nan"),
        "target_mode": target_mode,
        "direction_labeling": direction_labeling,
        **reg_metrics,
    }


def calc_ic_for_column(df: pd.DataFrame, score_col: str, ret_col: str, min_cross_section: int, group_col: str = "date") -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for dt, g in df.groupby(group_col):
        sub = g[[score_col, ret_col]].replace([np.inf, -np.inf], np.nan).dropna()
        n = len(sub)
        if n < min_cross_section:
            continue
        if sub[score_col].nunique() < 2 or sub[ret_col].nunique() < 2:
            continue
        ic = sub[score_col].corr(sub[ret_col], method="pearson")
        rank_ic = sub[score_col].corr(sub[ret_col], method="spearman")
        records.append({group_col: pd.Timestamp(dt), "ic": float(ic), "rank_ic": float(rank_ic), "n": int(n)})
    return pd.DataFrame(records).sort_values(group_col) if records else pd.DataFrame(columns=[group_col, "ic", "rank_ic", "n"])


def summarize_ic(ic_ts: pd.DataFrame) -> Dict[str, float]:
    if ic_ts.empty:
        return {
            "obs": 0.0,
            "ic_mean": float("nan"),
            "ic_std": float("nan"),
            "ic_ir": float("nan"),
            "ic_positive_ratio": float("nan"),
            "ic_tstat": float("nan"),
            "rank_ic_mean": float("nan"),
            "rank_ic_std": float("nan"),
            "rank_ic_ir": float("nan"),
            "rank_ic_positive_ratio": float("nan"),
            "rank_ic_tstat": float("nan"),
        }
    n = len(ic_ts)
    ic_mean = float(ic_ts["ic"].mean())
    ic_std = float(ic_ts["ic"].std(ddof=1)) if n > 1 else float("nan")
    rank_mean = float(ic_ts["rank_ic"].mean())
    rank_std = float(ic_ts["rank_ic"].std(ddof=1)) if n > 1 else float("nan")
    ic_ir = float(ic_mean / ic_std * np.sqrt(TRADING_DAYS_PER_YEAR)) if pd.notna(ic_std) and ic_std > 0 else float("nan")
    rank_ir = float(rank_mean / rank_std * np.sqrt(TRADING_DAYS_PER_YEAR)) if pd.notna(rank_std) and rank_std > 0 else float("nan")
    ic_tstat = float(ic_mean / (ic_std / np.sqrt(n))) if pd.notna(ic_std) and ic_std > 0 else float("nan")
    rank_tstat = float(rank_mean / (rank_std / np.sqrt(n))) if pd.notna(rank_std) and rank_std > 0 else float("nan")
    return {
        "obs": float(n),
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "ic_positive_ratio": float((ic_ts["ic"] > 0).mean()),
        "ic_tstat": ic_tstat,
        "rank_ic_mean": rank_mean,
        "rank_ic_std": rank_std,
        "rank_ic_ir": rank_ir,
        "rank_ic_positive_ratio": float((ic_ts["rank_ic"] > 0).mean()),
        "rank_ic_tstat": rank_tstat,
    }


def compute_factor_ic_statistics(
    pred_df: pd.DataFrame,
    factor_cols: List[str],
    ret_col: str,
    min_cross_section: int,
    group_col: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, object]] = []
    ts_rows: List[pd.DataFrame] = []
    for fac in factor_cols:
        if fac not in pred_df.columns:
            continue
        ic_tmp = calc_ic_for_column(pred_df, score_col=fac, ret_col=ret_col, min_cross_section=min_cross_section, group_col=group_col)
        summary = summarize_ic(ic_tmp)
        summary["factor"] = fac
        summary_rows.append(summary)
        if not ic_tmp.empty:
            ic_tmp = ic_tmp.copy()
            ic_tmp["factor"] = fac
            ts_rows.append(ic_tmp)
    summary_df = pd.DataFrame(summary_rows).sort_values("ic_ir", ascending=False) if summary_rows else pd.DataFrame(columns=["factor"])
    ts_df = pd.concat(ts_rows, ignore_index=True) if ts_rows else pd.DataFrame(columns=[group_col, "ic", "rank_ic", "n", "factor"])
    return summary_df, ts_df


def compute_score_spread(df: pd.DataFrame, score_col: str, ret_col: str, quantiles: int = 5, group_col: str = "date") -> Dict[str, float]:
    spreads: List[float] = []
    for _, g in df.groupby(group_col):
        sub = g[[score_col, ret_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < quantiles:
            continue
        try:
            q = pd.qcut(sub[score_col], q=quantiles, labels=False, duplicates="drop")
        except Exception:
            continue
        if q.nunique() < quantiles:
            continue
        top = sub.loc[q == q.max(), ret_col].mean()
        bot = sub.loc[q == q.min(), ret_col].mean()
        spreads.append(float(top - bot))

    if not spreads:
        return {"obs": 0.0, "spread_mean": float("nan"), "spread_std": float("nan"), "spread_ir": float("nan")}
    s = pd.Series(spreads, dtype=float)
    std = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
    ir = float(s.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR)) if pd.notna(std) and std > 0 else float("nan")
    return {"obs": float(len(s)), "spread_mean": float(s.mean()), "spread_std": std, "spread_ir": ir}
