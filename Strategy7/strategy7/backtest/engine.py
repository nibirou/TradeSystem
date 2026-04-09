"""Backtest engine with pluggable timing/portfolio/execution models."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import BacktestConfig
from ..core.constants import EPS
from ..core.time_utils import infer_periods_per_year
from ..data.loaders import lookup_index_period_return
from ..models.base import ExecutionModel, PortfolioModel, TimingModel
from .metrics import calc_trade_return, compute_return_stats


def _infer_periods_per_year(factor_freq: str, rebalance_stride: int) -> float:
    return infer_periods_per_year(factor_freq=factor_freq, stride=rebalance_stride)


def _supports_index_benchmark(factor_freq: str) -> bool:
    return str(factor_freq).strip().upper() in {"D", "W", "M"}


def _mean_or_zero(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0
    v = float(s.mean())
    return v if np.isfinite(v) else 0.0


def run_backtest(
    pred_df: pd.DataFrame,
    backtest_cfg: BacktestConfig,
    factor_freq: str,
    timing_model: TimingModel,
    portfolio_model: PortfolioModel,
    execution_model: ExecutionModel,
    index_benchmarks: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Run backtest using model scores already attached in pred_df as pred_score.

    The engine is intentionally stage-separated:
    1) stock picking by score threshold/top-k
    2) timing model -> portfolio exposure
    3) portfolio model -> target weights
    4) execution model -> fill ratio and realized return
    """
    data = pred_df.copy()
    if "pred_score" not in data.columns:
        raise ValueError("pred_df must contain pred_score.")
    if "entry_price" not in data.columns or "exit_price" not in data.columns:
        raise ValueError("pred_df must contain entry_price and exit_price.")

    data["gross_trade_ret"] = data["exit_price"] / (data["entry_price"] + EPS) - 1.0
    data["net_trade_ret"] = calc_trade_return(
        data["entry_price"],
        data["exit_price"],
        fee_bps=backtest_cfg.fee_bps,
        slippage_bps=backtest_cfg.slippage_bps,
    )

    signal_col = "signal_ts"
    overlap_possible = int(backtest_cfg.rebalance_stride < backtest_cfg.horizon)
    periods_per_year = _infer_periods_per_year(
        factor_freq=factor_freq,
        rebalance_stride=backtest_cfg.rebalance_stride,
    )
    index_benchmark_enabled = _supports_index_benchmark(factor_freq)
    rebalance_points = sorted(pd.to_datetime(data[signal_col].dropna().unique()))
    trade_records: List[Dict[str, object]] = []
    position_records: List[Dict[str, object]] = []
    prev_weights: Dict[str, float] = {}

    def _turnover(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
        all_codes = set(prev_w.keys()) | set(new_w.keys())
        return float(sum(abs(float(new_w.get(c, 0.0)) - float(prev_w.get(c, 0.0))) for c in all_codes))

    # Rebalance loop: each step is one "decision timestamp".
    for idx in range(0, len(rebalance_points), int(backtest_cfg.rebalance_stride)):
        dt = rebalance_points[idx]
        day_raw = data[pd.to_datetime(data[signal_col]) == dt].copy()
        day_all = day_raw.dropna(subset=["entry_price", "exit_price", "net_trade_ret", "future_ret_n"]).copy()

        if not day_all.empty:
            entry_ts = pd.to_datetime(day_all["entry_ts"], errors="coerce").mode().iloc[0]
            exit_ts = pd.to_datetime(day_all["exit_ts"], errors="coerce").mode().iloc[0]
        elif not day_raw.empty:
            entry_ts = pd.to_datetime(day_raw["entry_ts"], errors="coerce").dropna().iloc[0] if day_raw["entry_ts"].notna().any() else pd.NaT
            exit_ts = pd.to_datetime(day_raw["exit_ts"], errors="coerce").dropna().iloc[0] if day_raw["exit_ts"].notna().any() else pd.NaT
        else:
            entry_ts = pd.NaT
            exit_ts = pd.NaT

        benchmark_pool_ret = float(day_all["net_trade_ret"].mean()) if not day_all.empty else 0.0
        bench_size = int(len(day_all))

        # Index benchmark comparison is enabled only for D/W/M research frequencies.
        if index_benchmark_enabled:
            benchmark_hs300_ret = lookup_index_period_return(index_benchmarks.get("hs300", pd.DataFrame()), entry_ts, exit_ts)
            benchmark_zz500_ret = lookup_index_period_return(index_benchmarks.get("zz500", pd.DataFrame()), entry_ts, exit_ts)
            benchmark_zz1000_ret = lookup_index_period_return(index_benchmarks.get("zz1000", pd.DataFrame()), entry_ts, exit_ts)
        else:
            benchmark_hs300_ret = float("nan")
            benchmark_zz500_ret = float("nan")
            benchmark_zz1000_ret = float("nan")

        # Selection layer: threshold + top-k.
        day_pick = day_all[day_all["pred_score"] >= backtest_cfg.long_threshold].copy()
        day_pick = day_pick.sort_values("pred_score", ascending=False).head(backtest_cfg.top_k)

        timing_exposure, timing_diag = timing_model.predict_exposure(day_all if not day_all.empty else day_raw)
        portfolio_diag: Dict[str, float] = {}
        execution_diag: Dict[str, float] = {}
        portfolio_turnover = float("nan")
        portfolio_concentration = float("nan")
        portfolio_entropy = float("nan")

        if day_pick.empty or timing_exposure <= 0.0:
            strategy_ret = 0.0
            active_count = 0
            avg_score = float("nan")
            new_weights = {}
            portfolio_turnover = _turnover(prev_weights, new_weights)
            prev_weights = new_weights
        else:
            # Weighting layer.
            target_w, portfolio_diag = portfolio_model.compute_weights(
                day_pick=day_pick,
                day_universe=day_all,
                prev_weights=prev_weights,
                fee_bps=backtest_cfg.fee_bps,
                slippage_bps=backtest_cfg.slippage_bps,
            )
            if target_w.empty or target_w.sum() <= EPS:
                target_w = pd.Series(np.full(len(day_pick), 1.0 / len(day_pick), dtype=float), index=day_pick["code"].astype(str))
                portfolio_diag["optimizer_fallback"] = 1.0

            day_pick = day_pick.copy()
            day_pick["weight_target"] = day_pick["code"].astype(str).map(target_w).fillna(0.0)
            sw = float(day_pick["weight_target"].sum())
            if sw <= EPS:
                day_pick["weight_target"] = 1.0 / len(day_pick)
            else:
                day_pick["weight_target"] = day_pick["weight_target"] / (sw + EPS)

            # Timing is applied as top-down exposure scaling.
            day_pick["weight_target"] = day_pick["weight_target"] * float(timing_exposure)

            # Execution layer (fills/cost adjustments).
            day_exec, execution_diag = execution_model.apply_execution(
                day_pick=day_pick,
                weight_col="weight_target",
                fee_bps=backtest_cfg.fee_bps,
                slippage_bps=backtest_cfg.slippage_bps,
            )
            if "executed_weight" not in day_exec.columns:
                day_exec["executed_weight"] = day_exec["weight_target"]
            if "realized_trade_ret" not in day_exec.columns:
                day_exec["realized_trade_ret"] = day_exec["net_trade_ret"]

            strategy_ret = float((day_exec["executed_weight"] * day_exec["realized_trade_ret"]).sum())
            active_count = int((day_exec["executed_weight"] > 1e-6).sum())
            avg_score = float((day_exec["pred_score"] * day_exec["executed_weight"]).sum() / (day_exec["executed_weight"].sum() + EPS))

            portfolio_concentration = float(np.square(day_exec["executed_weight"]).sum())
            portfolio_entropy = float(-(day_exec["executed_weight"] * np.log(day_exec["executed_weight"] + EPS)).sum())

            new_weights = {
                str(code): float(w)
                for code, w in day_exec[["code", "executed_weight"]].itertuples(index=False, name=None)
                if w > 1e-8
            }
            portfolio_turnover = float(portfolio_diag.get("opt_turnover", _turnover(prev_weights, new_weights)))
            prev_weights = new_weights

            for _, r in day_exec.iterrows():
                position_records.append(
                    {
                        "signal_ts": pd.Timestamp(r["signal_ts"]),
                        "code": str(r["code"]),
                        "entry_ts": pd.Timestamp(r["entry_ts"]),
                        "exit_ts": pd.Timestamp(r["exit_ts"]),
                        "entry_price": float(r["entry_price"]),
                        "exit_price": float(r["exit_price"]),
                        "pred_score": float(r["pred_score"]),
                        "future_ret_ref": float(r["future_ret_n"]),
                        "gross_trade_ret": float(r["gross_trade_ret"]),
                        "net_trade_ret": float(r["net_trade_ret"]),
                        "weight_target": float(r.get("weight_target", np.nan)),
                        "executed_weight": float(r.get("executed_weight", np.nan)),
                        "fill_ratio": float(r.get("fill_ratio", np.nan)),
                        "realized_trade_ret": float(r.get("realized_trade_ret", np.nan)),
                        "weighting_mode": backtest_cfg.portfolio_mode,
                        "industry_bucket": str(r.get("industry_bucket", "")),
                        "board_type": str(r.get("board_type", "")),
                    }
                )

        row = {
            "trade_date": pd.Timestamp(dt),
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "hold_n": int(backtest_cfg.horizon),
            "active_stocks": active_count,
            "pool_size": bench_size,
            "avg_pred_score": avg_score,
            "strategy_ret": strategy_ret,
            "benchmark_pool_ret": benchmark_pool_ret,
            "benchmark_hs300_ret": benchmark_hs300_ret,
            "benchmark_zz500_ret": benchmark_zz500_ret,
            "benchmark_zz1000_ret": benchmark_zz1000_ret,
            "weighting_mode": backtest_cfg.portfolio_mode,
            "portfolio_turnover": portfolio_turnover,
            "portfolio_concentration": portfolio_concentration,
            "portfolio_weight_entropy": portfolio_entropy,
            "timing_exposure": float(timing_exposure),
        }
        for d in (timing_diag, portfolio_diag, execution_diag):
            for k, v in d.items():
                row[k] = float(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else v
        trade_records.append(row)

    # Build outputs: per-rebalance trades and per-position details.
    trades_df = pd.DataFrame(trade_records)
    positions_df = pd.DataFrame(position_records)
    if trades_df.empty:
        empty_stats = compute_return_stats(pd.Series(dtype=float), horizon=backtest_cfg.horizon, periods_per_year=periods_per_year)
        bench_hs300 = dict(empty_stats, enabled=bool(index_benchmark_enabled))
        bench_zz500 = dict(empty_stats, enabled=bool(index_benchmark_enabled))
        bench_zz1000 = dict(empty_stats, enabled=bool(index_benchmark_enabled))
        excess_hs300 = dict(empty_stats, enabled=bool(index_benchmark_enabled))
        excess_zz500 = dict(empty_stats, enabled=bool(index_benchmark_enabled))
        excess_zz1000 = dict(empty_stats, enabled=bool(index_benchmark_enabled))
        summary = {
            "strategy": empty_stats,
            "benchmark": empty_stats,
            "excess": empty_stats,
            "benchmark_pool": empty_stats,
            "benchmark_hs300": bench_hs300,
            "benchmark_zz500": bench_zz500,
            "benchmark_zz1000": bench_zz1000,
            "excess_vs_pool": empty_stats,
            "excess_vs_hs300": excess_hs300,
            "excess_vs_zz500": excess_zz500,
            "excess_vs_zz1000": excess_zz1000,
            "annualization_periods_per_year": periods_per_year,
            "rebalance_overlap_possible": float(overlap_possible),
            "active_trade_ratio": 0.0,
            "portfolio_weighting_mode": backtest_cfg.portfolio_mode,
            "portfolio_avg_turnover": 0.0,
            "portfolio_avg_concentration": 0.0,
            "portfolio_avg_entropy": 0.0,
            "timing_avg_exposure": 0.0,
            "execution_avg_fill": 0.0,
            "index_benchmark_enabled": bool(index_benchmark_enabled),
        }
        return trades_df, positions_df, pd.DataFrame(), summary

    trades_df = trades_df.sort_values("trade_date").reset_index(drop=True)
    trades_df["strategy_ret"] = pd.to_numeric(trades_df["strategy_ret"], errors="coerce").fillna(0.0)
    trades_df["benchmark_pool_ret"] = pd.to_numeric(trades_df["benchmark_pool_ret"], errors="coerce").fillna(0.0)
    for tag in ["hs300", "zz500", "zz1000"]:
        c = f"benchmark_{tag}_ret"
        trades_df[c] = pd.to_numeric(trades_df[c], errors="coerce")
        if index_benchmark_enabled:
            trades_df[c] = trades_df[c].fillna(0.0)

    trades_df["strategy_net"] = (1.0 + trades_df["strategy_ret"]).cumprod()
    trades_df["strategy_cum_return"] = trades_df["strategy_net"] - 1.0
    trades_df["strategy_cum_ret"] = trades_df["strategy_cum_return"]

    # Pool benchmark always enabled.
    trades_df["benchmark_pool_net"] = (1.0 + trades_df["benchmark_pool_ret"]).cumprod()
    trades_df["benchmark_pool_cum_return"] = trades_df["benchmark_pool_net"] - 1.0
    trades_df["benchmark_pool_cum_ret"] = trades_df["benchmark_pool_cum_return"]
    trades_df["excess_vs_pool_ret"] = trades_df["strategy_ret"] - trades_df["benchmark_pool_ret"]
    trades_df["excess_vs_pool_net"] = (1.0 + trades_df["excess_vs_pool_ret"]).cumprod()
    trades_df["excess_vs_pool_cum_return"] = trades_df["excess_vs_pool_net"] - 1.0
    trades_df["excess_vs_pool_cum_ret"] = trades_df["excess_vs_pool_cum_return"]

    for tag in ["hs300", "zz500", "zz1000"]:
        rc = f"benchmark_{tag}_ret"
        nc = f"benchmark_{tag}_net"
        cc = f"benchmark_{tag}_cum_return"
        c2 = f"benchmark_{tag}_cum_ret"
        ex_ret = f"excess_vs_{tag}_ret"
        ex_net = f"excess_vs_{tag}_net"
        ex_cum = f"excess_vs_{tag}_cum_return"
        ex_cum_ret = f"excess_vs_{tag}_cum_ret"
        if index_benchmark_enabled:
            trades_df[nc] = (1.0 + trades_df[rc]).cumprod()
            trades_df[cc] = trades_df[nc] - 1.0
            trades_df[c2] = trades_df[cc]
            trades_df[ex_ret] = trades_df["strategy_ret"] - trades_df[rc]
            trades_df[ex_net] = (1.0 + trades_df[ex_ret]).cumprod()
            trades_df[ex_cum] = trades_df[ex_net] - 1.0
            trades_df[ex_cum_ret] = trades_df[ex_cum]
        else:
            trades_df[nc] = np.nan
            trades_df[cc] = np.nan
            trades_df[c2] = np.nan
            trades_df[ex_ret] = np.nan
            trades_df[ex_net] = np.nan
            trades_df[ex_cum] = np.nan
            trades_df[ex_cum_ret] = np.nan

    trades_df["benchmark_ret"] = trades_df["benchmark_pool_ret"]
    trades_df["benchmark_net"] = trades_df["benchmark_pool_net"]
    trades_df["benchmark_cum_return"] = trades_df["benchmark_pool_cum_return"]
    trades_df["benchmark_cum_ret"] = trades_df["benchmark_cum_return"]
    trades_df["excess_ret"] = trades_df["excess_vs_pool_ret"]
    trades_df["excess_net"] = trades_df["excess_vs_pool_net"]
    trades_df["excess_cum_return"] = trades_df["excess_vs_pool_cum_return"]
    trades_df["excess_cum_ret"] = trades_df["excess_cum_return"]

    strategy_stats = compute_return_stats(trades_df["strategy_ret"], horizon=backtest_cfg.horizon, periods_per_year=periods_per_year)
    benchmark_pool_stats = compute_return_stats(
        trades_df["benchmark_pool_ret"],
        horizon=backtest_cfg.horizon,
        periods_per_year=periods_per_year,
    )
    excess_pool_stats = compute_return_stats(
        trades_df["excess_vs_pool_ret"],
        horizon=backtest_cfg.horizon,
        periods_per_year=periods_per_year,
    )

    if index_benchmark_enabled:
        benchmark_hs300_stats = compute_return_stats(
            trades_df["benchmark_hs300_ret"],
            horizon=backtest_cfg.horizon,
            periods_per_year=periods_per_year,
        )
        benchmark_zz500_stats = compute_return_stats(
            trades_df["benchmark_zz500_ret"],
            horizon=backtest_cfg.horizon,
            periods_per_year=periods_per_year,
        )
        benchmark_zz1000_stats = compute_return_stats(
            trades_df["benchmark_zz1000_ret"],
            horizon=backtest_cfg.horizon,
            periods_per_year=periods_per_year,
        )
        excess_hs300_stats = compute_return_stats(
            trades_df["excess_vs_hs300_ret"],
            horizon=backtest_cfg.horizon,
            periods_per_year=periods_per_year,
        )
        excess_zz500_stats = compute_return_stats(
            trades_df["excess_vs_zz500_ret"],
            horizon=backtest_cfg.horizon,
            periods_per_year=periods_per_year,
        )
        excess_zz1000_stats = compute_return_stats(
            trades_df["excess_vs_zz1000_ret"],
            horizon=backtest_cfg.horizon,
            periods_per_year=periods_per_year,
        )
        benchmark_hs300_stats["enabled"] = True
        benchmark_zz500_stats["enabled"] = True
        benchmark_zz1000_stats["enabled"] = True
        excess_hs300_stats["enabled"] = True
        excess_zz500_stats["enabled"] = True
        excess_zz1000_stats["enabled"] = True
    else:
        disabled_stats = compute_return_stats(pd.Series(dtype=float), horizon=backtest_cfg.horizon, periods_per_year=periods_per_year)
        benchmark_hs300_stats = dict(disabled_stats, enabled=False)
        benchmark_zz500_stats = dict(disabled_stats, enabled=False)
        benchmark_zz1000_stats = dict(disabled_stats, enabled=False)
        excess_hs300_stats = dict(disabled_stats, enabled=False)
        excess_zz500_stats = dict(disabled_stats, enabled=False)
        excess_zz1000_stats = dict(disabled_stats, enabled=False)

    summary = {
        "strategy": strategy_stats,
        "benchmark": benchmark_pool_stats,
        "excess": excess_pool_stats,
        "benchmark_pool": benchmark_pool_stats,
        "benchmark_hs300": benchmark_hs300_stats,
        "benchmark_zz500": benchmark_zz500_stats,
        "benchmark_zz1000": benchmark_zz1000_stats,
        "excess_vs_pool": excess_pool_stats,
        "excess_vs_hs300": excess_hs300_stats,
        "excess_vs_zz500": excess_zz500_stats,
        "excess_vs_zz1000": excess_zz1000_stats,
        "annualization_periods_per_year": periods_per_year,
        "rebalance_overlap_possible": float(overlap_possible),
        "active_trade_ratio": float((trades_df["active_stocks"] > 0).mean()),
        "portfolio_weighting_mode": backtest_cfg.portfolio_mode,
        "portfolio_avg_turnover": _mean_or_zero(trades_df["portfolio_turnover"]),
        "portfolio_avg_concentration": _mean_or_zero(trades_df["portfolio_concentration"]),
        "portfolio_avg_entropy": _mean_or_zero(trades_df["portfolio_weight_entropy"]),
        "timing_avg_exposure": _mean_or_zero(trades_df["timing_exposure"]),
        "execution_avg_fill": _mean_or_zero(trades_df.get("avg_fill_ratio", pd.Series(dtype=float))),
        "index_benchmark_enabled": bool(index_benchmark_enabled),
    }
    curve_df = trades_df.copy()
    return trades_df, positions_df, curve_df, summary

