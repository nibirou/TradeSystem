"""Backtest plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def plot_backtest_curves(
    curve_df: pd.DataFrame,
    output_main_png: Path,
    output_excess_png: Path,
    title_prefix: str,
) -> Dict[str, bool]:
    plot_status = {"main": False, "excess": False}
    if curve_df.empty or plt is None:
        return plot_status

    dt = pd.to_datetime(curve_df["trade_date"])
    fig1, axes1 = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    main_lines = [
        ("strategy_ret", "Strategy"),
        ("benchmark_pool_ret", "Benchmark-PoolMean"),
        ("benchmark_hs300_ret", "HS300"),
        ("benchmark_zz500_ret", "ZZ500"),
        ("benchmark_zz1000_ret", "ZZ1000"),
    ]
    for col, label in main_lines:
        if col in curve_df.columns:
            axes1[0].plot(dt, curve_df[col], label=label, linewidth=1.2)
    axes1[0].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    axes1[0].set_ylabel("Period Return")
    axes1[0].legend(loc="best")
    axes1[0].grid(alpha=0.2)

    main_net_lines = [
        ("strategy_net", "Strategy"),
        ("benchmark_pool_net", "Benchmark-PoolMean"),
        ("benchmark_hs300_net", "HS300"),
        ("benchmark_zz500_net", "ZZ500"),
        ("benchmark_zz1000_net", "ZZ1000"),
    ]
    for col, label in main_net_lines:
        if col in curve_df.columns:
            axes1[1].plot(dt, curve_df[col], label=label, linewidth=1.3)
    axes1[1].set_ylabel("Net Value")
    axes1[1].legend(loc="best")
    axes1[1].grid(alpha=0.2)

    main_cum_lines = [
        ("strategy_cum_ret", "Strategy"),
        ("benchmark_pool_cum_ret", "Benchmark-PoolMean"),
        ("benchmark_hs300_cum_ret", "HS300"),
        ("benchmark_zz500_cum_ret", "ZZ500"),
        ("benchmark_zz1000_cum_ret", "ZZ1000"),
    ]
    for col, label in main_cum_lines:
        if col in curve_df.columns:
            axes1[2].plot(dt, curve_df[col], label=label, linewidth=1.3)
    axes1[2].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    axes1[2].set_ylabel("Cum Return")
    axes1[2].set_xlabel("Date")
    axes1[2].legend(loc="best")
    axes1[2].grid(alpha=0.2)

    fig1.suptitle(f"{title_prefix} - Main", fontsize=14)
    fig1.tight_layout()
    fig1.savefig(output_main_png, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    plot_status["main"] = True

    fig2, axes2 = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    excess_ret_lines = [
        ("excess_vs_pool_ret", "Excess vs PoolMean"),
        ("excess_vs_hs300_ret", "Excess vs HS300"),
        ("excess_vs_zz500_ret", "Excess vs ZZ500"),
        ("excess_vs_zz1000_ret", "Excess vs ZZ1000"),
    ]
    for col, label in excess_ret_lines:
        if col in curve_df.columns:
            axes2[0].plot(dt, curve_df[col], label=label, linewidth=1.2)
    axes2[0].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    axes2[0].set_ylabel("Excess Period Return")
    axes2[0].legend(loc="best")
    axes2[0].grid(alpha=0.2)

    excess_net_lines = [
        ("excess_vs_pool_net", "Excess vs PoolMean"),
        ("excess_vs_hs300_net", "Excess vs HS300"),
        ("excess_vs_zz500_net", "Excess vs ZZ500"),
        ("excess_vs_zz1000_net", "Excess vs ZZ1000"),
    ]
    for col, label in excess_net_lines:
        if col in curve_df.columns:
            axes2[1].plot(dt, curve_df[col], label=label, linewidth=1.3)
    axes2[1].set_ylabel("Excess Net Value")
    axes2[1].legend(loc="best")
    axes2[1].grid(alpha=0.2)

    excess_cum_lines = [
        ("excess_vs_pool_cum_ret", "Excess vs PoolMean"),
        ("excess_vs_hs300_cum_ret", "Excess vs HS300"),
        ("excess_vs_zz500_cum_ret", "Excess vs ZZ500"),
        ("excess_vs_zz1000_cum_ret", "Excess vs ZZ1000"),
    ]
    for col, label in excess_cum_lines:
        if col in curve_df.columns:
            axes2[2].plot(dt, curve_df[col], label=label, linewidth=1.3)
    axes2[2].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    axes2[2].set_ylabel("Excess Cum Return")
    axes2[2].set_xlabel("Date")
    axes2[2].legend(loc="best")
    axes2[2].grid(alpha=0.2)

    fig2.suptitle(f"{title_prefix} - Excess", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(output_excess_png, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    plot_status["excess"] = True
    return plot_status

