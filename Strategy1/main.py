# main.py
import pandas as pd

from config import POOLS, DAILY_FREQ, INTRADAY_FREQ
from data_loader import load_daily_data, load_intraday_data
from features import (
    preprocess_daily,
    build_daily_factors,
    aggregate_intraday_features,
    merge_daily_intraday_factors,
    compute_forward_returns,
    cross_sectional_standardize,
)
from factor_model import compute_rolling_factor_weights, apply_factor_weights
from backtest import backtest_long_only
from reporting import compute_performance_stats


def main():
    # ===== 1. 加载数据 =====
    print("Loading daily data...")
    daily_raw = load_daily_data(pools=POOLS, freq=DAILY_FREQ)

    print("Loading intraday (5min) data...")
    intraday_raw = load_intraday_data(pools=POOLS, freq=INTRADAY_FREQ)

    # ===== 2. 日频预处理 =====
    print("Preprocessing daily data...")
    daily_prep = preprocess_daily(daily_raw)

    # ===== 3. 构造日频因子 =====
    print("Building daily factors...")
    daily_factors = build_daily_factors(daily_prep)

    # ===== 4. 构造日内因子并合并 =====
    print("Aggregating intraday features...")
    intraday_agg = aggregate_intraday_features(intraday_raw)

    print("Merging daily & intraday factors...")
    factor_df = merge_daily_intraday_factors(daily_factors, intraday_agg)

    # ===== 5. 计算前瞻收益（用于IC和模型评估，不用于回测交易） =====
    print("Computing forward returns...")
    factor_df = compute_forward_returns(factor_df, horizon=1)

    # ===== 6. 指定因子列并做截面标准化 =====
    factor_cols = [
        "mom_5", "mom_10", "mom_20",
        "rev_1",
        "vol_20",
        "turn_5", "turn_20",
        "illiq_20",
        "value_pe", "value_pb",
        "open_close_ret",
        "intraday_range",
        "last_hour_ret",
        "open_vol_frac",
        "vwap_gap",
    ]

    # 去掉完全缺失的因子
    factor_cols = [c for c in factor_cols if c in factor_df.columns]

    print(f"Factor columns used: {factor_cols}")

    print("Cross-sectional standardization...")
    factor_df = cross_sectional_standardize(factor_df, factor_cols)

    # ===== 7. 滚动计算因子IC & 权重 =====
    print("Computing rolling factor weights (IC-based)...")
    weight_df = compute_rolling_factor_weights(
        factor_df,
        factor_cols,
        ret_col="ret_fwd_1",
    )

    # ===== 8. 生成多因子综合得分 score =====
    print("Applying factor weights to compute score...")
    scored_df = apply_factor_weights(factor_df, factor_cols, weight_df)

    # 回测前过滤缺失值（如 price / score）
    scored_df = scored_df.dropna(subset=["open", "close", "volume", "amount", "score"])
    scored_df = scored_df.sort_values(["date", "code"]).reset_index(drop=True)

    # ===== 9. 回测 =====
    print("Starting backtest...")
    result = backtest_long_only(scored_df, factor_cols=factor_cols)

    nav_df = result.nav_df
    trades_df = result.trades

    print("Backtest finished.")
    print(nav_df.tail())
    print("Number of trades:", len(trades_df))

    # ===== 10. 绩效评估 =====
    stats = compute_performance_stats(nav_df)
    print("Performance stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    # ===== 11. 持久化输出（可选） =====
    nav_df.to_csv("backtest_nav.csv", index=False)
    trades_df.to_csv("backtest_trades.csv", index=False)
    print("Saved nav to backtest_nav.csv and trades to backtest_trades.csv")


if __name__ == "__main__":
    main()
