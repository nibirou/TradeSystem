import pandas as pd
import numpy as np
from moudle1_dataloader import DataConfig, PeriodConfig, load_data_bundle, load_data_bundle_update
from moudle2_factorengine import FactorEngine
from moudle4_backtest import Analyzer

from moudle6_backtestupdate import CostModel, ExecutionModel, StrategyConfigV2, BacktesterV2

def add_composite_score(factor_df: pd.DataFrame, cfg: StrategyConfigV2) -> pd.DataFrame:
    df = factor_df.copy()

    # 合成子类因子
    df["L_block"] = -df["L1"] * cfg.w_L1 + df["L2"] * cfg.w_L2
    df["S_block"] = (df["S1"] + df["S2"] + df["S3"] + df["S4"]) / 4.0
    df["F_block"] = (df["F1"] + df["F2"]) / 2.0
    df["R_block"] = df["R1"]

    # 标准化各 block，避免尺度差异
    for col in ["L_block", "S_block", "F_block", "R_block"]:
        df[col] = df.groupby("date")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )

    # 总得分：L + S + F - R
    df["alpha_score"] = (
        df["L_block"] * 1.0 +
        df["S_block"] * 1.0 +
        df["F_block"] * 0.8 -
        df["R_block"] * 0.5
    )

    return df

if __name__ == "__main__":
    # 1) 数据加载（模块 1）
    data_cfg = DataConfig(
        base_dir="./data_baostock",
    )
    period_cfg = PeriodConfig(
        factor_start="2025-09-01",
        factor_end="2025-12-19",
        backtest_start="2025-12-15",
        backtest_end="2025-12-16",
    )

    bundle = load_data_bundle_update(data_cfg, period_cfg, pools=("hs300", "zz500"))
    daily = bundle["daily"]
    minute5 = bundle["minute5"]
    trade_dates = bundle["trade_dates"]

    # 2) 因子计算（模块 2）
    fe = FactorEngine(daily, minute5, trade_dates)
    factor_df = fe.compute_all_factors()

    cost_model = CostModel(
        commission_rate=0.0002,
        stamp_tax_rate=0.0005,
        min_commission=5.0
    )
    exec_model = ExecutionModel(
        buy_vwap_bars=6,
        sell_vwap_bars=6,
        slippage_bps=2.0
    )

    strat_cfg_v2 = StrategyConfigV2(
        capital=80000,
        max_positions=6,
        max_new_positions_per_day=3,
        max_hold_days=10,
        stop_loss_pct=-0.05,
        take_profit_pct=0.10
    )

    factor_scored = add_composite_score(factor_df, strat_cfg_v2)

    bt2 = BacktesterV2(
        daily_df=daily,
        minute5_df=minute5,
        factor_scored_df=factor_scored,
        trade_dates=trade_dates,
        period_cfg=period_cfg,
        strat_cfg=strat_cfg_v2,
        cost_model=cost_model,
        exec_model=exec_model
    )

    nav_df, trades_df, daily_positions_df = bt2.run_backtest()

    # 保存
    import os
    out_dir = "backtest_output_v2"
    os.makedirs(out_dir, exist_ok=True)
    nav_df.to_csv(os.path.join(out_dir, "nav.csv"), index=False)
    trades_df.to_csv(os.path.join(out_dir, "trades_detailed.csv"), index=False)
    daily_positions_df.to_csv(os.path.join(out_dir, "daily_positions.csv"), index=False)

    print("✅ 已输出：nav.csv / trades_detailed.csv / daily_positions.csv")