import pandas as pd
import numpy as np
from moudle1_dataloader import DataConfig, PeriodConfig, load_data_bundle, load_data_bundle_update
from moudle2_factorengine import FactorEngine
from moudle4_backtest import Analyzer


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