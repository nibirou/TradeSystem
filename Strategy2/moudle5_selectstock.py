# 使用最新的T日数据，计算每只股票的策略因子综合得分，排名，给出T+1日的选股

import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from moudle1_dataloader import DataConfig, PeriodConfig, load_data_bundle, load_data_bundle_update
from moudle2_factorengine import FactorEngine
from moudle4_backtest import Analyzer

@dataclass
class StrategyConfig:
    capital: float = 80000.0           # 总资金 5-10 万，这里给个中间值
    max_positions: int = 6            # 最大持仓股票数
    max_new_positions_per_day: int = 3  # 每天最多新开仓数量

    max_hold_days: int = 10           # 最大持仓天数（1-15 天内）
    stop_loss_pct: float = -0.05      # 止损 -5%
    take_profit_pct: float = 0.10     # 止盈 +10%

    # 风控过滤参数
    max_vol_20d: float = 0.06         # 20 日波动率上限（比如日度 std < 6%）
    min_turn_20d: float = 0.005       # 20 日平均换手下限
    max_turn_20d: float = 0.30        # 20 日平均换手上限，过滤极端妖股

    # 低位过滤：取当日 L1 排名前 x% 低位
    low_position_quantile: float = 0.3

    # 选股时各类因子权重（你以后可以调参或做 IC 加权）
    w_L1: float = 0.4   # 低位重要
    w_L2: float = 0.2   # MA5/MA20
    w_S: float = 0.2    # 短期走强
    w_F: float = 0.15   # 资金介入
    w_R: float = 0.05   # 风险（负向）

def add_composite_score(factor_df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
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

    def _get_cross_section(self, trade_date: pd.Timestamp) -> pd.DataFrame:
        d = trade_date

        df_d = self.daily[self.daily[self.date_col] == d].copy()
        f_d = self.factor[self.factor["date"] == d].copy()

        cross = f_d.merge(df_d, on=["date", "code"], how="inner", suffixes=("", "_daily"))

        # 基础过滤：停牌 / ST / 价格、金额异常的
        cross = cross[cross["tradestatus"] == 1]
        if "isST" in cross.columns:
            cross = cross[cross["isST"] == 0]

        # 风控过滤：波动率 & 换手率
        cross = cross[
            (cross["vol_20d"] <= self.cfg.max_vol_20d) &
            (cross["turn_20d"] >= self.cfg.min_turn_20d) &
            (cross["turn_20d"] <= self.cfg.max_turn_20d)
        ]

        # 低位过滤：L1 越小越好，取前 x% 低位
        if len(cross) > 0:
            thresh = cross["L1"].quantile(self.cfg.low_position_quantile)
            cross = cross[cross["L1"] <= thresh]

        return cross


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

    print(trade_dates)

    # 2) 因子计算（模块 2）
    fe = FactorEngine(daily, minute5, trade_dates)
    factor_df = fe.compute_all_factors()

    # 3) 合成得分（模块 3.1）
    strat_cfg = StrategyConfig(
        capital=80000,
        max_positions=6,
        max_new_positions_per_day=3,
        max_hold_days=10,
        stop_loss_pct=-0.05,
        take_profit_pct=0.10,
    )
    factor_scored = add_composite_score(factor_df, strat_cfg)

    factor_today = factor_scored.loc[factor_scored[data_cfg.date_col] == period_cfg.factor_end]
    
    factor_today = factor_today.sort_values("alpha_score", ascending=False)
    
    print(factor_today)
    
    all_stock = pd.read_csv("./data_baostock/metadata/stock_list_all.csv")
    
    selected_stock = all_stock.loc[all_stock["code"].isin(factor_today.head(10)["code"])]
    
    print(f"{period_cfg.factor_end}日因子得分策略选股（前10名）：", selected_stock)