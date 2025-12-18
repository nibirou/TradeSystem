import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from moudle1_dataloader import DataConfig, PeriodConfig, load_data_bundle
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

# position = {
#     "code": str,
#     "entry_date": Timestamp,
#     "entry_price": float,
#     "shares": int,
#     "signal_date": Timestamp,
#     "buy_low": float,
#     "buy_high": float,
#     "stop_loss": float,
#     "take_profit": float,
# }
class Backtester:
    def __init__(self, daily_df: pd.DataFrame, factor_df: pd.DataFrame,
                 trade_dates: pd.DatetimeIndex,
                 period_cfg, strat_cfg: StrategyConfig):

        self.daily = daily_df.copy()
        # self.daily = (
        #     self.daily
        #         .sort_values(["code", "date"])
        #         .drop_duplicates(subset=["code", "date"], keep="last")
        #         .reset_index(drop=True)
        # )
        self.factor = factor_df.copy()
        self.trade_dates = trade_dates
        self.period = period_cfg
        self.cfg = strat_cfg

        self.code_col = "code"
        self.date_col = "date"

        # 预计算日收益 & 20 日波动率、20 日换手
        self._prepare_daily_features()

        # 当前持仓 + 交易记录
        self.positions = []   # list of dict
        # self.trades = []      # list of dict
        self.nav_records = [] # 每日净值
        
        # 新增：每日持仓快照
        self.daily_positions = []   # list of dict
        # 新增：详细交易日志（比 self.trades 更全）
        self.trade_logs = []

    # ----------------------------------------
    # 日频辅助特征
    # ----------------------------------------
    def _prepare_daily_features(self):
        df = self.daily
        df = df.sort_values([self.code_col, self.date_col])

        df["ret"] = df.groupby("code")["close"].pct_change()
        df["vol_20d"] = df.groupby("code")["ret"].rolling(20).std().reset_index(0, drop=True)
        df["turn_20d"] = df.groupby("code")["turn"].rolling(20).mean().reset_index(0, drop=True)

        self.daily = df

    # ----------------------------------------
    # 获取某日的截面数据（daily + factor）
    # ----------------------------------------
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

    # ----------------------------------------
    # 单日选股：返回当日信号 stock list（按 alpha_score 排序）
    # ----------------------------------------
    def _select_stocks(self, trade_date: pd.Timestamp, held_codes):
        cross = self._get_cross_section(trade_date)

        if cross.empty:
            return []

        # 排除已经持仓的股票
        cross = cross[~cross["code"].isin(held_codes)]

        cross = cross.sort_values("alpha_score", ascending=False)
        selected = cross.head(self.cfg.max_new_positions_per_day)

        return selected

    # ----------------------------------------
    # 获取下一交易日
    # ----------------------------------------
    def _get_next_trade_date(self, current_date: pd.Timestamp):
        dates = self.trade_dates
        idx = dates.get_loc(current_date)
        if idx >= len(dates) - 1:
            return None
        return dates[idx + 1]

    # ----------------------------------------
    # 更新现有持仓：止盈/止损/超期平仓
    # 使用当天收盘价来判断（可升级为用 high/low 模拟盘中触发）
    # ----------------------------------------
    def _update_positions(self, current_date: pd.Timestamp):
        df_d = self.daily[self.daily[self.date_col] == current_date]
        price_map = df_d.set_index("code")["close"].to_dict()

        new_positions = []
        daily_pnl = 0.0

        for pos in self.positions:
            code = pos["code"]
            if code not in price_map:
                new_positions.append(pos)
                continue

            close_price = price_map[code]
            ret = (close_price - pos["entry_price"]) / pos["entry_price"]
            holding_days = (current_date - pos["entry_date"]).days + 1

            exit_flag = False
            exit_reason = None

            if ret <= self.cfg.stop_loss_pct:
                exit_flag = True
                exit_reason = "stop_loss"
            elif ret >= self.cfg.take_profit_pct:
                exit_flag = True
                exit_reason = "take_profit"
            elif holding_days >= self.cfg.max_hold_days:
                exit_flag = True
                exit_reason = "max_hold"

            if exit_flag:
                # 平仓
                exit_value = close_price * pos["shares"]
                entry_value = pos["entry_price"] * pos["shares"]
                pnl = exit_value - entry_value
                daily_pnl += pnl

                self.trade_logs.append({
                    "code": code,
                    "signal_date": pos["signal_date"],
                    "entry_date": pos["entry_date"],
                    "exit_date": current_date,

                    "entry_price": pos["entry_price"],
                    "exit_price": close_price,
                    "shares": pos["shares"],

                    "holding_days": holding_days,
                    "pnl": pnl,
                    "ret": ret,
                    "exit_reason": exit_reason,

                    "buy_low": pos["buy_low"],
                    "buy_high": pos["buy_high"],
                    "stop_loss": pos["stop_loss"],
                    "take_profit": pos["take_profit"],

                    # 因子快照全部展开
                    **pos["factor_snapshot"],
                })

            else:
                new_positions.append(pos)

        self.positions = new_positions
        return daily_pnl

    # ----------------------------------------
    # 记录每日净值
    # ----------------------------------------
    def _record_nav(self, current_date: pd.Timestamp, daily_pnl: float, last_nav: float):
        # 计算当前持仓市值
        df_d = self.daily[self.daily[self.date_col] == current_date]
        price_map = df_d.set_index("code")["close"].to_dict()

        holdings_value = 0.0
        for pos in self.positions:
            code = pos["code"]
            if code not in price_map:
                continue
            holdings_value += price_map[code] * pos["shares"]

        cash = last_nav - holdings_value
        nav = holdings_value + cash + daily_pnl  # 简化处理：daily_pnl 叠加

        self.nav_records.append({
            "date": current_date,
            "nav": nav,
            "holdings_value": holdings_value,
            "cash": nav - holdings_value
        })

        return nav
    
    # ----------------------------------------
    # 记录每日持仓快照
    # ----------------------------------------
    def _record_daily_positions(self, current_date: pd.Timestamp, nav: float):
        # 由于hs300 zz500可能存在股票重合，因此df_d可能存在同一只股票有两条相同记录的情况，这时需要删去一条
        # hs300 / zz500 股票池交叉
        df_d = self.daily[self.daily[self.date_col] == current_date]
        
        # dup = df_d["code"].value_counts()
        # print(dup[dup > 1].head(10))
        
        price_map = df_d.set_index("code")[["open", "high", "low", "close"]].to_dict("index")

        for pos in self.positions:
            code = pos["code"]
            if code not in price_map:
                continue

            prices = price_map[code]
            close_price = prices["close"]

            holding_days = (current_date - pos["entry_date"]).days + 1
            cost = pos["entry_price"]
            pnl = (close_price - cost) * pos["shares"]
            ret = (close_price - cost) / cost
            position_value = close_price * pos["shares"]
            weight = position_value / nav if nav > 0 else 0

            self.daily_positions.append({
                "date": current_date,
                "code": code,

                # 价格
                "open": prices["open"],
                "high": prices["high"],
                "low": prices["low"],
                "close": close_price,

                # 持仓信息
                "entry_date": pos["entry_date"],
                "entry_price": cost,
                "shares": pos["shares"],
                "holding_days": holding_days,

                # 盈亏
                "position_value": position_value,
                "pnl": pnl,
                "ret": ret,
                "weight": weight,

                # 风控状态
                "stop_loss": pos["stop_loss"],
                "take_profit": pos["take_profit"],
                "hit_stop_loss": close_price <= pos["stop_loss"],
                "hit_take_profit": close_price >= pos["take_profit"],

                # 因子（用于事后分析）
                **pos["factor_snapshot"],
            })


    # ----------------------------------------
    # 主回测函数
    # ----------------------------------------
    def run_backtest(self):
        nav = self.cfg.capital
        self.positions = []
        # self.trades = []
        self.nav_records = []

        # 只在回测区间内循环
        bt_start = pd.to_datetime(self.period.backtest_start)
        bt_end = pd.to_datetime(self.period.backtest_end)

        for d in self.trade_dates:
            if not (bt_start <= d <= bt_end):
                continue

            # 1) 更新持仓（用当天收盘价）
            daily_pnl = self._update_positions(d)

            # 2) 选股 & 开新仓（在下一交易日 open 买入）
            held_codes = [p["code"] for p in self.positions]
            selected = self._select_stocks(d, held_codes)

            next_d = self._get_next_trade_date(d)
            if next_d is not None and len(selected) > 0:
                # 下一交易日开盘价
                df_next = self.daily[self.daily[self.date_col] == next_d]
                open_map = df_next.set_index("code")["open"].to_dict()

                # 剩余额度
                max_add = max(0, self.cfg.max_positions - len(self.positions))
                selected = selected.head(min(max_add, len(selected)))

                if len(selected) > 0 and max_add > 0:
                    capital_per_pos = nav / self.cfg.max_positions

                    for _, row in selected.iterrows():
                        code = row["code"]
                        if code not in open_map:
                            continue

                        open_price = open_map[code]
                        if open_price <= 0:
                            continue

                        shares = int(capital_per_pos / open_price)
                        if shares <= 0:
                            continue

                        buy_low = open_price * (1 - 0.003)
                        buy_high = open_price * (1 + 0.003)
                        stop_loss = open_price * (1 + self.cfg.stop_loss_pct)
                        take_profit = open_price * (1 + self.cfg.take_profit_pct)

                        pos = {
                            # 基础信息
                            "code": code,
                            "signal_date": d,
                            "entry_date": next_d,
                            "entry_price": open_price,
                            "shares": shares,

                            # 风控参数
                            "buy_low": buy_low,
                            "buy_high": buy_high,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,

                            # 状态字段
                            "holding_days": 0,
                            "exit_date": None,
                            "exit_price": None,
                            "exit_reason": None,

                            # 买入时的因子快照（非常重要）
                            "factor_snapshot": {
                                "L1": row["L1"],
                                "L2": row["L2"],
                                "S1": row["S1"],
                                "S2": row["S2"],
                                "S3": row["S3"],
                                "S4": row["S4"],
                                "F1": row["F1"],
                                "F2": row["F2"],
                                "R1": row["R1"],
                                "alpha_score": row["alpha_score"],
                            }
                        }
                        self.positions.append(pos)

            # 3) 记录净值
            nav = self._record_nav(d, daily_pnl, nav)
            
            # 4) 每日持仓
            self._record_daily_positions(d, nav)

        nav_df = pd.DataFrame(self.nav_records).sort_values("date")
        # trades_df = pd.DataFrame(self.trades)
        trade_logs_df = pd.DataFrame(self.trade_logs)
        daily_positions_df = pd.DataFrame(self.daily_positions)

        # return nav_df, trades_df, trade_logs_df, daily_positions_df
        return nav_df, trade_logs_df, daily_positions_df

def evaluate_future_window(trades_df: pd.DataFrame, daily_df: pd.DataFrame,
                           horizon: int = 15) -> pd.DataFrame:
    """
    对每一笔交易，以 entry_date 为起点，计算未来 horizon 天的最大 / 最小收益。
    用收盘价作为近似（如需更精细可用 high / low）。
    """
    if trades_df.empty:
        return trades_df

    daily = daily_df.copy()
    daily = daily.sort_values(["code", "date"])

    daily_group = daily.groupby("code")

    future_max_list = []
    future_min_list = []

    for idx, row in trades_df.iterrows():
        code = row["code"]
        entry_date = pd.to_datetime(row["entry_date"])
        entry_price = row["entry_price"]

        if code not in daily_group.groups:
            future_max_list.append(np.nan)
            future_min_list.append(np.nan)
            continue

        df_c = daily_group.get_group(code)
        df_c = df_c[df_c["date"] >= entry_date].copy()
        df_c = df_c.head(horizon)  # 未来 horizon 天

        if df_c.empty:
            future_max_list.append(np.nan)
            future_min_list.append(np.nan)
            continue

        max_close = df_c["close"].max()
        min_close = df_c["close"].min()

        future_max_ret = (max_close - entry_price) / entry_price
        future_min_ret = (min_close - entry_price) / entry_price

        future_max_list.append(future_max_ret)
        future_min_list.append(future_min_ret)

    trades_df = trades_df.copy()
    trades_df["future_max_ret_15d"] = future_max_list
    trades_df["future_min_ret_15d"] = future_min_list

    return trades_df

if __name__ == "__main__":
    # 1) 数据加载（模块 1）
    data_cfg = DataConfig(
        base_dir="./data_baostock",
    )
    period_cfg = PeriodConfig(
        factor_start="2025-09-01",
        factor_end="2025-12-01",
        backtest_start="2025-10-01",
        backtest_end="2025-11-30",
    )
    # bundle = load_data_bundle(data_cfg, period_cfg, pools=("hs300", "zz500"))
    bundle = load_data_bundle(data_cfg, period_cfg, pools=("hs300", "zz500"))
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

    # 4) 回测（模块 3.2）
    bt = Backtester(daily, factor_scored, trade_dates, period_cfg, strat_cfg)
    nav_df, trade_logs_df, daily_positions_df = bt.run_backtest()
    
    # 保存
    output_dir = "./Strategy2/backtest_output"
    os.makedirs(output_dir, exist_ok=True)

    nav_df.to_csv(f"{output_dir}/nav.csv", index=False)
    # trades_df.to_csv(f"{output_dir}/trades_simple.csv", index=False)
    trade_logs_df.to_csv(f"{output_dir}/trades_detailed.csv", index=False)
    daily_positions_df.to_csv(f"{output_dir}/daily_positions.csv", index=False)

    print("✅ 回测结果已保存到 backtest_output/")

    # 5) 评估未来 15 天是否曾走强（模块 3.3）
    trades_eval = evaluate_future_window(trade_logs_df, daily, horizon=15)

    # print(nav_df.tail()) # nav(net asset value) 净资产
    # print(trades_eval.head())
    
    print(nav_df) # nav(net asset value) 净资产
    print(trades_eval)

    # 回测分析
    an = Analyzer(nav_df, trade_logs_df)
    an.run_all()