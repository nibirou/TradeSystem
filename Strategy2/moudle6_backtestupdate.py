from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class CostModel:
    """
    更贴近 A 股：
    - 佣金：双边收取（买卖都有），通常万 1~万 3；这里默认万 2
    - 印花税：仅卖出收取，当前 A 股为 0.05%（千分之 0.5）——注意如未来政策变化你可改参数
    - 过户费：沪市有、深市没有；这里简化为 0（你可自行扩）
    """
    commission_rate: float = 0.0002       # 佣金
    stamp_tax_rate: float = 0.0005        # 印花税（卖出）
    min_commission: float = 5.0           # 最低 5 元（常见券商规则）

    def buy_cost(self, trade_value: float) -> float:
        comm = max(self.min_commission, trade_value * self.commission_rate)
        return comm

    def sell_cost(self, trade_value: float) -> float:
        comm = max(self.min_commission, trade_value * self.commission_rate)
        stamp = trade_value * self.stamp_tax_rate
        return comm + stamp


@dataclass
class ExecutionModel:
    """
    用 5min K 线来模拟“更像真实人/程序的成交价”。
    关键保证：成交价一定落在当日 high-low 之间。

    规则（默认）：
    - 买入：用开盘后前 N 根 5min 的 VWAP（更像挂单+成交均价）
    - 卖出：用收盘前最后 N 根 5min 的 VWAP（更像择机退出）
    - 止盈止损触发：用 5min high/low 顺序扫描，
        * 止损触发：用 stop_price（可加滑点）
        * 止盈触发：用 take_profit_price（可加滑点）
    """
    buy_vwap_bars: int = 6     # 开盘后 6 根 5min ≈ 30 分钟
    sell_vwap_bars: int = 6    # 收盘前 6 根 5min ≈ 30 分钟
    slippage_bps: float = 2.0  # 滑点（基点），2 bps = 0.02%

    def _apply_slippage(self, price: float, side: str) -> float:
        slip = self.slippage_bps / 10000.0
        if side.lower() == "buy":
            return price * (1 + slip)
        else:
            return price * (1 - slip)

    @staticmethod
    def vwap(df_5m: pd.DataFrame) -> float:
        # 用 close*volume 近似
        pv = (df_5m["close"] * df_5m["volume"]).sum()
        vv = df_5m["volume"].sum()
        if vv <= 0:
            return float(df_5m["close"].iloc[0])
        return float(pv / vv)

    def get_buy_price(self, day_5m: pd.DataFrame, fallback_open: float) -> float:
        if day_5m is None or day_5m.empty:
            return self._apply_slippage(float(fallback_open), "buy")
        n = min(self.buy_vwap_bars, len(day_5m))
        px = self.vwap(day_5m.iloc[:n])
        # 保证在当日范围内
        lo, hi = float(day_5m["low"].min()), float(day_5m["high"].max())
        px = float(np.clip(px, lo, hi))
        return self._apply_slippage(px, "buy")

    def get_sell_price(self, day_5m: pd.DataFrame, fallback_close: float) -> float:
        if day_5m is None or day_5m.empty:
            return self._apply_slippage(float(fallback_close), "sell")
        n = min(self.sell_vwap_bars, len(day_5m))
        px = self.vwap(day_5m.iloc[-n:])
        lo, hi = float(day_5m["low"].min()), float(day_5m["high"].max())
        px = float(np.clip(px, lo, hi))
        return self._apply_slippage(px, "sell")


@dataclass
class StrategyConfigV2:
    capital: float = 80000.0
    max_positions: int = 6
    max_new_positions_per_day: int = 3

    # 持有周期（交易日数量）
    max_hold_days: int = 10

    # 止盈止损（相对 entry_price）
    stop_loss_pct: float = -0.05
    take_profit_pct: float = 0.10

    # 过滤（可继续扩充）
    max_vol_20d: float = 0.06
    min_turn_20d: float = 0.005
    max_turn_20d: float = 0.30
    low_position_quantile: float = 0.3

    # 下单资金分配
    # - 等权：按可用现金和目标持仓数进行分配
    lot_size: int = 100            # A 股 1 手=100 股
    min_order_value: float = 2000  # 过小订单没意义，过滤
    
class BacktesterV2:
    def __init__(self,
                 daily_df: pd.DataFrame,
                 minute5_df: pd.DataFrame,
                 factor_scored_df: pd.DataFrame,
                 trade_dates: pd.DatetimeIndex,
                 period_cfg,
                 strat_cfg: StrategyConfigV2,
                 cost_model: CostModel,
                 exec_model: ExecutionModel):
        self.daily = daily_df.copy()
        self.m5 = minute5_df.copy()
        self.factor = factor_scored_df.copy()
        self.trade_dates = trade_dates
        self.period = period_cfg

        self.cfg = strat_cfg
        self.cost = cost_model
        self.exec = exec_model

        self.code_col = "code"
        self.date_col = "date"
        self.dt_col = "datetime"

        # 基础清洗与排序
        self.daily[self.date_col] = pd.to_datetime(self.daily[self.date_col])
        self.m5[self.date_col] = pd.to_datetime(self.m5[self.date_col])
        self.factor["date"] = pd.to_datetime(self.factor["date"])

        self.daily = self.daily.sort_values([self.code_col, self.date_col])
        self.m5 = self.m5.sort_values([self.code_col, self.dt_col])
        self.factor = self.factor.sort_values([self.code_col, "date"])

        # 预计算日辅助（波动、换手）
        self._prepare_daily_features()

        # 建 5min 索引：按 (code,date) 快速取当日 5min
        self._build_m5_index()

        # 账户状态
        self.cash = float(self.cfg.capital)
        self.positions = {}  # code -> position dict

        # 记录
        self.nav_records = []
        self.trades_detailed = []
        self.daily_positions = []

    def _prepare_daily_features(self):
        df = self.daily
        df["ret"] = df.groupby("code")["close"].pct_change()
        df["vol_20d"] = df.groupby("code")["ret"].rolling(20).std().reset_index(0, drop=True)
        if "turn" in df.columns:
            df["turn_20d"] = df.groupby("code")["turn"].rolling(20).mean().reset_index(0, drop=True)
        else:
            df["turn_20d"] = np.nan
        self.daily = df

    def _build_m5_index(self):
        # groupby 对象缓存：后续 day_5m = self.m5_groups.get((code,date))
        self.m5_groups = {}
        for (code, d), g in self.m5.groupby([self.code_col, self.date_col]):
            self.m5_groups[(code, pd.to_datetime(d))] = g.sort_values(self.dt_col)

    def _get_day5m(self, code: str, d: pd.Timestamp) -> pd.DataFrame:
        return self.m5_groups.get((code, pd.to_datetime(d)))

    def _get_daily_row(self, code: str, d: pd.Timestamp):
        df = self.daily[(self.daily[self.code_col] == code) & (self.daily[self.date_col] == d)]
        if df.empty:
            return None
        # 若异常重复，取最后一条
        return df.iloc[-1]

    def _get_cross_section(self, d: pd.Timestamp) -> pd.DataFrame:
        # 因子截面
        f = self.factor[self.factor["date"] == d].copy()
        # 日频截面（用于 tradestatus/isST/vol/turn）
        dd = self.daily[self.daily[self.date_col] == d].copy()
        cross = f.merge(dd, on=["date", "code"], how="inner")

        # 过滤：停牌/ST
        if "tradestatus" in cross.columns:
            cross = cross[cross["tradestatus"] == 1]
        if "isST" in cross.columns:
            cross = cross[cross["isST"] == 0]

        # 风控过滤：波动/换手
        cross = cross[
            (cross["vol_20d"].notna()) &
            (cross["vol_20d"] <= self.cfg.max_vol_20d)
        ]
        if "turn_20d" in cross.columns:
            cross = cross[
                (cross["turn_20d"].notna()) &
                (cross["turn_20d"] >= self.cfg.min_turn_20d) &
                (cross["turn_20d"] <= self.cfg.max_turn_20d)
            ]

        # 低位过滤：L1 取前 x 分位（L1 越小越低位）
        if len(cross) > 0 and "L1" in cross.columns:
            thr = cross["L1"].quantile(self.cfg.low_position_quantile)
            cross = cross[cross["L1"] <= thr]

        # 排序：alpha_score 越大越好
        cross = cross.sort_values("alpha_score", ascending=False)
        return cross

    def _next_trade_date(self, d: pd.Timestamp):
        idx = self.trade_dates.get_loc(d)
        if idx >= len(self.trade_dates) - 1:
            return None
        return self.trade_dates[idx + 1]

    # ---------------------------
    # 关键：A 股下单股数（100股一手）
    # ---------------------------
    def _calc_lot_shares(self, budget: float, price: float) -> int:
        if price <= 0:
            return 0
        raw = int(budget / price)
        lots = raw // self.cfg.lot_size
        return lots * self.cfg.lot_size

    # ---------------------------
    # 按天扫描止盈止损（5min high/low 顺序）
    # 返回：是否触发、触发价格、触发时间、原因
    # ---------------------------
    def _scan_intraday_exit(self, code: str, d: pd.Timestamp, pos: dict):
        day5m = self._get_day5m(code, d)
        if day5m is None or day5m.empty:
            return None  # 无 5min，交给到期/收盘卖逻辑

        stop_px = pos["stop_price"]
        tp_px = pos["take_profit_price"]

        # 顺序扫描，模拟盘中触发
        for _, bar in day5m.iterrows():
            lo = float(bar["low"])
            hi = float(bar["high"])
            t = bar[self.dt_col]

            # 同一根K线内同时触发的处理：保守起见先止损（更贴近风控优先）
            if lo <= stop_px:
                px = self.exec._apply_slippage(float(stop_px), "sell")
                return {"reason": "stop_loss", "price": px, "time": t}
            if hi >= tp_px:
                px = self.exec._apply_slippage(float(tp_px), "sell")
                return {"reason": "take_profit", "price": px, "time": t}

        return None

    # ---------------------------
    # 每日：先处理卖出（含止盈止损/到期），再处理买入（昨日信号）
    # ---------------------------
    def run_backtest(self):
        bt_start = pd.to_datetime(self.period.backtest_start)
        bt_end = pd.to_datetime(self.period.backtest_end)

        # 交易日循环
        for d in self.trade_dates:
            if not (bt_start <= d <= bt_end):
                continue

            # 1) 先处理卖出（更符合现实：先腾现金再买）
            self._process_exits(d)

            # 2) 再处理买入：用 d-1 的信号，在 d 执行
            prev_d = self._prev_trade_date(d)
            if prev_d is not None and (bt_start <= prev_d <= bt_end):
                self._process_entries(signal_date=prev_d, exec_date=d)

            # 3) 记录每日持仓快照 + NAV
            self._record_daily_snapshot(d)

        nav_df = pd.DataFrame(self.nav_records).sort_values("date").reset_index(drop=True)
        trades_df = pd.DataFrame(self.trades_detailed).sort_values(["entry_date", "code"]).reset_index(drop=True)
        daily_pos_df = pd.DataFrame(self.daily_positions).sort_values(["date", "code"]).reset_index(drop=True)
        return nav_df, trades_df, daily_pos_df

    def _prev_trade_date(self, d: pd.Timestamp):
        idx = self.trade_dates.get_loc(d)
        if idx <= 0:
            return None
        return self.trade_dates[idx - 1]

    # ---------------------------
    # 卖出逻辑
    # ---------------------------
    def _process_exits(self, d: pd.Timestamp):
        to_close = []

        for code, pos in self.positions.items():
            # 计算持有天数（按交易日计数更合理）
            pos["hold_days"] += 1

            # 先扫描当日 5min 是否触发止盈止损
            hit = self._scan_intraday_exit(code, d, pos)
            if hit is not None:
                to_close.append((code, hit["price"], hit["reason"], hit["time"]))
                continue

            # 若到期，则用当日“卖出 VWAP”（收盘前 N 根 5min）模拟退出
            if pos["hold_days"] >= self.cfg.max_hold_days:
                day5m = self._get_day5m(code, d)
                daily_row = self._get_daily_row(code, d)
                if daily_row is None:
                    continue
                fallback_close = float(daily_row["close"])
                px = self.exec.get_sell_price(day5m, fallback_close)
                to_close.append((code, px, "max_hold", None))

        for code, exit_price, reason, exit_time in to_close:
            self._close_position(code, d, exit_price, reason, exit_time)

    def _close_position(self, code: str, d: pd.Timestamp, exit_price: float, reason: str, exit_time):
        pos = self.positions.get(code)
        if pos is None:
            return

        shares = int(pos["shares"])
        trade_value = float(exit_price) * shares
        cost = self.cost.sell_cost(trade_value)
        cash_in = trade_value - cost

        self.cash += cash_in

        pnl = (exit_price - pos["entry_price"]) * shares - pos["entry_cost"] - cost
        ret = pnl / (pos["entry_price"] * shares + 1e-9)

        rec = {
            "code": code,
            "signal_date": pos["signal_date"],
            "entry_date": pos["entry_date"],
            "exit_date": d,
            "entry_price": pos["entry_price"],
            "exit_price": float(exit_price),
            "shares": shares,
            "entry_value": pos["entry_price"] * shares,
            "exit_value": trade_value,
            "entry_cost": pos["entry_cost"],
            "exit_cost": cost,
            "pnl": pnl,
            "ret": ret,
            "hold_days": pos["hold_days"],
            "exit_reason": reason,
            "exit_time": exit_time,
            # 风控线
            "stop_price": pos["stop_price"],
            "take_profit_price": pos["take_profit_price"],
        }
        # 因子快照（可选：很多列）
        for k, v in pos["factor_snapshot"].items():
            rec[f"fac_{k}"] = v

        self.trades_detailed.append(rec)
        del self.positions[code]

    # ---------------------------
    # 买入逻辑：signal_date 出信号，exec_date 执行
    # ---------------------------
    def _process_entries(self, signal_date: pd.Timestamp, exec_date: pd.Timestamp):
        # 持仓容量限制
        capacity = self.cfg.max_positions - len(self.positions)
        if capacity <= 0:
            return

        cross = self._get_cross_section(signal_date)
        if cross.empty:
            return

        # 排除已持仓
        cross = cross[~cross["code"].isin(self.positions.keys())]
        if cross.empty:
            return

        # 最多当日新开仓
        n_new = min(self.cfg.max_new_positions_per_day, capacity, len(cross))
        candidates = cross.head(n_new).copy()

        # 当日可用现金分配（等权分配到“目标仓位数”）
        # 注意：只用 cash 分配，保证现金不为负
        # 目标：尽量把资金均匀分到 max_positions
        per_budget = max(0.0, self.cash / max(1, capacity))

        for _, row in candidates.iterrows():
            code = row["code"]

            # 取执行日的日频、5min
            daily_row = self._get_daily_row(code, exec_date)
            if daily_row is None:
                continue
            day5m = self._get_day5m(code, exec_date)

            fallback_open = float(daily_row["open"])
            buy_price = self.exec.get_buy_price(day5m, fallback_open)

            # 预算：取 per_budget，但不能超过现金
            budget = min(per_budget, self.cash)
            if budget < self.cfg.min_order_value:
                continue

            shares = self._calc_lot_shares(budget, buy_price)
            if shares <= 0:
                continue

            trade_value = buy_price * shares
            entry_cost = self.cost.buy_cost(trade_value)
            total_cost = trade_value + entry_cost

            # 现金检查：不允许 cash 负数
            if total_cost > self.cash:
                # 现金不足：尝试缩小一档（减一手）直到可买
                while shares > 0:
                    shares -= self.cfg.lot_size
                    if shares <= 0:
                        break
                    trade_value = buy_price * shares
                    entry_cost = self.cost.buy_cost(trade_value)
                    total_cost = trade_value + entry_cost
                    if total_cost <= self.cash:
                        break
                if shares <= 0 or total_cost > self.cash:
                    continue

            # 扣现金
            self.cash -= total_cost

            # 建仓
            pos = {
                "code": code,
                "signal_date": signal_date,
                "entry_date": exec_date,
                "entry_price": float(buy_price),
                "shares": int(shares),
                "entry_cost": float(entry_cost),
                "hold_days": 0,

                "stop_price": float(buy_price * (1 + self.cfg.stop_loss_pct)),
                "take_profit_price": float(buy_price * (1 + self.cfg.take_profit_pct)),

                # 因子快照（越全越好）
                "factor_snapshot": {
                    "L1": row.get("L1", np.nan),
                    "L2": row.get("L2", np.nan),
                    "S1": row.get("S1", np.nan),
                    "S2": row.get("S2", np.nan),
                    "S3": row.get("S3", np.nan),
                    "S4": row.get("S4", np.nan),
                    "F1": row.get("F1", np.nan),
                    "F2": row.get("F2", np.nan),
                    "R1": row.get("R1", np.nan),
                    "alpha_score": row.get("alpha_score", np.nan),
                }
            }
            self.positions[code] = pos

    # ---------------------------
    # 每日记账：NAV、持仓快照（不会把未成交仓位算进去）
    # ---------------------------
    def _record_daily_snapshot(self, d: pd.Timestamp):
        # 计算持仓市值：只统计 entry_date <= d 的真实持仓
        holdings_value = 0.0
        for code, pos in self.positions.items():
            daily_row = self._get_daily_row(code, d)
            if daily_row is None:
                continue
            close_px = float(daily_row["close"])
            holdings_value += close_px * pos["shares"]

        nav = self.cash + holdings_value
        self.nav_records.append({
            "date": d,
            "nav": nav,
            "cash": self.cash,
            "holdings_value": holdings_value,
            "num_positions": len(self.positions)
        })

        # 记录每日持仓明细（用于复盘）
        for code, pos in self.positions.items():
            daily_row = self._get_daily_row(code, d)
            if daily_row is None:
                continue
            open_px = float(daily_row["open"])
            high_px = float(daily_row["high"])
            low_px = float(daily_row["low"])
            close_px = float(daily_row["close"])

            mv = close_px * pos["shares"]
            pnl = (close_px - pos["entry_price"]) * pos["shares"] - pos["entry_cost"]
            ret = pnl / (pos["entry_price"] * pos["shares"] + 1e-9)

            rec = {
                "date": d,
                "code": code,
                "entry_date": pos["entry_date"],
                "signal_date": pos["signal_date"],
                "entry_price": pos["entry_price"],
                "shares": pos["shares"],
                "hold_days": pos["hold_days"],

                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,

                "market_value": mv,
                "pnl": pnl,
                "ret": ret,
                "weight": (mv / nav) if nav > 0 else 0.0,

                "stop_price": pos["stop_price"],
                "take_profit_price": pos["take_profit_price"],
                "hit_stop_eod": close_px <= pos["stop_price"],
                "hit_tp_eod": close_px >= pos["take_profit_price"],
            }
            for k, v in pos["factor_snapshot"].items():
                rec[f"fac_{k}"] = v

            self.daily_positions.append(rec)

