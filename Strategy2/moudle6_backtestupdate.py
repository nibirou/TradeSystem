from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class PriceLimitModel:
    main_limit: float = 0.10     # 主板
    st_limit: float = 0.05       # ST
    gem_limit: float = 0.20      # 创业板 300
    star_limit: float = 0.20     # 科创板 688

    round_tick: float = 0.01     # A股价格最小变动单位通常为 0.01（绝大多数情况）

    def _limit_rate(self, code: str, is_st: int) -> float:
        if int(is_st) == 1:
            return self.st_limit
        if code.startswith("sz.300"):
            return self.gem_limit
        if code.startswith("sh.688"):
            return self.star_limit
        return self.main_limit

    def _round_price(self, px: float) -> float:
        # 四舍五入到分
        return float(np.round(px / self.round_tick) * self.round_tick)

    def limit_prices(self, code: str, preclose: float, is_st: int) -> tuple[float, float]:
        r = self._limit_rate(code, is_st)
        up = self._round_price(preclose * (1 + r))
        dn = self._round_price(preclose * (1 - r))
        return dn, up

    def is_one_price_limit_up(self, high: float, low: float, limit_up: float, eps=1e-6) -> bool:
        return abs(high - limit_up) < eps and abs(low - limit_up) < eps and abs(high - low) < eps

    def is_one_price_limit_down(self, high: float, low: float, limit_down: float, eps=1e-6) -> bool:
        return abs(high - limit_down) < eps and abs(low - limit_down) < eps and abs(high - low) < eps
    
@dataclass
class RangeFillModel:
    slippage_bps: float = 2.0  # 0.02%

    def _apply_slippage(self, px: float, side: str) -> float:
        slip = self.slippage_bps / 10000.0
        return px * (1 + slip) if side == "buy" else px * (1 - slip)

    @staticmethod
    def bar_vwap(bar_df: pd.DataFrame) -> float:
        pv = (bar_df["close"] * bar_df["volume"]).sum()
        vv = bar_df["volume"].sum()
        if vv <= 0:
            return float(bar_df["close"].iloc[0])
        return float(pv / vv)

    def try_fill_range_order(self, day5m: pd.DataFrame, side: str,
                             price_low: float, price_high: float):
        """
        返回: None（不成交）
        或 dict(price=成交价, time=成交时间, bar_index=第几根bar)
        """
        if day5m is None or day5m.empty:
            return None

        lo_target = min(price_low, price_high)
        hi_target = max(price_low, price_high)

        # 顺序扫描 5min
        for i, (_, bar) in enumerate(day5m.iterrows()):
            bar_low = float(bar["low"])
            bar_high = float(bar["high"])

            # 是否触达区间：两个区间有交集
            if bar_high < lo_target or bar_low > hi_target:
                continue

            # 成交价：用该bar的“近似 vwap”，再 clip 到目标区间
            # 这里用单行bar，只能用 close 近似，想更精细可用 bar 内成交分布模型
            px = float(bar["close"])
            px = float(np.clip(px, lo_target, hi_target))

            px = self._apply_slippage(px, side)
            return {"price": px, "time": bar["datetime"], "bar_index": i}

        return None

@dataclass
class Order:
    code: str
    side: str            # "buy" or "sell"
    submit_date: pd.Timestamp
    valid_date: pd.Timestamp  # 当日有效（也可扩展为多日）
    qty: int

    # 区间单
    price_low: float
    price_high: float

    # 订单来源/原因
    reason: str          # "entry", "stop_loss", "take_profit", "max_hold", "manual"

    # 记录因子/关联pos等
    signal_date: pd.Timestamp | None = None
    ref_entry_price: float | None = None

class BacktesterV3:
    def __init__(self,
                 daily_df: pd.DataFrame,
                 minute5_df: pd.DataFrame,
                 factor_scored_df: pd.DataFrame,
                 trade_dates: pd.DatetimeIndex,
                 period_cfg,
                 strat_cfg,
                 cost_model,
                 price_limit_model: PriceLimitModel,
                 range_fill_model: RangeFillModel):

        self.daily = daily_df.copy()
        self.m5 = minute5_df.copy()
        self.factor = factor_scored_df.copy()
        self.trade_dates = trade_dates
        self.period = period_cfg

        self.cfg = strat_cfg
        self.cost = cost_model
        self.lim = price_limit_model
        self.fill = range_fill_model

        self.code_col = "code"
        self.date_col = "date"
        self.dt_col = "datetime"

        # 标准化
        self.daily[self.date_col] = pd.to_datetime(self.daily[self.date_col])
        self.m5[self.date_col] = pd.to_datetime(self.m5[self.date_col])
        self.factor["date"] = pd.to_datetime(self.factor["date"])

        self.daily = self.daily.sort_values([self.code_col, self.date_col])
        self.m5 = self.m5.sort_values([self.code_col, self.dt_col])
        self.factor = self.factor.sort_values([self.code_col, "date"])

        # 日辅助特征
        self._prepare_daily_features()

        # 5min 索引
        self._build_m5_index()

        # 账户状态
        self.cash = float(self.cfg.capital)
        self.reserved_cash = 0.0  # 下单占用资金（未成交）
        self.positions = {}       # code -> pos dict
        self.pending_orders: list[Order] = []

        # 记录
        self.nav_records = []
        self.daily_positions = []
        self.trades_detailed = []
        self.order_logs = []

        self.factor_map = {}
        for _, r in self.factor.iterrows():
            self.factor_map[(r["date"], r["code"])] = r.to_dict()

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
        self.m5_groups = {}
        for (code, d), g in self.m5.groupby([self.code_col, self.date_col]):
            self.m5_groups[(code, pd.to_datetime(d))] = g.sort_values(self.dt_col)

    def _get_day5m(self, code: str, d: pd.Timestamp):
        return self.m5_groups.get((code, pd.to_datetime(d)))

    def _get_daily_row(self, code: str, d: pd.Timestamp):
        df = self.daily[(self.daily["code"] == code) & (self.daily["date"] == d)]
        if df.empty:
            return None
        return df.iloc[-1]

    def _prev_trade_date(self, d: pd.Timestamp):
        idx = self.trade_dates.get_loc(d)
        if idx <= 0:
            return None
        return self.trade_dates[idx - 1]

    def _calc_lot_shares(self, budget: float, price: float) -> int:
        raw = int(budget / price)
        lots = raw // self.cfg.lot_size
        return lots * self.cfg.lot_size

    # ----------------------------
    # 因子截面过滤 + 选股（同你之前逻辑）
    # ----------------------------
    def _get_cross_section(self, d: pd.Timestamp) -> pd.DataFrame:
        f = self.factor[self.factor["date"] == d].copy()
        dd = self.daily[self.daily["date"] == d].copy()
        cross = f.merge(dd, on=["date", "code"], how="inner")

        if "tradestatus" in cross.columns:
            cross = cross[cross["tradestatus"] == 1]
        if "isST" in cross.columns:
            cross = cross[cross["isST"] == 0]

        cross = cross[(cross["vol_20d"].notna()) & (cross["vol_20d"] <= self.cfg.max_vol_20d)]
        if "turn_20d" in cross.columns:
            cross = cross[
                (cross["turn_20d"].notna()) &
                (cross["turn_20d"] >= self.cfg.min_turn_20d) &
                (cross["turn_20d"] <= self.cfg.max_turn_20d)
            ]

        if len(cross) > 0 and "L1" in cross.columns:
            thr = cross["L1"].quantile(self.cfg.low_position_quantile)
            cross = cross[cross["L1"] <= thr]

        cross = cross.sort_values("alpha_score", ascending=False)
        return cross

    # ----------------------------
    # 生成买入区间（可用你的 buy_low/buy_high 逻辑）
    # 这里给更“交易员风格”的区间：
    # - 用执行日开盘价为中心，设置一个小偏离区间（比如±0.3%）
    # - 你也可以改为基于 5min vwap/波动率动态区间
    # ----------------------------
    def _build_buy_range(self, exec_open: float) -> tuple[float, float]:
        return exec_open * (1 - 0.003), exec_open * (1 + 0.003)

    def _build_sell_range(self, ref_price: float) -> tuple[float, float]:
        # 默认给一个“接近市价”的区间，保证可成交但仍在高低内
        return ref_price * (1 - 0.002), ref_price * (1 + 0.002)

    # ----------------------------
    # 核心：订单撮合（考虑涨跌停/一字板）
    # ----------------------------
    def _match_order(self, order: Order, d: pd.Timestamp):
        dr = self._get_daily_row(order.code, d)
        if dr is None:
            return None, "no_daily"

        preclose = float(dr.get("preclose", np.nan))
        is_st = int(dr.get("isST", 0)) if "isST" in dr.index else 0
        high = float(dr["high"]); low = float(dr["low"])
        open_px = float(dr["open"]); close_px = float(dr["close"])

        if np.isnan(preclose) or preclose <= 0:
            # 没有 preclose 无法计算涨跌停，降级为不限制（你也可以选择拒绝成交）
            limit_dn, limit_up = -np.inf, np.inf
        else:
            limit_dn, limit_up = self.lim.limit_prices(order.code, preclose, is_st)

        # 一字板判断（流动性极差）：买单在一字涨停几乎不可成交；卖单在一字跌停几乎不可成交
        if order.side == "buy" and self.lim.is_one_price_limit_up(high, low, limit_up):
            return None, "one_price_limit_up_no_buy"
        if order.side == "sell" and self.lim.is_one_price_limit_down(high, low, limit_dn):
            return None, "one_price_limit_down_no_sell"

        # 用 5min 做区间撮合
        day5m = self._get_day5m(order.code, d)

        fill = self.fill.try_fill_range_order(day5m, order.side, order.price_low, order.price_high)
        if fill is None:
            # 当天未触达区间，视为未成交（当日有效撤单）
            return None, "range_not_touched"

        # 成交价再 clip 到当日 high-low（理论上已经在区间内，但加一道保险）
        px = float(np.clip(fill["price"], low, high))
        return {"price": px, "time": fill["time"]}, "filled"

    # ----------------------------
    # 处理止盈止损：触发后生成“卖出区间单”
    # 真实情况：触发后你也要下单，不一定成交（比如跌停卖不出去）
    # ----------------------------
    def _generate_exit_order_if_needed(self, code: str, d: pd.Timestamp):
        pos = self.positions[code]
        dr = self._get_daily_row(code, d)
        if dr is None:
            return

        # 用 5min 顺序扫描触发（用高低价）
        day5m = self._get_day5m(code, d)
        if day5m is None or day5m.empty:
            return

        stop_px = pos["stop_price"]
        tp_px = pos["take_profit_price"]

        # 扫描触发点（保守先止损）
        hit_reason = None
        hit_time = None
        for _, bar in day5m.iterrows():
            lo = float(bar["low"]); hi = float(bar["high"])
            if lo <= stop_px:
                hit_reason = "stop_loss"
                hit_time = bar[self.dt_col]
                ref = stop_px
                break
            if hi >= tp_px:
                hit_reason = "take_profit"
                hit_time = bar[self.dt_col]
                ref = tp_px
                break

        if hit_reason is None:
            return

        # 触发后生成卖出区间单：围绕触发价给个小区间
        sell_low, sell_high = self._build_sell_range(ref)
        qty = pos["shares"]

        self.pending_orders.append(Order(
            code=code, side="sell",
            submit_date=d, valid_date=d,
            qty=qty,
            price_low=sell_low, price_high=sell_high,
            reason=hit_reason,
            signal_date=pos["signal_date"],
            ref_entry_price=pos["entry_price"]
        ))

        self.order_logs.append({
            "date": d, "code": code, "side": "sell",
            "reason": hit_reason, "submit_time": hit_time,
            "price_low": sell_low, "price_high": sell_high,
            "qty": qty
        })

    # ----------------------------
    # 到期卖出：生成卖出区间单（收盘附近）
    # ----------------------------
    def _generate_max_hold_exit(self, code: str, d: pd.Timestamp):
        pos = self.positions[code]
        dr = self._get_daily_row(code, d)
        if dr is None:
            return
        # 以当日 close 为参考给卖出区间
        ref = float(dr["close"])
        sell_low, sell_high = self._build_sell_range(ref)

        self.pending_orders.append(Order(
            code=code, side="sell",
            submit_date=d, valid_date=d,
            qty=pos["shares"],
            price_low=sell_low, price_high=sell_high,
            reason="max_hold",
            signal_date=pos["signal_date"],
            ref_entry_price=pos["entry_price"]
        ))

        self.order_logs.append({
            "date": d, "code": code, "side": "sell",
            "reason": "max_hold",
            "price_low": sell_low, "price_high": sell_high,
            "qty": pos["shares"]
        })

    # ----------------------------
    # 每日流程：先生成卖出订单 → 撮合卖出 → 再生成买入订单 → 撮合买入 → 记账
    # ----------------------------
    def run_backtest(self):
        bt_start = pd.to_datetime(self.period.backtest_start)
        bt_end = pd.to_datetime(self.period.backtest_end)

        for d in self.trade_dates:
            if not (bt_start <= d <= bt_end):
                continue

            # 0) 清理当日有效订单容器（当天会重新生成）
            self.pending_orders = []

            # 1) 为已有持仓生成：止盈止损卖单 + 到期卖单
            for code in list(self.positions.keys()):
                pos = self.positions[code]
                pos["hold_days"] += 1

                # 先止盈止损触发（可能当天卖不出去）
                self._generate_exit_order_if_needed(code, d)

                # 再到期退出（如果当天已经触发止盈止损并卖掉，这里不会重复；卖不掉也可能叠加多张单——
                # 为简化，下面撮合时我们对同code只执行一张“优先级最高”的卖单）
                if pos["hold_days"] >= self.cfg.max_hold_days:
                    self._generate_max_hold_exit(code, d)

            # 2) 撮合卖出（先卖后买更真实）
            self._execute_sell_orders(d)

            # 3) 用前一交易日信号生成买单，并撮合买入
            prev_d = self._prev_trade_date(d)
            if prev_d is not None and (bt_start <= prev_d <= bt_end):
                self._generate_entry_orders(signal_date=prev_d, exec_date=d)
                self._execute_buy_orders(d)

            # 4) 日终记账：NAV & 持仓快照（只含真实持仓）
            self._record_daily_snapshot(d)

        nav_df = pd.DataFrame(self.nav_records).sort_values("date").reset_index(drop=True)
        trades_df = pd.DataFrame(self.trades_detailed).sort_values(["entry_date", "code"]).reset_index(drop=True)
        daily_pos_df = pd.DataFrame(self.daily_positions).sort_values(["date", "code"]).reset_index(drop=True)
        orders_df = pd.DataFrame(self.order_logs).sort_values(["date", "code"]).reset_index(drop=True)
        return nav_df, trades_df, daily_pos_df, orders_df

    # ----------------------------
    # 卖单执行：同一股票可能产生多张卖单（止损/止盈/到期）
    # 真实交易也会有冲突，这里定义优先级：止损 > 止盈 > 到期
    # ----------------------------
    def _execute_sell_orders(self, d: pd.Timestamp):
        sell_orders = [o for o in self.pending_orders if o.side == "sell" and o.valid_date == d]
        if not sell_orders:
            return

        priority = {"stop_loss": 0, "take_profit": 1, "max_hold": 2, "manual": 3}
        sell_orders.sort(key=lambda o: (priority.get(o.reason, 9)))

        executed_codes = set()
        for o in sell_orders:
            if o.code not in self.positions:
                continue
            if o.code in executed_codes:
                continue  # 同code只执行一张卖单

            fill, status = self._match_order(o, d)
            if status != "filled":
                # 卖不出去：真实世界会出现（尤其跌停），这里记录但不平仓
                self.order_logs.append({"date": d, "code": o.code, "side": "sell",
                                        "reason": o.reason, "status": status})
                continue

            exit_price = fill["price"]
            self._close_position(o.code, d, exit_price, o.reason, fill.get("time"))
            executed_codes.add(o.code)

    def _close_position(self, code: str, d: pd.Timestamp, exit_price: float, reason: str, exit_time):
        pos = self.positions[code]
        shares = int(pos["shares"])
        trade_value = float(exit_price) * shares
        exit_cost = self.cost.sell_cost(trade_value)
        cash_in = trade_value - exit_cost
        self.cash += cash_in

        pnl = (exit_price - pos["entry_price"]) * shares - pos["entry_cost"] - exit_cost
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
            "exit_cost": exit_cost,
            "pnl": pnl,
            "ret": ret,
            "hold_days": pos["hold_days"],
            "exit_reason": reason,
            "exit_time": exit_time,
            "stop_price": pos["stop_price"],
            "take_profit_price": pos["take_profit_price"],
        }
        for k, v in pos["factor_snapshot"].items():
            rec[f"fac_{k}"] = v
        self.trades_detailed.append(rec)
        del self.positions[code]

    # ----------------------------
    # 生成买单：signal_date 选股，在 exec_date 挂区间买单
    # 现金占用：下单时先预占预算，避免“理论现金够但实际重复下单”
    # ----------------------------
    def _generate_entry_orders(self, signal_date: pd.Timestamp, exec_date: pd.Timestamp):
        capacity = self.cfg.max_positions - len(self.positions)
        if capacity <= 0:
            return

        cross = self._get_cross_section(signal_date)
        if cross.empty:
            return

        cross = cross[~cross["code"].isin(self.positions.keys())]
        if cross.empty:
            return

        n_new = min(self.cfg.max_new_positions_per_day, capacity, len(cross))
        candidates = cross.head(n_new).copy()

        # 可用于新开仓的资金：现金 - 预占
        free_cash = max(0.0, self.cash - self.reserved_cash)
        if free_cash <= 0:
            return

        # 等权分配到“剩余容量”
        per_budget = free_cash / max(1, capacity)

        for _, row in candidates.iterrows():
            code = row["code"]
            dr = self._get_daily_row(code, exec_date)
            if dr is None:
                continue
            exec_open = float(dr["open"])

            buy_low, buy_high = self._build_buy_range(exec_open)

            # 预算不能太小
            budget = min(per_budget, free_cash)
            if budget < self.cfg.min_order_value:
                continue

            # 先用区间中值估算股数（更保守：用 buy_high）
            est_price = buy_high
            shares = self._calc_lot_shares(budget, est_price)
            if shares <= 0:
                continue

            # 预估买入总成本（用 buy_high 保守估算）
            est_value = shares * est_price
            est_cost = self.cost.buy_cost(est_value)
            reserve_need = est_value + est_cost

            if reserve_need > free_cash:
                continue

            # 预占现金
            self.reserved_cash += reserve_need
            free_cash -= reserve_need

            self.pending_orders.append(Order(
                code=code, side="buy",
                submit_date=exec_date, valid_date=exec_date,
                qty=shares,
                price_low=buy_low, price_high=buy_high,
                reason="entry",
                signal_date=signal_date
            ))

            self.order_logs.append({
                "date": exec_date, "code": code, "side": "buy",
                "reason": "entry",
                "price_low": buy_low, "price_high": buy_high,
                "qty": shares,
                "reserved_cash": reserve_need
            })

    def _execute_buy_orders(self, d: pd.Timestamp):
        buy_orders = [o for o in self.pending_orders if o.side == "buy" and o.valid_date == d]
        if not buy_orders:
            # 当天无买单，释放预占（理论上不该发生；这里防御）
            self.reserved_cash = 0.0
            return

        actually_reserved_left = self.reserved_cash

        for o in buy_orders:
            # 撮合
            fill, status = self._match_order(o, d)
            if status != "filled":
                self.order_logs.append({"date": d, "code": o.code, "side": "buy", "reason": o.reason, "status": status})
                continue

            buy_price = fill["price"]
            shares = int(o.qty)
            trade_value = buy_price * shares
            entry_cost = self.cost.buy_cost(trade_value)
            total = trade_value + entry_cost

            # 再次检查现金（理论上预占保证了足够，但价格可能略高于估计）
            if total > self.cash:
                self.order_logs.append({"date": d, "code": o.code, "side": "buy", "reason": o.reason, "status": "cash_insufficient_at_fill"})
                continue

            # 真正扣现金
            self.cash -= total

            fac = self.factor_map.get((o.signal_date, o.code), {})

            # 建仓
            self.positions[o.code] = {
                "code": o.code,
                "signal_date": o.signal_date,
                "entry_date": d,
                "entry_price": float(buy_price),
                "shares": shares,
                "entry_cost": float(entry_cost),
                "hold_days": 0,

                "stop_price": float(buy_price * (1 + self.cfg.stop_loss_pct)),
                "take_profit_price": float(buy_price * (1 + self.cfg.take_profit_pct)),

                "factor_snapshot": {
                    # "L1": float(o.signal_date is not None) and np.nan,  # 可在此处注入 row 因子（见下方说明）
                    k: fac.get(k, np.nan) for k in ["L1","L2","S1","S2","S3","S4","F1","F2","R1","alpha_score"]
                }
            }

            # 记录买入成交
            self.order_logs.append({"date": d, "code": o.code, "side": "buy",
                                    "status": "filled", "price": buy_price, "qty": shares, "time": fill.get("time")})

        # 日终释放所有预占（因为订单当日有效）
        self.reserved_cash = 0.0

    # ----------------------------
    # 记账：NAV & 每日持仓快照（只含真实持仓）
    # ----------------------------
    def _record_daily_snapshot(self, d: pd.Timestamp):
        holdings_value = 0.0
        for code, pos in self.positions.items():
            dr = self._get_daily_row(code, d)
            if dr is None:
                continue
            holdings_value += float(dr["close"]) * pos["shares"]

        nav = self.cash + holdings_value

        self.nav_records.append({
            "date": d,
            "nav": nav,
            "cash": self.cash,
            "holdings_value": holdings_value,
            "num_positions": len(self.positions)
        })

        for code, pos in self.positions.items():
            dr = self._get_daily_row(code, d)
            if dr is None:
                continue
            open_px, high_px, low_px, close_px = map(float, [dr["open"], dr["high"], dr["low"], dr["close"]])

            mv = close_px * pos["shares"]
            pnl = (close_px - pos["entry_price"]) * pos["shares"] - pos["entry_cost"]
            ret = pnl / (pos["entry_price"] * pos["shares"] + 1e-9)

            self.daily_positions.append({
                "date": d, "code": code,
                "signal_date": pos["signal_date"],
                "entry_date": pos["entry_date"],
                "entry_price": pos["entry_price"],
                "shares": pos["shares"],
                "hold_days": pos["hold_days"],

                "open": open_px, "high": high_px, "low": low_px, "close": close_px,

                "market_value": mv,
                "pnl": pnl,
                "ret": ret,
                "weight": mv / nav if nav > 0 else 0.0,

                "stop_price": pos["stop_price"],
                "take_profit_price": pos["take_profit_price"],
            })
