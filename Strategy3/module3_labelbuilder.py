# module4_labelbuilder.py
import pandas as pd
import numpy as np

class LabelBuilder:
    """
    研究级 LabelBuilder：
    - 买入 / 卖出价格全部来自真实 5min bar
    - 明确处理停牌、缺失、异常
    - 不制造“虚假成交”
    """

    def __init__(self, minute5: pd.DataFrame, trade_dates: pd.DatetimeIndex):
        self.minute = minute5.copy()
        self.trade_dates = trade_dates

        # 基础清洗
        self.minute = self.minute.dropna(subset=["code", "datetime", "close"])
        self.minute.sort_values(["code", "datetime"], inplace=True)

    def _get_future_trade_dates(self, t: pd.Timestamp, n: int):
        idx = self.trade_dates.get_loc(t)
        return self.trade_dates[idx + 1: idx + 1 + n]

    # =========================
    # 买入价规则
    # =========================
    def _get_entry_price(self, df_day: pd.DataFrame) -> float | None:
        """
        默认：当日第一根可交易 5min bar 的 open
        fallback: open -> close
        """
        if df_day.empty:
            return None

        first_bar = df_day.iloc[0]

        if pd.notna(first_bar.get("open")) and first_bar["open"] > 0:
            return float(first_bar["open"])

        if pd.notna(first_bar.get("close")) and first_bar["close"] > 0:
            return float(first_bar["close"])

        return None

    # =========================
    # 构造标签
    # =========================
    def build_labels(
        self,
        trade_date: pd.Timestamp,
        horizon_n: int = 10,
        stop_loss: float = -0.10
    ) -> pd.DataFrame:
        """
        返回 index=code, columns=[y, exit_type]
        """
        t = pd.to_datetime(trade_date)
        future_days = self._get_future_trade_dates(t, horizon_n)

        results = []

        # === t 日所有股票的 5min 数据 ===
        day_t = self.minute[self.minute["date"] == t]
        if day_t.empty:
            return pd.DataFrame()

        for code, df_t in day_t.groupby("code"):
            # ---------- 买入价 ----------
            buy_px = self._get_entry_price(df_t)
            if buy_px is None or not np.isfinite(buy_px) or buy_px <= 0:
                continue

            # ---------- 未来 N 日 ----------
            df_fut = self.minute[
                (self.minute["code"] == code) &
                (self.minute["date"].isin(future_days))
            ].sort_values("datetime")

            if df_fut.empty:
                continue

            # ---------- 止损路径检查 ----------
            exited = False
            for _, bar in df_fut.iterrows():
                low = bar.get("low")
                if pd.isna(low):
                    continue
                if low / buy_px - 1 <= stop_loss:
                    sell_px = bar.get("close")
                    if pd.isna(sell_px) or sell_px <= 0:
                        sell_px = buy_px * (1 + stop_loss)
                    y = sell_px / buy_px - 1
                    results.append((code, y, "stop_loss"))
                    exited = True
                    break

            if exited:
                continue

            # ---------- 正常卖出：未来窗口内最优 close ----------
            df_fut_valid = df_fut[df_fut["close"] > 0]
            if df_fut_valid.empty:
                continue

            best_close = df_fut_valid["close"].max()
            if not np.isfinite(best_close):
                continue

            y = best_close / buy_px - 1
            results.append((code, y, "take_best"))

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(
            results,
            columns=["code", "y", "exit_type"]
        ).set_index("code")
