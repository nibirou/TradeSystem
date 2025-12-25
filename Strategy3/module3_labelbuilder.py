# module4_labelbuilder.py
import pandas as pd
import numpy as np

class LabelBuilder:
    def __init__(self, minute5: pd.DataFrame, trade_dates: pd.DatetimeIndex):
        self.minute = minute5.copy()
        self.trade_dates = trade_dates

        self.minute.sort_values(["code", "datetime"], inplace=True)

    def _get_future_trade_dates(self, t: pd.Timestamp, n: int):
        idx = self.trade_dates.get_loc(t)
        return self.trade_dates[idx + 1: idx + 1 + n]

    def build_labels(
        self,
        trade_date: pd.Timestamp,
        horizon_n: int = 10,
        stop_loss: float = -0.10,
        entry_time: str = "09:35"
    ) -> pd.DataFrame:
        """
        返回 index=code, columns=[y, exit_type]
        """
        t = pd.to_datetime(trade_date)
        future_days = self._get_future_trade_dates(t, horizon_n)

        # --- t日买入价 ---
        entry = self.minute[
            (self.minute["date"] == t) &
            (self.minute["datetime"].dt.strftime("%H:%M") == entry_time)
        ][["code", "open"]].rename(columns={"open": "buy_px"})

        if entry.empty:
            return pd.DataFrame()

        results = []

        for code, buy_px in entry.set_index("code")["buy_px"].items():
            df_fut = self.minute[
                (self.minute["code"] == code) &
                (self.minute["date"].isin(future_days))
            ].sort_values("datetime")

            if df_fut.empty:
                continue

            # --- 止损检测 ---
            min_low = df_fut["low"].min()
            if min_low / buy_px - 1 <= stop_loss:
                y = stop_loss
                exit_type = "stop_loss"
            else:
                best_close = df_fut["close"].max()
                y = best_close / buy_px - 1
                exit_type = "take_best"

            results.append((code, y, exit_type))

        return pd.DataFrame(
            results,
            columns=["code", "y", "exit_type"]
        ).set_index("code")
