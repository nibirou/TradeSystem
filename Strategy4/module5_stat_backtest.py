# module6_stat_backtest.py
import pandas as pd
import numpy as np
from module4_factor_regression import CrossSectionRegression

class StatBacktester:
    def __init__(
        self,
        factor_df: pd.DataFrame,
        label_builder,
        trade_dates: pd.DatetimeIndex,
        factor_cols,
        date_stride: int = 10,         # <<< 新增：每隔N天取样一次
        fit_mode: str = "fmb_mean",    # "pooled" 或 "fmb_mean"
        topk: int = 30,
    ):
        self.factor_df = factor_df.copy()
        self.label_builder = label_builder
        self.trade_dates = trade_dates
        self.factor_cols = factor_cols
        self.date_stride = max(1, int(date_stride))
        self.fit_mode = fit_mode
        self.topk = topk

    def _sample_dates(self, start_date, end_date):
        dates = self.trade_dates[
            (self.trade_dates >= pd.to_datetime(start_date)) &
            (self.trade_dates <= pd.to_datetime(end_date))
        ]
        if len(dates) == 0:
            return dates
        # stride采样：t0, t0+N, t0+2N...
        return dates[::self.date_stride]

    @staticmethod
    def _date_demean(df: pd.DataFrame, cols: list, date_col="date"):
        """对每个日期做去均值（相当于日期固定效应），用于 pooled 回归更稳"""
        out = df.copy()
        for c in cols:
            out[c] = out[c] - out.groupby(date_col)[c].transform("mean")
        return out

    def build_dataset(self, start_date, end_date, horizon_n=10, stop_loss=-0.10):
        """构造 stride 采样后的面板数据：columns=[date, code, y] + factor_cols"""
        sample_dates = self._sample_dates(start_date, end_date)
        all_rows = []

        for t in sample_dates:
            y_df = self.label_builder.build_labels(
                t, horizon_n=horizon_n, stop_loss=stop_loss
            )
            if y_df.empty:
                continue

            fac_t = self.factor_df[self.factor_df["date"] == t]
            if fac_t.empty:
                continue

            df = fac_t.merge(
                y_df.reset_index().rename(columns={"index": "code"}),
                on="code", how="inner"
            )
            if df.empty:
                continue

            all_rows.append(df[["date", "code", "y"] + self.factor_cols])

        if not all_rows:
            return pd.DataFrame()
        return pd.concat(all_rows, ignore_index=True)

    def fit_global_beta(self, panel_df: pd.DataFrame):
        """
        根据 fit_mode 输出一个全局 beta：
          - pooled: 一次回归（建议先 date_demean）
          - fmb_mean: 每个date截面回归 -> beta取均值
        """
        if panel_df.empty:
            return None

        reg = CrossSectionRegression(self.factor_cols)

        if self.fit_mode == "pooled":
            df = panel_df.dropna(subset=["y"] + self.factor_cols).copy()
            # 强烈建议：日期去均值（等价日期固定效应）
            df = self._date_demean(df, ["y"] + self.factor_cols, date_col="date")

            # 一次 pooled OLS
            _, beta = reg.fit_one_day(df.rename(columns={"y": "y"}))
            return beta

        if self.fit_mode == "fmb_mean":
            betas = []
            for d, g in panel_df.groupby("date"):
                _, beta_d = reg.fit_one_day(g)
                if beta_d is not None:
                    betas.append(beta_d)
            if not betas:
                return None
            beta_global = pd.concat(betas, axis=1).mean(axis=1)
            return beta_global

        raise ValueError(f"Unknown fit_mode={self.fit_mode}")

    def evaluate_in_sample(self, panel_df: pd.DataFrame, beta_global: pd.Series):
        """
        第一类回测：给定 beta_global，对每个取样日做预测、排名、统计（仍然是 in-sample 评估）
        """
        res = []
        for d, g in panel_df.groupby("date"):
            g = g.dropna(subset=["y"] + self.factor_cols).copy()
            if g.empty:
                continue
            g["y_hat"] = g[self.factor_cols].dot(beta_global)

            g = g.sort_values("y_hat", ascending=False)
            top = g.head(min(self.topk, len(g)))
            bottom = g.tail(min(self.topk, len(g)))

            ic = g["y_hat"].corr(g["y"])
            ric = g["y_hat"].rank().corr(g["y"].rank())

            res.append({
                "date": d,
                "IC": ic,
                "RankIC": ric,
                "TopK_mean_y": top["y"].mean(),
                "TopK_winrate": (top["y"] > 0).mean(),
                "LongShort": top["y"].mean() - bottom["y"].mean()
            })

        return pd.DataFrame(res).set_index("date").sort_index()
