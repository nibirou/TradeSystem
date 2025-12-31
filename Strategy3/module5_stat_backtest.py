# module6_stat_backtest.py
import pandas as pd
import numpy as np
from module4_factor_regression import CrossSectionRegression

class StatBacktester:
    # pooled 直接把一段时间内所有股票的所有因子和收益率数据拿来回归
    # fmb_mean 在一段时间内，设置一个时间间隔N，把每个时间间隔内所有股票的所有因子和收益率数据拿来回归，之后对回归得到的收益率、残差取均值
    # per_stock_ts 按股票时间序列回归，在一段时间内，设置一个时间间隔N，把每个时间间隔内对每只股票分别做回归，之后对回归得到的收益率取均值，每只股票的残差各自保留
    def __init__(
        self,
        factor_df: pd.DataFrame,
        label_builder,
        trade_dates: pd.DatetimeIndex,
        factor_cols,
        date_stride: int = 10,         # <<< 新增：每隔N天取样一次，对于fmb_mean 和 per_stock_ts 
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
        if self.date_stride == 0:
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

    def build_dataset(self, start_date, end_date, horizon_n=10, stop_loss=-0.10, label="train"):
        """构造 stride 采样后的面板数据：columns=[date, code, y] + factor_cols"""
        if label == "inference":
            fac_t = self.factor_df[self.factor_df["date"] == start_date]
            if fac_t.empty:
                return pd.DataFrame()
            return fac_t

        if label == "backtest" or "train":
            sample_dates = self._sample_dates(start_date, end_date)
            print("sample_dates:", sample_dates)
            all_rows = []

            # 对每个采样日期（解释变量因子值纳入多因子截面回归模型的日期，选取sample_date+1到sample_date+horizon_n+1，计算预期收益率）
            # date_stride 是选取解释变量因子值纳入多因子截面回归模型的日期，的采样间隔
            # horizon_n 是对应每个采样日期，计算未来horizon_n个交易日内的预期收益率Y
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

                # print(df[["date", "code", "y"] + self.factor_cols])

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
        
        if self.fit_mode == "per_stock_ts":
            beta_global, beta_by_stock, residuals_by_stock = self.fit_per_stock_ts(
                panel_df,
                min_obs=20
            )
            # print(beta_global)
            # print(beta_by_stock)
            # print(residuals_by_stock)
            # print(f"per_stock_ts: valid stocks = {len(beta_by_stock)}")
            self.beta_by_stock = beta_by_stock
            self.residuals_by_stock = residuals_by_stock
            return beta_global, beta_by_stock, residuals_by_stock

        raise ValueError(f"Unknown fit_mode={self.fit_mode}")

    def fit_per_stock_ts(
        self,
        panel_df: pd.DataFrame,
        min_obs: int = 20 # 最小样本量
    ):
        """
        panel_df:
            columns = [date, code, y] + factor_cols
            已经是 stride = timegap 采样后的数据

        返回：
            beta_global: Series(index=factor_cols)
            beta_by_stock: dict[code] = Series(beta_i)
            residuals_by_stock: dict[code] = DataFrame(date, resid)
        """
        from module4_factor_regression import CrossSectionRegression

        factor_cols = self.factor_cols
        reg = CrossSectionRegression(factor_cols)

        beta_by_stock = {}
        residuals_by_stock = {}

        for code, g in panel_df.groupby("code"):
            g = g.dropna(subset=["y"] + factor_cols).sort_values("date")
            # 对参加回归的数据条数添加最小限制要求
            # if len(g) < max(min_obs, len(factor_cols) + 5):
            #     continue

            # === 时间序列回归（对单只股票）===
            y = g["y"].values
            X = g[factor_cols].values
            X = np.c_[np.ones(len(X)), X]

            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue

            alpha_i = coef[0]
            beta_i = pd.Series(coef[1:], index=factor_cols)
            beta_by_stock[code] = beta_i

            # === 残差保留 ===
            y_hat = X @ coef
            resid = y - y_hat

            residuals_by_stock[code] = pd.DataFrame({
                "date": g["date"].values,
                "resid": resid
            })

        if not beta_by_stock:
            return None, None, None

        beta_df = pd.concat(beta_by_stock.values(), axis=1)
        beta_global = beta_df.mean(axis=1)

        return beta_global, beta_by_stock, residuals_by_stock
    
    def evaluate_in_sample(self, panel_df: pd.DataFrame, beta_global: pd.Series):
        """
        第一类回测（in-sample）：
        - pooled / fmb_mean:
            y_hat = X · beta_global
        - per_stock_ts:
            y_hat = X · beta_global + residual_{i,t}
        """
        print("beta_global:", beta_global)
        if beta_global is None or len(beta_global) == 0:
            return pd.DataFrame()

        # === 对齐因子 ===
        beta = pd.Series(beta_global).reindex(self.factor_cols)
        valid_factors = beta.dropna().index.tolist()
        if not valid_factors:
            return pd.DataFrame()

        beta = beta.loc[valid_factors]

        # === 残差表（仅 per_stock_ts 使用） ===
        resid_long = None
        if self.fit_mode == "per_stock_ts":
            if not hasattr(self, "residuals_by_stock"):
                raise RuntimeError("per_stock_ts requires residuals_by_stock")

            pieces = []
            for code, rdf in self.residuals_by_stock.items():
                if rdf is None or rdf.empty:
                    continue
                tmp = rdf.copy()
                tmp["code"] = code
                tmp["date"] = pd.to_datetime(tmp["date"])
                pieces.append(tmp[["date", "code", "resid"]])

            if pieces:
                resid_long = pd.concat(pieces, ignore_index=True)

        res = []

        for d, g in panel_df.groupby("date"):
            g = g.dropna(subset=["y"] + valid_factors).copy()
            if g.empty:
                continue

            g["code"] = g["code"].astype(str)
            d_ts = pd.to_datetime(d)

            # === 系统性预测：X · beta_global ===
            X = g[valid_factors].astype(float).values
            g["y_hat_sys"] = X @ beta.values

            # === 个股残差项 ===
            if self.fit_mode == "per_stock_ts":
                if resid_long is None:
                    continue

                g = g.merge(
                    resid_long[resid_long["date"] == d_ts],
                    on=["date", "code"],
                    how="left"
                )
                # 如果某股票该日没有残差，用 0（不引入额外 alpha）
                g["resid"] = g["resid"].fillna(0.0)
                g["y_hat"] = g["y_hat_sys"] + g["resid"]
            else:
                g["y_hat"] = g["y_hat_sys"]

            g = g.dropna(subset=["y_hat"])
            if g.empty:
                continue

            # === 排序 排名 & 统计 ===
            g = g.sort_values("y_hat", ascending=False)
            k = min(self.topk, len(g))
            top = g.head(k)

            print(f"{d_ts}日多因子截面回归策略选股（前{k}名）：", top)

            bottom = g.tail(k)

            ic = g["y_hat"].corr(g["y"])
            ric = g["y_hat"].rank().corr(g["y"].rank())

            res.append({
                "date": d_ts,
                "IC": ic,
                "RankIC": ric,
                "TopK_mean_y": float(top["y"].mean()),
                "TopK_winrate": float((top["y"] > 0).mean()),
                "LongShort": float(top["y"].mean() - bottom["y"].mean()),
                "N": int(len(g)),
                "N_resid_used": int((g["resid"] != 0).sum()) if self.fit_mode == "per_stock_ts" else 0
            })
            print("res pre day:", res)

        if not res:
            return pd.DataFrame()

        return pd.DataFrame(res).set_index("date").sort_index()


    def inference_in_sample(self, panel_df: pd.DataFrame, beta_global: pd.Series):
        """
        In-sample 预测模块：
        - 输入：某一日（或多日）的 panel_df，只需要因子值
        - 输出：code + 预测收益率 y_hat（按 y_hat 降序）
        """

        if beta_global is None or len(beta_global) == 0:
            return pd.DataFrame(columns=["code", "y_hat"])

        # ===== 因子对齐 =====
        beta = pd.Series(beta_global).reindex(self.factor_cols)
        valid_factors = beta.dropna().index.tolist()
        if not valid_factors:
            return pd.DataFrame(columns=["code", "y_hat"])

        beta = beta.loc[valid_factors]

        df = panel_df.copy()
        if df.empty:
            return pd.DataFrame(columns=["code", "y_hat"])

        # 基础清洗
        df = df.dropna(subset=["code"] + valid_factors)
        if df.empty:
            return pd.DataFrame(columns=["code", "y_hat"])

        df["code"] = df["code"].astype(str)

        # ===== 系统性预测部分 =====
        X = df[valid_factors].astype(float).values
        df["y_hat_sys"] = X @ beta.values

        # ===== per_stock_ts：叠加个股残差 =====
        if self.fit_mode == "per_stock_ts":
            if not hasattr(self, "residuals_by_stock"):
                raise RuntimeError("per_stock_ts requires residuals_by_stock")

            # 构造 residuals long 表
            pieces = []
            for code, rdf in self.residuals_by_stock.items():
                if rdf is None or rdf.empty:
                    continue
                tmp = rdf.copy()
                tmp["code"] = code
                tmp["date"] = pd.to_datetime(tmp["date"])
                pieces.append(tmp[["date", "code", "resid"]])

            if pieces and "date" in df.columns:
                resid_long = pd.concat(pieces, ignore_index=True)
                df["date"] = pd.to_datetime(df["date"])

                df = df.merge(
                    resid_long,
                    on=["date", "code"],
                    how="left"
                )
                df["resid"] = df["resid"].fillna(0.0)
            else:
                # 没有残差就等价于 0
                df["resid"] = 0.0

            df["y_hat"] = df["y_hat_sys"] + df["resid"]

        else:
            df["y_hat"] = df["y_hat_sys"]

        # ===== 输出 =====
        out = (
            df[["code", "y_hat"]]
            .dropna(subset=["y_hat"])
            .sort_values("y_hat", ascending=False)
            .reset_index(drop=True)
        )

        return out
