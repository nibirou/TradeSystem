import pandas as pd
import numpy as np


# =====================
# 通用工具函数
# =====================

def winsorize(series, limit=0.01):
    """去极值：1%~99% winsorize"""
    lower = series.quantile(limit)
    upper = series.quantile(1 - limit)
    return series.clip(lower, upper)


def standardize(series):
    """Z-score 标准化"""
    return (series - series.mean()) / (series.std() + 1e-9)


def neutralize_factor(df, factor_col, industry_col=None):
    """
    对因子进行中性化（这里先不对行业中性，
    后续如果你引入行业标签，可以扩展）
    """
    s = df[factor_col]
    return s - s.mean()

class FactorEngine:
    """
    因子引擎：计算所有 Alpha 因子
    输入：
        - daily_df
        - minute5_df
        - trade_dates
    输出：
        - factor_panel: dict[date][code] = 所有因子值
    """

    def __init__(self, daily_df, minute5_df, trade_dates):
        self.daily = daily_df.copy()
        self.minute = minute5_df.copy()
        self.trade_dates = trade_dates

        self.code_col = "code"
        self.date_col = "date"
        self.datetime_col = "datetime"

        # 统一排序
        self.daily.sort_values([self.code_col, self.date_col], inplace=True)
        self.minute.sort_values([self.code_col, self.datetime_col], inplace=True)

    # ============================================================
    # 一、低位因子 Low-Position Factors
    # ============================================================

    def factor_L1_low_position(self):
        """
        20日价格区间位置（分位版）
        值越小，表示越低位
        """
        df = self.daily.copy()
        df["low_20"] = df.groupby("code")["close"].rolling(20).min().reset_index(0, drop=True)
        df["high_20"] = df.groupby("code")["close"].rolling(20).max().reset_index(0, drop=True)
        df["L1"] = (df["close"] - df["low_20"]) / (df["high_20"] - df["low_20"] + 1e-9)
        return df[["date", "code", "L1"]]

    def factor_L2_ma_ratio(self):
        """
        MA5 / MA20 偏离因子（低位反转常见）
        """
        df = self.daily.copy()
        df["MA5"] = df.groupby("code")["close"].rolling(5).mean().reset_index(0, drop=True)
        df["MA20"] = df.groupby("code")["close"].rolling(20).mean().reset_index(0, drop=True)
        df["L2"] = df["MA5"] / df["MA20"] - 1
        return df[["date", "code", "L2"]]

    # ============================================================
    # 二、走强迹象因子 Strength Factors
    # ============================================================

    def factor_S1_momentum10_skip2(self):
        """
        跳空动量：收益计算跳过最近 2 天，避免反转干扰
        """
        df = self.daily.copy()
        df["S1"] = df.groupby("code")["close"].shift(3) / df.groupby("code")["close"].shift(10) - 1
        return df[["date", "code", "S1"]]

    def factor_S2_vwap_deviation_5min(self):
        """
        使用 5min 构造 VWAP 偏离因子
        """
        df = self.minute.copy()

        # VWAP = sum(price * volume) / sum(volume)
        df["pv"] = df["close"] * df["volume"]
        minute_vwap = df.groupby(["code", df[self.datetime_col].dt.date]).apply(
            lambda x: x["pv"].sum() / (x["volume"].sum() + 1e-9)
        ).reset_index()
        minute_vwap.columns = ["code", "date", "vwap_5min"]

        # 合并进 daily
        d = self.daily.merge(minute_vwap, on=["code", "date"], how="left")

        d["S2"] = (d["close"] - d["vwap_5min"]) / (d["vwap_5min"] + 1e-9)
        return d[["date", "code", "S2"]]

    def factor_S3_volume_spike(self):
        """
        突然放量（minute5 对比前 48 根）
        """
        df = self.minute.copy()
        df["vol_ma48"] = df.groupby("code")["volume"].rolling(48).mean().reset_index(0, drop=True)
        df["vol_spike"] = df["volume"] / (df["vol_ma48"] + 1e-9)

        # 转成日频
        daily_spike = df.groupby(["code", df[self.datetime_col].dt.date])["vol_spike"].max()
        daily_spike = daily_spike.reset_index()
        daily_spike.columns = ["code", "date", "S3"]

        return daily_spike

    def factor_S4_volume_contraction(self):
        """
        缩量止跌：volume / MA20(volume)
        """
        df = self.daily.copy()
        df["vol_ma20"] = df.groupby("code")["volume"].rolling(20).mean().reset_index(0, drop=True)
        df["S4"] = df["volume"] / (df["vol_ma20"] + 1e-9)
        return df[["date", "code", "S4"]]

    # ============================================================
    # 三、资金行为因子 Fund Flow Factors
    # ============================================================

    def factor_F1_active_buy_ratio(self):
        """
        主动买入占比 —— 分钟线没有拆 bid/ask，
        用 替代方法：涨的时候 volume 较大
        """
        df = self.minute.copy()
        df["return"] = df.groupby("code")["close"].pct_change()
        df["active_buy"] = (df["return"] > 0).astype(int) * df["volume"]

        buy_ratio = df.groupby(["code", df[self.datetime_col].dt.date]).apply(
            lambda x: x["active_buy"].sum() / (x["volume"].sum() + 1e-9)
        ).reset_index()
        buy_ratio.columns = ["code", "date", "F1"]

        return buy_ratio

    def factor_F2_turnover_abnormal(self):
        df = self.daily.copy()
        df["turn_ma20"] = df.groupby("code")["turn"].rolling(20).mean().reset_index(0, drop=True)
        df["F2"] = df["turn"] / (df["turn_ma20"] + 1e-9)
        return df[["date", "code", "F2"]]

    # ============================================================
    # 四、风险因子
    # ============================================================

    def factor_R1_volatility_compression(self):
        """
        波动压缩：std5 / std20 越小越接近突破前状态
        """
        df = self.daily.copy()
        df["ret"] = df.groupby("code")["close"].pct_change()

        df["std5"] = df.groupby("code")["ret"].rolling(5).std().reset_index(0, drop=True)
        df["std20"] = df.groupby("code")["ret"].rolling(20).std().reset_index(0, drop=True)

        df["R1"] = df["std5"] / (df["std20"] + 1e-9)
        return df[["date", "code", "R1"]]

    # ============================================================
    # 五、汇总因子
    # ============================================================

    def compute_all_factors(self):
        print(">>> 计算低位因子 ...")
        L1 = self.factor_L1_low_position()
        L2 = self.factor_L2_ma_ratio()

        print(">>> 计算走强迹象因子 ...")
        S1 = self.factor_S1_momentum10_skip2()
        S2 = self.factor_S2_vwap_deviation_5min()
        S3 = self.factor_S3_volume_spike()
        S4 = self.factor_S4_volume_contraction()

        print(">>> 计算资金行为因子 ...")
        F1 = self.factor_F1_active_buy_ratio()
        F2 = self.factor_F2_turnover_abnormal()

        print(">>> 计算风险因子 ...")
        R1 = self.factor_R1_volatility_compression()

        # 合并
        dfs = [L1, L2, S1, S2, S3, S4, F1, F2, R1]
        factor_df = dfs[0]
        for df in dfs[1:]:
            factor_df = factor_df.merge(df, on=["date", "code"], how="left")

        # 去极值 & 标准化
        print(">>> 进行去极值、标准化 ...")
        for col in ["L1","L2","S1","S2","S3","S4","F1","F2","R1"]:
            factor_df[col] = factor_df.groupby("date")[col].transform(winsorize)
            factor_df[col] = factor_df.groupby("date")[col].transform(standardize)

        return factor_df
