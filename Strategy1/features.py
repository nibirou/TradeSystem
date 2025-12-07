# features.py
import numpy as np
import pandas as pd
from typing import List, Optional

from config import (
    FACTOR_WINSOR_LOWER,
    FACTOR_WINSOR_UPPER,
)


def _to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def preprocess_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """
    日频基础预处理：
      - 类型转换
      - 剔除停牌、ST
      - 计算日收益/对数收益
    """
    df = daily.copy()

    # 类型转换
    df = _to_numeric(
        df,
        [
            "open", "high", "low", "close", "preclose",
            "volume", "amount", "turn",
            "peTTM", "psTTM", "pcfNcfTTM", "pbMRQ",
            "tradestatus", "isST"
        ],
    )

    # 只保留正常交易日 & 非ST
    df = df[df["tradestatus"] == 1]
    df = df[df["isST"] == 0]

    # 收益
    df["ret"] = df["close"] / df["preclose"] - 1.0
    df["ln_ret"] = np.log(df["close"] / df["preclose"].replace(0, np.nan))

    # 基本排序
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    return df


def build_daily_factors(daily: pd.DataFrame) -> pd.DataFrame:
    """
    基于日频构造价量/波动/估值/流动性相关因子。
    """
    df = daily.copy()
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    g = df.groupby("code", group_keys=False)

    # 动量因子：5日/10日/20日收益
    df["mom_5"] = g["close"].apply(lambda x: x.pct_change(5))
    df["mom_10"] = g["close"].apply(lambda x: x.pct_change(10))
    df["mom_20"] = g["close"].apply(lambda x: x.pct_change(20))

    # 短期反转：昨日收益取负
    df["rev_1"] = -g["ret"].shift(1)

    # 历史波动率：20日标准差（对数收益）
    df["vol_20"] = g["ln_ret"].apply(lambda x: x.rolling(20, min_periods=10).std())

    # 换手率相关
    df["turn_5"] = g["turn"].apply(lambda x: x.rolling(5, min_periods=3).mean())
    df["turn_20"] = g["turn"].apply(lambda x: x.rolling(20, min_periods=10).mean())

    # 流动性因子：Amihud illiquidity (简单版)
    # illiq_t = |ret_t| / amount_t
    df["illiq"] = df["ret"].abs() / (df["amount"].replace(0, np.nan))

    # 20日平均 illiq
    df["illiq_20"] = g["illiq"].apply(
        lambda x: x.rolling(20, min_periods=10).mean()
    )

    # 估值因子：价值因子 = 1 / 估值
    df["value_pe"] = 1.0 / df["peTTM"].replace(0, np.nan)
    df["value_pb"] = 1.0 / df["pbMRQ"].replace(0, np.nan)

    return df


def aggregate_intraday_features(
    intraday: pd.DataFrame,
) -> pd.DataFrame:
    """
    将5分钟数据聚合为日内因子：
      - 日内 VWAP
      - open_close_ret: 收盘相对于开盘收益
      - intraday_range: (high - low) / vwap
      - last_hour_ret: 最后一小时收益近似（最后 N bar）
      - open_vol_frac: 开盘前N个bar成交量 / 当日成交量
    """
    df = intraday.copy()
    df = df.sort_values(["code", "date", "time"]).reset_index(drop=True)

    # 类型转换
    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 先算 daily 聚合，用于 open_close_ret / 日内高低
    agg_daily = df.groupby(["date", "code"]).agg(
        day_open=("open", "first"),
        day_close=("close", "last"),
        day_high=("high", "max"),
        day_low=("low", "min"),
        day_volume=("volume", "sum"),
        day_amount=("amount", "sum"),
    )

    # VWAP
    df["px_vol"] = df["close"] * df["volume"]
    vwap = df.groupby(["date", "code"]).apply(
        lambda x: x["px_vol"].sum() / (x["volume"].sum() + 1e-9)
    )
    vwap.name = "vwap"

    # 日内收益
    agg_daily["open_close_ret"] = (
        agg_daily["day_close"] / agg_daily["day_open"] - 1.0
    )

    # 日内振幅相对于vwap
    agg_daily = agg_daily.join(vwap)
    agg_daily["intraday_range"] = (
        (agg_daily["day_high"] - agg_daily["day_low"]) /
        agg_daily["vwap"].replace(0, np.nan)
    )

    # 最后一小时收益（假设 5 分钟一个bar，一小时 12 个bar）
    def _last_hour_ret(x: pd.DataFrame) -> float:
        x = x.sort_values("time")
        if len(x) < 12:
            # 不足12个bar，就用最后一个bar相对于开盘
            if len(x) < 2:
                return np.nan
            return x["close"].iloc[-1] / x["open"].iloc[0] - 1.0
        last = x.iloc[-12:]
        return last["close"].iloc[-1] / last["open"].iloc[0] - 1.0

    last_hour_ret = df.groupby(["date", "code"]).apply(_last_hour_ret)
    last_hour_ret.name = "last_hour_ret"

    # 开盘阶段成交量占比（前N个bar）
    def _open_vol_frac(x: pd.DataFrame, n: int = 6) -> float:
        x = x.sort_values("time")
        n = min(n, len(x))
        first = x.iloc[:n]
        return first["volume"].sum() / (x["volume"].sum() + 1e-9)

    open_vol_frac = df.groupby(["date", "code"]).apply(_open_vol_frac)
    open_vol_frac.name = "open_vol_frac"

    agg_daily = agg_daily.join([last_hour_ret, open_vol_frac])

    agg_daily = agg_daily.reset_index()
    return agg_daily


def merge_daily_intraday_factors(
    daily_factors: pd.DataFrame,
    intraday_agg: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    将日频因子和日内聚合因子合并。
    """
    if intraday_agg is None:
        return daily_factors

    df = daily_factors.merge(
        intraday_agg[
            [
                "date",
                "code",
                "vwap",
                "open_close_ret",
                "intraday_range",
                "last_hour_ret",
                "open_vol_frac",
            ]
        ],
        on=["date", "code"],
        how="left",
    )

    # VWAP gap: 收盘相对vwap的偏离
    df["vwap_gap"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, np.nan)

    return df


def compute_forward_returns(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    计算前瞻收益：ret_fwd_h = close_{t+h} / close_t - 1
    作为因子IC与多因子模型的目标变量。
    """
    out = df.copy()
    out = out.sort_values(["code", "date"]).reset_index(drop=True)
    g = out.groupby("code", group_keys=False)
    out[f"ret_fwd_{horizon}"] = g["close"].apply(
        lambda x: x.shift(-horizon) / x - 1.0
    )
    return out


def winsorize_series(x: pd.Series, lower: float, upper: float) -> pd.Series:
    q_low = x.quantile(lower)
    q_high = x.quantile(upper)
    return x.clip(q_low, q_high)


def cross_sectional_standardize(
    df: pd.DataFrame,
    factor_cols: List[str],
    lower: float = FACTOR_WINSOR_LOWER,
    upper: float = FACTOR_WINSOR_UPPER,
) -> pd.DataFrame:
    """
    对因子做截面去极值+标准化（z-score），按date分组。
    """
    def _process_one_day(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        for col in factor_cols:
            if col not in g.columns:
                continue
            s = g[col].astype(float)
            s = winsorize_series(s, lower, upper)
            std = s.std()
            if std == 0 or np.isnan(std):
                g[col] = 0.0
            else:
                g[col] = (s - s.mean()) / std
        return g

    df_proc = df.groupby("date", group_keys=False).apply(_process_one_day)
    df_proc = df_proc.reset_index(drop=True)
    return df_proc