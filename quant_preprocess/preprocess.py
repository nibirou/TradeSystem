# 模块 2：统一数据清洗/对齐/中性化接口 quant_preprocess/preprocess.py
# 现在的 winsorize/standardize/neutralize 统一，并补上常用的：缺失值、重复值、inf、对齐交易日网格
# quant_preprocess/preprocess.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def winsorize(s: pd.Series, limit: float = 0.01) -> pd.Series:
    lo = s.quantile(limit)
    hi = s.quantile(1 - limit)
    return s.clip(lo, hi)


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() + 1e-9)


def neutralize_simple(df: pd.DataFrame, factor_col: str) -> pd.Series:
    # 预留扩展：行业/市值/风格中性化（回归残差）
    return df[factor_col] - df[factor_col].mean()


def clean_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """安全数值化 + inf->nan"""
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c] = out[c].replace([np.inf, -np.inf], np.nan)
    return out


def dedup_fact(df: pd.DataFrame, keys: list[str], keep: str = "last") -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(keys).drop_duplicates(keys, keep=keep).reset_index(drop=True)


def align_panel_to_trade_dates(
    df: pd.DataFrame,
    trade_dates: pd.DatetimeIndex,
    code_col: str = "code",
    date_col: str = "date",
    ffill_limit: Optional[int] = None
) -> pd.DataFrame:
    """
    给任何 [date, code, ...] 面板补齐 date x code 网格，并按 code ffill（不引入未来信息）
    """
    if df.empty:
        return df

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    trade_dates = pd.DatetimeIndex(pd.to_datetime(trade_dates)).sort_values()

    codes = df[code_col].astype(str).unique()
    full_index = pd.MultiIndex.from_product([trade_dates, codes], names=[date_col, code_col])

    out = df.set_index([date_col, code_col]).reindex(full_index).reset_index()

    # 对每只股票 ffill（常用于 close/基本面/文本因子等）
    val_cols = [c for c in out.columns if c not in (date_col, code_col)]
    out[val_cols] = out.groupby(code_col)[val_cols].ffill(limit=ffill_limit)

    return out


def _clean_price_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.where(s > 0, np.nan)
    return s


# -----------------------------
# A) 日频补齐：date x code 网格 + ffill
# -----------------------------
def ensure_daily_complete(
    daily: pd.DataFrame,
    trade_dates: pd.DatetimeIndex,
    code_col: str = "code",
    date_col: str = "date",
    price_cols: tuple[str, ...] = ("open", "high", "low", "close"),
    other_ffill_cols: tuple[str, ...] = ("volume", "turn", "tradestatus"),
    keep_last_on_dup: bool = True,
) -> pd.DataFrame:
    """
    目标：保证每个 (date,code) 都有一行。缺失用上一交易日 ffill。
    """
    if daily is None or daily.empty:
        return pd.DataFrame(columns=[date_col, code_col])

    df = daily.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[code_col] = df[code_col].astype(str)

    # 先去重，避免 non-unique multi-index
    df = df.sort_values([code_col, date_col])
    if keep_last_on_dup:
        df = df.drop_duplicates([date_col, code_col], keep="last")
    else:
        df = df.drop_duplicates([date_col, code_col], keep="first")

    # 清洗价格列
    for c in price_cols:
        if c in df.columns:
            df[c] = _clean_price_series(df[c])

    # 交易日范围限制
    td = pd.DatetimeIndex(pd.to_datetime(trade_dates)).sort_values()
    df = df[df[date_col].isin(td)]

    codes = df[code_col].unique()
    full_index = pd.MultiIndex.from_product([td, codes], names=[date_col, code_col])

    out = df.set_index([date_col, code_col]).reindex(full_index).reset_index()
    out = out.sort_values([code_col, date_col])

    # ffill：当天缺失用上一交易日补
    ffill_cols = []
    for c in price_cols + other_ffill_cols:
        if c in out.columns:
            ffill_cols.append(c)

    if ffill_cols:
        out[ffill_cols] = out.groupby(code_col)[ffill_cols].ffill()

    return out.reset_index(drop=True)


# -----------------------------
# B) 5分钟补齐：对每只股票，构建 “交易日 x 5min模板” 网格 + ffill
# -----------------------------
def infer_intraday_time_template(minute_df: pd.DataFrame, dt_col: str = "datetime") -> pd.DatetimeIndex:
    """
    从现有 minute 数据推断一个“日内 5min bar 时间模板”（只保留时分秒，不含日期）。
    方式：取出现频率最高的一组 time-of-day 序列（通常是标准交易日模板）。
    """
    if minute_df is None or minute_df.empty:
        return pd.DatetimeIndex([])

    dt = pd.to_datetime(minute_df[dt_col], errors="coerce").dropna()
    if dt.empty:
        return pd.DatetimeIndex([])

    # time-of-day（用一个虚拟日期承载）
    tod = dt.dt.strftime("%H:%M:%S")
    # 取 top 300 个最常见时间点（足够覆盖全日）
    vc = tod.value_counts()
    # 大部分 A 股 5min 全日 ~48 根上下；为了稳健，取出现次数最多的那些 time
    # 这里用阈值：出现次数 >= 中位数的一半（避免把噪声时间戳带进来）
    thr = max(1, int(vc.median() * 0.5))
    common = vc[vc >= thr].index.tolist()
    if not common:
        common = vc.head(60).index.tolist()

    # 排序成日内顺序
    common_sorted = sorted(common)
    # 用虚拟日期 2000-01-01 生成模板
    tmpl = pd.to_datetime(["2000-01-01 " + t for t in common_sorted])
    return pd.DatetimeIndex(tmpl)


def ensure_minute_complete(
    minute5: pd.DataFrame,
    trade_dates: pd.DatetimeIndex,
    code_col: str = "code",
    date_col: str = "date",
    dt_col: str = "datetime",
    price_cols: tuple[str, ...] = ("open", "high", "low", "close"),
    vol_cols: tuple[str, ...] = ("volume",),
    keep_last_on_dup: bool = True,
) -> pd.DataFrame:
    """
    目标：
    - 保证每个交易日、每个应有的 5min 时间点都存在
    - 缺 5min -> 用上一 5min ffill
    - 缺整天 -> 用上一交易日整天数据补（通过跨日 ffill 实现）
    """
    if minute5 is None or minute5.empty:
        return pd.DataFrame(columns=[date_col, code_col, dt_col])

    df = minute5.copy()
    df[code_col] = df[code_col].astype(str)
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[code_col, dt_col])
    df[date_col] = df[dt_col].dt.normalize()

    # 去重，确保 (code,datetime) 唯一
    df = df.sort_values([code_col, dt_col])
    if keep_last_on_dup:
        df = df.drop_duplicates([code_col, dt_col], keep="last")
    else:
        df = df.drop_duplicates([code_col, dt_col], keep="first")

    # 清洗价格列
    for c in price_cols:
        if c in df.columns:
            df[c] = _clean_price_series(df[c])
    for c in vol_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    td = pd.DatetimeIndex(pd.to_datetime(trade_dates)).sort_values()
    df = df[df[date_col].isin(td)]

    # 推断日内时间模板（只要时分秒序列）
    tmpl = infer_intraday_time_template(df, dt_col=dt_col)
    if tmpl.empty:
        # 模板推断失败就不补齐，直接返回已去重数据
        return df.reset_index(drop=True)

    # 将模板映射到每个交易日：date + tod
    # tmpl 是 2000-01-01 HH:MM:SS，我们取其 time，再加到真实日期上
    tod = tmpl.time
    # 为每个交易日生成全日 datetime grid
    # 注意：这里按你的数据时间戳口径生成，不硬编码交易时段
    all_datetimes = []
    for d in td:
        base = pd.Timestamp(d).to_pydatetime().date()
        all_datetimes.extend([pd.Timestamp.combine(base, t) for t in tod])
    all_datetimes = pd.DatetimeIndex(all_datetimes)

    # 对每只股票构造 full index：datetime x code（注意这里用 datetime 为主轴）
    codes = df[code_col].unique()
    full_index = pd.MultiIndex.from_product([codes, all_datetimes], names=[code_col, dt_col])

    out = df.set_index([code_col, dt_col]).reindex(full_index).reset_index()

    # 补 date
    out[date_col] = out[dt_col].dt.normalize()
    out = out.sort_values([code_col, dt_col])

    # 先跨日 ffill（缺整天 -> 用上一交易日最后一个bar延续）
    ffill_cols = []
    for c in price_cols + vol_cols:
        if c in out.columns:
            ffill_cols.append(c)

    if ffill_cols:
        out[ffill_cols] = out.groupby(code_col)[ffill_cols].ffill()

    return out.reset_index(drop=True)