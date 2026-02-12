# quant_preprocess/market_loader.py
# 模块 1：交易日历 + 股票池 + 量价数据加载 quant_preprocess/market_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import PathsConfig, SchemaConfig, PeriodConfig, LoadRequest


def read_trade_dates(csv_path: str) -> pd.DatetimeIndex:
    """
    兼容：
      1) 有表头：calendar_date,is_trading_day
      2) 无表头：2025-11-10,1
    """
    df = pd.read_csv(csv_path)
    if not set(["calendar_date", "is_trading_day"]).issubset(df.columns):
        df = pd.read_csv(csv_path, header=None, names=["calendar_date", "is_trading_day"])

    df["calendar_date"] = pd.to_datetime(df["calendar_date"], errors="coerce")
    df["is_trading_day"] = pd.to_numeric(df["is_trading_day"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["calendar_date"])

    trade_days = (
        df[df["is_trading_day"] == 1]
        .sort_values("calendar_date")["calendar_date"]
        .unique()
    )
    trade_dates = pd.DatetimeIndex(trade_days)
    if trade_dates.empty:
        raise ValueError(f"交易日历为空，请检查文件: {csv_path}")
    return trade_dates


def _shift_trade_date(trade_dates: pd.DatetimeIndex, anchor: pd.Timestamp, n_back: int) -> pd.Timestamp:
    anchor = pd.to_datetime(anchor)
    idx = trade_dates.get_loc(anchor)
    return trade_dates[max(0, idx - int(n_back))]


def compute_load_window(period: PeriodConfig, full_trade_dates: pd.DatetimeIndex, buffer_n: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    fs = pd.to_datetime(period.factor_start)
    fe = pd.to_datetime(period.factor_end)
    load_start = _shift_trade_date(full_trade_dates, fs, buffer_n)
    load_end = fe
    return load_start, load_end


def load_stock_pool(paths: PathsConfig, schema: SchemaConfig, pool: str) -> pd.DataFrame:
    meta_dir = Path(paths.base_dir) / paths.meta_dir
    file_map = {"hs300": paths.hs300_list, "zz500": paths.zz500_list}
    fp = meta_dir / file_map[pool]
    if not fp.exists():
        raise FileNotFoundError(f"股票池文件不存在: {fp}")
    df = pd.read_csv(fp)
    df[schema.code_col] = df[schema.code_col].astype(str)
    return df


def load_daily_bars(paths: PathsConfig, schema: SchemaConfig, pool: str,
                    start: Optional[pd.Timestamp] = None,
                    end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    root = Path(paths.base_dir) / paths.hist_dir / pool / paths.daily_freq_dir
    if not root.exists():
        raise FileNotFoundError(f"日频目录不存在: {root}")

    start = pd.to_datetime(start) if start is not None else None
    end = pd.to_datetime(end) if end is not None else None

    dfs = []
    for fp in sorted(root.glob("*.csv")):
        df = pd.read_csv(fp)
        if df.empty:
            continue

        df[schema.date_col] = pd.to_datetime(df[schema.date_col], errors="coerce")
        df[schema.code_col] = df[schema.code_col].astype(str)
        df = df.dropna(subset=[schema.date_col, schema.code_col])

        if start is not None:
            df = df[df[schema.date_col] >= start]
        if end is not None:
            df = df[df[schema.date_col] <= end]
        if df.empty:
            continue

        df["source_pool"] = pool
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"未加载到任何日频数据: {root}")

    out = pd.concat(dfs, ignore_index=True)
    out = (out.sort_values([schema.code_col, schema.date_col])
              .drop_duplicates([schema.code_col, schema.date_col], keep="last")
              .reset_index(drop=True))
    return out


def load_minute5_bars(paths: PathsConfig, schema: SchemaConfig, pool: str,
                      start: Optional[pd.Timestamp] = None,
                      end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    root = Path(paths.base_dir) / paths.hist_dir / pool / paths.minute5_freq_dir
    if not root.exists():
        raise FileNotFoundError(f"5分钟目录不存在: {root}")

    start = pd.to_datetime(start) if start is not None else None
    end = pd.to_datetime(end) if end is not None else None

    dfs = []
    for fp in sorted(root.glob("*.csv")):
        df = pd.read_csv(fp)
        if df.empty:
            continue

        df[schema.date_col] = pd.to_datetime(df[schema.date_col], errors="coerce")
        df[schema.time_col] = df[schema.time_col].astype(str)
        df[schema.code_col] = df[schema.code_col].astype(str)

        # 先按 date 粗裁剪（你原来的关键优化点）
        df = df.dropna(subset=[schema.date_col, schema.code_col])
        if start is not None:
            df = df[df[schema.date_col] >= start]
        if end is not None:
            df = df[df[schema.date_col] <= end]
        if df.empty:
            continue

        # 解析 baostock 5min time: YYYYMMDDHHMMSSmmm -> 取前14位
        df[schema.datetime_col] = pd.to_datetime(
            df[schema.time_col].str.slice(0, 14),
            format="%Y%m%d%H%M%S",
            errors="coerce"
        )
        df = df.dropna(subset=[schema.datetime_col])
        if df.empty:
            continue

        # 统一补 date（用于按日聚合/label）
        df[schema.date_col] = df[schema.datetime_col].dt.normalize()

        df["source_pool"] = pool
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"未加载到任何5分钟数据: {root}")

    out = pd.concat(dfs, ignore_index=True)
    out = (out.sort_values([schema.code_col, schema.datetime_col])
              .drop_duplicates([schema.code_col, schema.datetime_col], keep="last")
              .reset_index(drop=True))
    return out


@dataclass
class MarketBundle:
    daily: pd.DataFrame
    minute5: Optional[pd.DataFrame]
    trade_dates: pd.DatetimeIndex
    load_start: pd.Timestamp
    load_end: pd.Timestamp


class MarketDataLoader:
    def __init__(self, paths: PathsConfig, schema: SchemaConfig):
        self.paths = paths
        self.schema = schema

    def load(self, period: PeriodConfig, req: LoadRequest) -> MarketBundle:
        full_trade_dates = read_trade_dates(str(Path(self.paths.base_dir) / self.paths.trade_calendar_csv))

        # 决定加载窗口：优先 req.start/end，否则 period + buffer 自动推
        if req.start is not None and req.end is not None:
            load_start = pd.to_datetime(req.start)
            load_end = pd.to_datetime(req.end)
        else:
            load_start, load_end = compute_load_window(period, full_trade_dates, period.factor_buffer_n)

        daily_list = []
        min_list = []

        for p in req.pools:
            _ = load_stock_pool(self.paths, self.schema, p)  # 保留“存在性校验”，你也可以返回代码宇宙
            if req.load_daily:
                daily_list.append(load_daily_bars(self.paths, self.schema, p, load_start, load_end))
            if req.load_minute5:
                min_list.append(load_minute5_bars(self.paths, self.schema, p, load_start, load_end))

        daily = pd.concat(daily_list, ignore_index=True) if daily_list else pd.DataFrame()
        minute5 = pd.concat(min_list, ignore_index=True) if min_list else None

        # trade_dates 用因子区间（而不是 load_start）——你后续计算因子/label更清晰
        td = full_trade_dates[(full_trade_dates >= period.factor_start) & (full_trade_dates <= period.factor_end)]

        return MarketBundle(
            daily=daily,
            minute5=minute5,
            trade_dates=pd.DatetimeIndex(td).sort_values(),
            load_start=load_start,
            load_end=load_end,
        )
