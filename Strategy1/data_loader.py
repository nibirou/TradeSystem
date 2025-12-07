# data_loader.py
import os
import pandas as pd
from typing import Iterable, Optional
from config import HIST_DIR, POOLS, DAILY_FREQ, INTRADAY_FREQ


def load_daily_data(
    pools: Iterable[str] = POOLS,
    freq: str = DAILY_FREQ,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    读取日频数据（parquet），合并沪深300 & 中证500；
    返回 DataFrame: [date, code, ..., pool]
    """
    dfs = []
    for pool in pools:
        path = os.path.join(HIST_DIR, pool, freq)
        if not os.path.exists(path):
            continue
        for fname in os.listdir(path):
            if not fname.endswith(".parquet"):
                continue
            fpath = os.path.join(path, fname)
            df = pd.read_parquet(fpath)
            df["pool"] = pool
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No daily parquet files found under HIST_DIR")

    daily = pd.concat(dfs, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["date", "code"]).reset_index(drop=True)

    if start_date:
        daily = daily[daily["date"] >= pd.to_datetime(start_date)]
    if end_date:
        daily = daily[daily["date"] <= pd.to_datetime(end_date)]

    return daily


def load_intraday_data(
    pools: Iterable[str] = POOLS,
    freq: str = INTRADAY_FREQ,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    读取5分钟数据（parquet），合并沪深300 & 中证500；
    返回 DataFrame: [date, time, code, ..., pool]
    """
    dfs = []
    for pool in pools:
        path = os.path.join(HIST_DIR, pool, freq)
        if not os.path.exists(path):
            continue
        for fname in os.listdir(path):
            if not fname.endswith(".parquet"):
                continue
            fpath = os.path.join(path, fname)
            df = pd.read_parquet(fpath)
            df["pool"] = pool
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No intraday parquet files found under HIST_DIR")

    intraday = pd.concat(dfs, ignore_index=True)
    intraday["date"] = pd.to_datetime(intraday["date"])
    intraday = intraday.sort_values(["date", "code", "time"]).reset_index(drop=True)

    if start_date:
        intraday = intraday[intraday["date"] >= pd.to_datetime(start_date)]
    if end_date:
        intraday = intraday[intraday["date"] <= pd.to_datetime(end_date)]

    return intraday
