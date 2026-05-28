# -*- coding: utf-8 -*-
"""Shared helpers for the new data download scripts.

The existing scripts in this repository intentionally remain untouched.  This
module only serves the new ED4/ED2 scripts so they can share one path convention
and one BaoStock stock-pool implementation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import os
import random
import time
from typing import Iterable, Optional

import baostock as bs
import pandas as pd


POOL_CHOICES = ("sz50", "hs300", "zz500", "all")


def default_base_dir() -> Path:
    env_value = (
        os.getenv("DATA_BAOSTOCK_ROOT")
        or os.getenv("STRATEGY7_DATA_BAOSTOCK_ROOT")
        or os.getenv("QUANT_DATA_ROOT")
    )
    if env_value:
        return Path(env_value).expanduser()

    cwd = Path.cwd().resolve()
    candidates = [
        cwd / "data_baostock",
        cwd.parent / "data_baostock",
        cwd.parent.parent / "data_baostock",
        Path("D:/PythonProject/Quant/data_baostock"),
        Path("/workspace/Quant/data_baostock"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return Path("D:/PythonProject/Quant/data_baostock")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def polite_sleep(low: float = 0.2, high: float = 0.8) -> None:
    time.sleep(random.uniform(low, high))


def save_table(df: pd.DataFrame, stem: Path, *, csv_encoding: str = "utf-8") -> None:
    ensure_dir(stem.parent)
    df.to_csv(stem.with_suffix(".csv"), index=False, encoding=csv_encoding)
    df.to_parquet(stem.with_suffix(".parquet"), index=False)


def read_existing_table(stem: Path) -> pd.DataFrame:
    csv_path = stem.with_suffix(".csv")
    parquet_path = stem.with_suffix(".parquet")
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False)
    return pd.DataFrame()


def bs_login() -> None:
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {lg.error_msg}")


def bs_logout() -> None:
    bs.logout()


def bs_query_to_df(rs) -> pd.DataFrame:
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def latest_trade_date_bs() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    start = f"{datetime.now().year}-01-01"
    if today <= f"{datetime.now().year}-01-04":
        start = f"{datetime.now().year - 1}-01-01"
    rs = bs.query_trade_dates(start_date=start, end_date=today)
    df = bs_query_to_df(rs)
    if df.empty:
        raise RuntimeError("BaoStock query_trade_dates returned empty data")
    df["calendar_date"] = pd.to_datetime(df["calendar_date"], errors="coerce")
    df["is_trading_day"] = pd.to_numeric(df["is_trading_day"], errors="coerce").fillna(0).astype(int)
    trade_days = df[df["is_trading_day"] == 1]["calendar_date"].dropna().tolist()
    if not trade_days:
        raise RuntimeError("No trading day found from BaoStock calendar")
    return pd.Timestamp(trade_days[-1]).strftime("%Y-%m-%d")


def get_stock_list_bs(pool: str = "hs300", day: Optional[str] = None) -> pd.DataFrame:
    pool = str(pool).strip().lower()
    if pool == "sz50":
        rs = bs.query_sz50_stocks()
    elif pool == "hs300":
        rs = bs.query_hs300_stocks()
    elif pool == "zz500":
        rs = bs.query_zz500_stocks()
    elif pool == "all":
        rs = bs.query_all_stock(day=day or latest_trade_date_bs())
    else:
        raise ValueError(f"unsupported pool: {pool}")
    df = bs_query_to_df(rs)
    if df.empty:
        return pd.DataFrame(columns=["code", "name"])
    if "code_name" in df.columns:
        df = df.rename(columns={"code_name": "name"})
    if "name" not in df.columns:
        df["name"] = df["code"]
    return df[["code", "name"]].drop_duplicates("code").sort_values("code").reset_index(drop=True)


def save_stock_snapshot(stocks: pd.DataFrame, base_dir: Path, pool: str) -> None:
    meta_dir = ensure_dir(base_dir / "metadata")
    snap_dir = ensure_dir(meta_dir / "stock_snapshots")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stocks.to_csv(meta_dir / f"stock_list_{pool}.csv", index=False, encoding="utf-8")
    stocks.to_csv(snap_dir / f"{pool}_{ts}.csv", index=False, encoding="utf-8")


def codes_from_history_dir(base_dir: Path, pool: str, freq: str = "d") -> set[str]:
    hist_dir = base_dir / "stock_hist" / pool / freq
    if not hist_dir.exists():
        return set()
    out = set()
    for fp in hist_dir.glob("*.csv"):
        stem = fp.stem
        if stem.endswith(f"_{freq}"):
            stem = stem[: -(len(freq) + 1)]
        out.add(stem.replace("_", "."))
    return out


def load_last_snapshot_codes(base_dir: Path, pool: str) -> set[str]:
    fp = base_dir / "metadata" / f"stock_list_{pool}.csv"
    if not fp.exists():
        return set()
    df = pd.read_csv(fp, dtype=str)
    if "code" not in df.columns:
        return set()
    return set(df["code"].dropna().astype(str).tolist())


def load_pool_codes(base_dir: Path, pool: str, *, include_history_freqs: Iterable[str] = ("d",)) -> pd.DataFrame:
    bs_login()
    try:
        stocks = get_stock_list_bs(pool, day=latest_trade_date_bs() if pool == "all" else None)
    finally:
        bs_logout()
    save_stock_snapshot(stocks, base_dir, pool)

    codes = set(stocks["code"].astype(str).tolist())
    for freq in include_history_freqs:
        codes |= codes_from_history_dir(base_dir, pool, freq)
    codes |= load_last_snapshot_codes(base_dir, pool)

    name_map = dict(zip(stocks["code"].astype(str), stocks["name"].astype(str)))
    out = pd.DataFrame({"code": sorted(codes)})
    out["name"] = out["code"].map(name_map).fillna(out["code"])
    return out


def split_exchange_code(code: str) -> tuple[str, str]:
    s = str(code).strip().lower().replace("_", ".")
    if "." in s:
        left, right = s.split(".", 1)
        if left in {"sh", "sz"}:
            market, symbol = left, right
        else:
            symbol, market = left, right
    else:
        symbol = "".join(ch for ch in s if ch.isdigit())[-6:].zfill(6)
        market = "sh" if symbol.startswith(("5", "6", "9")) else "sz"
    symbol = "".join(ch for ch in symbol if ch.isdigit())[-6:].zfill(6)
    if market not in {"sh", "sz"}:
        market = "sh" if symbol.startswith(("5", "6", "9")) else "sz"
    return market, symbol


def to_bs_code(code: str) -> str:
    market, symbol = split_exchange_code(code)
    return f"{market}.{symbol}"


def to_symbol_key(code: str) -> str:
    return to_bs_code(code).replace(".", "_")


def to_plain_code(code: str) -> str:
    return split_exchange_code(code)[1]


def to_tencent_code(code: str) -> str:
    market, symbol = split_exchange_code(code)
    return f"{market}{symbol}"


def to_xt_code(code: str) -> str:
    market, symbol = split_exchange_code(code)
    return f"{symbol}.{market.upper()}"


def to_ak_suffix_code(code: str) -> str:
    market, symbol = split_exchange_code(code)
    return f"{symbol}.{market.upper()}"


def ymd(date_like: object) -> str:
    return pd.to_datetime(date_like).strftime("%Y%m%d")


def ymd_dash(date_like: object) -> str:
    return pd.to_datetime(date_like).strftime("%Y-%m-%d")
