# -*- coding: utf-8 -*-
"""Robust benchmark index downloader for Strategy7.

Writes the files consumed by Strategy7:
    data_baostock/ak_index/hs300_price.csv
    data_baostock/ak_index/zz500_price.csv
    data_baostock/ak_index/zz1000_price.csv

AkShare is used first; BaoStock is a fallback for daily index OHLCV where
available. Constituent snapshots for sz50/hs300/zz500 are also saved.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

import akshare as ak
import baostock as bs
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_fetch_common import (  # noqa: E402
    bs_login,
    bs_logout,
    bs_query_to_df,
    default_base_dir,
    ensure_dir,
    get_stock_list_bs,
    latest_trade_date_bs,
    save_table,
)


INDEX_MAP = {
    "hs300": {"ak": "sh000300", "bs": "sh.000300", "stem": "hs300_price"},
    "zz500": {"ak": "sh000905", "bs": "sh.000905", "stem": "zz500_price"},
    "zz1000": {"ak": "sh000852", "bs": "sh.000852", "stem": "zz1000_price"},
}


def normalize_index_df(raw: pd.DataFrame, code: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    rename = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "pctChg",
    }
    df = df.rename(columns={c: rename.get(str(c), c) for c in df.columns})
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["code"] = code
    for col in ["open", "high", "low", "close", "volume", "amount", "pctChg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = [c for c in ["date", "code", "open", "high", "low", "close", "volume", "amount", "pctChg"] if c in df.columns]
    return df[keep].dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")


def fetch_index_ak(symbol: str) -> pd.DataFrame:
    return ak.stock_zh_index_daily(symbol=symbol)


def fetch_index_bs(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    fields = "date,code,open,high,low,close,volume,amount,pctChg"
    rs = bs.query_history_k_data_plus(
        code,
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3",
    )
    if rs.error_code != "0":
        raise RuntimeError(rs.error_msg)
    return bs_query_to_df(rs)


def update_index_prices(base_dir: Path, start_date: str = "2006-01-01", end_date: Optional[str] = None) -> None:
    out_dir = ensure_dir(base_dir / "ak_index")
    end_date = end_date or latest_trade_date_bs()
    for name, meta in INDEX_MAP.items():
        df = pd.DataFrame()
        try:
            df = normalize_index_df(fetch_index_ak(meta["ak"]), meta["bs"])
            df = df[df["date"] >= start_date]
            print(f"[index][akshare] {name}: rows={len(df)}")
        except Exception as exc:
            print(f"[index][akshare-fail] {name}: {exc}")
        if df.empty:
            try:
                bs_login()
                try:
                    raw = fetch_index_bs(meta["bs"], start_date=start_date, end_date=end_date)
                finally:
                    bs_logout()
                df = normalize_index_df(raw, meta["bs"])
                print(f"[index][baostock] {name}: rows={len(df)}")
            except Exception as exc:
                print(f"[index][baostock-fail] {name}: {exc}")
        if not df.empty:
            save_table(df.reset_index(drop=True), out_dir / meta["stem"])


def update_constituents(base_dir: Path) -> None:
    meta_dir = ensure_dir(base_dir / "metadata" / "index_constituents")
    date_tag = latest_trade_date_bs().replace("-", "")
    bs_login()
    try:
        for pool in ["sz50", "hs300", "zz500"]:
            df = get_stock_list_bs(pool)
            if not df.empty:
                save_table(df, meta_dir / f"{pool}_{date_tag}")
                df.to_csv(base_dir / "metadata" / f"stock_list_{pool}.csv", index=False, encoding="utf-8")
                print(f"[constituents] {pool}: rows={len(df)}")
    finally:
        bs_logout()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=str(default_base_dir()))
    parser.add_argument("--start-date", default="2006-01-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--skip-constituents", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base = Path(args.base_dir)
    update_index_prices(base, start_date=args.start_date, end_date=args.end_date or None)
    if not args.skip_constituents:
        update_constituents(base)
