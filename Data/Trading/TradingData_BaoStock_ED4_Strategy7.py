# -*- coding: utf-8 -*-
"""BaoStock downloader aligned with Strategy7.

Compared with the older ED scripts:
- No hard-coded query date for all-stock pools.
- Saves directly to data_baostock/stock_hist/<pool>/<freq>/<sh_600000>_<freq>.
- Keeps daily and 5-minute files in the exact layout Strategy7 loads.
- Uses incremental resume and type cleanup before Parquet writes.

BaoStock's Python client keeps global session state, so this script is
sequential by default for stability. For large all-market 5m backfills, prefer
running several pools/date ranges in separate terminals instead of sharing one
BaoStock session across threads.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import sys
from typing import Optional

import baostock as bs
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_fetch_common import (  # noqa: E402
    bs_login,
    bs_logout,
    bs_query_to_df,
    default_base_dir,
    ensure_dir,
    load_pool_codes,
    polite_sleep,
    read_existing_table,
    save_table,
    to_bs_code,
    to_symbol_key,
)


def fields_for_freq(freq: str) -> str:
    if freq == "d":
        return "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST"
    if freq in {"w", "m"}:
        return "date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg"
    if freq in {"5", "15", "30", "60"}:
        return "date,time,code,open,high,low,close,volume,amount,adjustflag"
    raise ValueError(f"unsupported freq: {freq}")


def min_start_date(code: str, freq: str) -> str:
    is_index = code.startswith("sh.000") or code.startswith("sz.399") or code.startswith("sh.880")
    if freq in {"d", "w", "m"}:
        return "2006-01-01" if is_index else "1990-12-19"
    if is_index:
        return ""
    return "2019-01-01"


def clean_baostock_df(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.replace("", pd.NA).copy()
    for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg", "peTTM", "psTTM", "pcfNcfTTM", "pbMRQ", "tradestatus", "isST", "adjustflag"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "time" in out.columns:
        out["time"] = out["time"].astype("string")
    subset = ["date", "time"] if "time" in out.columns else ["date"]
    out = out.drop_duplicates(subset=subset, keep="last").sort_values(subset).reset_index(drop=True)
    return out


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def query_history(code: str, fields: str, start_date: str, end_date: str, freq: str, adjustflag: str):
    rs = bs.query_history_k_data_plus(
        code,
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency=freq,
        adjustflag=adjustflag,
    )
    if rs.error_code != "0":
        raise RuntimeError(rs.error_msg)
    return rs


def infer_resume_start(old: pd.DataFrame, fallback: str, freq: str) -> str:
    if old.empty or "date" not in old.columns:
        return fallback
    if freq in {"5", "15", "30", "60"} and "time" in old.columns:
        t = old["time"].astype(str).str.zfill(17).str.slice(0, 14)
        last_dt = pd.to_datetime(t, format="%Y%m%d%H%M%S", errors="coerce").max()
        if pd.notna(last_dt):
            return (last_dt + pd.Timedelta(minutes=int(freq))).strftime("%Y-%m-%d")
    max_date = pd.to_datetime(old["date"], errors="coerce").max()
    if pd.isna(max_date):
        return fallback
    return (max_date + timedelta(days=1)).strftime("%Y-%m-%d")


def update_one(
    code: str,
    *,
    base_dir: Path,
    pool: str,
    freq: str,
    start_date: Optional[str],
    end_date: str,
    adjustflag: str,
) -> str:
    code = to_bs_code(code)
    earliest = min_start_date(code, freq)
    if not earliest:
        return f"skip-index-minute:{code}"
    start = start_date or earliest
    start = max(pd.to_datetime(start).date(), pd.to_datetime(earliest).date()).strftime("%Y-%m-%d")

    stem = base_dir / "stock_hist" / pool / freq / f"{to_symbol_key(code)}_{freq}"
    old = read_existing_table(stem)
    start = infer_resume_start(old, start, freq)
    if pd.to_datetime(start) > pd.to_datetime(end_date):
        return f"skip-current:{code}"

    rs = query_history(code, fields_for_freq(freq), start, end_date, freq, adjustflag)
    df = bs_query_to_df(rs)
    if df.empty:
        return f"empty:{code}"
    df = clean_baostock_df(df, freq)
    if not old.empty:
        df = clean_baostock_df(pd.concat([old, df], ignore_index=True), freq)
    save_table(df, stem)
    polite_sleep(0.05, 0.2)
    return f"ok:{code}:{len(df)}"


def run_history_download(
    pool: str = "hs300",
    freq: str = "d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    adjustflag: str = "2",
    base_dir: Optional[Path] = None,
) -> None:
    base_dir = Path(base_dir or default_base_dir())
    ensure_dir(base_dir / "stock_hist" / pool / freq)
    end_date = end_date or pd.Timestamp.now().strftime("%Y-%m-%d")
    stocks = load_pool_codes(base_dir, pool, include_history_freqs=(freq,))
    codes = stocks["code"].astype(str).tolist()
    print(f"[baostock-ed4] pool={pool} freq={freq} codes={len(codes)} end={end_date} adjustflag={adjustflag}")
    bs_login()
    try:
        for code in tqdm(codes, desc=f"{pool}-{freq}"):
            try:
                msg = update_one(
                    code,
                    base_dir=base_dir,
                    pool=pool,
                    freq=freq,
                    start_date=start_date,
                    end_date=end_date,
                    adjustflag=adjustflag,
                )
                if msg.startswith(("empty", "skip-index")):
                    print(msg)
            except Exception as exc:
                print(f"[fail] {code}: {type(exc).__name__}: {exc}")
    finally:
        bs_logout()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="hs300", choices=["sz50", "hs300", "zz500", "all"])
    parser.add_argument("--freq", default="d", choices=["d", "w", "m", "5", "15", "30", "60"])
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--adjustflag", default="2", choices=["1", "2", "3"])
    parser.add_argument("--base-dir", default=str(default_base_dir()))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_history_download(
        pool=args.pool,
        freq=args.freq,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        adjustflag=args.adjustflag,
        base_dir=Path(args.base_dir),
    )
