# -*- coding: utf-8 -*-
"""ETF minute, NAV, scale, and dividend downloader via AKShare.

This complements the existing ETF daily-history script with intraday bars and
fund-level metadata useful for ETF rotation and hedging research.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional

import akshare as ak
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_fetch_common import default_base_dir, ensure_dir, polite_sleep, read_existing_table, save_table, ymd  # noqa: E402


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_etf_list() -> pd.DataFrame:
    return ak.fund_etf_spot_em()


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_etf_minute(symbol: str, start_dt: str, end_dt: str, period: str, adjust: str) -> pd.DataFrame:
    return ak.fund_etf_hist_min_em(symbol=symbol, start_date=start_dt, end_date=end_dt, period=period, adjust=adjust)


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_etf_info(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    return ak.fund_etf_fund_info_em(fund=symbol, start_date=start_date, end_date=end_date)


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_etf_daily_table() -> pd.DataFrame:
    return ak.fund_etf_fund_daily_em()


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_etf_dividend_sina(sina_symbol: str) -> pd.DataFrame:
    return ak.fund_etf_dividend_sina(symbol=sina_symbol)


def _sina_symbol(code: str) -> str:
    code = str(code).zfill(6)
    return ("sh" if code.startswith("5") else "sz") + code


def normalize_minute(raw: pd.DataFrame, code: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    rename = {
        "时间": "datetime",
        "日期": "datetime",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = raw.rename(columns={c: rename.get(str(c), c) for c in raw.columns}).copy()
    if "datetime" not in df.columns:
        return pd.DataFrame()
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    out = pd.DataFrame(
        {
            "date": dt.dt.strftime("%Y-%m-%d"),
            "time": dt.dt.strftime("%Y%m%d%H%M%S") + "000",
            "code": str(code).zfill(6),
        }
    )
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    return out.dropna(subset=["date", "time", "open", "close"]).sort_values(["date", "time"])


def update_one_etf(
    code: str,
    *,
    base_dir: Path,
    period: str,
    adjust: str,
    start_dt: str,
    end_dt: str,
    start_date: str,
    end_date: str,
    with_info: bool,
    with_dividend: bool,
) -> str:
    code = str(code).zfill(6)
    wrote = 0
    minute_stem = base_dir / "etf_hist" / "eastmoney" / "minute" / period / (adjust or "raw") / code
    old = read_existing_table(minute_stem)
    raw = fetch_etf_minute(code, start_dt, end_dt, period, adjust)
    minute = normalize_minute(raw, code)
    if not minute.empty:
        merged = pd.concat([old, minute], ignore_index=True) if not old.empty else minute
        merged = merged.drop_duplicates(["date", "time"], keep="last").sort_values(["date", "time"])
        save_table(merged.reset_index(drop=True), minute_stem)
        wrote += len(minute)
    polite_sleep(0.1, 0.35)

    if with_info:
        info_stem = base_dir / "etf_info" / "eastmoney" / code
        old_info = read_existing_table(info_stem)
        info = fetch_etf_info(code, start_date, end_date)
        if info is not None and not info.empty:
            info["code"] = code
            merged = pd.concat([old_info, info], ignore_index=True) if not old_info.empty else info
            save_table(merged.drop_duplicates().reset_index(drop=True), info_stem)
            wrote += len(info)
        polite_sleep(0.1, 0.35)

    if with_dividend:
        div_stem = base_dir / "etf_info" / "sina_dividend" / code
        old_div = read_existing_table(div_stem)
        div = fetch_etf_dividend_sina(_sina_symbol(code))
        if div is not None and not div.empty:
            div["code"] = code
            merged = pd.concat([old_div, div], ignore_index=True) if not old_div.empty else div
            save_table(merged.drop_duplicates().reset_index(drop=True), div_stem)
            wrote += len(div)
        polite_sleep(0.1, 0.35)
    return f"ok:{code}:{wrote}" if wrote else f"empty:{code}"


def run_download(
    period: str = "5",
    adjust: str = "",
    lookback_days: int = 15,
    workers: int = 8,
    base_dir: Optional[Path] = None,
    with_info: bool = True,
    with_dividend: bool = True,
) -> None:
    base_dir = Path(base_dir or default_base_dir())
    ensure_dir(base_dir / "etf_hist" / "eastmoney" / "minute" / period / (adjust or "raw"))
    etf = fetch_etf_list()
    if etf is None or etf.empty:
        raise RuntimeError("ETF list is empty")
    code_col = "代码" if "代码" in etf.columns else etf.columns[0]
    name_col = "名称" if "名称" in etf.columns else None
    codes = etf[code_col].dropna().astype(str).str.zfill(6).drop_duplicates().tolist()
    meta = pd.DataFrame({"code": codes})
    if name_col:
        meta["name"] = etf.drop_duplicates(code_col).set_index(code_col).reindex(codes)[name_col].astype(str).values
    save_table(meta, base_dir / "metadata" / "etf_metadata" / "etf_list_eastmoney")

    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=int(lookback_days))
    print(f"[etf-minute] codes={len(codes)} period={period} start={start} end={end}")
    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as ex:
        futs = [
            ex.submit(
                update_one_etf,
                code,
                base_dir=base_dir,
                period=period,
                adjust=adjust,
                start_dt=start.strftime("%Y-%m-%d %H:%M:%S"),
                end_dt=end.strftime("%Y-%m-%d %H:%M:%S"),
                start_date=ymd(start),
                end_date=ymd(end),
                with_info=with_info,
                with_dividend=with_dividend,
            )
            for code in codes
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"etf-{period}m"):
            try:
                msg = fut.result()
                if msg.startswith("empty"):
                    print(msg)
            except Exception as exc:
                print(f"[fail] {type(exc).__name__}: {exc}")

    try:
        daily = fetch_etf_daily_table()
        if daily is not None and not daily.empty:
            save_table(daily, base_dir / "etf_info" / "eastmoney_daily" / f"daily_{pd.Timestamp.now().strftime('%Y%m%d')}")
    except Exception as exc:
        print(f"[etf-daily-info-fail] {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="5", choices=["1", "5", "15", "30", "60"])
    parser.add_argument("--adjust", default="", choices=["", "qfq", "hfq"])
    parser.add_argument("--lookback-days", type=int, default=15)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-dir", default=str(default_base_dir()))
    parser.add_argument("--skip-info", action="store_true")
    parser.add_argument("--skip-dividend", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_download(
        period=args.period,
        adjust=args.adjust,
        lookback_days=args.lookback_days,
        workers=args.workers,
        base_dir=Path(args.base_dir),
        with_info=not args.skip_info,
        with_dividend=not args.skip_dividend,
    )
