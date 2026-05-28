# -*- coding: utf-8 -*-
"""Corporate-action and ownership-event downloader.

These event tables are useful for avoiding label leakage and for building
event/risk factors: dividends, share-capital changes, ST status, and top
shareholder concentration.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
from typing import Optional

import akshare as ak
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_fetch_common import (  # noqa: E402
    default_base_dir,
    ensure_dir,
    load_pool_codes,
    polite_sleep,
    read_existing_table,
    save_table,
    to_bs_code,
    to_plain_code,
    to_symbol_key,
    ymd,
)


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_share_change(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    return ak.stock_share_change_cninfo(symbol=symbol, start_date=start_date, end_date=end_date)


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_dividend(symbol: str, indicator: str) -> pd.DataFrame:
    return ak.stock_history_dividend_detail(symbol=symbol, indicator=indicator, date="")


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_st_list() -> pd.DataFrame:
    return ak.stock_zh_a_st_em()


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_top10_float_holders(date: str) -> pd.DataFrame:
    return ak.stock_gdfx_free_holding_analyse_em(date=date)


def _add_event_identity(df: pd.DataFrame, code: str, source: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy().replace("", pd.NA)
    out["code"] = to_bs_code(code)
    out["plain_code"] = to_plain_code(code)
    out["source"] = source
    return out


def update_one_stock(
    code: str,
    *,
    base_dir: Path,
    pool: str,
    start_date: str,
    end_date: str,
    include_share_change: bool,
    include_dividend: bool,
) -> str:
    key = to_symbol_key(code)
    wrote = 0
    if include_share_change:
        stem = base_dir / "corporate_actions" / "share_change" / pool / key
        old = read_existing_table(stem)
        df = _add_event_identity(fetch_share_change(to_plain_code(code), start_date, end_date), code, "cninfo_share_change")
        merged = pd.concat([old, df], ignore_index=True) if not old.empty else df
        if not merged.empty:
            merged = merged.drop_duplicates().reset_index(drop=True)
            save_table(merged, stem)
            wrote += len(df)
        polite_sleep(0.15, 0.45)

    if include_dividend:
        parts = []
        for indicator in ["分红", "配股"]:
            try:
                part = _add_event_identity(fetch_dividend(to_plain_code(code), indicator), code, f"akshare_dividend_{indicator}")
                if not part.empty:
                    part["indicator"] = indicator
                    parts.append(part)
            except Exception as exc:
                print(f"[dividend-fail] {key} {indicator}: {exc}")
            polite_sleep(0.15, 0.45)
        if parts:
            stem = base_dir / "corporate_actions" / "dividend" / pool / key
            old = read_existing_table(stem)
            df = pd.concat(parts, ignore_index=True)
            merged = pd.concat([old, df], ignore_index=True) if not old.empty else df
            merged = merged.drop_duplicates().reset_index(drop=True)
            save_table(merged, stem)
            wrote += len(df)
    return f"ok:{key}:{wrote}" if wrote else f"empty:{key}"


def run_stock_events(
    pool: str = "hs300",
    start_date: str = "20000101",
    end_date: Optional[str] = None,
    workers: int = 6,
    base_dir: Optional[Path] = None,
    include_share_change: bool = True,
    include_dividend: bool = True,
) -> None:
    base_dir = Path(base_dir or default_base_dir())
    end_date = end_date or pd.Timestamp.now().strftime("%Y%m%d")
    stocks = load_pool_codes(base_dir, pool, include_history_freqs=("d",))
    codes = stocks["code"].astype(str).tolist()
    print(f"[corp-actions] pool={pool} codes={len(codes)} start={start_date} end={end_date}")
    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as ex:
        futs = [
            ex.submit(
                update_one_stock,
                code,
                base_dir=base_dir,
                pool=pool,
                start_date=start_date,
                end_date=end_date,
                include_share_change=include_share_change,
                include_dividend=include_dividend,
            )
            for code in codes
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{pool}-corp"):
            try:
                msg = fut.result()
                if msg.startswith("empty"):
                    print(msg)
            except Exception as exc:
                print(f"[fail] {type(exc).__name__}: {exc}")


def run_market_events(base_dir: Optional[Path] = None, holder_dates: Optional[list[str]] = None) -> None:
    base_dir = Path(base_dir or default_base_dir())
    today = pd.Timestamp.now().strftime("%Y%m%d")
    st = fetch_st_list()
    if st is not None and not st.empty:
        save_table(st, base_dir / "corporate_actions" / "st_status" / f"st_list_{today}")

    for date in holder_dates or []:
        df = fetch_top10_float_holders(ymd(date))
        if df is not None and not df.empty:
            save_table(df, base_dir / "corporate_actions" / "shareholder_top10_float" / f"holders_{ymd(date)}")
        polite_sleep(0.3, 0.8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="hs300", choices=["sz50", "hs300", "zz500", "all"])
    parser.add_argument("--start-date", default="20000101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--base-dir", default=str(default_base_dir()))
    parser.add_argument("--skip-share-change", action="store_true")
    parser.add_argument("--skip-dividend", action="store_true")
    parser.add_argument("--market-events", action="store_true")
    parser.add_argument("--holder-dates", default="", help="Comma separated quarter-end dates, e.g. 20240331,20240630")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    holder_dates = [x.strip() for x in args.holder_dates.split(",") if x.strip()]
    if args.market_events:
        run_market_events(base_dir=Path(args.base_dir), holder_dates=holder_dates)
    run_stock_events(
        pool=args.pool,
        start_date=ymd(args.start_date),
        end_date=ymd(args.end_date) if args.end_date else None,
        workers=args.workers,
        base_dir=Path(args.base_dir),
        include_share_change=not args.skip_share_change,
        include_dividend=not args.skip_dividend,
    )
