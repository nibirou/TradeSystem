# -*- coding: utf-8 -*-
"""Money-flow and LHB event downloader.

This complements the text-news scripts with structured sentiment/flow inputs:
- EastMoney per-stock capital flow, near 100 trading days.
- EastMoney all-market capital-flow ranks.
- EastMoney LHB statistics.

The raw event files can later be merged through Strategy7's custom source hooks
or mined as external factors.
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
    load_pool_codes,
    polite_sleep,
    read_existing_table,
    save_table,
    to_bs_code,
    to_plain_code,
    to_symbol_key,
)


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_individual_flow(code: str, market: str) -> pd.DataFrame:
    return ak.stock_individual_fund_flow(stock=code, market=market)


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_flow_rank(indicator: str) -> pd.DataFrame:
    return ak.stock_individual_fund_flow_rank(indicator=indicator)


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_lhb_stat(symbol: str) -> pd.DataFrame:
    return ak.stock_lhb_stock_statistic_em(symbol=symbol)


def _market_for_code(code: str) -> str:
    return "sh" if to_bs_code(code).startswith("sh.") else "sz"


def normalize_flow(raw: pd.DataFrame, code: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if "日期" in df.columns:
        df["date"] = pd.to_datetime(df["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    elif "date" not in df.columns:
        return pd.DataFrame()
    df["code"] = to_bs_code(code)
    df["plain_code"] = to_plain_code(code)
    df["source"] = "eastmoney_individual_fund_flow"
    return df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")


def update_one_flow(code: str, *, base_dir: Path, pool: str) -> str:
    key = to_symbol_key(code)
    stem = base_dir / "data_fund_flow_eastmoney" / pool / key
    old = read_existing_table(stem)
    df = normalize_flow(fetch_individual_flow(to_plain_code(code), _market_for_code(code)), code)
    if df.empty and old.empty:
        return f"empty:{key}"
    merged = pd.concat([old, df], ignore_index=True) if not old.empty else df
    if not merged.empty:
        subset = [c for c in ["date", "code"] if c in merged.columns]
        merged = merged.drop_duplicates(subset=subset or None, keep="last").sort_values(subset or list(merged.columns[:1]))
        save_table(merged.reset_index(drop=True), stem)
    polite_sleep(0.1, 0.35)
    return f"ok:{key}:{len(df)}"


def run_individual_flow(pool: str = "hs300", workers: int = 8, base_dir: Optional[Path] = None) -> None:
    base_dir = Path(base_dir or default_base_dir())
    stocks = load_pool_codes(base_dir, pool, include_history_freqs=("d",))
    codes = stocks["code"].astype(str).tolist()
    print(f"[fund-flow] pool={pool} codes={len(codes)}")
    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as ex:
        futs = [ex.submit(update_one_flow, code, base_dir=base_dir, pool=pool) for code in codes]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{pool}-flow"):
            try:
                msg = fut.result()
                if msg.startswith("empty"):
                    print(msg)
            except Exception as exc:
                print(f"[fail] {type(exc).__name__}: {exc}")


def run_market_flow_and_lhb(base_dir: Optional[Path] = None) -> None:
    base_dir = Path(base_dir or default_base_dir())
    today = pd.Timestamp.now().strftime("%Y%m%d")
    for indicator in ["今日", "3日", "5日", "10日"]:
        try:
            df = fetch_flow_rank(indicator)
            if df is not None and not df.empty:
                df["indicator"] = indicator
                save_table(df, base_dir / "data_fund_flow_eastmoney" / "rank" / f"rank_{indicator}_{today}")
        except Exception as exc:
            print(f"[rank-fail] {indicator}: {exc}")
        polite_sleep(0.2, 0.6)

    for symbol in ["近一月", "近三月", "近六月", "近一年"]:
        try:
            df = fetch_lhb_stat(symbol)
            if df is not None and not df.empty:
                df["window"] = symbol
                save_table(df, base_dir / "data_lhb_eastmoney" / f"lhb_{symbol}_{today}")
        except Exception as exc:
            print(f"[lhb-fail] {symbol}: {exc}")
        polite_sleep(0.2, 0.6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="hs300", choices=["sz50", "hs300", "zz500", "all"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-dir", default=str(default_base_dir()))
    parser.add_argument("--market", action="store_true", help="Also download ranks and LHB market tables")
    parser.add_argument("--skip-individual", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.skip_individual:
        run_individual_flow(pool=args.pool, workers=args.workers, base_dir=Path(args.base_dir))
    if args.market:
        run_market_flow_and_lhb(base_dir=Path(args.base_dir))
