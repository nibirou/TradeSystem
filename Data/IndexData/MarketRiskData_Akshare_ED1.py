# -*- coding: utf-8 -*-
"""Market-risk and liquidity supplements from AkShare.

Outputs:
- market_risk/akshare/margin_sse
- market_risk/akshare/margin_szse/<date>
- market_risk/akshare/hsgt/*
- market_risk/akshare/qvix/*

The data are useful for risk regime, liquidity, northbound flow, and option
implied-volatility features.
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
from data_fetch_common import default_base_dir, ensure_dir, save_table, ymd  # noqa: E402


QVIX_FUNCS = [
    "index_option_50etf_qvix",
    "index_option_300etf_qvix",
    "index_option_500etf_qvix",
    "index_option_100etf_qvix",
    "index_option_1000index_qvix",
    "index_option_300index_qvix",
    "index_option_50index_qvix",
    "index_option_cyb_qvix",
    "index_option_kcb_qvix",
]


def _stamp(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = df.copy()
    out["provider"] = "akshare"
    out["dataset"] = dataset
    out["download_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return out


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_margin_sse(start_date: str, end_date: str) -> pd.DataFrame:
    return ak.stock_margin_sse(start_date=ymd(start_date), end_date=ymd(end_date))


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_margin_szse_one(date: str) -> pd.DataFrame:
    return ak.stock_margin_szse(date=ymd(date))


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_noarg(func_name: str) -> pd.DataFrame:
    func = getattr(ak, func_name)
    df = func()
    return pd.DataFrame() if df is None else df


def update_margin(base_dir: Path, start_date: str, end_date: str, workers: int) -> None:
    sse = fetch_margin_sse(start_date, end_date)
    if not sse.empty:
        save_table(_stamp(sse, "stock_margin_sse"), base_dir / "market_risk" / "akshare" / "margin_sse")
        print(f"[margin-sse] saved {len(sse)} rows")

    dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq="B")
    ensure_dir(base_dir / "market_risk" / "akshare" / "margin_szse")
    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as ex:
        futs = {ex.submit(fetch_margin_szse_one, d.strftime("%Y%m%d")): d.strftime("%Y%m%d") for d in dates}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="margin-szse"):
            day = futs[fut]
            try:
                df = fut.result()
                if not df.empty:
                    save_table(_stamp(df, "stock_margin_szse"), base_dir / "market_risk" / "akshare" / "margin_szse" / day)
            except Exception as exc:
                print(f"[margin-szse-fail] {day}: {type(exc).__name__}: {exc}")


def update_hsgt(base_dir: Path) -> None:
    out_dir = base_dir / "market_risk" / "akshare" / "hsgt"
    ensure_dir(out_dir)
    datasets = {
        "stock_hsgt_fund_flow_summary_em": lambda: ak.stock_hsgt_fund_flow_summary_em(),
        "stock_hsgt_hist_em": lambda: ak.stock_hsgt_hist_em(),
    }
    for name, func in datasets.items():
        try:
            df = func()
            if df is not None and not df.empty:
                save_table(_stamp(df, name), out_dir / name)
                print(f"[hsgt] {name}: {len(df)} rows")
        except Exception as exc:
            print(f"[hsgt-fail] {name}: {type(exc).__name__}: {exc}")


def update_qvix(base_dir: Path, funcs: Optional[list[str]] = None) -> None:
    out_dir = base_dir / "market_risk" / "akshare" / "qvix"
    ensure_dir(out_dir)
    for name in funcs or QVIX_FUNCS:
        if not hasattr(ak, name):
            print(f"[qvix-skip] missing {name}")
            continue
        try:
            df = fetch_noarg(name)
            if not df.empty:
                save_table(_stamp(df, name), out_dir / name)
                print(f"[qvix] {name}: {len(df)} rows")
        except Exception as exc:
            print(f"[qvix-fail] {name}: {type(exc).__name__}: {exc}")


def run_download(
    start_date: str,
    end_date: str,
    base_dir: Optional[Path] = None,
    workers: int = 2,
    include_margin: bool = True,
    include_hsgt: bool = True,
    include_qvix: bool = True,
) -> None:
    base_dir = Path(base_dir or default_base_dir())
    ensure_dir(base_dir / "market_risk" / "akshare")
    if include_margin:
        update_margin(base_dir, start_date, end_date, workers)
    if include_hsgt:
        update_hsgt(base_dir)
    if include_qvix:
        update_qvix(base_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    today = datetime.now().strftime("%Y%m%d")
    start = (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime("%Y%m%d")
    parser.add_argument("--start-date", default=start)
    parser.add_argument("--end-date", default=today)
    parser.add_argument("--base-dir", default=str(default_base_dir()))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-margin", action="store_true")
    parser.add_argument("--no-hsgt", action="store_true")
    parser.add_argument("--no-qvix", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_download(
        start_date=args.start_date,
        end_date=args.end_date,
        base_dir=Path(args.base_dir),
        workers=args.workers,
        include_margin=not args.no_margin,
        include_hsgt=not args.no_hsgt,
        include_qvix=not args.no_qvix,
    )
