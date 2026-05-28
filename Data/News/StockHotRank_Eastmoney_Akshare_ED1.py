# -*- coding: utf-8 -*-
"""EastMoney stock popularity and keyword downloader via AkShare."""

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
from data_fetch_common import (  # noqa: E402
    default_base_dir,
    ensure_dir,
    load_pool_codes,
    polite_sleep,
    save_table,
    split_exchange_code,
    to_symbol_key,
)


def to_em_hot_symbol(code: str) -> str:
    market, symbol = split_exchange_code(code)
    return f"{market.upper()}{symbol}"


def _stamp(df: pd.DataFrame, code: str, dataset: str) -> pd.DataFrame:
    out = df.copy()
    out["code"] = code
    out["provider"] = "akshare_eastmoney"
    out["dataset"] = dataset
    out["download_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return out


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_rank(code: str) -> pd.DataFrame:
    return ak.stock_hot_rank_detail_realtime_em(symbol=to_em_hot_symbol(code))


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_keywords(code: str) -> pd.DataFrame:
    return ak.stock_hot_keyword_em(symbol=to_em_hot_symbol(code))


def update_one(code: str, *, base_dir: Path, pool: str, include_keywords: bool) -> str:
    key = to_symbol_key(code)
    day = datetime.now().strftime("%Y%m%d")
    saved = 0
    rank = fetch_rank(code)
    if rank is not None and not rank.empty:
        save_table(
            _stamp(rank, code, "stock_hot_rank_detail_realtime_em"),
            base_dir / "sentiment" / "eastmoney_hot_rank" / pool / day / f"{key}_rank",
        )
        saved += len(rank)
    if include_keywords:
        keywords = fetch_keywords(code)
        if keywords is not None and not keywords.empty:
            save_table(
                _stamp(keywords, code, "stock_hot_keyword_em"),
                base_dir / "sentiment" / "eastmoney_hot_rank" / pool / day / f"{key}_keywords",
            )
            saved += len(keywords)
    polite_sleep(0.05, 0.25)
    return f"ok:{key}:{saved}" if saved else f"empty:{key}"


def run_download(
    pool: str = "hs300",
    workers: int = 4,
    base_dir: Optional[Path] = None,
    include_keywords: bool = True,
) -> None:
    base_dir = Path(base_dir or default_base_dir())
    ensure_dir(base_dir / "sentiment" / "eastmoney_hot_rank" / pool)
    stocks = load_pool_codes(base_dir, pool, include_history_freqs=("d",))
    codes = stocks["code"].astype(str).tolist()
    print(f"[hot-rank] pool={pool} codes={len(codes)} keywords={include_keywords}")
    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as ex:
        futs = [
            ex.submit(update_one, code, base_dir=base_dir, pool=pool, include_keywords=include_keywords)
            for code in codes
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="eastmoney-hot"):
            try:
                msg = fut.result()
                if not msg.startswith("ok"):
                    print(msg)
            except Exception as exc:
                print(f"[fail] {type(exc).__name__}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="hs300", choices=["sz50", "hs300", "zz500", "all"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-dir", default=str(default_base_dir()))
    parser.add_argument("--no-keywords", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_download(
        pool=args.pool,
        workers=args.workers,
        base_dir=Path(args.base_dir),
        include_keywords=not args.no_keywords,
    )
