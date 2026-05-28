# -*- coding: utf-8 -*-
"""Tencent tick-trade downloader via AKShare.

This is a free near-term supplement, not a historical tick warehouse.  AKShare's
Tencent interface returns the latest available trading day's transaction tape.
The script stores one file per code per trade date.
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
from data_fetch_common import (  # noqa: E402
    default_base_dir,
    ensure_dir,
    load_pool_codes,
    polite_sleep,
    save_table,
    to_bs_code,
    to_symbol_key,
    to_tencent_code,
)


RENAME_MAP = {
    "成交时间": "time",
    "时间": "time",
    "成交价格": "price",
    "成交价": "price",
    "价格变动": "price_change",
    "成交量": "volume",
    "成交额": "amount",
    "性质": "side",
}


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_tick_tx(symbol: str) -> pd.DataFrame:
    return ak.stock_zh_a_tick_tx_js(symbol=symbol)


def normalize_tick_df(raw: pd.DataFrame, code: str, trade_date: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.rename(columns={c: RENAME_MAP.get(str(c), c) for c in raw.columns}).copy()
    if "time" not in df.columns:
        return pd.DataFrame()
    date_text = pd.to_datetime(trade_date).strftime("%Y-%m-%d")
    out = pd.DataFrame(
        {
            "date": date_text,
            "time": df["time"].astype(str),
            "code": to_bs_code(code),
            "source": "tencent",
        }
    )
    out["datetime"] = pd.to_datetime(date_text + " " + out["time"], errors="coerce")
    for col in ["price", "price_change", "volume", "amount"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    if "side" in df.columns:
        out["side"] = df["side"].astype(str)
    out = out.dropna(subset=["datetime", "price"])
    return out.sort_values("datetime").drop_duplicates(["datetime", "price", "volume", "amount"], keep="last")


def update_one(code: str, *, base_dir: Path, pool: str, trade_date: str) -> str:
    key = to_symbol_key(code)
    day = pd.to_datetime(trade_date).strftime("%Y%m%d")
    stem = base_dir / "tick_trades" / "tencent" / pool / day / key
    raw = fetch_tick_tx(to_tencent_code(code))
    df = normalize_tick_df(raw, code, trade_date)
    if df.empty:
        return f"empty:{key}"
    save_table(df.reset_index(drop=True), stem)
    polite_sleep(0.05, 0.25)
    return f"ok:{key}:{len(df)}"


def run_download(
    pool: str = "hs300",
    trade_date: Optional[str] = None,
    workers: int = 8,
    base_dir: Optional[Path] = None,
) -> None:
    base_dir = Path(base_dir or default_base_dir())
    trade_date = trade_date or datetime.now().strftime("%Y-%m-%d")
    ensure_dir(base_dir / "tick_trades" / "tencent" / pool / pd.to_datetime(trade_date).strftime("%Y%m%d"))
    stocks = load_pool_codes(base_dir, pool, include_history_freqs=("d",))
    codes = stocks["code"].astype(str).tolist()
    print(f"[tick] pool={pool} trade_date={trade_date} codes={len(codes)}")
    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as ex:
        futs = [ex.submit(update_one, code, base_dir=base_dir, pool=pool, trade_date=trade_date) for code in codes]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{pool}-ticks"):
            try:
                msg = fut.result()
                if msg.startswith("empty"):
                    print(msg)
            except Exception as exc:
                print(f"[fail] {type(exc).__name__}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="hs300", choices=["sz50", "hs300", "zz500", "all"])
    parser.add_argument("--trade-date", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-dir", default=str(default_base_dir()))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_download(
        pool=args.pool,
        trade_date=args.trade_date or None,
        workers=args.workers,
        base_dir=Path(args.base_dir),
    )
