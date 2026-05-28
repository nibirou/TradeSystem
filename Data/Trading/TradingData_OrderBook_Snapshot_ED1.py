# -*- coding: utf-8 -*-
"""Best-effort order-book snapshot downloader.

Free public web sources usually expose only current best bid/ask snapshots. True
historical Level2 10-level order book, order-by-order trades, and order queue
data normally require broker/Level2 permission. This script stores current
snapshots from:
- akshare: stock_bid_ask_em, broadly available but typically not 10-level.
- xtquant: optional local QMT feed if the terminal and permission are available.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Optional

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
    to_plain_code,
    to_symbol_key,
    to_xt_code,
)


def _flatten_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "|".join(str(x) for x in value)
    if isinstance(value, dict):
        return "|".join(f"{k}:{v}" for k, v in value.items())
    return "" if value is None else str(value)


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_ak_bid_ask(code: str) -> pd.DataFrame:
    return ak.stock_bid_ask_em(symbol=to_plain_code(code))


def normalize_ak_bid_ask(raw: pd.DataFrame, code: str, snapshot_time: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if len(df.columns) >= 2:
        key_col, value_col = df.columns[0], df.columns[1]
        out = df[[key_col, value_col]].rename(columns={key_col: "item", value_col: "value"})
    else:
        out = df.reset_index().rename(columns={"index": "item", df.columns[0]: "value"})
    out["value"] = out["value"].map(_flatten_value)
    out["code"] = code
    out["snapshot_time"] = snapshot_time
    out["provider"] = "akshare_stock_bid_ask_em"
    return out[["snapshot_time", "code", "provider", "item", "value"]]


def fetch_xtquant_full_tick(code: str) -> pd.DataFrame:
    from xtquant import xtdata

    xt_code = to_xt_code(code)
    data = xtdata.get_full_tick([xt_code])
    tick = data.get(xt_code) if isinstance(data, dict) else None
    if not tick:
        return pd.DataFrame()
    rows = []
    for key, value in tick.items():
        rows.append({"item": str(key), "value": _flatten_value(value)})
    return pd.DataFrame(rows)


def normalize_xtquant_snapshot(raw: pd.DataFrame, code: str, snapshot_time: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    out = raw[["item", "value"]].copy()
    out["code"] = code
    out["snapshot_time"] = snapshot_time
    out["provider"] = "xtquant_get_full_tick"
    return out[["snapshot_time", "code", "provider", "item", "value"]]


def update_one(code: str, *, base_dir: Path, pool: str, provider: str, snapshot_time: str) -> str:
    key = to_symbol_key(code)
    day = pd.to_datetime(snapshot_time).strftime("%Y%m%d")
    clock = pd.to_datetime(snapshot_time).strftime("%H%M%S")
    stem = base_dir / "order_book_snapshot" / provider / pool / day / f"{key}_{clock}"
    if provider == "akshare":
        raw = fetch_ak_bid_ask(code)
        df = normalize_ak_bid_ask(raw, code, snapshot_time)
    elif provider == "xtquant":
        raw = fetch_xtquant_full_tick(code)
        df = normalize_xtquant_snapshot(raw, code, snapshot_time)
    else:
        raise ValueError(f"unsupported provider: {provider}")
    if df.empty:
        return f"empty:{key}"
    save_table(df, stem)
    polite_sleep(0.03, 0.18)
    return f"ok:{key}:{len(df)}"


def run_snapshot(
    pool: str = "hs300",
    provider: str = "akshare",
    workers: int = 8,
    base_dir: Optional[Path] = None,
) -> None:
    base_dir = Path(base_dir or default_base_dir())
    snapshot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stocks = load_pool_codes(base_dir, pool, include_history_freqs=("d",))
    codes = stocks["code"].astype(str).tolist()
    ensure_dir(base_dir / "order_book_snapshot" / provider / pool)
    print(f"[orderbook] provider={provider} pool={pool} snapshot_time={snapshot_time} codes={len(codes)}")
    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as ex:
        futs = [
            ex.submit(update_one, code, base_dir=base_dir, pool=pool, provider=provider, snapshot_time=snapshot_time)
            for code in codes
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{pool}-{provider}"):
            try:
                msg = fut.result()
                if msg.startswith("empty"):
                    print(msg)
            except Exception as exc:
                print(f"[fail] {type(exc).__name__}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="hs300", choices=["sz50", "hs300", "zz500", "all"])
    parser.add_argument("--provider", default="akshare", choices=["akshare", "xtquant"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-dir", default=str(default_base_dir()))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_snapshot(pool=args.pool, provider=args.provider, workers=args.workers, base_dir=Path(args.base_dir))
