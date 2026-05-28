# -*- coding: utf-8 -*-
"""EastMoney minute-bar downloader for Strategy7.

Free boundary:
- ak.stock_zh_a_hist_min_em supports 1/5/15/30/60 minute bars.
- EastMoney normally exposes only recent intraday history, especially for 1m.
  Treat this as a rolling supplement rather than a full historical L2 source.

Output shape matches Strategy7:
    data_baostock/stock_hist/<pool>/<period>/<sh_600000>_<period>.csv
with columns:
    date,time,code,open,high,low,close,volume,amount
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
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
)


COL_MAP = {
    "时间": "datetime",
    "日期": "datetime",
    "day": "datetime",
    "date": "datetime",
    "开盘": "open",
    "open": "open",
    "收盘": "close",
    "close": "close",
    "最高": "high",
    "high": "high",
    "最低": "low",
    "low": "low",
    "成交量": "volume",
    "volume": "volume",
    "成交额": "amount",
    "amount": "amount",
}


def _parse_minute_time(df: pd.DataFrame) -> pd.Series:
    if "datetime" in df.columns:
        return pd.to_datetime(df["datetime"], errors="coerce")
    for col in df.columns:
        if "时间" in str(col) or str(col).lower() in {"date", "day", "datetime"}:
            return pd.to_datetime(df[col], errors="coerce")
    return pd.Series(pd.NaT, index=df.index)


def normalize_minute_df(raw: pd.DataFrame, code: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.rename(columns={c: COL_MAP.get(str(c), c) for c in raw.columns}).copy()
    dt = _parse_minute_time(df)
    out = pd.DataFrame(
        {
            "date": dt.dt.strftime("%Y-%m-%d"),
            "time": dt.dt.strftime("%Y%m%d%H%M%S") + "000",
            "code": to_bs_code(code),
        }
    )
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[col] = pd.NA
    if out["amount"].isna().all() and {"close", "volume"}.issubset(out.columns):
        out["amount"] = pd.to_numeric(out["close"], errors="coerce") * pd.to_numeric(out["volume"], errors="coerce")
    out = out.dropna(subset=["date", "time", "open", "high", "low", "close"])
    return out.sort_values(["date", "time"]).drop_duplicates(["date", "time"], keep="last").reset_index(drop=True)


def _last_local_datetime(old: pd.DataFrame) -> Optional[pd.Timestamp]:
    if old.empty or "date" not in old.columns:
        return None
    if "time" in old.columns:
        t = old["time"].astype(str).str.zfill(17).str.slice(0, 14)
        dt = pd.to_datetime(t, format="%Y%m%d%H%M%S", errors="coerce")
    else:
        dt = pd.to_datetime(old["date"], errors="coerce")
    dt = dt.dropna()
    if dt.empty:
        return None
    return pd.Timestamp(dt.max())


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_minute(symbol: str, start_dt: str, end_dt: str, period: str, adjust: str) -> pd.DataFrame:
    return ak.stock_zh_a_hist_min_em(
        symbol=symbol,
        start_date=start_dt,
        end_date=end_dt,
        period=period,
        adjust=adjust,
    )


def update_one(
    code: str,
    *,
    base_dir: Path,
    pool: str,
    period: str,
    adjust: str,
    start_dt: str,
    end_dt: str,
) -> str:
    key = to_symbol_key(code)
    stem = base_dir / "stock_hist" / pool / period / f"{key}_{period}"
    old = read_existing_table(stem)
    effective_start = pd.to_datetime(start_dt)
    last_dt = _last_local_datetime(old)
    if last_dt is not None:
        effective_start = max(effective_start, last_dt + timedelta(minutes=int(period)))
    effective_end = pd.to_datetime(end_dt)
    if effective_start > effective_end:
        return f"skip:{key}"

    raw = fetch_minute(
        symbol=to_plain_code(code),
        start_dt=effective_start.strftime("%Y-%m-%d %H:%M:%S"),
        end_dt=effective_end.strftime("%Y-%m-%d %H:%M:%S"),
        period=period,
        adjust=adjust,
    )
    new = normalize_minute_df(raw, code)
    if new.empty:
        return f"empty:{key}"
    if not old.empty:
        keep = [c for c in ["date", "time", "code", "open", "high", "low", "close", "volume", "amount"] if c in old.columns]
        old = old[keep].copy()
        merged = pd.concat([old, new], ignore_index=True)
    else:
        merged = new
    merged = merged.sort_values(["date", "time"]).drop_duplicates(["date", "time"], keep="last").reset_index(drop=True)
    save_table(merged, stem)
    polite_sleep(0.05, 0.25)
    return f"ok:{key}:{len(new)}"


def run_download(
    pool: str = "hs300",
    period: str = "5",
    adjust: str = "",
    lookback_days: int = 15,
    workers: int = 8,
    base_dir: Optional[Path] = None,
    end_dt: Optional[str] = None,
) -> None:
    base_dir = Path(base_dir or default_base_dir())
    ensure_dir(base_dir / "stock_hist" / pool / period)
    end = pd.to_datetime(end_dt) if end_dt else pd.Timestamp.now()
    start = end - pd.Timedelta(days=int(lookback_days))
    stocks = load_pool_codes(base_dir, pool, include_history_freqs=("d", period))
    codes = stocks["code"].astype(str).tolist()

    print(f"[minute] pool={pool} period={period} codes={len(codes)} start={start} end={end}")
    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as ex:
        futs = [
            ex.submit(
                update_one,
                code,
                base_dir=base_dir,
                pool=pool,
                period=period,
                adjust=adjust,
                start_dt=start.strftime("%Y-%m-%d %H:%M:%S"),
                end_dt=end.strftime("%Y-%m-%d %H:%M:%S"),
            )
            for code in codes
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{pool}-{period}m"):
            try:
                msg = fut.result()
                if msg.startswith(("empty", "fail")):
                    print(msg)
            except Exception as exc:
                print(f"[fail] {type(exc).__name__}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="hs300", choices=["sz50", "hs300", "zz500", "all"])
    parser.add_argument("--period", default="5", choices=["1", "5", "15", "30", "60"])
    parser.add_argument("--adjust", default="", choices=["", "qfq", "hfq"])
    parser.add_argument("--lookback-days", type=int, default=15)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-dir", default=str(default_base_dir()))
    parser.add_argument("--end-dt", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_download(
        pool=args.pool,
        period=args.period,
        adjust=args.adjust,
        lookback_days=args.lookback_days,
        workers=args.workers,
        base_dir=Path(args.base_dir),
        end_dt=args.end_dt or None,
    )
