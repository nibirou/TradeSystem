# -*- coding: utf-8 -*-
"""Optional efinance downloader.

`efinance` is another open-source wrapper around public market endpoints.  Keep
it isolated because some versions create a cache directory inside site-packages
on import, which can fail in locked conda environments.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Optional

import pandas as pd
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


KLT_MAP = {"1": 1, "5": 5, "15": 15, "30": 30, "60": 60, "d": 101, "w": 102, "m": 103}
FQT_MAP = {"": 0, "qfq": 1, "hfq": 2}

COL_MAP = {
    "日期": "datetime",
    "股票代码": "source_code",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turn",
    "date": "datetime",
    "time": "datetime",
    "open": "open",
    "close": "close",
    "high": "high",
    "low": "low",
    "volume": "volume",
    "amount": "amount",
}


def _load_efinance() -> Any:
    try:
        import efinance as ef
    except ModuleNotFoundError as exc:
        raise SystemExit("missing optional package: efinance; install it in env_quant") from exc
    except PermissionError as exc:
        raise SystemExit(
            "efinance import failed while creating its package cache directory. "
            "Run from a writable env or reinstall/patch efinance so its cache is writable."
        ) from exc
    return ef


def _normalize(raw: pd.DataFrame, code: str, freq: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.rename(columns={c: COL_MAP.get(str(c), c) for c in raw.columns}).copy()
    if "datetime" not in df.columns:
        for col in df.columns:
            if "日期" in str(col) or str(col).lower() in {"date", "time", "datetime"}:
                df = df.rename(columns={col: "datetime"})
                break
    dt = pd.to_datetime(df.get("datetime"), errors="coerce")
    out = pd.DataFrame({"date": dt.dt.strftime("%Y-%m-%d"), "code": to_bs_code(code)})
    if freq not in {"d", "w", "m"}:
        out["time"] = dt.dt.strftime("%Y%m%d%H%M%S") + "000"
    for col in ["open", "high", "low", "close", "volume", "amount", "turn"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    required = ["date", "open", "high", "low", "close"]
    out = out.dropna(subset=[c for c in required if c in out.columns])
    sort_cols = ["date", "time"] if "time" in out.columns else ["date"]
    dedupe_cols = sort_cols
    return out.sort_values(sort_cols).drop_duplicates(dedupe_cols, keep="last").reset_index(drop=True)


def fetch_history(code: str, start_date: str, end_date: str, freq: str, adjust: str) -> pd.DataFrame:
    ef = _load_efinance()
    raw = ef.stock.get_quote_history(
        stock_codes=to_plain_code(code),
        beg=ymd(start_date),
        end=ymd(end_date),
        klt=KLT_MAP[freq],
        fqt=FQT_MAP[adjust],
    )
    if isinstance(raw, dict):
        raw = next(iter(raw.values()), pd.DataFrame())
    return _normalize(raw, code, freq)


def update_one(
    code: str,
    *,
    base_dir: Path,
    pool: str,
    freq: str,
    adjust: str,
    start_date: str,
    end_date: str,
    strategy7_layout: bool,
) -> str:
    key = to_symbol_key(code)
    root = base_dir / "stock_hist" if strategy7_layout else base_dir / "stock_hist_efinance"
    stem = root / pool / freq / f"{key}_{freq}"
    old = read_existing_table(stem)
    new = fetch_history(code, start_date, end_date, freq, adjust)
    if new.empty:
        return f"empty:{key}"
    merged = pd.concat([old, new], ignore_index=True) if not old.empty else new
    sort_cols = ["date", "time"] if "time" in merged.columns else ["date"]
    merged = merged.sort_values(sort_cols).drop_duplicates(sort_cols, keep="last").reset_index(drop=True)
    save_table(merged, stem)
    polite_sleep(0.05, 0.25)
    return f"ok:{key}:{len(new)}"


def run_download(
    pool: str,
    freq: str,
    adjust: str,
    start_date: str,
    end_date: str,
    workers: int,
    base_dir: Optional[Path] = None,
    strategy7_layout: bool = False,
) -> None:
    base_dir = Path(base_dir or default_base_dir())
    _load_efinance()
    root = base_dir / "stock_hist" if strategy7_layout else base_dir / "stock_hist_efinance"
    ensure_dir(root / pool / freq)
    stocks = load_pool_codes(base_dir, pool, include_history_freqs=("d", freq))
    codes = stocks["code"].astype(str).tolist()
    print(f"[efinance] pool={pool} freq={freq} adjust={adjust} codes={len(codes)}")
    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as ex:
        futs = [
            ex.submit(
                update_one,
                code,
                base_dir=base_dir,
                pool=pool,
                freq=freq,
                adjust=adjust,
                start_date=start_date,
                end_date=end_date,
                strategy7_layout=strategy7_layout,
            )
            for code in codes
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="efinance"):
            try:
                msg = fut.result()
                if not msg.startswith("ok"):
                    print(msg)
            except Exception as exc:
                print(f"[fail] {type(exc).__name__}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="hs300", choices=["sz50", "hs300", "zz500", "all"])
    parser.add_argument("--freq", default="d", choices=list(KLT_MAP.keys()))
    parser.add_argument("--adjust", default="", choices=list(FQT_MAP.keys()))
    parser.add_argument("--start-date", default="20150101")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-dir", default=str(default_base_dir()))
    parser.add_argument("--strategy7-layout", action="store_true", help="write into stock_hist instead of stock_hist_efinance")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_download(
        pool=args.pool,
        freq=args.freq,
        adjust=args.adjust,
        start_date=args.start_date,
        end_date=args.end_date,
        workers=args.workers,
        base_dir=Path(args.base_dir),
        strategy7_layout=args.strategy7_layout,
    )
