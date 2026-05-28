# -*- coding: utf-8 -*-
"""Optional TongDaXin/xmtdx data downloader for Strategy7.

This script is intentionally optional.  The main repository does not depend on
`xmtdx` or `pytdx`, but they can be very useful public TDX supplements:

- K bars: daily and 1/5/15/30/60 minute bars.
- Quotes: current 5-level quote snapshots, when the TDX server provides them.
- Transactions: current or historical trade prints, depending on server support.

Install one of the optional clients in env_quant before running:
    pip install xmtdx
or:
    pip install pytdx
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterable, Optional

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
    split_exchange_code,
    to_bs_code,
    to_plain_code,
    to_symbol_key,
    ymd,
)


DEFAULT_TDX_HOST = "180.153.18.170"
DEFAULT_TDX_PORT = 7709


def require_provider(provider: str) -> None:
    package = {"xmtdx": "xmtdx", "pytdx": "pytdx"}[provider]
    if importlib.util.find_spec(package) is None:
        raise SystemExit(f"missing optional package: {package}; install it in env_quant")


def _obj_get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _market_for_xmtdx(code: str) -> Any:
    from xmtdx import Market

    market, _ = split_exchange_code(code)
    return Market.SH if market == "sh" else Market.SZ


def _market_for_pytdx(code: str) -> int:
    market, _ = split_exchange_code(code)
    return 1 if market == "sh" else 0


def _category_for_xmtdx(freq: str) -> Any:
    from xmtdx import KlineCategory

    mapping = {
        "1": "MIN_1",
        "5": "MIN_5",
        "15": "MIN_15",
        "30": "MIN_30",
        "60": "MIN_60",
        "d": "DAY",
        "w": "WEEK",
        "m": "MONTH",
    }
    return getattr(KlineCategory, mapping[freq])


def _category_for_pytdx(freq: str) -> int:
    # pytdx follows the classic TDX category convention.
    return {
        "5": 0,
        "15": 1,
        "30": 2,
        "60": 3,
        "d": 9,
        "w": 5,
        "m": 6,
        "1": 7,
    }[freq]


def _tdx_datetime(obj: Any, freq: str) -> pd.Timestamp:
    raw_dt = _obj_get(obj, "datetime")
    if raw_dt is not None:
        dt = pd.to_datetime(raw_dt, errors="coerce")
        if pd.notna(dt):
            return pd.Timestamp(dt)

    year = int(_obj_get(obj, "year", 1970) or 1970)
    month = int(_obj_get(obj, "month", 1) or 1)
    day = int(_obj_get(obj, "day", 1) or 1)
    hour = int(_obj_get(obj, "hour", 0) or 0)
    minute = int(_obj_get(obj, "minute", 0) or 0)
    if freq in {"d", "w", "m"}:
        hour = 0
        minute = 0
    return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)


def _normalize_bars(records: Iterable[Any], code: str, freq: str) -> pd.DataFrame:
    rows = []
    for rec in records:
        dt = _tdx_datetime(rec, freq)
        if pd.isna(dt):
            continue
        rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "time": dt.strftime("%Y%m%d%H%M%S") + "000",
                "code": to_bs_code(code),
                "open": _obj_get(rec, "open"),
                "high": _obj_get(rec, "high"),
                "low": _obj_get(rec, "low"),
                "close": _obj_get(rec, "close"),
                "volume": _obj_get(rec, "vol", _obj_get(rec, "volume")),
                "amount": _obj_get(rec, "amount"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if freq in {"d", "w", "m"}:
        df = df[["date", "code", "open", "high", "low", "close", "volume", "amount"]]
        return df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").drop_duplicates("date")
    return (
        df[["date", "time", "code", "open", "high", "low", "close", "volume", "amount"]]
        .dropna(subset=["date", "time", "open", "high", "low", "close"])
        .sort_values(["date", "time"])
        .drop_duplicates(["date", "time"], keep="last")
    )


def _open_xmtdx_client(host: str, port: int) -> Any:
    from xmtdx import TdxClient

    try:
        return TdxClient(host, port=port)
    except TypeError:
        return TdxClient(host)


def fetch_bars_xmtdx(code: str, freq: str, host: str, port: int, max_pages: int) -> pd.DataFrame:
    market = _market_for_xmtdx(code)
    symbol = to_plain_code(code)
    category = _category_for_xmtdx(freq)
    records = []
    with _open_xmtdx_client(host, port) as client:
        for page in range(max_pages):
            batch = client.get_security_bars(market, symbol, category, page * 800, 800)
            if not batch:
                break
            records.extend(batch)
            if len(batch) < 800:
                break
    return _normalize_bars(records, code, freq)


def fetch_bars_pytdx(code: str, freq: str, host: str, port: int, max_pages: int) -> pd.DataFrame:
    from pytdx.hq import TdxHq_API

    market = _market_for_pytdx(code)
    symbol = to_plain_code(code)
    category = _category_for_pytdx(freq)
    api = TdxHq_API()
    records = []
    with api.connect(host, port):
        for page in range(max_pages):
            batch = api.get_security_bars(category, market, symbol, page * 800, 800)
            if not batch:
                break
            records.extend(batch)
            if len(batch) < 800:
                break
    return _normalize_bars(records, code, freq)


def update_bars_one(
    code: str,
    *,
    provider: str,
    freq: str,
    base_dir: Path,
    pool: str,
    host: str,
    port: int,
    max_pages: int,
) -> str:
    key = to_symbol_key(code)
    stem = base_dir / "stock_hist" / pool / freq / f"{key}_{freq}"
    old = read_existing_table(stem)
    if provider == "xmtdx":
        new = fetch_bars_xmtdx(code, freq, host, port, max_pages)
    elif provider == "pytdx":
        new = fetch_bars_pytdx(code, freq, host, port, max_pages)
    else:
        raise ValueError(f"unsupported provider: {provider}")
    if new.empty:
        return f"empty:{key}"
    merged = pd.concat([old, new], ignore_index=True) if not old.empty else new
    sort_cols = ["date", "time"] if "time" in merged.columns else ["date"]
    dedupe_cols = ["date", "time"] if "time" in merged.columns else ["date"]
    merged = merged.sort_values(sort_cols).drop_duplicates(dedupe_cols, keep="last").reset_index(drop=True)
    save_table(merged, stem)
    polite_sleep(0.03, 0.18)
    return f"ok:{key}:{len(new)}"


def _quote_to_row(obj: Any, snapshot_time: str) -> dict[str, Any]:
    row = {"snapshot_time": snapshot_time}
    for name in [
        "market",
        "code",
        "price",
        "pre_close",
        "open",
        "high",
        "low",
        "vol",
        "cur_vol",
        "amount",
        "s_vol",
        "b_vol",
        "server_time",
    ]:
        row[name] = _obj_get(obj, name)
    for level in range(1, 6):
        row[f"bid{level}"] = _obj_get(obj, f"bid{level}")
        row[f"ask{level}"] = _obj_get(obj, f"ask{level}")
        row[f"bid_vol{level}"] = _obj_get(obj, f"bid_vol{level}")
        row[f"ask_vol{level}"] = _obj_get(obj, f"ask_vol{level}")
    return row


def run_quotes_xmtdx(pool: str, base_dir: Path, host: str, port: int) -> None:
    require_provider("xmtdx")
    stocks = load_pool_codes(base_dir, pool, include_history_freqs=("d",))
    pairs = [(_market_for_xmtdx(code), to_plain_code(code)) for code in stocks["code"].astype(str)]
    snapshot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    with _open_xmtdx_client(host, port) as client:
        for start in range(0, len(pairs), 80):
            rows.extend(_quote_to_row(q, snapshot_time) for q in client.get_security_quotes(pairs[start : start + 80]))
    df = pd.DataFrame(rows)
    if df.empty:
        print("[quotes] empty")
        return
    day = pd.to_datetime(snapshot_time).strftime("%Y%m%d")
    clock = pd.to_datetime(snapshot_time).strftime("%H%M%S")
    save_table(df, base_dir / "quote_snapshot" / "xmtdx" / pool / day / f"quotes_{clock}")
    print(f"[quotes] saved {len(df)} rows")


def _normalize_transactions(records: Iterable[Any], code: str, trade_date: str) -> pd.DataFrame:
    rows = []
    day = ymd(trade_date)
    for rec in records:
        hour = int(_obj_get(rec, "hour", 0) or 0)
        minute = int(_obj_get(rec, "minute", 0) or 0)
        second = int(_obj_get(rec, "second", 0) or 0)
        rows.append(
            {
                "date": pd.to_datetime(day).strftime("%Y-%m-%d"),
                "time": f"{day}{hour:02d}{minute:02d}{second:02d}000",
                "code": to_bs_code(code),
                "price": _obj_get(rec, "price"),
                "volume": _obj_get(rec, "vol", _obj_get(rec, "volume")),
                "amount": _obj_get(rec, "amount"),
                "side": _obj_get(rec, "buyorsell", _obj_get(rec, "side")),
                "provider": "xmtdx",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ["price", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["amount"].isna().all():
        df["amount"] = df["price"] * df["volume"]
    return df.sort_values("time").reset_index(drop=True)


def fetch_transactions_xmtdx(code: str, trade_date: str, host: str, port: int, max_pages: int) -> pd.DataFrame:
    market = _market_for_xmtdx(code)
    symbol = to_plain_code(code)
    records = []
    day = ymd(trade_date)
    with _open_xmtdx_client(host, port) as client:
        for page in range(max_pages):
            batch = client.get_history_transaction_data(market, symbol, day, page * 800, 800)
            if not batch:
                break
            records.extend(batch)
            if len(batch) < 800:
                break
    return _normalize_transactions(records, code, day)


def update_transactions_one(
    code: str,
    *,
    trade_date: str,
    base_dir: Path,
    pool: str,
    host: str,
    port: int,
    max_pages: int,
) -> str:
    key = to_symbol_key(code)
    day = ymd(trade_date)
    df = fetch_transactions_xmtdx(code, day, host, port, max_pages)
    if df.empty:
        return f"empty:{key}"
    stem = base_dir / "tick_trades" / "xmtdx" / pool / day / f"{key}_{day}"
    save_table(df, stem)
    polite_sleep(0.03, 0.18)
    return f"ok:{key}:{len(df)}"


def run_bars(
    *,
    pool: str,
    provider: str,
    freq: str,
    workers: int,
    base_dir: Path,
    host: str,
    port: int,
    max_pages: int,
) -> None:
    require_provider(provider)
    stocks = load_pool_codes(base_dir, pool, include_history_freqs=("d", freq))
    codes = stocks["code"].astype(str).tolist()
    ensure_dir(base_dir / "stock_hist" / pool / freq)
    print(f"[tdx-bars] provider={provider} pool={pool} freq={freq} codes={len(codes)}")
    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as ex:
        futs = [
            ex.submit(
                update_bars_one,
                code,
                provider=provider,
                freq=freq,
                base_dir=base_dir,
                pool=pool,
                host=host,
                port=port,
                max_pages=max_pages,
            )
            for code in codes
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{provider}-{freq}"):
            try:
                msg = fut.result()
                if not msg.startswith("ok"):
                    print(msg)
            except ModuleNotFoundError as exc:
                raise SystemExit(f"missing optional package: {exc.name}; install it in env_quant") from exc
            except Exception as exc:
                print(f"[fail] {type(exc).__name__}: {exc}")


def run_transactions(
    *,
    pool: str,
    trade_date: str,
    workers: int,
    base_dir: Path,
    host: str,
    port: int,
    max_pages: int,
) -> None:
    require_provider("xmtdx")
    stocks = load_pool_codes(base_dir, pool, include_history_freqs=("d",))
    codes = stocks["code"].astype(str).tolist()
    print(f"[tdx-transactions] provider=xmtdx pool={pool} date={ymd(trade_date)} codes={len(codes)}")
    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as ex:
        futs = [
            ex.submit(
                update_transactions_one,
                code,
                trade_date=trade_date,
                base_dir=base_dir,
                pool=pool,
                host=host,
                port=port,
                max_pages=max_pages,
            )
            for code in codes
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="xmtdx-transactions"):
            try:
                msg = fut.result()
                if not msg.startswith("ok"):
                    print(msg)
            except ModuleNotFoundError as exc:
                raise SystemExit(f"missing optional package: {exc.name}; install xmtdx in env_quant") from exc
            except Exception as exc:
                print(f"[fail] {type(exc).__name__}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="bars", choices=["bars", "quotes", "transactions"])
    parser.add_argument("--provider", default="xmtdx", choices=["xmtdx", "pytdx"])
    parser.add_argument("--pool", default="hs300", choices=["sz50", "hs300", "zz500", "all"])
    parser.add_argument("--freq", default="d", choices=["1", "5", "15", "30", "60", "d", "w", "m"])
    parser.add_argument("--trade-date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--host", default=DEFAULT_TDX_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_TDX_PORT)
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--base-dir", default=str(default_base_dir()))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.base_dir)
    if args.mode == "bars":
        run_bars(
            pool=args.pool,
            provider=args.provider,
            freq=args.freq,
            workers=args.workers,
            base_dir=out_dir,
            host=args.host,
            port=args.port,
            max_pages=args.max_pages,
        )
    elif args.mode == "quotes":
        if args.provider != "xmtdx":
            raise SystemExit("quotes mode currently requires --provider xmtdx")
        run_quotes_xmtdx(args.pool, out_dir, args.host, args.port)
    elif args.mode == "transactions":
        if args.provider != "xmtdx":
            raise SystemExit("transactions mode currently requires --provider xmtdx")
        run_transactions(
            pool=args.pool,
            trade_date=args.trade_date,
            workers=args.workers,
            base_dir=out_dir,
            host=args.host,
            port=args.port,
            max_pages=args.max_pages,
        )
