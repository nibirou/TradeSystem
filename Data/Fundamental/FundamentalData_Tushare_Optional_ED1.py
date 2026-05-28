# -*- coding: utf-8 -*-
"""Optional Tushare Pro downloader.

Tushare is useful as a cross-check and for structured fundamentals, corporate
actions, adjustment factors, and index data.  It requires a token and some APIs
have point/frequency limits, so this script is optional and isolated.

Usage:
    set TUSHARE_TOKEN=your_token
    conda run -n env_quant python Data/Fundamental/FundamentalData_Tushare_Optional_ED1.py --mode stock_basic
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Optional

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_fetch_common import default_base_dir, ensure_dir, load_pool_codes, save_table, split_exchange_code, ymd  # noqa: E402


def _load_tushare(token: Optional[str] = None):
    try:
        import tushare as ts
    except ModuleNotFoundError as exc:
        raise SystemExit("missing optional package: tushare; install it in env_quant") from exc
    token = token or os.getenv("TUSHARE_TOKEN")
    if not token:
        raise SystemExit("TUSHARE_TOKEN is required for this optional downloader")
    ts.set_token(token)
    return ts.pro_api(token)


def to_ts_code(code: str) -> str:
    market, symbol = split_exchange_code(code)
    return f"{symbol}.{market.upper()}"


def _stamp(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = df.copy()
    out["provider"] = "tushare"
    out["dataset"] = dataset
    out["download_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return out


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def pro_call(pro, name: str, **kwargs) -> pd.DataFrame:
    func = getattr(pro, name)
    df = func(**kwargs)
    return pd.DataFrame() if df is None else df


def save_stock_basic(pro, base_dir: Path) -> None:
    fields = "ts_code,symbol,name,area,industry,market,list_date,exchange,curr_type,list_status,delist_date,is_hs"
    df = pro_call(pro, "stock_basic", exchange="", list_status="L", fields=fields)
    if not df.empty:
        save_table(_stamp(df, "stock_basic"), base_dir / "tushare" / "stock_basic" / "stock_basic")
        print(f"[stock_basic] saved {len(df)} rows")


def save_daily_basic(pro, base_dir: Path, trade_date: str) -> None:
    fields = (
        "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,"
        "pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,"
        "free_share,total_mv,circ_mv"
    )
    df = pro_call(pro, "daily_basic", trade_date=ymd(trade_date), fields=fields)
    if not df.empty:
        save_table(_stamp(df, "daily_basic"), base_dir / "tushare" / "daily_basic" / ymd(trade_date))
        print(f"[daily_basic] saved {len(df)} rows")


def save_index_daily(pro, base_dir: Path, start_date: str, end_date: str) -> None:
    indices = {"hs300": "000300.SH", "zz500": "000905.SH", "zz1000": "000852.SH", "sse": "000001.SH"}
    for name, ts_code in indices.items():
        df = pro_call(pro, "index_daily", ts_code=ts_code, start_date=ymd(start_date), end_date=ymd(end_date))
        if not df.empty:
            save_table(_stamp(df, f"index_daily_{name}"), base_dir / "tushare" / "index_daily" / name)
            print(f"[index_daily] {name}: {len(df)} rows")


def save_per_stock_dataset(
    pro,
    dataset: str,
    codes: list[str],
    base_dir: Path,
    start_date: str,
    end_date: str,
    pool: str,
) -> None:
    out_dir = ensure_dir(base_dir / "tushare" / dataset / pool)
    for code in tqdm(codes, desc=f"tushare-{dataset}"):
        ts_code = to_ts_code(code)
        try:
            kwargs = {"ts_code": ts_code}
            if dataset in {"adj_factor", "income", "balancesheet", "cashflow", "fina_indicator", "suspend_d"}:
                kwargs.update({"start_date": ymd(start_date), "end_date": ymd(end_date)})
            df = pro_call(pro, dataset, **kwargs)
            if df.empty:
                continue
            safe = ts_code.replace(".", "_").lower()
            save_table(_stamp(df, dataset), out_dir / safe)
        except Exception as exc:
            print(f"[fail] {dataset} {ts_code}: {type(exc).__name__}: {exc}")


def run_download(
    mode: str,
    pool: str,
    start_date: str,
    end_date: str,
    base_dir: Optional[Path] = None,
    token: Optional[str] = None,
) -> None:
    base_dir = Path(base_dir or default_base_dir())
    ensure_dir(base_dir / "tushare")
    pro = _load_tushare(token)
    if mode == "stock_basic":
        save_stock_basic(pro, base_dir)
        return
    if mode == "daily_basic":
        save_daily_basic(pro, base_dir, end_date)
        return
    if mode == "index_daily":
        save_index_daily(pro, base_dir, start_date, end_date)
        return

    stocks = load_pool_codes(base_dir, pool, include_history_freqs=("d",))
    codes = stocks["code"].astype(str).tolist()
    save_per_stock_dataset(pro, mode, codes, base_dir, start_date, end_date, pool)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="stock_basic",
        choices=[
            "stock_basic",
            "daily_basic",
            "adj_factor",
            "income",
            "balancesheet",
            "cashflow",
            "fina_indicator",
            "dividend",
            "suspend_d",
            "index_daily",
        ],
    )
    parser.add_argument("--pool", default="hs300", choices=["sz50", "hs300", "zz500", "all"])
    parser.add_argument("--start-date", default="20150101")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--base-dir", default=str(default_base_dir()))
    parser.add_argument("--token", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_download(
        mode=args.mode,
        pool=args.pool,
        start_date=args.start_date,
        end_date=args.end_date,
        base_dir=Path(args.base_dir),
        token=args.token or None,
    )
