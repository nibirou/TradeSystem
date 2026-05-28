# -*- coding: utf-8 -*-
"""Macro and liquidity data downloader based on AkShare.

These tables are not consumed directly by Strategy7 today, but they are useful
for regime filters, market timing, risk-on/risk-off features, and later text or
cross-asset research.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import inspect
import sys
from typing import Callable, Optional

import akshare as ak
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_fetch_common import default_base_dir, ensure_dir, save_table  # noqa: E402


DATASETS = [
    "macro_china_lpr",
    "macro_china_shibor_all",
    "macro_china_cpi",
    "macro_china_ppi",
    "macro_china_pmi",
    "macro_china_non_man_pmi",
    "macro_china_cx_pmi_yearly",
    "macro_china_gdp",
    "macro_china_money_supply",
    "macro_china_m2_yearly",
    "macro_china_new_financial_credit",
    "macro_china_reserve_requirement_ratio",
    "macro_china_fx_reserves_yearly",
    "macro_china_trade_balance",
    "macro_china_exports_yoy",
    "macro_china_imports_yoy",
    "macro_china_stock_market_cap",
    "macro_china_market_margin_sh",
    "macro_china_market_margin_sz",
    "macro_china_real_estate",
    "macro_china_consumer_goods_retail",
    "macro_china_industrial_production_yoy",
    "macro_china_urban_unemployment",
    "macro_china_bank_financing",
    "macro_china_commodity_price_index",
]


def _callable_without_required_args(func: Callable[..., pd.DataFrame]) -> bool:
    sig = inspect.signature(func)
    return all(
        p.default is not inspect.Parameter.empty
        or p.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        for p in sig.parameters.values()
    )


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_dataset(name: str) -> pd.DataFrame:
    if not hasattr(ak, name):
        raise AttributeError(f"akshare has no dataset function: {name}")
    func = getattr(ak, name)
    if not _callable_without_required_args(func):
        raise TypeError(f"{name} requires explicit parameters; add it as a custom dataset")
    df = func()
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    out["provider"] = "akshare"
    out["dataset"] = name
    out["download_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return out


def update_one(name: str, base_dir: Path) -> str:
    df = fetch_dataset(name)
    if df.empty:
        return f"empty:{name}"
    save_table(df, base_dir / "macro" / "akshare" / name)
    return f"ok:{name}:{len(df)}"


def run_download(
    datasets: Optional[list[str]] = None,
    workers: int = 2,
    base_dir: Optional[Path] = None,
    list_only: bool = False,
) -> None:
    names = datasets or DATASETS
    if list_only:
        for name in DATASETS:
            print(name)
        return
    base_dir = Path(base_dir or default_base_dir())
    ensure_dir(base_dir / "macro" / "akshare")
    print(f"[macro] datasets={len(names)} base_dir={base_dir}")
    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as ex:
        futs = [ex.submit(update_one, name, base_dir) for name in names]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="macro-akshare"):
            try:
                msg = fut.result()
                if not msg.startswith("ok"):
                    print(msg)
            except Exception as exc:
                print(f"[fail] {type(exc).__name__}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None, help="default: all curated datasets")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--base-dir", default=str(default_base_dir()))
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_download(args.datasets, args.workers, Path(args.base_dir), args.list)
