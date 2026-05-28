# -*- coding: utf-8 -*-
"""Lightweight EastMoney research-report downloader via AKShare.

The existing report scripts scrape EastMoney/Wencai detail pages. This script
keeps a smaller, more stable per-stock report list that Strategy7 can ingest as
text events when --text-root-report-em points to this output root.
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
def fetch_report_list(symbol: str) -> pd.DataFrame:
    return ak.stock_research_report_em(symbol=symbol)


def _first_col(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def normalize_reports(raw: pd.DataFrame, code: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    date_col = _first_col(df, ["日期", "发布时间", "publish_time", "time", "date"])
    title_col = _first_col(df, ["报告名称", "研报标题", "标题", "title"])
    org_col = _first_col(df, ["机构", "机构名称", "orgName"])
    rating_col = _first_col(df, ["东财评级", "评级", "rating"])
    pdf_col = _first_col(df, ["报告PDF链接", "pdf链接", "pdf", "url"])
    if date_col is None:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    out["publish_time"] = pd.to_datetime(df[date_col], errors="coerce")
    out["code"] = to_bs_code(code)
    out["股票代码"] = to_plain_code(code)
    out["title"] = df[title_col].astype(str) if title_col else ""
    out["content"] = out["title"]
    out["organization"] = df[org_col].astype(str) if org_col else ""
    out["rating_change"] = df[rating_col].astype(str) if rating_col else ""
    out["url"] = df[pdf_col].astype(str) if pdf_col else ""
    out["source"] = "akshare_stock_research_report_em"
    return out.dropna(subset=["date"]).sort_values(["date", "title"]).drop_duplicates(["date", "code", "title"], keep="last")


def update_one(code: str, *, base_dir: Path, pool: str) -> str:
    key = to_symbol_key(code)
    stem = base_dir / "data_ak_reports" / pool / key
    old = read_existing_table(stem)
    df = normalize_reports(fetch_report_list(to_plain_code(code)), code)
    if df.empty and old.empty:
        return f"empty:{key}"
    merged = pd.concat([old, df], ignore_index=True) if not old.empty else df
    merged = merged.drop_duplicates(["date", "code", "title"], keep="last").sort_values(["date", "title"])
    save_table(merged.reset_index(drop=True), stem)
    polite_sleep(0.15, 0.45)
    return f"ok:{key}:{len(df)}"


def run_download(pool: str = "hs300", workers: int = 6, base_dir: Optional[Path] = None) -> None:
    base_dir = Path(base_dir or default_base_dir())
    stocks = load_pool_codes(base_dir, pool, include_history_freqs=("d",))
    codes = stocks["code"].astype(str).tolist()
    print(f"[ak-reports] pool={pool} codes={len(codes)}")
    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as ex:
        futs = [ex.submit(update_one, code, base_dir=base_dir, pool=pool) for code in codes]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{pool}-ak-reports"):
            try:
                msg = fut.result()
                if msg.startswith("empty"):
                    print(msg)
            except Exception as exc:
                print(f"[fail] {type(exc).__name__}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="hs300", choices=["sz50", "hs300", "zz500", "all"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--base-dir", default=str(default_base_dir()))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_download(pool=args.pool, workers=args.workers, base_dir=Path(args.base_dir))
