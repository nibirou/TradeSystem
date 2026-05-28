# -*- coding: utf-8 -*-
"""CNInfo disclosure downloader.

CNInfo is the primary public disclosure portal in China. This script complements
the existing EastMoney announcement script and writes per-stock event files that
Strategy7 can load as notice text data when --text-root-notice points here.
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


DEFAULT_CATEGORIES = [
    "年报",
    "半年报",
    "一季报",
    "三季报",
    "业绩预告",
    "权益分派",
    "风险提示",
    "特别处理和退市",
    "补充更正",
]


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_cninfo(symbol: str, category: str, start_date: str, end_date: str) -> pd.DataFrame:
    return ak.stock_zh_a_disclosure_report_cninfo(
        symbol=symbol,
        market="沪深京",
        category=category,
        start_date=start_date,
        end_date=end_date,
    )


def normalize_cninfo(raw: pd.DataFrame, code: str, category: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    rename = {
        "代码": "raw_code",
        "简称": "name",
        "公告标题": "title",
        "公告时间": "publish_time",
        "公告链接": "url",
    }
    df = df.rename(columns={c: rename.get(str(c), c) for c in df.columns})
    publish = pd.to_datetime(
        df["publish_time"] if "publish_time" in df.columns else pd.Series(pd.NaT, index=df.index),
        errors="coerce",
    )
    title = df["title"].astype(str) if "title" in df.columns else pd.Series("", index=df.index, dtype=str)
    url = df["url"].astype(str) if "url" in df.columns else pd.Series("", index=df.index, dtype=str)
    name = df["name"].astype(str) if "name" in df.columns else pd.Series("", index=df.index, dtype=str)
    out = pd.DataFrame()
    out["date"] = publish.dt.strftime("%Y-%m-%d")
    out["publish_time"] = publish
    out["code"] = to_bs_code(code)
    out["股票代码"] = to_plain_code(code)
    out["公告日期"] = out["date"]
    out["title"] = title
    out["content"] = title
    out["url"] = url
    out["name"] = name
    out["category"] = category
    out["source"] = "cninfo"
    out = out.dropna(subset=["date"])
    return out.drop_duplicates(["date", "code", "title", "url"], keep="last").sort_values(["date", "title"])


def update_one(
    code: str,
    *,
    base_dir: Path,
    pool: str,
    categories: list[str],
    start_date: str,
    end_date: str,
) -> str:
    key = to_symbol_key(code)
    stem = base_dir / "data_cninfo_notices" / pool / key
    old = read_existing_table(stem)
    parts = []
    for cat in categories:
        raw = fetch_cninfo(to_plain_code(code), cat, start_date, end_date)
        part = normalize_cninfo(raw, code, cat)
        if not part.empty:
            parts.append(part)
        polite_sleep(0.15, 0.5)
    if not parts and old.empty:
        return f"empty:{key}"
    new = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    merged = pd.concat([old, new], ignore_index=True) if not old.empty else new
    if merged.empty:
        return f"empty:{key}"
    merged = (
        merged.drop_duplicates(["date", "code", "title", "url", "category"], keep="last")
        .sort_values(["date", "category", "title"])
        .reset_index(drop=True)
    )
    save_table(merged, stem)
    return f"ok:{key}:{len(new)}"


def run_download(
    pool: str = "hs300",
    start_date: str = "20150101",
    end_date: Optional[str] = None,
    categories: Optional[list[str]] = None,
    workers: int = 6,
    base_dir: Optional[Path] = None,
) -> None:
    base_dir = Path(base_dir or default_base_dir())
    end_date = end_date or pd.Timestamp.now().strftime("%Y%m%d")
    categories = categories or DEFAULT_CATEGORIES
    ensure_dir(base_dir / "data_cninfo_notices" / pool)
    stocks = load_pool_codes(base_dir, pool, include_history_freqs=("d",))
    codes = stocks["code"].astype(str).tolist()
    print(f"[cninfo] pool={pool} codes={len(codes)} start={start_date} end={end_date} categories={len(categories)}")
    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as ex:
        futs = [
            ex.submit(
                update_one,
                code,
                base_dir=base_dir,
                pool=pool,
                categories=categories,
                start_date=start_date,
                end_date=end_date,
            )
            for code in codes
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{pool}-cninfo"):
            try:
                msg = fut.result()
                if msg.startswith("empty"):
                    print(msg)
            except Exception as exc:
                print(f"[fail] {type(exc).__name__}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="hs300", choices=["sz50", "hs300", "zz500", "all"])
    parser.add_argument("--start-date", default="20150101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--base-dir", default=str(default_base_dir()))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cats = [x.strip() for x in args.categories.split(",") if x.strip()]
    run_download(
        pool=args.pool,
        start_date=ymd(args.start_date),
        end_date=ymd(args.end_date) if args.end_date else None,
        categories=cats,
        workers=args.workers,
        base_dir=Path(args.base_dir),
    )
