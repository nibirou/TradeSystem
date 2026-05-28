# -*- coding: utf-8 -*-
"""Optional pywencai generic query downloader.

Use this for free-form iWenCai factor/event exploration, then promote stable
queries into dedicated production scripts once the fields are confirmed.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import sys
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_fetch_common import default_base_dir, ensure_dir, save_table  # noqa: E402


PRESETS = {
    "research": "近30日机构调研次数排名，A股",
    "buyback": "近一年回购金额排名，A股",
    "pledge": "股权质押比例排名，A股",
    "analyst": "近90日券商研报买入评级数量排名，A股",
    "limit_up": "今日涨停股票，所属概念，封单金额，A股",
}


def _load_pywencai():
    try:
        import pywencai
    except ModuleNotFoundError as exc:
        raise SystemExit("missing optional package: pywencai; install it in env_quant") from exc
    return pywencai


def _safe_name(text: str) -> str:
    name = re.sub(r"[^0-9A-Za-z_\-]+", "_", text).strip("_")
    return name[:80] or "wencai_query"


def _console_escape(text: str) -> str:
    return text.encode("unicode_escape").decode("ascii")


def run_query(
    query: str,
    base_dir: Optional[Path] = None,
    query_type: str = "stock",
    loop: bool = True,
    name: str = "",
    cookie: str = "",
) -> None:
    pywencai = _load_pywencai()
    base_dir = Path(base_dir or default_base_dir())
    day = datetime.now().strftime("%Y%m%d")
    out_dir = ensure_dir(base_dir / "wencai_query" / day)
    kwargs = {"query": query, "query_type": query_type, "loop": loop}
    if cookie:
        kwargs["cookie"] = cookie
    result = pywencai.get(**kwargs)
    df = pd.DataFrame(result)
    if df.empty:
        print("[wencai] empty")
        return
    df["provider"] = "pywencai"
    df["query"] = query
    df["download_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stem = out_dir / (name or _safe_name(query))
    save_table(df, stem)
    print(f"[wencai] saved {len(df)} rows to {stem.with_suffix('.csv')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="")
    parser.add_argument("--query-type", default="stock")
    parser.add_argument("--base-dir", default=str(default_base_dir()))
    parser.add_argument("--name", default="")
    parser.add_argument("--cookie", default="", help="or set IWENCAI_COOKIE")
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--list-presets", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.list_presets:
        for key, query in PRESETS.items():
            print(f"{key}: {_console_escape(query)}")
        raise SystemExit(0)
    query_text = args.query or PRESETS.get(args.preset, "")
    if not query_text:
        raise SystemExit("provide --query or --preset")
    run_query(
        query=query_text,
        base_dir=Path(args.base_dir),
        query_type=args.query_type,
        loop=not args.no_loop,
        name=args.name,
        cookie=args.cookie or os.getenv("IWENCAI_COOKIE", ""),
    )
