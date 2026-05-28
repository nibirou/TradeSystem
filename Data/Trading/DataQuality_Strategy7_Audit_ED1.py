# -*- coding: utf-8 -*-
"""Audit Strategy7 market-data files without changing them."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_fetch_common import default_base_dir, ensure_dir, save_table  # noqa: E402


DAILY_REQUIRED = ["date", "code", "open", "high", "low", "close", "volume", "amount"]
MINUTE_REQUIRED = ["date", "time", "code", "open", "high", "low", "close", "volume", "amount"]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _audit_file(path: Path, required_cols: list[str]) -> dict[str, object]:
    row: dict[str, object] = {
        "file": str(path),
        "ok": False,
        "rows": 0,
        "start_date": "",
        "end_date": "",
        "duplicate_keys": 0,
        "missing_cols": "",
        "bad_price_rows": 0,
        "error": "",
    }
    try:
        df = _read_table(path)
        row["rows"] = len(df)
        missing = [c for c in required_cols if c not in df.columns]
        row["missing_cols"] = ",".join(missing)
        if "date" in df.columns:
            dt = pd.to_datetime(df["date"], errors="coerce")
            if dt.notna().any():
                row["start_date"] = dt.min().strftime("%Y-%m-%d")
                row["end_date"] = dt.max().strftime("%Y-%m-%d")
        key_cols = [c for c in ["date", "time"] if c in df.columns]
        if key_cols:
            row["duplicate_keys"] = int(df.duplicated(key_cols).sum())
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        if price_cols:
            px = df[price_cols].apply(pd.to_numeric, errors="coerce")
            row["bad_price_rows"] = int(((px <= 0) | px.isna()).any(axis=1).sum())
        row["ok"] = not missing and row["duplicate_keys"] == 0 and row["bad_price_rows"] == 0
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def run_audit(pool: str, freq: str, base_dir: Optional[Path] = None, prefer_parquet: bool = True) -> None:
    base_dir = Path(base_dir or default_base_dir())
    hist_dir = base_dir / "stock_hist" / pool / freq
    if not hist_dir.exists():
        raise SystemExit(f"history directory does not exist: {hist_dir}")
    pattern = "*.parquet" if prefer_parquet else "*.csv"
    files = sorted(hist_dir.glob(pattern))
    if not files and prefer_parquet:
        files = sorted(hist_dir.glob("*.csv"))
    required = DAILY_REQUIRED if freq in {"d", "w", "m"} else MINUTE_REQUIRED
    rows = [_audit_file(fp, required) for fp in tqdm(files, desc=f"audit-{pool}-{freq}")]
    report = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_stem = base_dir / "quality_reports" / f"strategy7_market_quality_{pool}_{freq}_{ts}"
    ensure_dir(out_stem.parent)
    save_table(report, out_stem)
    summary = {
        "pool": pool,
        "freq": freq,
        "files": len(files),
        "ok_files": int(report["ok"].sum()) if not report.empty else 0,
        "bad_files": int((~report["ok"]).sum()) if not report.empty else 0,
        "output": str(out_stem.with_suffix(".csv")),
    }
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="hs300", choices=["sz50", "hs300", "zz500", "all"])
    parser.add_argument("--freq", default="d")
    parser.add_argument("--base-dir", default=str(default_base_dir()))
    parser.add_argument("--csv", action="store_true", help="audit CSV instead of parquet when both exist")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_audit(args.pool, args.freq, Path(args.base_dir), prefer_parquet=not args.csv)
