# -*- coding: utf-8 -*-
"""Industry and concept membership downloader.

Adds EastMoney/THS industry and concept boards beside the existing Shenwan
third-level script. The output is intentionally raw + normalized so it can be
used both for research inspection and as future Strategy7 custom factors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import akshare as ak
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_fetch_common import default_base_dir, polite_sleep, save_table  # noqa: E402


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def safe_call(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def _code_col(df: pd.DataFrame) -> str | None:
    for col in ["代码", "股票代码", "成分股代码", "code"]:
        if col in df.columns:
            return col
    return None


def _name_col(df: pd.DataFrame) -> str | None:
    for col in ["名称", "股票名称", "成分股名称", "name"]:
        if col in df.columns:
            return col
    return None


def normalize_members(raw: pd.DataFrame, board_name: str, board_code: str, source: str, board_type: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    ccol = _code_col(df)
    ncol = _name_col(df)
    out = df.copy()
    if ccol:
        out["plain_code"] = out[ccol].astype(str).str.extract(r"(\d{6})", expand=False)
        out["code"] = out["plain_code"].map(lambda x: ("sh." if str(x).startswith("6") else "sz.") + str(x) if pd.notna(x) else "")
    else:
        out["code"] = ""
        out["plain_code"] = ""
    out["name"] = out[ncol].astype(str) if ncol else ""
    out["board_name"] = board_name
    out["board_code"] = board_code
    out["board_type"] = board_type
    out["source"] = source
    return out


def download_em_boards(base_dir: Path, board_type: str) -> None:
    if board_type == "industry":
        boards = safe_call(ak.stock_board_industry_name_em)
        member_fn = ak.stock_board_industry_cons_em
        out_root = base_dir / "board_membership" / "eastmoney_industry"
        name_candidates = ["板块名称", "名称"]
    else:
        boards = safe_call(ak.stock_board_concept_name_em)
        member_fn = ak.stock_board_concept_cons_em
        out_root = base_dir / "board_membership" / "eastmoney_concept"
        name_candidates = ["板块名称", "概念名称", "名称"]
    if boards is None or boards.empty:
        return
    save_table(boards, out_root / "_board_list")
    frames = []
    for _, row in tqdm(boards.iterrows(), total=len(boards), desc=f"em-{board_type}"):
        board_name = next((str(row[c]) for c in name_candidates if c in boards.columns), "")
        if not board_name:
            continue
        try:
            raw = safe_call(member_fn, symbol=board_name)
            part = normalize_members(raw, board_name, board_name, f"eastmoney_{board_type}", board_type)
            if not part.empty:
                frames.append(part)
                safe_name = "".join(ch if ch.isalnum() else "_" for ch in board_name)[:80]
                save_table(part, out_root / "by_board" / safe_name)
        except Exception as exc:
            print(f"[board-fail] {board_name}: {exc}")
        polite_sleep(0.15, 0.45)
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        save_table(all_df, out_root / "all_members")


def download_ths_boards(base_dir: Path, board_type: str) -> None:
    if board_type == "industry":
        boards = safe_call(ak.stock_board_industry_name_ths)
        member_fn = ak.stock_board_industry_cons_ths if hasattr(ak, "stock_board_industry_cons_ths") else None
        out_root = base_dir / "board_membership" / "ths_industry"
    else:
        boards = safe_call(ak.stock_board_concept_name_ths)
        member_fn = ak.stock_board_concept_cons_ths if hasattr(ak, "stock_board_concept_cons_ths") else None
        out_root = base_dir / "board_membership" / "ths_concept"
    if boards is not None and not boards.empty:
        save_table(boards, out_root / "_board_list")
    if member_fn is None or boards is None or boards.empty:
        return
    name_col = _name_col(boards) or boards.columns[0]
    frames = []
    for _, row in tqdm(boards.iterrows(), total=len(boards), desc=f"ths-{board_type}"):
        board_name = str(row[name_col])
        try:
            raw = safe_call(member_fn, symbol=board_name)
            part = normalize_members(raw, board_name, board_name, f"ths_{board_type}", board_type)
            if not part.empty:
                frames.append(part)
        except Exception as exc:
            print(f"[ths-board-fail] {board_name}: {exc}")
        polite_sleep(0.2, 0.5)
    if frames:
        save_table(pd.concat(frames, ignore_index=True), out_root / "all_members")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=str(default_base_dir()))
    parser.add_argument("--source", default="em", choices=["em", "ths", "both"])
    parser.add_argument("--board-type", default="both", choices=["industry", "concept", "both"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base = Path(args.base_dir)
    board_types = ["industry", "concept"] if args.board_type == "both" else [args.board_type]
    for bt in board_types:
        if args.source in {"em", "both"}:
            download_em_boards(base, bt)
        if args.source in {"ths", "both"}:
            download_ths_boards(base, bt)
