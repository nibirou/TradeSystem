"""Common utility helpers."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


_LOG_LEVEL = "normal"  # quiet | normal | verbose


def parse_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).normalize()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def import_module_from_file(module_path: str, module_name: str = "custom_module") -> object:
    path = Path(module_path)
    if not path.exists():
        raise FileNotFoundError(f"module file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def to_jsonable_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def set_log_level(level: str = "normal", quiet: bool = False, verbose: bool = False) -> str:
    """Configure progress-log level globally.

    Priority:
    1) quiet=True  -> quiet
    2) verbose=True -> verbose
    3) explicit level (quiet/normal/verbose)
    """
    global _LOG_LEVEL
    if bool(quiet):
        _LOG_LEVEL = "quiet"
        return _LOG_LEVEL
    if bool(verbose):
        _LOG_LEVEL = "verbose"
        return _LOG_LEVEL
    lv = str(level).strip().lower()
    if lv not in {"quiet", "normal", "verbose"}:
        lv = "normal"
    _LOG_LEVEL = lv
    return _LOG_LEVEL


def log_progress(message: str, module: str = "core", level: str = "info") -> None:
    """Print a timestamped progress log line.

    This is intentionally lightweight (stdout print) so it works in both
    scripts and notebooks without extra logging dependencies.
    """
    lv = str(level).strip().lower()
    if _LOG_LEVEL == "quiet":
        return
    if _LOG_LEVEL != "verbose" and lv in {"debug", "trace"}:
        return
    ts = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}][{module}] {message}")


def symbol_key_from_filename(filename: str) -> Optional[str]:
    m = re.match(r"^([a-z]{2}_\d{6})_(d|5)\.(csv|parquet)$", filename.lower())
    return m.group(1) if m else None


def split_exchange_code(code_or_key: str) -> Tuple[str, str]:
    s = str(code_or_key).strip().lower()
    if "_" in s:
        ex, code = s.split("_", 1)
    elif "." in s:
        left, right = s.split(".", 1)
        # common forms:
        # 1) sh.600000 / sz.000001
        # 2) 600000.sh / 000001.sz
        if left in {"sh", "sz"}:
            ex, code = left, right
        elif right in {"sh", "sz"}:
            ex, code = right, left
        else:
            code, ex = left, right
    else:
        code, ex = s, ""

    code = re.sub(r"[^0-9]", "", code)
    code = code[-6:].zfill(6) if code else ""
    ex = ex.strip().lower()
    if ex not in {"sh", "sz"}:
        if code.startswith(("6", "9")):
            ex = "sh"
        elif code.startswith(("0", "1", "2", "3")):
            ex = "sz"
    return ex, code


def is_main_board_symbol(code_or_key: str) -> bool:
    ex, code = split_exchange_code(code_or_key)
    if ex == "sh":
        return code.startswith(("600", "601", "603", "605"))
    if ex == "sz":
        return code.startswith(("000", "001", "002", "003"))
    return False


def infer_board_type(code_or_key: str) -> str:
    ex, code = split_exchange_code(code_or_key)
    if ex == "sh":
        if code.startswith(("600", "601", "603", "605")):
            return "main_board"
        if code.startswith("688"):
            return "star_board"
    if ex == "sz":
        if code.startswith(("000", "001", "002", "003")):
            return "main_board"
        if code.startswith("300"):
            return "chi_next"
    return "other_board"


def infer_industry_bucket(code_or_key: str) -> str:
    ex, code = split_exchange_code(code_or_key)
    if not code:
        return "unknown"
    return f"{ex}_{code[:2]}"
