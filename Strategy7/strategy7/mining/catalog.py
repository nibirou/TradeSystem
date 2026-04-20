"""Catalog I/O and Strategy7 integration for mined/custom factors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from ..core.utils import ensure_dir
from ..factors.base import FactorLibrary


def _json_default(obj):
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if np.isfinite(v):
            return v
        return None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _normalize_catalog(catalog: Dict[str, object] | None = None) -> Dict[str, object]:
    if not isinstance(catalog, dict):
        catalog = {}
    entries = catalog.get("entries", [])
    if not isinstance(entries, list):
        entries = []
    return {
        "version": int(catalog.get("version", 1)),
        "entries": entries,
    }


def load_catalog(catalog_path: str | Path | None) -> Dict[str, object]:
    if catalog_path is None:
        return _normalize_catalog()
    p = Path(catalog_path)
    if not p.exists():
        return _normalize_catalog()
    try:
        # Support both UTF-8 and UTF-8-BOM files.
        obj = json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return _normalize_catalog()
    return _normalize_catalog(obj)


def save_catalog(catalog_path: str | Path, catalog: Dict[str, object]) -> None:
    p = Path(catalog_path)
    ensure_dir(p.parent)
    p.write_text(
        json.dumps(_normalize_catalog(catalog), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _entry_key(entry: Dict[str, object]) -> tuple[str, str]:
    return str(entry.get("name", "")).strip(), str(entry.get("freq", "")).strip()


def _split_csv_items(expr: str | Sequence[str] | None) -> List[str]:
    if expr is None:
        return []
    if isinstance(expr, str):
        parts = [x.strip() for x in expr.split(",")]
        return [x for x in parts if x]
    out: List[str] = []
    for x in expr:
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def _entry_factor_packages(entry: Dict[str, object]) -> Set[str]:
    out: Set[str] = set()
    primary = str(entry.get("factor_package", "")).strip()
    if primary:
        out.add(primary)

    packs_expr = str(entry.get("factor_packages", "")).strip()
    if packs_expr:
        out.update(_split_csv_items(packs_expr))

    category = str(entry.get("category", "")).strip()
    if category:
        out.add(category)
    return {x for x in out if x}


def upsert_catalog_entries(catalog_path: str | Path, new_entries: Iterable[Dict[str, object]]) -> Dict[str, object]:
    catalog = load_catalog(catalog_path)
    entries: List[Dict[str, object]] = list(catalog.get("entries", []))
    index: Dict[tuple[str, str], int] = {}
    for i, e in enumerate(entries):
        k = _entry_key(e)
        if k[0] and k[1]:
            index[k] = i

    for e in new_entries:
        item = dict(e)
        k = _entry_key(item)
        if not k[0] or not k[1]:
            continue
        if k in index:
            entries[index[k]] = {**entries[index[k]], **item}
        else:
            index[k] = len(entries)
            entries.append(item)

    catalog["entries"] = entries
    save_catalog(catalog_path, catalog)
    return catalog


def load_active_catalog_entries(
    catalog_path: str | Path | None,
    freq: str | None = None,
    *,
    factor_names: Sequence[str] | None = None,
    package_expr: str = "",
) -> List[Dict[str, object]]:
    catalog = load_catalog(catalog_path)
    out: List[Dict[str, object]] = []

    name_set = {str(x).strip() for x in (factor_names or []) if str(x).strip()}
    requested_packs = _split_csv_items(package_expr)
    requested_pack_set = {x for x in requested_packs if x.lower() != "all"}
    package_filter_on = bool(requested_pack_set) and ("all" not in {x.lower() for x in requested_packs})

    for e in catalog.get("entries", []):
        status = str(e.get("status", "active")).strip().lower()
        if status != "active":
            continue
        if freq is not None and str(e.get("freq", "")).strip() != str(freq):
            continue
        name = str(e.get("name", "")).strip()
        value_col = str(e.get("value_col", name)).strip()
        table_path = str(e.get("table_path", "")).strip()
        if not name or not value_col or not table_path:
            continue
        if name_set and name not in name_set:
            continue

        row = dict(e)
        row_packs = _entry_factor_packages(row)
        if package_filter_on and not (row_packs & requested_pack_set):
            continue
        if not str(row.get("factor_package", "")).strip():
            row["factor_package"] = str(row.get("category", "")).strip()
        if not str(row.get("factor_packages", "")).strip():
            row["factor_packages"] = ",".join(sorted(row_packs))
        out.append(row)
    return out


def list_catalog_factor_packages(
    catalog_path: str | Path | None,
    *,
    freq: str | None = None,
) -> List[str]:
    entries = load_active_catalog_entries(catalog_path=catalog_path, freq=freq)
    out: Set[str] = set()
    for e in entries:
        out.update(_entry_factor_packages(e))
    return sorted(x for x in out if x)


def _read_factor_table(path: str, usecols: List[str] | None = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        if p.suffix.lower() == ".parquet":
            return pd.read_parquet(p, columns=usecols)
        if usecols is None:
            return pd.read_csv(p)
        return pd.read_csv(p, usecols=lambda c: c in set(usecols))
    except Exception:
        return pd.DataFrame()


def _find_join_keys(base_panel: pd.DataFrame, table: pd.DataFrame) -> List[str]:
    if "code" not in table.columns or "code" not in base_panel.columns:
        return []
    if "datetime" in base_panel.columns and "datetime" in table.columns:
        return ["code", "datetime"]
    if "date" in base_panel.columns and "date" in table.columns:
        return ["code", "date"]
    return []


def _sanitize_join_frame(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    out = df.copy()
    if "code" in keys:
        out["code"] = out["code"].astype(str).str.strip()
    if "date" in keys and "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    if "datetime" in keys and "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=keys)
    if out.empty:
        return out
    return out.sort_values(keys).drop_duplicates(keys, keep="last")


def merge_catalog_factors(
    base_panel: pd.DataFrame,
    catalog_path: str | Path | None,
    factor_freq: str,
    *,
    factor_names: Sequence[str] | None = None,
    package_expr: str = "",
) -> Tuple[pd.DataFrame, Dict[str, int], List[Dict[str, object]]]:
    """Merge active catalog factors into panel and return loaded catalog entries."""
    out = base_panel.copy()
    notes: Dict[str, int] = {}
    entries = load_active_catalog_entries(
        catalog_path,
        freq=factor_freq,
        factor_names=factor_names,
        package_expr=package_expr,
    )
    if not entries:
        return out, notes, []

    by_table: Dict[str, List[Dict[str, object]]] = {}
    for e in entries:
        by_table.setdefault(str(e.get("table_path", "")).strip(), []).append(e)

    loaded_entries: List[Dict[str, object]] = []
    for table_path, t_entries in by_table.items():
        val_cols = sorted({str(e.get("value_col", e.get("name", ""))).strip() for e in t_entries})
        usecols = ["date", "datetime", "code", *val_cols]
        table = _read_factor_table(table_path, usecols=usecols)
        if table.empty:
            notes[table_path] = 0
            continue

        keys = _find_join_keys(out, table)
        if not keys:
            notes[table_path] = 0
            continue

        table = _sanitize_join_frame(table, keys)
        if table.empty:
            notes[table_path] = 0
            continue

        rename_map: Dict[str, str] = {}
        keep_cols = list(keys)
        table_entries_loaded: List[Dict[str, object]] = []
        for e in t_entries:
            name = str(e.get("name", "")).strip()
            value_col = str(e.get("value_col", name)).strip()
            if not name or value_col not in table.columns:
                continue
            target_col = name if name not in keep_cols else f"catalog_{name}"
            if value_col != target_col:
                rename_map[value_col] = target_col
            keep_cols.append(target_col)
            table_entries_loaded.append({**e, "column_name": target_col})

        if not table_entries_loaded:
            notes[table_path] = 0
            continue

        table = table.rename(columns=rename_map)
        keep_cols = [c for c in keep_cols if c in table.columns]
        table = table[keep_cols].copy()
        out = out.merge(table, on=keys, how="left")
        loaded_entries.extend(table_entries_loaded)
        notes[table_path] = int(len(table))

    return out, notes, loaded_entries


def register_catalog_factors(library: FactorLibrary, entries: Iterable[Dict[str, object]]) -> List[str]:
    registered: List[str] = []
    for e in entries:
        name = str(e.get("name", "")).strip()
        freq = str(e.get("freq", "")).strip()
        col = str(e.get("column_name", e.get("value_col", name))).strip()
        if not name or not freq or not col:
            continue
        category = str(e.get("category", "catalog_mined")).strip() or "catalog_mined"
        desc = str(e.get("description", "")).strip() or f"catalog factor from {e.get('framework', 'unknown')}"
        library.register(name=name, category=category, description=desc, func=lambda d, c=col: d[c], freq=freq)
        registered.append(f"{freq}:{name}")
    return sorted(set(registered))
