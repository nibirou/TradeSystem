"""Pluggable external data sources (fundamental, NLP, custom factor tables)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Protocol

import pandas as pd

from ..core.registry import Registry
from ..core.utils import import_module_from_file


class ExternalDataSource(Protocol):
    """Return a [date, code, ...] daily aligned panel."""

    name: str

    def load(self, trade_dates: pd.DatetimeIndex, code_universe: List[str]) -> pd.DataFrame:
        ...


@dataclass
class TableFileSource:
    """Load one external factor/fundamental/text table file."""

    name: str
    path: str
    date_col: str = "date"
    code_col: str = "code"
    prefix: str = ""
    file_format: str = "auto"

    def load(self, trade_dates: pd.DatetimeIndex, code_universe: List[str]) -> pd.DataFrame:
        fp = Path(self.path)
        if not fp.exists():
            return pd.DataFrame()
        if self.file_format == "parquet" or (self.file_format == "auto" and fp.suffix.lower() == ".parquet"):
            df = pd.read_parquet(fp)
        else:
            df = pd.read_csv(fp)
        if self.date_col not in df.columns or self.code_col not in df.columns:
            return pd.DataFrame()
        out = df.copy()
        out[self.date_col] = pd.to_datetime(out[self.date_col], errors="coerce").dt.normalize()
        out[self.code_col] = out[self.code_col].astype(str)
        out = out.dropna(subset=[self.date_col, self.code_col])
        out = out[out[self.date_col].isin(pd.DatetimeIndex(trade_dates))]
        out = out[out[self.code_col].isin(set(code_universe))]
        if out.empty:
            return out
        rename_map: Dict[str, str] = {}
        for c in out.columns:
            if c in {self.date_col, self.code_col}:
                continue
            if self.prefix:
                rename_map[c] = f"{self.prefix}_{c}"
        out = out.rename(columns=rename_map)
        out = out.rename(columns={self.date_col: "date", self.code_col: "code"})
        return out


@dataclass
class DirectoryTableSource:
    """Load multiple table files from a folder and vertically concatenate."""

    name: str
    folder: str
    pattern: str = "*.csv"
    date_col: str = "date"
    code_col: str = "code"
    prefix: str = ""

    def load(self, trade_dates: pd.DatetimeIndex, code_universe: List[str]) -> pd.DataFrame:
        p = Path(self.folder)
        if not p.exists():
            return pd.DataFrame()
        frames = []
        for fp in sorted(p.glob(self.pattern)):
            src = TableFileSource(
                name=f"{self.name}:{fp.name}",
                path=str(fp),
                date_col=self.date_col,
                code_col=self.code_col,
                prefix=self.prefix,
                file_format="auto",
            )
            df = src.load(trade_dates, code_universe)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values(["date", "code"]).drop_duplicates(["date", "code"], keep="last").reset_index(drop=True)


class DataSourceRegistry(Registry[ExternalDataSource]):
    def __init__(self) -> None:
        super().__init__("external_data_source")


def load_custom_source_module(registry: DataSourceRegistry, module_path: str) -> None:
    mod = import_module_from_file(module_path, module_name="strategy7_custom_sources")
    if not hasattr(mod, "register_sources"):
        raise RuntimeError("custom source module must provide register_sources(registry).")
    mod.register_sources(registry)


def merge_external_sources(
    base_panel: pd.DataFrame,
    registry: DataSourceRegistry,
    trade_dates: pd.DatetimeIndex,
    code_universe: List[str],
) -> tuple[pd.DataFrame, Dict[str, int]]:
    out = base_panel.copy()
    notes: Dict[str, int] = {}
    for name, source in registry.items():
        df = source.load(trade_dates, code_universe)
        if df is None or df.empty:
            notes[name] = 0
            continue
        df = df.copy()
        if "date" not in df.columns or "code" not in df.columns:
            notes[name] = 0
            continue
        out = out.merge(df, on=["date", "code"], how="left")
        notes[name] = int(len(df))
    return out, notes

