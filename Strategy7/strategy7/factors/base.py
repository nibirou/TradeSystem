"""Pluggable factor library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import pandas as pd

from ..core.utils import import_module_from_file

FactorFunc = Callable[[pd.DataFrame], pd.Series]


@dataclass
class FactorDef:
    name: str
    category: str
    description: str
    func: FactorFunc
    freq: str


class FactorLibrary:
    """Frequency-aware factor registry."""

    def __init__(self) -> None:
        self._factors: Dict[str, FactorDef] = {}

    def _key(self, freq: str, name: str) -> str:
        return f"{freq}:{name}"

    def register(self, name: str, category: str, description: str, func: FactorFunc, freq: str = "D") -> None:
        self._factors[self._key(freq, name)] = FactorDef(name=name, category=category, description=description, func=func, freq=freq)

    def names(self, freq: str) -> List[str]:
        return sorted(v.name for v in self._factors.values() if v.freq == freq)

    def get(self, freq: str, name: str) -> FactorDef:
        k = self._key(freq, name)
        if k not in self._factors:
            raise KeyError(f"factor not found: {name} @ {freq}")
        return self._factors[k]

    def has(self, freq: str, name: str) -> bool:
        return self._key(freq, name) in self._factors

    def metadata(self, freq: str | None = None) -> pd.DataFrame:
        rows = []
        for fd in self._factors.values():
            if freq is not None and fd.freq != freq:
                continue
            rows.append(
                {
                    "factor": fd.name,
                    "freq": fd.freq,
                    "category": fd.category,
                    "description": fd.description,
                }
            )
        return pd.DataFrame(rows).sort_values(["freq", "factor"]) if rows else pd.DataFrame(columns=["factor", "freq", "category", "description"])


def load_custom_factor_module(library: FactorLibrary, module_path: str) -> None:
    mod = import_module_from_file(module_path, module_name="strategy7_custom_factor_module")
    if not hasattr(mod, "register_factors"):
        raise RuntimeError("custom factor module must provide register_factors(library)")
    mod.register_factors(library)


def resolve_selected_factors(library: FactorLibrary, freq: str, factor_list_arg: str, default_set: List[str]) -> List[str]:
    if factor_list_arg.strip():
        selected = [x.strip() for x in factor_list_arg.split(",") if x.strip()]
    else:
        selected = default_set.copy()
    available = set(library.names(freq))
    missing = [f for f in selected if f not in available]
    if missing:
        raise ValueError(f"missing factors for freq={freq}: {missing}")
    return selected


def compute_factor_panel(base_df: pd.DataFrame, library: FactorLibrary, freq: str, selected_factors: List[str]) -> pd.DataFrame:
    panel = base_df.copy()
    for fac in selected_factors:
        try:
            values = library.get(freq, fac).func(panel)
        except Exception as exc:
            raise RuntimeError(f"factor computation failed: {fac} @ {freq}") from exc
        if isinstance(values, pd.Series):
            aligned = values.reindex(panel.index)
        else:
            aligned = pd.Series(values)
            if len(aligned) != len(panel):
                raise ValueError(
                    f"factor result length mismatch for {fac} @ {freq}: "
                    f"expected={len(panel)}, got={len(aligned)}"
                )
            aligned.index = panel.index
        panel[fac] = pd.to_numeric(aligned, errors="coerce")
    return panel


def register_passthrough_panel_factors(
    library: FactorLibrary,
    base_df: pd.DataFrame,
    freq: str,
    *,
    category: str = "auto_panel",
    description_prefix: str = "auto passthrough",
) -> List[str]:
    """Register numeric panel columns as direct-usage factors for the target frequency."""
    if base_df.empty:
        return []

    exclude = {
        "date",
        "datetime",
        "code",
        "entry_date",
        "exit_date",
        "entry_ts",
        "exit_ts",
        "target_date",
        "future_ret_n",
        "target_up",
        "target_return",
        "target_volatility",
        "signal_ts",
        "time_freq",
    }
    registered: List[str] = []
    for c in base_df.columns:
        if c in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(base_df[c]):
            continue
        if library.has(freq, c):
            continue
        library.register(
            name=str(c),
            category=category,
            description=f"{description_prefix}: {c}",
            func=lambda d, col=c: pd.to_numeric(d[col], errors="coerce"),
            freq=freq,
        )
        registered.append(str(c))
    return sorted(registered)
