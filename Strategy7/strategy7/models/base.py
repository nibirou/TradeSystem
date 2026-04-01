"""Base model interfaces for four pluggable modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


class PersistableModel(ABC):
    @abstractmethod
    def save(self, folder: Path, run_tag: str) -> Dict[str, str]:
        raise NotImplementedError


class StockSelectionModel(PersistableModel, ABC):
    @abstractmethod
    def fit(self, train_df: pd.DataFrame, factor_cols: list[str], target_col: str) -> "StockSelectionModel":
        raise NotImplementedError

    @abstractmethod
    def predict_score(self, df: pd.DataFrame, factor_cols: list[str]) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def fill_values(self) -> pd.Series:
        raise NotImplementedError


class TimingModel(PersistableModel, ABC):
    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "TimingModel":
        raise NotImplementedError

    @abstractmethod
    def predict_exposure(self, day_df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """Return exposure in [0,1]."""
        raise NotImplementedError


class PortfolioModel(PersistableModel, ABC):
    @abstractmethod
    def compute_weights(
        self,
        day_pick: pd.DataFrame,
        day_universe: pd.DataFrame,
        prev_weights: Dict[str, float],
        fee_bps: float,
        slippage_bps: float,
    ) -> Tuple[pd.Series, Dict[str, float]]:
        raise NotImplementedError


class ExecutionModel(PersistableModel, ABC):
    @abstractmethod
    def apply_execution(
        self,
        day_pick: pd.DataFrame,
        weight_col: str,
        fee_bps: float,
        slippage_bps: float,
    ) -> tuple[pd.DataFrame, Dict[str, float]]:
        raise NotImplementedError

