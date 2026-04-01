"""Template custom stock model plugin.

Usage:
python run_strategy7.py --custom-stock-model-py ./strategy7/plugins/custom_stock_model_template.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from strategy7.models.base import StockSelectionModel


class ConstantScoreStockModel(StockSelectionModel):
    def fit(self, train_df: pd.DataFrame, factor_cols: list[str], target_col: str) -> "ConstantScoreStockModel":
        return self

    def predict_score(self, df: pd.DataFrame, factor_cols: list[str]) -> pd.Series:
        return pd.Series(np.full(len(df), 0.5, dtype=float), index=df.index, name="pred_score")

    def fill_values(self) -> pd.Series:
        return pd.Series(dtype=float)

    def save(self, folder: Path, run_tag: str) -> Dict[str, str]:
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"custom_stock_constant_{run_tag}.txt"
        p.write_text("custom stock model: constant score", encoding="utf-8")
        return {"note_file": str(p)}


def build_model(cfg) -> StockSelectionModel:
    return ConstantScoreStockModel()

