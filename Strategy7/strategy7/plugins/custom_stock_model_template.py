"""Template custom stock model plugin.

Usage:
python run_strategy7.py --custom-stock-model-py ./strategy7/plugins/custom_stock_model_template.py

# load mode:
# python run_strategy7.py \
#   --model-run-mode load \
#   --custom-stock-model-py ./strategy7/plugins/custom_stock_model_template.py \
#   --stock-model-path <your_saved_json_path>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from strategy7.models.base import StockSelectionModel


class ConstantScoreStockModel(StockSelectionModel):
    def __init__(self, constant_score: float = 0.5) -> None:
        self.constant_score = float(constant_score)
        self._factor_cols: list[str] = []

    def fit(self, train_df: pd.DataFrame, factor_cols: list[str], target_col: str) -> "ConstantScoreStockModel":
        self._factor_cols = [str(x) for x in factor_cols if str(x).strip()]
        return self

    def predict_score(self, df: pd.DataFrame, factor_cols: list[str]) -> pd.Series:
        score = float(np.clip(self.constant_score, 0.0, 1.0))
        return pd.Series(np.full(len(df), score, dtype=float), index=df.index, name="pred_score")

    def fill_values(self) -> pd.Series:
        return pd.Series(dtype=float)

    def save(self, folder: Path, run_tag: str) -> Dict[str, str]:
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"custom_stock_constant_{run_tag}.json"
        payload = {
            "model_type": "custom_stock_constant",
            "constant_score": float(self.constant_score),
            "factor_cols": list(self._factor_cols),
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"meta_json": str(p)}


def build_model(cfg) -> StockSelectionModel:
    return ConstantScoreStockModel()


def load_model(cfg, model_path: str | None) -> StockSelectionModel:
    score = 0.5
    factor_cols: list[str] = []
    if model_path:
        p = Path(model_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"custom stock model file not found: {p}")
        payload = json.loads(p.read_text(encoding="utf-8"))
        score = float(payload.get("constant_score", score))
        factor_cols = [str(x) for x in payload.get("factor_cols", []) if str(x).strip()]
    model = ConstantScoreStockModel(constant_score=score)
    model._factor_cols = factor_cols
    return model
