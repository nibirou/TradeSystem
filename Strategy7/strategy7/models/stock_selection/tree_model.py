"""Decision-tree based stock selection model."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ...core.constants import EPS
from ...core.utils import dump_json
from ..base import StockSelectionModel


@dataclass
class TreeStockModel(StockSelectionModel):
    max_depth: int = 6
    min_samples_leaf: int = 200
    random_state: int = 42

    _model: object | None = None
    _is_classifier: bool = True
    _fill_values: pd.Series | None = None
    _target_col: str = "target_up"
    _factor_cols: list[str] | None = None

    def fit(self, train_df: pd.DataFrame, factor_cols: list[str], target_col: str) -> "TreeStockModel":
        self._factor_cols = list(factor_cols)
        x = train_df[factor_cols].replace([np.inf, -np.inf], np.nan)
        self._fill_values = x.median(numeric_only=True)
        x = x.fillna(self._fill_values)
        y_raw = pd.to_numeric(train_df[target_col], errors="coerce")
        valid = y_raw.notna()
        x = x.loc[valid]
        y = y_raw.loc[valid]
        self._target_col = target_col

        y_unique = sorted(pd.Series(y).dropna().unique().tolist())
        self._is_classifier = set(y_unique).issubset({0, 1}) and len(y_unique) <= 2
        if self._is_classifier:
            model = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                class_weight="balanced",
            )
            model.fit(x, y.astype(int))
            self._model = model
        else:
            model = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            model.fit(x, y.astype(float))
            self._model = model
        return self

    def predict_score(self, df: pd.DataFrame, factor_cols: list[str]) -> pd.Series:
        if self._model is None or self._fill_values is None:
            raise RuntimeError("TreeStockModel is not fitted.")
        x = df[factor_cols].replace([np.inf, -np.inf], np.nan).fillna(self._fill_values)
        if self._is_classifier:
            prob = self._model.predict_proba(x)
            if prob.shape[1] == 1:
                cls = int(self._model.classes_[0])
                s = np.ones(len(x), dtype=float) if cls == 1 else np.zeros(len(x), dtype=float)
            else:
                class_to_idx = {int(c): i for i, c in enumerate(self._model.classes_)}
                s = prob[:, class_to_idx.get(1, 0)]
            return pd.Series(s, index=df.index, name="pred_score")

        # regression score -> map to [0,1] by rank percentile
        pred = pd.Series(self._model.predict(x), index=df.index)
        rank = pred.rank(pct=True).fillna(0.5)
        return rank.rename("pred_score")

    def fill_values(self) -> pd.Series:
        if self._fill_values is None:
            return pd.Series(dtype=float)
        return self._fill_values

    def save(self, folder: Path, run_tag: str) -> Dict[str, str]:
        folder.mkdir(parents=True, exist_ok=True)
        model_path = folder / f"stock_model_tree_{run_tag}.pkl"
        meta_path = folder / f"stock_model_tree_{run_tag}.json"
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "fill_values": self._fill_values,
                    "is_classifier": self._is_classifier,
                    "target_col": self._target_col,
                    "factor_cols": list(self._factor_cols or []),
                },
                f,
            )
        dump_json(
            meta_path,
            {
                "model_type": "decision_tree",
                "is_classifier": bool(self._is_classifier),
                "max_depth": int(self.max_depth),
                "min_samples_leaf": int(self.min_samples_leaf),
                "random_state": int(self.random_state),
                "factor_count": int(len(self._factor_cols or [])),
            },
        )
        return {"model_pkl": str(model_path), "meta_json": str(meta_path)}
