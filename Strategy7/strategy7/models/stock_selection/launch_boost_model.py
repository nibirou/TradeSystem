"""Launch-boost stock selection model.

Designed for "bottoming now, launching within N bars" research:
1. Classification head learns up/down probability (`target_up`).
2. Return head learns forward-return ranking (`target_return`).
3. A lightweight launch-signal calibration blends dedicated factors.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from ...core.utils import dump_json
from ..base import StockSelectionModel


def _as_float_series(values: object, index: pd.Index) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").reindex(index)
    return pd.Series(values, index=index, dtype=float)


def _rank01(s: pd.Series) -> pd.Series:
    if s.empty:
        return pd.Series(dtype=float, index=s.index)
    return s.rank(method="average", pct=True).clip(0.0, 1.0)


@dataclass
class LaunchBoostStockModel(StockSelectionModel):
    max_depth: int = 4
    min_samples_leaf: int = 120
    learning_rate: float = 0.05
    max_iter: int = 320
    l2_regularization: float = 1.0
    return_head_weight: float = 0.35
    random_state: int = 42

    _cls_model: object | None = None
    _reg_model: object | None = None
    _has_return_head: bool = False
    _constant_cls_prob: float | None = None
    _fill_values: pd.Series | None = None
    _target_col: str = "target_return"
    _factor_cols: list[str] | None = None
    _train_summary: Dict[str, float] | None = None

    def fit(self, train_df: pd.DataFrame, factor_cols: list[str], target_col: str) -> "LaunchBoostStockModel":
        self._factor_cols = list(factor_cols)
        x = train_df[factor_cols].replace([np.inf, -np.inf], np.nan)
        self._fill_values = x.median(numeric_only=True)
        x = x.fillna(self._fill_values)
        self._target_col = str(target_col)

        # Classification head: prefer native target_up, fallback to target sign.
        if "target_up" in train_df.columns:
            y_cls_raw = pd.to_numeric(train_df["target_up"], errors="coerce")
        else:
            y_ref = pd.to_numeric(train_df.get("target_return", train_df[target_col]), errors="coerce")
            y_cls_raw = (y_ref > 0.0).astype(float)
        valid_cls = y_cls_raw.notna()
        x_cls = x.loc[valid_cls]
        y_cls = y_cls_raw.loc[valid_cls].astype(int)

        # Return head: prefer target_return, fallback to target_col when continuous.
        y_ret_raw = pd.to_numeric(train_df.get("target_return", train_df[target_col]), errors="coerce")
        valid_ret = y_ret_raw.notna()
        x_ret = x.loc[valid_ret]
        y_ret = y_ret_raw.loc[valid_ret].astype(float)

        pos_rate = float((y_cls == 1).mean()) if len(y_cls) > 0 else 0.5
        if len(y_cls) < 32 or y_cls.nunique() < 2:
            self._constant_cls_prob = float(np.clip(pos_rate, 1e-3, 1.0 - 1e-3))
            self._cls_model = None
        else:
            self._constant_cls_prob = None
            pos_rate = float(np.clip(pos_rate, 1e-3, 1.0 - 1e-3))
            w_pos = 0.5 / pos_rate
            w_neg = 0.5 / (1.0 - pos_rate)
            sample_weight = np.where(y_cls.to_numpy(dtype=int) == 1, w_pos, w_neg)
            cls_leaf = max(8, min(int(self.min_samples_leaf), max(8, int(len(y_cls) * 0.2))))
            cls = HistGradientBoostingClassifier(
                learning_rate=float(self.learning_rate),
                max_depth=int(self.max_depth),
                max_iter=int(self.max_iter),
                min_samples_leaf=int(cls_leaf),
                l2_regularization=float(self.l2_regularization),
                random_state=int(self.random_state),
            )
            cls.fit(x_cls, y_cls, sample_weight=sample_weight)
            self._cls_model = cls

        if len(y_ret) >= 64 and y_ret.nunique() >= 8:
            reg_leaf = max(8, min(int(self.min_samples_leaf), max(8, int(len(y_ret) * 0.2))))
            reg = HistGradientBoostingRegressor(
                learning_rate=float(self.learning_rate),
                max_depth=int(self.max_depth),
                max_iter=int(self.max_iter),
                min_samples_leaf=int(reg_leaf),
                l2_regularization=float(self.l2_regularization),
                random_state=int(self.random_state),
            )
            reg.fit(x_ret, y_ret)
            self._reg_model = reg
            self._has_return_head = True
        else:
            self._reg_model = None
            self._has_return_head = False

        self._train_summary = {
            "train_rows": float(len(train_df)),
            "cls_rows": float(len(y_cls)),
            "ret_rows": float(len(y_ret)),
            "cls_pos_rate": float(pos_rate),
            "ret_std": float(y_ret.std(ddof=1)) if len(y_ret) > 1 else 0.0,
            "has_return_head": float(self._has_return_head),
            "effective_cls_min_samples_leaf": float(
                max(8, min(int(self.min_samples_leaf), max(8, int(len(y_cls) * 0.2))))
            )
            if len(y_cls) > 0
            else 0.0,
            "effective_ret_min_samples_leaf": float(
                max(8, min(int(self.min_samples_leaf), max(8, int(len(y_ret) * 0.2))))
            )
            if len(y_ret) > 0
            else 0.0,
        }
        return self

    def _predict_launch_prior(self, df: pd.DataFrame) -> pd.Series:
        # Dedicated launch-signal factors are optional; fallback keeps neutral prior.
        prior_components: list[pd.Series] = []
        for c in [
            "launch_quality_score",
            "launch_breakout_pressure_10",
            "launch_trend_turn_5_20",
            "bottom_rebound_20",
            "bottom_rebound_10",
            "launch_volume_dryup_20",
            "launch_intraday_confirm",
            "launch_crowding_clear",
        ]:
            if c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce")
                prior_components.append(_rank01(s))
        if not prior_components:
            return pd.Series(0.5, index=df.index, dtype=float)
        prior = pd.concat(prior_components, axis=1).mean(axis=1, skipna=True)
        return prior.fillna(0.5).clip(0.0, 1.0)

    def predict_score(self, df: pd.DataFrame, factor_cols: list[str]) -> pd.Series:
        if self._fill_values is None:
            raise RuntimeError("LaunchBoostStockModel is not fitted.")
        x = df[factor_cols].replace([np.inf, -np.inf], np.nan).fillna(self._fill_values)

        if self._constant_cls_prob is not None:
            p_up = pd.Series(float(self._constant_cls_prob), index=df.index, dtype=float)
        else:
            if self._cls_model is None:
                raise RuntimeError("LaunchBoostStockModel classifier head is unavailable.")
            prob = self._cls_model.predict_proba(x)
            class_to_idx = {int(c): i for i, c in enumerate(self._cls_model.classes_)}
            idx_up = class_to_idx.get(1, 0)
            p_up = pd.Series(prob[:, idx_up], index=df.index, dtype=float).clip(0.0, 1.0)

        if self._has_return_head and self._reg_model is not None:
            ret_hat = pd.Series(self._reg_model.predict(x), index=df.index, dtype=float)
            ret_rank = _rank01(ret_hat).fillna(0.5)
        else:
            ret_rank = _rank01(p_up).fillna(0.5)

        w_ret = float(np.clip(self.return_head_weight, 0.0, 0.9))
        score = (1.0 - w_ret) * p_up + w_ret * ret_rank

        # Final launch-prior calibration.
        launch_prior = self._predict_launch_prior(df)
        score = 0.85 * score + 0.15 * launch_prior
        return score.clip(0.0, 1.0).rename("pred_score")

    def fill_values(self) -> pd.Series:
        if self._fill_values is None:
            return pd.Series(dtype=float)
        return self._fill_values

    def save(self, folder: Path, run_tag: str) -> Dict[str, str]:
        folder.mkdir(parents=True, exist_ok=True)
        model_path = folder / f"stock_model_launch_boost_{run_tag}.pkl"
        meta_path = folder / f"stock_model_launch_boost_{run_tag}.json"
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "cls_model": self._cls_model,
                    "reg_model": self._reg_model,
                    "has_return_head": bool(self._has_return_head),
                    "constant_cls_prob": self._constant_cls_prob,
                    "fill_values": self._fill_values,
                    "target_col": self._target_col,
                    "factor_cols": list(self._factor_cols or []),
                    "train_summary": dict(self._train_summary or {}),
                },
                f,
            )
        dump_json(
            meta_path,
            {
                "model_type": "launch_boost",
                "max_depth": int(self.max_depth),
                "min_samples_leaf": int(self.min_samples_leaf),
                "learning_rate": float(self.learning_rate),
                "max_iter": int(self.max_iter),
                "l2_regularization": float(self.l2_regularization),
                "return_head_weight": float(self.return_head_weight),
                "random_state": int(self.random_state),
                "factor_count": int(len(self._factor_cols or [])),
                "train_summary": dict(self._train_summary or {}),
            },
        )
        return {"model_pkl": str(model_path), "meta_json": str(meta_path)}
