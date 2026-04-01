"""Timing model plugins."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ...core.utils import dump_json
from ..base import TimingModel


def _safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            return s.fillna(float(s.median()))
    return pd.Series(np.full(len(df), default), index=df.index, dtype=float)


def _robust_z(value: float, hist: List[float]) -> float:
    h = pd.Series(hist, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(h) < 8 or not np.isfinite(value):
        return 0.0
    med = float(h.median())
    mad = float((h - med).abs().median())
    scale = 1.4826 * mad if mad > 1e-12 else float(h.std(ddof=0))
    if scale <= 1e-12 or not np.isfinite(scale):
        return 0.0
    return float(np.clip((value - med) / (scale + 1e-12), -4.0, 4.0))


@dataclass
class NoTimingModel(TimingModel):
    def fit(self, train_df: pd.DataFrame) -> "NoTimingModel":
        return self

    def predict_exposure(self, day_df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        return 1.0, {"timing_enabled": 0.0, "timing_exposure": 1.0}

    def save(self, folder: Path, run_tag: str) -> Dict[str, str]:
        folder.mkdir(parents=True, exist_ok=True)
        meta = folder / f"timing_none_{run_tag}.json"
        dump_json(meta, {"model_type": "none"})
        return {"meta_json": str(meta)}


@dataclass
class VolatilityRegimeTimingModel(TimingModel):
    vol_threshold: float = 0.0
    momentum_threshold: float = 0.0
    history_vol: List[float] = field(default_factory=list)
    history_crowding: List[float] = field(default_factory=list)
    history_momentum: List[float] = field(default_factory=list)

    def fit(self, train_df: pd.DataFrame) -> "VolatilityRegimeTimingModel":
        vol = _safe_series(train_df, "realized_vol_20", default=0.0)
        mom = _safe_series(train_df, "ret_20d", default=0.0)
        if self.vol_threshold <= 0.0 and vol.notna().any():
            self.vol_threshold = float(vol.quantile(0.75))
        if self.momentum_threshold == 0.0 and mom.notna().any():
            self.momentum_threshold = float(mom.quantile(0.35))
        return self

    def predict_exposure(self, day_df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        vol = float(_safe_series(day_df, "realized_vol_20", default=0.0).median())
        crowd = float(_safe_series(day_df, "crowding_proxy_raw", default=0.0).median())
        mom = float(_safe_series(day_df, "ret_20d", default=0.0).median())

        vol_z = _robust_z(vol, self.history_vol)
        crowd_z = _robust_z(crowd, self.history_crowding)
        mom_z = _robust_z(mom, self.history_momentum)

        score = 1.0
        if vol > self.vol_threshold:
            score -= 0.25
        if mom < self.momentum_threshold:
            score -= 0.25
        score -= 0.10 * max(vol_z, 0.0)
        score -= 0.10 * max(crowd_z, 0.0)
        score += 0.08 * max(mom_z, 0.0)
        exposure = float(np.clip(score, 0.15, 1.0))

        self.history_vol.append(vol)
        self.history_crowding.append(crowd)
        self.history_momentum.append(mom)
        self.history_vol = self.history_vol[-300:]
        self.history_crowding = self.history_crowding[-300:]
        self.history_momentum = self.history_momentum[-300:]

        diag = {
            "timing_enabled": 1.0,
            "timing_exposure": exposure,
            "timing_market_vol": vol,
            "timing_crowding": crowd,
            "timing_momentum": mom,
            "timing_vol_z": vol_z,
            "timing_crowding_z": crowd_z,
            "timing_momentum_z": mom_z,
            "timing_vol_threshold": float(self.vol_threshold),
            "timing_momentum_threshold": float(self.momentum_threshold),
        }
        return exposure, diag

    def save(self, folder: Path, run_tag: str) -> Dict[str, str]:
        folder.mkdir(parents=True, exist_ok=True)
        pkl = folder / f"timing_vol_regime_{run_tag}.pkl"
        meta = folder / f"timing_vol_regime_{run_tag}.json"
        with open(pkl, "wb") as f:
            pickle.dump(self, f)
        dump_json(
            meta,
            {
                "model_type": "volatility_regime",
                "vol_threshold": float(self.vol_threshold),
                "momentum_threshold": float(self.momentum_threshold),
            },
        )
        return {"model_pkl": str(pkl), "meta_json": str(meta)}

