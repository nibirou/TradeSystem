"""Template custom timing model plugin.

Usage:
python run_strategy7.py --custom-timing-model-py ./strategy7/plugins/custom_timing_model_template.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from strategy7.models.base import TimingModel


class FixedExposureTimingModel(TimingModel):
    def __init__(self, exposure: float = 0.8) -> None:
        self.exposure = float(exposure)

    def fit(self, train_df: pd.DataFrame) -> "FixedExposureTimingModel":
        return self

    def predict_exposure(self, day_df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        e = max(0.0, min(1.0, self.exposure))
        return e, {"timing_enabled": 1.0, "timing_exposure": e}

    def save(self, folder: Path, run_tag: str) -> Dict[str, str]:
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"custom_timing_fixed_{run_tag}.txt"
        p.write_text(f"fixed exposure={self.exposure}", encoding="utf-8")
        return {"note_file": str(p)}


def build_model(cfg) -> TimingModel:
    return FixedExposureTimingModel(exposure=0.8)

