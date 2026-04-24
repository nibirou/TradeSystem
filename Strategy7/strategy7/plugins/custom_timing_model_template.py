"""Template custom timing model plugin.

Usage:
python run_strategy7.py --custom-timing-model-py ./strategy7/plugins/custom_timing_model_template.py

# load mode:
# python run_strategy7.py \
#   --model-run-mode load \
#   --custom-timing-model-py ./strategy7/plugins/custom_timing_model_template.py \
#   --timing-model-path <your_saved_json_path>
"""

from __future__ import annotations

import json
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
        p = folder / f"custom_timing_fixed_{run_tag}.json"
        payload = {
            "model_type": "custom_timing_fixed",
            "exposure": float(self.exposure),
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"meta_json": str(p)}


def build_model(cfg) -> TimingModel:
    return FixedExposureTimingModel(exposure=0.8)


def load_model(cfg, model_path: str | None) -> TimingModel:
    exposure = 0.8
    if model_path:
        p = Path(model_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"custom timing model file not found: {p}")
        payload = json.loads(p.read_text(encoding="utf-8"))
        exposure = float(payload.get("exposure", exposure))
    return FixedExposureTimingModel(exposure=exposure)
