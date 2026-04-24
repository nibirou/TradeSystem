"""Template custom execution model plugin.

Usage:
python run_strategy7.py --custom-execution-model-py ./strategy7/plugins/custom_execution_model_template.py

# load mode:
# python run_strategy7.py \
#   --model-run-mode load \
#   --custom-execution-model-py ./strategy7/plugins/custom_execution_model_template.py \
#   --execution-model-path <your_saved_json_path>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from strategy7.models.base import ExecutionModel


class HalfFillExecutionModel(ExecutionModel):
    def __init__(self, fill_ratio: float = 0.5) -> None:
        self.fill_ratio = float(fill_ratio)

    def apply_execution(
        self,
        day_pick: pd.DataFrame,
        weight_col: str,
        fee_bps: float,
        slippage_bps: float,
    ) -> tuple[pd.DataFrame, Dict[str, float]]:
        out = day_pick.copy()
        fill_ratio = max(0.0, min(1.0, self.fill_ratio))
        out["fill_ratio"] = fill_ratio
        out["executed_weight"] = pd.to_numeric(out[weight_col], errors="coerce").fillna(0.0) * fill_ratio
        out["realized_trade_ret"] = pd.to_numeric(out["net_trade_ret"], errors="coerce").fillna(0.0)
        return out, {"execution_model": 9.0, "avg_fill_ratio": float(fill_ratio), "cash_drag_weight": float(max(0.0, 1.0 - out["executed_weight"].sum())), "extra_cost_bps": 0.0}

    def save(self, folder: Path, run_tag: str) -> Dict[str, str]:
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"custom_execution_half_fill_{run_tag}.json"
        payload = {
            "model_type": "custom_execution_half_fill",
            "fill_ratio": float(self.fill_ratio),
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"meta_json": str(p)}


def build_model(cfg) -> ExecutionModel:
    return HalfFillExecutionModel()


def load_model(cfg, model_path: str | None) -> ExecutionModel:
    fill_ratio = 0.5
    if model_path:
        p = Path(model_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"custom execution model file not found: {p}")
        payload = json.loads(p.read_text(encoding="utf-8"))
        fill_ratio = float(payload.get("fill_ratio", fill_ratio))
    return HalfFillExecutionModel(fill_ratio=fill_ratio)
