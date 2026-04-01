"""Template custom execution model plugin.

Usage:
python run_strategy7.py --custom-execution-model-py ./strategy7/plugins/custom_execution_model_template.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from strategy7.models.base import ExecutionModel


class HalfFillExecutionModel(ExecutionModel):
    def apply_execution(
        self,
        day_pick: pd.DataFrame,
        weight_col: str,
        fee_bps: float,
        slippage_bps: float,
    ) -> tuple[pd.DataFrame, Dict[str, float]]:
        out = day_pick.copy()
        out["fill_ratio"] = 0.5
        out["executed_weight"] = pd.to_numeric(out[weight_col], errors="coerce").fillna(0.0) * 0.5
        out["realized_trade_ret"] = pd.to_numeric(out["net_trade_ret"], errors="coerce").fillna(0.0)
        return out, {"execution_model": 9.0, "avg_fill_ratio": 0.5, "cash_drag_weight": float(max(0.0, 1.0 - out["executed_weight"].sum())), "extra_cost_bps": 0.0}

    def save(self, folder: Path, run_tag: str) -> Dict[str, str]:
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"custom_execution_half_fill_{run_tag}.txt"
        p.write_text("half fill execution model", encoding="utf-8")
        return {"note_file": str(p)}


def build_model(cfg) -> ExecutionModel:
    return HalfFillExecutionModel()

