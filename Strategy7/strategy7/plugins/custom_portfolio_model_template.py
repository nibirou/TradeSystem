"""Template custom portfolio model plugin.

Usage:
python run_strategy7.py --custom-portfolio-model-py ./strategy7/plugins/custom_portfolio_model_template.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from strategy7.models.base import PortfolioModel


class Top1PortfolioModel(PortfolioModel):
    def compute_weights(
        self,
        day_pick: pd.DataFrame,
        day_universe: pd.DataFrame,
        prev_weights: Dict[str, float],
        fee_bps: float,
        slippage_bps: float,
    ) -> Tuple[pd.Series, Dict[str, float]]:
        if day_pick.empty:
            return pd.Series(dtype=float), {}
        first_code = str(day_pick.iloc[0]["code"])
        w = pd.Series(np.zeros(len(day_pick), dtype=float), index=day_pick["code"].astype(str))
        w.loc[first_code] = 1.0
        return w, {"portfolio_mode": 9.0}

    def save(self, folder: Path, run_tag: str) -> Dict[str, str]:
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"custom_portfolio_top1_{run_tag}.txt"
        p.write_text("top1 portfolio model", encoding="utf-8")
        return {"note_file": str(p)}


def build_model(cfg) -> PortfolioModel:
    return Top1PortfolioModel()

