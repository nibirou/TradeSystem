"""Execution engine plugins for backtest fill simulation."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ...config import ExecutionModelConfig
from ...core.constants import EPS
from ...core.utils import dump_json
from ..base import ExecutionModel


def _safe_numeric(df: pd.DataFrame, col: str, fill_value: float = 0.0) -> pd.Series:
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            return s.fillna(float(s.median()))
    return pd.Series(np.full(len(df), fill_value), index=df.index, dtype=float)


@dataclass
class IdealFillExecutionModel(ExecutionModel):
    def apply_execution(
        self,
        day_pick: pd.DataFrame,
        weight_col: str,
        fee_bps: float,
        slippage_bps: float,
    ) -> tuple[pd.DataFrame, Dict[str, float]]:
        out = day_pick.copy()
        out["fill_ratio"] = 1.0
        out["executed_weight"] = out[weight_col].astype(float)
        out["realized_trade_ret"] = out["net_trade_ret"].astype(float)
        diag = {
            "execution_model": 0.0,
            "avg_fill_ratio": 1.0,
            "cash_drag_weight": float(max(0.0, 1.0 - out["executed_weight"].sum())),
            "extra_cost_bps": 0.0,
        }
        return out, diag

    def save(self, folder: Path, run_tag: str) -> Dict[str, str]:
        folder.mkdir(parents=True, exist_ok=True)
        meta = folder / f"execution_ideal_{run_tag}.json"
        dump_json(meta, {"model_type": "ideal_fill"})
        return {"meta_json": str(meta)}


@dataclass
class RealisticFillExecutionModel(ExecutionModel):
    cfg: ExecutionModelConfig

    def apply_execution(
        self,
        day_pick: pd.DataFrame,
        weight_col: str,
        fee_bps: float,
        slippage_bps: float,
    ) -> tuple[pd.DataFrame, Dict[str, float]]:
        out = day_pick.copy()
        if out.empty:
            return out, {"execution_model": 1.0, "avg_fill_ratio": float("nan"), "cash_drag_weight": 1.0, "extra_cost_bps": 0.0}

        w = pd.to_numeric(out[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        if w.sum() <= EPS:
            w = pd.Series(np.full(len(out), 1.0 / len(out), dtype=float), index=out.index)
        else:
            w = w / (w.sum() + EPS)
        out[weight_col] = w

        liq = _safe_numeric(out, "amount_ma20", fill_value=0.0).clip(lower=0.0)
        if liq.sum() <= EPS:
            liq = _safe_numeric(out, "amount", fill_value=0.0).clip(lower=0.0)
        liq_share = liq / (liq.sum() + EPS) if liq.sum() > EPS else pd.Series(np.full(len(out), 1.0 / len(out)), index=out.index)

        # Each stock has max participation budget proportional to its liquidity share.
        cap = self.cfg.max_participation_rate * liq_share
        raw_fill = cap / (w + EPS)
        fill_ratio = raw_fill.clip(lower=0.0, upper=1.0)

        vol = _safe_numeric(out, "realized_vol_20", fill_value=0.0)
        crowd = _safe_numeric(out, "crowding_proxy_raw", fill_value=0.0).abs()
        vol_z = (vol - float(vol.mean())) / (float(vol.std(ddof=0)) + EPS)
        crowd_z = (crowd - float(crowd.mean())) / (float(crowd.std(ddof=0)) + EPS)
        state_penalty = (0.10 * vol_z.clip(lower=0.0) + 0.10 * crowd_z.clip(lower=0.0)).clip(lower=0.0, upper=0.5)

        fill_ratio = (fill_ratio * self.cfg.base_fill_rate * (1.0 - state_penalty)).clip(lower=0.0, upper=1.0)
        out["fill_ratio"] = fill_ratio
        out["executed_weight"] = w * fill_ratio

        extra_cost_bps_each = (state_penalty * (slippage_bps + 0.5 * fee_bps)).astype(float)
        extra_cost_ret_each = extra_cost_bps_each / 10000.0
        out["extra_cost_bps"] = extra_cost_bps_each
        out["realized_trade_ret"] = pd.to_numeric(out["net_trade_ret"], errors="coerce").fillna(0.0) - extra_cost_ret_each

        diag = {
            "execution_model": 1.0,
            "avg_fill_ratio": float(out["fill_ratio"].mean()),
            "cash_drag_weight": float(max(0.0, 1.0 - out["executed_weight"].sum())),
            "extra_cost_bps": float(out["extra_cost_bps"].mean()),
        }
        return out, diag

    def save(self, folder: Path, run_tag: str) -> Dict[str, str]:
        folder.mkdir(parents=True, exist_ok=True)
        pkl = folder / f"execution_realistic_{run_tag}.pkl"
        meta = folder / f"execution_realistic_{run_tag}.json"
        with open(pkl, "wb") as f:
            pickle.dump(self, f)
        dump_json(meta, {"model_type": "realistic_fill", "config": self.cfg.__dict__})
        return {"model_pkl": str(pkl), "meta_json": str(meta)}

