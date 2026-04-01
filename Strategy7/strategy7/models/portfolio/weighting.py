"""Portfolio weighting models: equal-weight and dynamic optimization."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ...config import PortfolioOptConfig
from ...core.constants import EPS
from ...core.utils import dump_json, infer_industry_bucket
from ..base import PortfolioModel


def cross_section_zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    m = float(x.mean()) if x.notna().any() else 0.0
    sd = float(x.std(ddof=0)) if x.notna().sum() > 1 else 0.0
    if sd <= EPS:
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    return (x - m) / (sd + EPS)


def robust_history_zscore(value: float, history: List[float]) -> float:
    h = pd.Series(history, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(h) < 5 or not np.isfinite(value):
        return 0.0
    med = float(h.median())
    mad = float((h - med).abs().median())
    scale = 1.4826 * mad if mad > EPS else float(h.std(ddof=0))
    if scale <= EPS or not np.isfinite(scale):
        return 0.0
    return float(np.clip((value - med) / (scale + EPS), -4.0, 4.0))


def ensure_feasible_caps(caps: np.ndarray, target_sum: float = 1.0, hard_cap: float = 1.0) -> np.ndarray:
    u = np.clip(np.asarray(caps, dtype=float), 0.0, hard_cap)
    if u.sum() >= target_sum - 1e-10:
        return u
    slack = np.clip(hard_cap - u, 0.0, None)
    deficit = target_sum - float(u.sum())
    if slack.sum() > EPS:
        u = u + slack * min(1.0, deficit / (float(slack.sum()) + EPS))
    if u.sum() < target_sum - 1e-10:
        n = len(u)
        uniform_cap = min(hard_cap, max(target_sum / max(n, 1), np.max(u) if n > 0 else 1.0))
        u = np.full(n, uniform_cap, dtype=float)
    return u


def project_to_capped_simplex(v: np.ndarray, caps: np.ndarray, target_sum: float = 1.0) -> np.ndarray:
    if len(v) == 0:
        return np.array([], dtype=float)
    u = ensure_feasible_caps(caps, target_sum=target_sum, hard_cap=1.0)
    x = np.asarray(v, dtype=float)
    lo = float(np.min(x - u))
    hi = float(np.max(x))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        w_mid = np.clip(x - mid, 0.0, u)
        if w_mid.sum() > target_sum:
            lo = mid
        else:
            hi = mid
    w = np.clip(x - hi, 0.0, u)
    residual = target_sum - float(w.sum())
    if abs(residual) > 1e-9:
        room = np.where(residual > 0.0, u - w, w)
        total_room = float(room[room > 0].sum())
        if total_room > EPS:
            adjust = room / (total_room + EPS) * residual
            w = np.clip(w + adjust, 0.0, u)
    total = float(w.sum())
    return w / (total + EPS) if total > EPS else np.full(len(w), 1.0 / len(w))


def _to_numeric_series(df: pd.DataFrame, col: str, fill_value: float = 0.0) -> pd.Series:
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            return s.fillna(float(s.median()))
    return pd.Series(np.full(len(df), fill_value, dtype=float), index=df.index)


def _build_style_matrix(selected_df: pd.DataFrame, universe_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    style_cols = [
        "barra_size_proxy",
        "barra_momentum_proxy",
        "barra_volatility_proxy",
        "barra_liquidity_proxy",
        "barra_beta_proxy",
    ]
    if selected_df.empty:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float)
    u = universe_df.copy()
    u["code"] = u["code"].astype(str)
    sel_codes = selected_df["code"].astype(str)
    exposures: List[np.ndarray] = []
    targets: List[float] = []
    for col in style_cols:
        uni_series = _to_numeric_series(u, col, fill_value=0.0)
        uni_z = cross_section_zscore(uni_series)
        code_to_exp = dict(zip(u["code"], uni_z))
        exp = sel_codes.map(code_to_exp).fillna(0.0).astype(float).to_numpy()
        exposures.append(exp)
        targets.append(float(uni_z.mean()))
    style_matrix = np.column_stack(exposures) if exposures else np.zeros((len(selected_df), 0), dtype=float)
    return style_matrix, np.asarray(targets, dtype=float)


def _build_industry_matrix(selected_df: pd.DataFrame, universe_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if selected_df.empty:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float)
    if "industry_bucket" in selected_df.columns:
        sel_ind = selected_df["industry_bucket"].astype(str).fillna("unknown")
    else:
        sel_ind = selected_df["code"].astype(str).map(infer_industry_bucket)
    if "industry_bucket" in universe_df.columns:
        uni_ind = universe_df["industry_bucket"].astype(str).fillna("unknown")
    else:
        uni_ind = universe_df["code"].astype(str).map(infer_industry_bucket)
    industries = sorted(set(sel_ind.tolist()) | set(uni_ind.tolist()))
    mat = np.column_stack([(sel_ind == ind).astype(float).to_numpy() for ind in industries]).astype(float)
    bench = uni_ind.value_counts(normalize=True)
    target = np.asarray([float(bench.get(ind, 0.0)) for ind in industries], dtype=float)
    return mat, target


def _compute_market_state(universe_df: pd.DataFrame, state_tracker: Dict[str, List[float]]) -> Dict[str, float]:
    market_vol = float(_to_numeric_series(universe_df, "realized_vol_20", fill_value=0.0).median())
    crowd = (
        0.45 * _to_numeric_series(universe_df, "vol_ratio_20", fill_value=0.0).abs()
        + 0.35 * _to_numeric_series(universe_df, "turn_ratio_5", fill_value=0.0).abs()
        + 0.20 * _to_numeric_series(universe_df, "ret_vol_corr_20", fill_value=0.0).abs()
    )
    crowd_level = float(crowd.median())
    style_disp = float(
        np.nanmean(
            [
                cross_section_zscore(_to_numeric_series(universe_df, "barra_size_proxy", fill_value=0.0)).std(ddof=0),
                cross_section_zscore(_to_numeric_series(universe_df, "barra_momentum_proxy", fill_value=0.0)).std(ddof=0),
                cross_section_zscore(_to_numeric_series(universe_df, "barra_volatility_proxy", fill_value=0.0)).std(ddof=0),
            ]
        )
    )
    return {
        "market_vol": market_vol,
        "crowding": crowd_level,
        "style_dispersion": style_disp,
        "vol_z": robust_history_zscore(market_vol, state_tracker.get("market_vol", [])),
        "crowding_z": robust_history_zscore(crowd_level, state_tracker.get("crowding", [])),
        "style_z": robust_history_zscore(style_disp, state_tracker.get("style_disp", [])),
    }


@dataclass
class EqualWeightPortfolioModel(PortfolioModel):
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
        w = pd.Series(np.full(len(day_pick), 1.0 / len(day_pick), dtype=float), index=day_pick["code"].astype(str))
        return w, {"portfolio_mode": 0.0}

    def save(self, folder: Path, run_tag: str) -> Dict[str, str]:
        folder.mkdir(parents=True, exist_ok=True)
        meta = folder / f"portfolio_equal_{run_tag}.json"
        dump_json(meta, {"model_type": "equal_weight"})
        return {"meta_json": str(meta)}


@dataclass
class DynamicOptimizationPortfolioModel(PortfolioModel):
    cfg: PortfolioOptConfig
    state_tracker: Dict[str, List[float]] = field(default_factory=lambda: {"market_vol": [], "crowding": [], "style_disp": []})

    def compute_weights(
        self,
        day_pick: pd.DataFrame,
        day_universe: pd.DataFrame,
        prev_weights: Dict[str, float],
        fee_bps: float,
        slippage_bps: float,
    ) -> Tuple[pd.Series, Dict[str, float]]:
        n = len(day_pick)
        if n == 0:
            return pd.Series(dtype=float), {}
        if n == 1:
            code = str(day_pick.iloc[0]["code"])
            return pd.Series({code: 1.0}), {"opt_iterations": 0.0, "opt_converged": 1.0}

        pick = day_pick.copy().reset_index(drop=True)
        uni = day_universe.copy()
        pick_codes = pick["code"].astype(str).tolist()
        pick_code_set = set(pick_codes)

        market_state = _compute_market_state(uni, self.state_tracker)
        vol_z = market_state["vol_z"]
        crowd_z = market_state["crowding_z"]
        style_z = market_state["style_z"]

        expected_scale = float(np.clip(1.0 + 0.18 * style_z - 0.25 * max(vol_z, 0.0) - 0.18 * max(crowd_z, 0.0), 0.4, 1.8))
        risk_scale = float(np.clip(1.0 + 0.40 * max(vol_z, 0.0) + 0.25 * max(crowd_z, 0.0), 0.7, 2.8))
        style_scale = float(np.clip(1.0 + 0.20 * max(crowd_z, 0.0) + 0.15 * abs(style_z), 0.8, 2.4))
        industry_scale = float(np.clip(1.0 + 0.20 * abs(style_z), 0.8, 2.2))
        tc_scale = float(np.clip(1.0 + 0.30 * max(crowd_z, 0.0), 0.8, 2.5))

        alpha_prob = cross_section_zscore(_to_numeric_series(pick, "pred_score", fill_value=0.5))
        alpha_mom = cross_section_zscore(_to_numeric_series(pick, "ret_20d", fill_value=0.0))
        alpha_micro = cross_section_zscore(_to_numeric_series(pick, "morning_momentum_30m", fill_value=0.0))
        mu = (0.65 * alpha_prob + 0.25 * alpha_mom + 0.10 * alpha_micro).to_numpy(dtype=float)

        vol_series = _to_numeric_series(pick, "realized_vol_20", fill_value=0.0).clip(lower=0.0)
        vol_zs = cross_section_zscore(vol_series).to_numpy(dtype=float)
        risk_diag = np.clip(1.0 + 0.5 * vol_zs, 0.2, 3.0)

        style_mat, style_target = _build_style_matrix(pick, uni)
        ind_mat, ind_target = _build_industry_matrix(pick, uni)
        crowding = cross_section_zscore(_to_numeric_series(pick, "crowding_proxy_raw", fill_value=0.0)).to_numpy(dtype=float)

        liq_amt = _to_numeric_series(pick, "amount_ma20", fill_value=0.0).clip(lower=0.0)
        if liq_amt.sum() <= EPS:
            liq_amt = _to_numeric_series(pick, "amount", fill_value=0.0).clip(lower=0.0)
        liq_share = (liq_amt / (liq_amt.sum() + EPS)).to_numpy(dtype=float) if liq_amt.sum() > EPS else np.full(n, 1.0 / n, dtype=float)

        effective_max_weight = float(np.clip(max(self.cfg.max_weight, 1.0 / n + 1e-6), 0.01, 1.0))
        caps = ensure_feasible_caps(np.clip(liq_share * max(self.cfg.liquidity_scale, 0.1), 0.01, effective_max_weight), target_sum=1.0, hard_cap=effective_max_weight)

        prev_vec = np.asarray([float(prev_weights.get(c, 0.0)) for c in pick_codes], dtype=float)
        prev_dropped = float(sum(v for c, v in prev_weights.items() if c not in pick_code_set and v > 0))
        prev_total = float(prev_vec.sum())
        if prev_total <= EPS:
            w = np.full(n, 1.0 / n, dtype=float)
        else:
            carry = max(0.0, 1.0 - prev_dropped)
            w = prev_vec / (prev_total + EPS) * carry + ((1.0 - carry) / n if carry < 1.0 else 0.0)
        w = project_to_capped_simplex(w, caps)

        ret_coef = max(self.cfg.expected_return_weight * expected_scale, 0.0)
        risk_coef = max(self.cfg.risk_aversion * risk_scale, 0.0)
        style_coef = max(self.cfg.style_penalty * style_scale, 0.0)
        industry_coef = max(self.cfg.industry_penalty * industry_scale, 0.0)
        crowd_coef = max(self.cfg.crowding_penalty * (1.0 + 0.25 * max(crowd_z, 0.0)), 0.0)
        tc_bps = (fee_bps + slippage_bps) / 10000.0
        tc_coef = max(self.cfg.transaction_cost_penalty * tc_scale * tc_bps * 100.0, 0.0)
        step = max(self.cfg.step_size, 1e-4)

        converged = 0.0
        iterations = 0
        for i in range(max(self.cfg.max_iter, 1)):
            iterations = i + 1
            grad = ret_coef * mu - 2.0 * risk_coef * risk_diag * w
            if style_mat.size > 0:
                style_gap = style_mat.T @ w - style_target
                grad = grad - 2.0 * style_coef * (style_mat @ style_gap)
            if ind_mat.size > 0:
                ind_gap = ind_mat.T @ w - ind_target
                grad = grad - 2.0 * industry_coef * (ind_mat @ ind_gap)
            crowd_exp = float(np.dot(crowding, w))
            grad = grad - 2.0 * crowd_coef * crowding * crowd_exp
            if prev_vec.size == w.size:
                grad = grad - tc_coef * np.sign(w - prev_vec)

            w_next = project_to_capped_simplex(w + step * grad, caps)

            turnover = prev_dropped + float(np.abs(w_next - prev_vec).sum())
            if turnover > self.cfg.max_turnover + 1e-10:
                allowed = max(self.cfg.max_turnover - prev_dropped, 0.0)
                cur_turnover = float(np.abs(w_next - prev_vec).sum())
                if cur_turnover > EPS and allowed < cur_turnover:
                    blend = allowed / (cur_turnover + EPS)
                    w_next = prev_vec + blend * (w_next - prev_vec)
                    w_next = project_to_capped_simplex(w_next, caps)

            if float(np.abs(w_next - w).sum()) < self.cfg.tolerance:
                converged = 1.0
                w = w_next
                break
            w = w_next

        style_dev = float(np.linalg.norm((style_mat.T @ w - style_target), ord=2)) if style_mat.size > 0 else 0.0
        ind_dev = float(np.linalg.norm((ind_mat.T @ w - ind_target), ord=2)) if ind_mat.size > 0 else 0.0
        crowd_exp = float(np.dot(crowding, w))
        liq_util = float(np.max(w / (caps + EPS)))
        turnover = prev_dropped + float(np.abs(w - prev_vec).sum())
        exp_ret_score = float(np.dot(mu, w))

        self.state_tracker.setdefault("market_vol", []).append(market_state["market_vol"])
        self.state_tracker.setdefault("crowding", []).append(market_state["crowding"])
        self.state_tracker.setdefault("style_disp", []).append(market_state["style_dispersion"])
        for k in ("market_vol", "crowding", "style_disp"):
            self.state_tracker[k] = self.state_tracker[k][-240:]

        diag = {
            "opt_iterations": float(iterations),
            "opt_converged": float(converged),
            "state_market_vol": market_state["market_vol"],
            "state_crowding": market_state["crowding"],
            "state_style_dispersion": market_state["style_dispersion"],
            "state_vol_z": vol_z,
            "state_crowding_z": crowd_z,
            "state_style_z": style_z,
            "dynamic_expected_scale": expected_scale,
            "dynamic_risk_scale": risk_scale,
            "dynamic_style_scale": style_scale,
            "dynamic_industry_scale": industry_scale,
            "dynamic_tc_scale": tc_scale,
            "opt_turnover": turnover,
            "opt_expected_ret_score": exp_ret_score,
            "opt_style_exposure_dev": style_dev,
            "opt_industry_dev": ind_dev,
            "opt_crowding_exposure": crowd_exp,
            "opt_liquidity_utilization": liq_util,
            "opt_effective_max_weight": effective_max_weight,
        }
        return pd.Series(w, index=pick_codes, dtype=float), diag

    def save(self, folder: Path, run_tag: str) -> Dict[str, str]:
        folder.mkdir(parents=True, exist_ok=True)
        pkl = folder / f"portfolio_dynamic_{run_tag}.pkl"
        meta = folder / f"portfolio_dynamic_{run_tag}.json"
        with open(pkl, "wb") as f:
            pickle.dump(self, f)
        dump_json(meta, {"model_type": "dynamic_opt", "config": self.cfg.__dict__})
        return {"model_pkl": str(pkl), "meta_json": str(meta)}

