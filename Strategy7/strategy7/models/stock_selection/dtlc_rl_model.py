"""DTLC_RL stock-selection model.

This module implements the Southwest Securities report
"Decoupled Temporal Contrastive Learning with Reinforcement Learning":

1) beta space: TCN encoder for market/systematic-risk features.
2) alpha space: multi-scale Transformer encoder for price-volume features.
3) theta space: gated residual MLP encoder for fundamental features.
4) supervised IC/MSE training with InfoNCE and orthogonal penalties.
5) PPO controller that dynamically generates beta/alpha/theta fusion weights.

The report uses a fixed daily feature recipe. Strategy7 passes a configurable
factor panel, so the implementation maps factor names into three spaces by
heuristics and falls back safely when a space is unavailable. This keeps the
model usable across daily, intraday, weekly, monthly, catalog, engineered, and
custom factor sets.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from ...core.constants import EPS
from ...core.utils import dump_json
from ..base import StockSelectionModel


def _parse_int_list(raw: str | Sequence[int], default: Sequence[int]) -> List[int]:
    if isinstance(raw, str):
        parts = [x.strip() for x in raw.replace("，", ",").replace(";", ",").split(",") if x.strip()]
        vals = [int(float(x)) for x in parts]
    else:
        vals = [int(x) for x in raw]
    out: List[int] = []
    for v in vals:
        if v > 0 and v not in out:
            out.append(v)
    return out or [int(x) for x in default]


@dataclass
class _TrainSample:
    code: str
    time_key: pd.Timestamp
    beta_seq: np.ndarray
    alpha_seq: np.ndarray
    theta_vec: np.ndarray
    market_state: np.ndarray
    target: float


@dataclass
class _PPOTransition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    log_prob: float
    value: float
    done: bool


@dataclass
class DTLCRLStockModel(StockSelectionModel):
    """Decoupled temporal contrastive learning model with PPO fusion."""

    seq_len: int = 60
    hidden_size: int = 64
    latent_size: int = 32
    num_heads: int = 4
    encoder_layers: int = 2
    grn_layers: int = 2
    ffn_mult: int = 4
    tcn_kernel_size: int = 3
    alpha_scales: str | Sequence[int] = "20,40,60"
    dropout: float = 0.10

    pretrain_epochs: int = 80
    ppo_epochs: int = 30
    lr: float = 1e-4
    ppo_lr: float = 3e-4
    weight_decay: float = 1e-4
    early_stop: int = 20
    per_epoch_batch: int = 100
    batch_size: int = -1
    label_transform: str = "cszscore"
    input_clip: float = 3.0
    mse_weight: float = 0.05
    ic_loss_weight: float = 1.0
    contrastive_weight: float = 0.05
    orthogonal_weight: float = 0.05
    contrastive_tau: float = 0.10
    positive_rank_pct: float = 0.20

    ppo_clip: float = 0.20
    gae_lambda: float = 0.95
    gamma: float = 0.99
    ppo_update_epochs: int = 3
    ppo_batch_size: int = 32
    entropy_weight: float = 0.01
    value_weight: float = 0.50
    stable_weight: float = 0.05
    diversity_weight: float = 0.02
    min_cross_section: int = 8

    random_state: int = 42
    device: str = "auto"

    _model: Any = field(default=None, init=False, repr=False)
    _factor_cols: List[str] = field(default_factory=list, init=False, repr=False)
    _space_cols: Dict[str, List[str]] = field(default_factory=dict, init=False, repr=False)
    _space_indices: Dict[str, List[int]] = field(default_factory=dict, init=False, repr=False)
    _fill_values: pd.Series | None = field(default=None, init=False, repr=False)
    _time_col: str | None = field(default=None, init=False, repr=False)
    _history_by_code: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _market_state_mean: pd.Series | None = field(default=None, init=False, repr=False)
    _market_state_std: pd.Series | None = field(default=None, init=False, repr=False)
    _market_state_lookup: Dict[pd.Timestamp, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _state_cols: List[str] = field(
        default_factory=lambda: ["market_return", "market_volatility", "market_liquidity", "market_dispersion"],
        init=False,
        repr=False,
    )
    _device_used: str = field(default="cpu", init=False, repr=False)
    _target_col: str = field(default="target_return", init=False, repr=False)
    _train_summary: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _score_sign: float = field(default=1.0, init=False, repr=False)

    @staticmethod
    def _require_torch() -> tuple[Any, Any, Any]:
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except Exception as exc:
            raise RuntimeError("DTLC_RL requires PyTorch. Please install torch first.") from exc
        return torch, nn, F

    @staticmethod
    def _resolve_time_col(df: pd.DataFrame) -> str:
        for c in ["signal_ts", "datetime", "date"]:
            if c in df.columns:
                return c
        raise ValueError("DTLC_RL requires one of ['signal_ts', 'datetime', 'date'] columns.")

    @staticmethod
    def _time_anchor(ts: pd.Series) -> pd.Series:
        dt = pd.to_datetime(ts, errors="coerce")
        normalized = dt.dt.normalize()
        has_intraday = bool(((dt - normalized).dt.total_seconds().fillna(0.0) != 0.0).any())
        return dt if has_intraday else normalized

    @staticmethod
    def _zscore(x: pd.Series) -> pd.Series:
        v = pd.to_numeric(x, errors="coerce")
        std = float(v.std(ddof=0)) if v.notna().sum() > 1 else 0.0
        if std <= EPS:
            return pd.Series(np.zeros(len(v), dtype=float), index=v.index)
        return (v - float(v.mean())) / (std + EPS)

    def _choose_device(self, torch: Any) -> str:
        req = str(self.device).lower().strip()
        if req == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if req == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return req

    def _set_seed(self, torch: Any) -> None:
        np.random.seed(int(self.random_state))
        torch.manual_seed(int(self.random_state))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.random_state))

    def _alpha_scales(self) -> List[int]:
        return _parse_int_list(self.alpha_scales, default=[20, 40, int(self.seq_len)])

    @staticmethod
    def _matches_any(name: str, tokens: Sequence[str]) -> bool:
        s = str(name).lower()
        return any(tok in s for tok in tokens)

    def _resolve_space_cols(self, factor_cols: List[str]) -> Dict[str, List[str]]:
        beta_tokens = [
            "beta",
            "mkt",
            "market",
            "context",
            "vol",
            "rv",
            "realized",
            "liq",
            "turn",
            "amount",
            "size",
            "sent",
            "crowding",
        ]
        theta_tokens = [
            "fund",
            "pe",
            "pb",
            "roe",
            "roic",
            "eps",
            "dividend",
            "profit",
            "margin",
            "cashflow",
            "leverage",
            "valuation",
            "growth",
            "quality",
            "btop",
        ]
        theta_cols = [c for c in factor_cols if self._matches_any(c, theta_tokens)]
        beta_cols = [c for c in factor_cols if self._matches_any(c, beta_tokens) and c not in theta_cols]
        alpha_cols = [c for c in factor_cols if c not in set(beta_cols) and c not in set(theta_cols)]

        if not beta_cols:
            beta_cols = list(factor_cols)
        if not alpha_cols:
            alpha_cols = list(factor_cols)
        if not theta_cols:
            theta_cols = list(factor_cols)
        return {"beta": beta_cols, "alpha": alpha_cols, "theta": theta_cols}

    def _set_space_cols(self, space_cols: Dict[str, List[str]]) -> None:
        self._space_cols = {k: [str(x) for x in v if str(x).strip()] for k, v in space_cols.items()}
        col_pos = {c: i for i, c in enumerate(self._factor_cols)}
        self._space_indices = {
            k: [int(col_pos[c]) for c in cols if c in col_pos]
            for k, cols in self._space_cols.items()
        }

    def _build_target(self, df: pd.DataFrame, target_col: str, anchor: pd.Series) -> pd.Series:
        base_col = target_col
        y_raw = pd.to_numeric(df[target_col], errors="coerce") if target_col in df.columns else pd.Series(np.nan, index=df.index)
        uniq = set(pd.Series(y_raw.dropna().unique()).tolist())
        if uniq.issubset({0, 1}) and "target_return" in df.columns:
            base_col = "target_return"
            y_raw = pd.to_numeric(df[base_col], errors="coerce")

        self._target_col = base_col
        transform = str(self.label_transform).lower().strip()
        if transform == "raw":
            return y_raw
        if transform == "csrank":
            return y_raw.groupby(anchor, sort=False, observed=True).transform(lambda s: s.rank(pct=True, method="average"))
        if transform == "csranknorm":
            ranked = y_raw.groupby(anchor, sort=False, observed=True).transform(lambda s: s.rank(pct=True, method="average"))
            ranked = ranked.replace([np.inf, -np.inf], np.nan)
            return ranked.groupby(anchor, sort=False, observed=True).transform(self._zscore)
        return y_raw.groupby(anchor, sort=False, observed=True).transform(self._zscore)

    def _transform_feature_frame(self, x: pd.DataFrame) -> pd.DataFrame:
        out = x.replace([np.inf, -np.inf], np.nan)
        if self._fill_values is not None:
            out = out.fillna(self._fill_values.reindex(out.columns))
        out = out.fillna(0.0)
        clip = float(self.input_clip)
        if clip > 0.0:
            out = out.clip(lower=-clip, upper=clip)
        return out

    @staticmethod
    def _pick_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _build_market_state_frame(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=self._state_cols, dtype=float)
        out = df.copy()
        out["_state_time"] = self._time_anchor(out[time_col])
        out = out.dropna(subset=["_state_time"])
        if out.empty:
            return pd.DataFrame(columns=self._state_cols, dtype=float)
        idx = pd.DatetimeIndex(sorted(pd.to_datetime(out["_state_time"]).drop_duplicates().tolist()))
        state = pd.DataFrame(index=idx, columns=self._state_cols, dtype=float)

        ret_col = self._pick_first_existing_column(out, ["ret_1d", "ret_1", "mom_5", "ret_3", "ret_6"])
        if ret_col is not None:
            ret = pd.to_numeric(out[ret_col], errors="coerce")
            state["market_return"] = ret.groupby(out["_state_time"]).mean().reindex(idx)
            state["market_dispersion"] = ret.groupby(out["_state_time"]).std(ddof=0).reindex(idx)
        else:
            state["market_return"] = 0.0
            state["market_dispersion"] = 0.0

        vol_col = self._pick_first_existing_column(out, ["realized_vol_20", "rv_12", "intraday_range", "range_norm"])
        if vol_col is not None:
            state["market_volatility"] = pd.to_numeric(out[vol_col], errors="coerce").abs().groupby(out["_state_time"]).median().reindex(idx)
        else:
            state["market_volatility"] = 0.0

        liq_col = self._pick_first_existing_column(out, ["turn", "turn_ratio_5", "amount_ratio_20", "amount_ratio_12", "amount", "volume"])
        if liq_col is not None:
            state["market_liquidity"] = pd.to_numeric(out[liq_col], errors="coerce").groupby(out["_state_time"]).median().reindex(idx)
        else:
            state["market_liquidity"] = 0.0

        return state.replace([np.inf, -np.inf], np.nan).sort_index().ffill().bfill().fillna(0.0)

    def _fit_market_state(self, df: pd.DataFrame) -> Dict[pd.Timestamp, np.ndarray]:
        state = self._build_market_state_frame(df, time_col=self._time_col)
        if state.empty:
            self._market_state_mean = pd.Series([0.0] * len(self._state_cols), index=self._state_cols, dtype=float)
            self._market_state_std = pd.Series([1.0] * len(self._state_cols), index=self._state_cols, dtype=float)
            self._market_state_lookup = {}
            return {}
        mean = state.mean(axis=0)
        std = state.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        norm = ((state - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self._market_state_mean = mean
        self._market_state_std = std
        self._market_state_lookup = {
            pd.Timestamp(k): norm.loc[k, self._state_cols].to_numpy(dtype=np.float32) for k in norm.index
        }
        return dict(self._market_state_lookup)

    def _normalize_market_state_frame(self, state: pd.DataFrame) -> pd.DataFrame:
        if state.empty:
            return state
        if self._market_state_mean is None or self._market_state_std is None:
            return state.fillna(0.0)
        mean = self._market_state_mean.reindex(self._state_cols).fillna(0.0)
        std = self._market_state_std.reindex(self._state_cols).replace(0.0, 1.0).fillna(1.0)
        return ((state.reindex(columns=self._state_cols) - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _state_for_key(self, key: pd.Timestamp, lookup: Dict[pd.Timestamp, np.ndarray]) -> np.ndarray:
        arr = lookup.get(pd.Timestamp(key))
        if arr is not None:
            return arr.astype(np.float32)
        if key == key.normalize():
            arr = lookup.get(pd.Timestamp(key).normalize())
            if arr is not None:
                return arr.astype(np.float32)
        return np.zeros(len(self._state_cols), dtype=np.float32)

    def _build_train_samples(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        target: pd.Series,
        state_lookup: Dict[pd.Timestamp, np.ndarray],
    ) -> Dict[pd.Timestamp, List[_TrainSample]]:
        keep_cols = list(dict.fromkeys(["code", self._time_col, *factor_cols]))
        out = df[keep_cols].copy()
        out["_row"] = np.arange(len(out), dtype=int)
        out["_model_time"] = pd.to_datetime(out[self._time_col], errors="coerce")
        out["_target"] = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
        out = out.dropna(subset=["code", "_model_time"]).sort_values(["code", "_model_time"])

        x_all = df[factor_cols].to_numpy(dtype=np.float32)
        beta_idx = self._space_indices["beta"]
        alpha_idx = self._space_indices["alpha"]
        theta_idx = self._space_indices["theta"]
        seq_len = int(self.seq_len)

        grouped: Dict[pd.Timestamp, List[_TrainSample]] = defaultdict(list)
        self._history_by_code = {}
        for code, g in out.groupby("code", sort=False, observed=True):
            rows = g["_row"].to_numpy(dtype=int)
            x = x_all[rows]
            y = g["_target"].to_numpy(dtype=np.float32)
            t = pd.to_datetime(g["_model_time"], errors="coerce")
            n = len(g)
            if n == 0:
                continue
            self._history_by_code[str(code)] = x[max(0, n - seq_len + 1) :].copy()
            if n < seq_len:
                continue
            for i in range(seq_len - 1, n):
                if np.isnan(y[i]):
                    continue
                full_seq = x[i - seq_len + 1 : i + 1]
                if full_seq.shape[0] != seq_len or np.isnan(full_seq).any():
                    continue
                key = pd.Timestamp(t.iloc[i])
                if key == key.normalize():
                    key = key.normalize()
                grouped[key].append(
                    _TrainSample(
                        code=str(code),
                        time_key=key,
                        beta_seq=full_seq[:, beta_idx].astype(np.float32),
                        alpha_seq=full_seq[:, alpha_idx].astype(np.float32),
                        theta_vec=x[i, theta_idx].astype(np.float32),
                        market_state=self._state_for_key(key, state_lookup),
                        target=float(y[i]),
                    )
                )
        return grouped

    @staticmethod
    def _stack_batch(samples: Iterable[_TrainSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ss = list(samples)
        beta = np.stack([s.beta_seq for s in ss], axis=0).astype(np.float32)
        alpha = np.stack([s.alpha_seq for s in ss], axis=0).astype(np.float32)
        theta = np.stack([s.theta_vec for s in ss], axis=0).astype(np.float32)
        state = np.stack([s.market_state for s in ss], axis=0).astype(np.float32)
        y = np.asarray([s.target for s in ss], dtype=np.float32)
        return beta, alpha, theta, state, y

    def _build_network(
        self,
        *,
        beta_dim: int,
        alpha_dim: int,
        theta_dim: int,
        state_dim: int,
        torch: Any,
        nn: Any,
        F: Any,
    ) -> Any:
        hidden_size = int(self.hidden_size)
        latent_size = int(self.latent_size)
        num_heads = int(self.num_heads)
        encoder_layers = int(self.encoder_layers)
        grn_layers = int(self.grn_layers)
        ffn_mult = int(self.ffn_mult)
        dropout = float(self.dropout)
        kernel_size = int(self.tcn_kernel_size)
        seq_len = int(self.seq_len)
        scales = []
        for s in self._alpha_scales():
            v = min(int(s), seq_len)
            if v not in scales:
                scales.append(v)
        attn_heads = max(1, min(num_heads, hidden_size))
        while hidden_size % attn_heads != 0 and attn_heads > 1:
            attn_heads -= 1

        class Chomp1d(nn.Module):
            def __init__(self, chomp_size: int) -> None:
                super().__init__()
                self.chomp_size = int(chomp_size)

            def forward(self, x: Any) -> Any:
                if self.chomp_size <= 0:
                    return x
                return x[:, :, : -self.chomp_size].contiguous()

        class TCNBlock(nn.Module):
            def __init__(self, channels: int, dilation: int) -> None:
                super().__init__()
                pad = (kernel_size - 1) * dilation
                self.net = nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
                    Chomp1d(pad),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
                    Chomp1d(pad),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )

            def forward(self, x: Any) -> Any:
                return x + self.net(x)

        class BetaTCNEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(beta_dim, hidden_size)
                self.block1 = TCNBlock(hidden_size, dilation=1)
                self.block2 = TCNBlock(hidden_size, dilation=2)
                self.out = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, latent_size), nn.Tanh())

            def forward(self, x: Any) -> Any:
                h = self.proj(x).transpose(1, 2).contiguous()
                h = self.block1(h)
                h = self.block2(h)
                pooled = h.mean(dim=-1)
                return self.out(pooled)

        class AlphaTransformerEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_proj = nn.Linear(alpha_dim, hidden_size)
                self.pos = nn.ParameterDict(
                    {str(s): nn.Parameter(torch.zeros(1, int(s), hidden_size)) for s in scales}
                )
                for p in self.pos.values():
                    nn.init.normal_(p, mean=0.0, std=0.02)
                self.encoders = nn.ModuleDict()
                for s in scales:
                    layer = nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=attn_heads,
                        dim_feedforward=hidden_size * max(1, ffn_mult),
                        dropout=dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    self.encoders[str(s)] = nn.TransformerEncoder(layer, num_layers=max(1, encoder_layers))
                self.gate = nn.Parameter(torch.zeros(len(scales)))
                self.out = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, latent_size), nn.Tanh())

            def forward(self, x: Any) -> Any:
                outs = []
                for s in scales:
                    xs = x[:, -int(s) :, :]
                    h = self.in_proj(xs) + self.pos[str(s)][:, -xs.shape[1] :, :]
                    h = self.encoders[str(s)](h)
                    if h.shape[1] != seq_len:
                        h_up = F.interpolate(h.transpose(1, 2), size=seq_len, mode="linear", align_corners=False).transpose(1, 2)
                    else:
                        h_up = h
                    outs.append(h_up)
                w = torch.softmax(self.gate, dim=0)
                fused = torch.stack(outs, dim=0)
                h = (w.reshape(-1, 1, 1, 1) * fused).sum(dim=0)
                pooled = h[:, -1, :]
                return self.out(pooled)

        class GRNBlock(nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.main = nn.Sequential(
                    nn.Linear(dim, dim * max(1, ffn_mult)),
                    nn.ELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * max(1, ffn_mult), dim),
                )
                self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
                self.norm = nn.LayerNorm(dim)

            def forward(self, x: Any) -> Any:
                s = self.main(x)
                g = self.gate(x)
                return self.norm(x + s * g)

        class ThetaGRNEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(theta_dim, hidden_size)
                self.blocks = nn.ModuleList([GRNBlock(hidden_size) for _ in range(max(1, grn_layers))])
                self.out = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, latent_size), nn.Tanh())

            def forward(self, x: Any) -> Any:
                h = self.proj(x)
                for block in self.blocks:
                    h = block(h)
                return self.out(h)

        class PPOActor(nn.Module):
            def __init__(self, in_dim: int) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                )
                self.out = nn.Linear(hidden_size, 3)

            def concentration(self, state: Any) -> Any:
                h = self.net(state)
                return F.softplus(self.out(h)) + 1e-3

            def dist(self, state: Any) -> Any:
                return torch.distributions.Dirichlet(self.concentration(state))

            def deterministic(self, state: Any) -> Any:
                conc = self.concentration(state)
                return conc / (conc.sum(dim=-1, keepdim=True) + 1e-12)

        class PPOCritic(nn.Module):
            def __init__(self, in_dim: int) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1),
                )

            def forward(self, state: Any) -> Any:
                return self.net(state).squeeze(-1)

        class DTLCRLNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.beta_encoder = BetaTCNEncoder()
                self.alpha_encoder = AlphaTransformerEncoder()
                self.theta_encoder = ThetaGRNEncoder()
                self.beta_head = nn.Linear(latent_size, 1)
                self.alpha_head = nn.Linear(latent_size, 1)
                self.theta_head = nn.Linear(latent_size, 1)
                self.linear_gate = nn.Linear(latent_size * 3, 3)
                self.pred_head = nn.Sequential(
                    nn.LayerNorm(latent_size),
                    nn.Linear(latent_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 1),
                )
                actor_state_dim = latent_size * 3 + state_dim
                self.actor = PPOActor(actor_state_dim)
                self.critic = PPOCritic(actor_state_dim)

            def encode(self, beta_seq: Any, alpha_seq: Any, theta_vec: Any) -> Dict[str, Any]:
                beta_z = self.beta_encoder(beta_seq)
                alpha_z = self.alpha_encoder(alpha_seq)
                theta_z = self.theta_encoder(theta_vec)
                return {"beta": beta_z, "alpha": alpha_z, "theta": theta_z}

            def day_state_from_z(self, z: Dict[str, Any], market_state: Any) -> Any:
                mkt = market_state.mean(dim=0)
                return torch.cat([z["beta"].mean(dim=0), z["alpha"].mean(dim=0), z["theta"].mean(dim=0), mkt], dim=0)

            def predict_with_weights(self, z: Dict[str, Any], weights: Any) -> Any:
                fused = weights[0] * z["beta"] + weights[1] * z["alpha"] + weights[2] * z["theta"]
                return self.pred_head(fused).squeeze(-1)

            def forward(self, beta_seq: Any, alpha_seq: Any, theta_vec: Any, market_state: Any) -> Dict[str, Any]:
                z = self.encode(beta_seq, alpha_seq, theta_vec)
                concat = torch.cat([z["beta"], z["alpha"], z["theta"]], dim=1)
                linear_w = torch.softmax(self.linear_gate(concat), dim=1)
                linear_fused = (
                    linear_w[:, 0:1] * z["beta"] + linear_w[:, 1:2] * z["alpha"] + linear_w[:, 2:3] * z["theta"]
                )
                day_state = self.day_state_from_z(z, market_state)
                actor_w = self.actor.deterministic(day_state.unsqueeze(0)).squeeze(0)
                return {
                    "beta_z": z["beta"],
                    "alpha_z": z["alpha"],
                    "theta_z": z["theta"],
                    "beta_pred": self.beta_head(z["beta"]).squeeze(-1),
                    "alpha_pred": self.alpha_head(z["alpha"]).squeeze(-1),
                    "theta_pred": self.theta_head(z["theta"]).squeeze(-1),
                    "linear_pred": self.pred_head(linear_fused).squeeze(-1),
                    "fused_pred": self.predict_with_weights(z, actor_w),
                    "weights": actor_w,
                    "day_state": day_state,
                }

            def predict_raw(self, beta_seq: Any, alpha_seq: Any, theta_vec: Any, market_state: Any) -> tuple[Any, Any]:
                z = self.encode(beta_seq, alpha_seq, theta_vec)
                day_state = self.day_state_from_z(z, market_state)
                weights = self.actor.deterministic(day_state.unsqueeze(0)).squeeze(0)
                return self.predict_with_weights(z, weights), weights

        return DTLCRLNet()

    @staticmethod
    def _ic_loss(pred: Any, target: Any, torch: Any) -> Any:
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        pred_c = pred - pred.mean()
        target_c = target - target.mean()
        pred_std = torch.sqrt(torch.mean(pred_c * pred_c) + 1e-12)
        target_std = torch.sqrt(torch.mean(target_c * target_c) + 1e-12)
        ic = torch.mean((pred_c / pred_std) * (target_c / target_std))
        return 1.0 - ic

    @staticmethod
    def _rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) < 3:
            return float("nan")
        s1 = pd.Series(y_true).rank(method="average")
        s2 = pd.Series(y_pred).rank(method="average")
        c = s1.corr(s2)
        return float(c) if pd.notna(c) else float("nan")

    def _contrastive_loss(self, z: Any, target: Any, torch: Any, F: Any) -> Any:
        n = int(z.shape[0])
        if n < 4:
            return torch.tensor(0.0, dtype=z.dtype, device=z.device)
        z_norm = F.normalize(z, dim=1)
        logits = torch.matmul(z_norm, z_norm.t()) / max(float(self.contrastive_tau), 1e-6)
        eye = torch.eye(n, dtype=torch.bool, device=z.device)
        order = torch.argsort(torch.argsort(target))
        dist = (order.reshape(-1, 1) - order.reshape(1, -1)).abs()
        width = max(1, int(math.ceil(n * float(self.positive_rank_pct))))
        pos_mask = (dist <= width) & (~eye)
        denom_mask = ~eye
        losses = []
        for i in range(n):
            if not bool(pos_mask[i].any()):
                continue
            denom = torch.logsumexp(logits[i][denom_mask[i]], dim=0)
            pos = torch.logsumexp(logits[i][pos_mask[i]], dim=0)
            losses.append(-(pos - denom))
        if not losses:
            return torch.tensor(0.0, dtype=z.dtype, device=z.device)
        return torch.stack(losses).mean()

    @staticmethod
    def _orthogonal_loss(z1: Any, z2: Any, torch: Any) -> Any:
        if int(z1.shape[0]) < 3:
            return torch.tensor(0.0, dtype=z1.dtype, device=z1.device)
        a = z1 - z1.mean(dim=0, keepdim=True)
        b = z2 - z2.mean(dim=0, keepdim=True)
        cov = torch.matmul(a.t(), b) / max(int(z1.shape[0]) - 1, 1)
        return torch.mean(cov * cov)

    def _evaluate_ic(self, net: Any, groups: Dict[pd.Timestamp, List[_TrainSample]], torch: Any, device: str) -> float:
        if not groups:
            return float("nan")
        net.eval()
        ics: List[float] = []
        with torch.no_grad():
            for key in sorted(groups.keys()):
                samples = groups[key]
                if len(samples) < self.min_cross_section:
                    continue
                beta, alpha, theta, state, y = self._stack_batch(samples)
                pred, _weights = net.predict_raw(
                    torch.tensor(beta, dtype=torch.float32, device=device),
                    torch.tensor(alpha, dtype=torch.float32, device=device),
                    torch.tensor(theta, dtype=torch.float32, device=device),
                    torch.tensor(state, dtype=torch.float32, device=device),
                )
                ic = self._rank_ic(y, pred.detach().cpu().numpy())
                if np.isfinite(ic):
                    ics.append(float(ic))
        return float(np.mean(ics)) if ics else float("nan")

    @staticmethod
    def _copy_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v.detach().cpu().clone() for k, v in state_dict.items()}

    def _pretrain_supervised(
        self,
        net: Any,
        train_groups: Dict[pd.Timestamp, List[_TrainSample]],
        val_groups: Dict[pd.Timestamp, List[_TrainSample]],
        *,
        torch: Any,
        F: Any,
        device: str,
    ) -> Dict[str, float]:
        optimizer = torch.optim.Adam(net.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay))
        rng = np.random.default_rng(int(self.random_state))
        train_keys = sorted(train_groups.keys())
        best_ic = -np.inf
        best_state: Dict[str, Any] | None = None
        best_epoch = -1
        patience = 0
        last_loss = float("nan")
        val_ic = float("nan")

        for epoch in range(1, int(self.pretrain_epochs) + 1):
            net.train()
            if int(self.per_epoch_batch) > 0 and len(train_keys) > int(self.per_epoch_batch):
                sampled_keys = rng.choice(train_keys, size=int(self.per_epoch_batch), replace=False).tolist()
            else:
                sampled_keys = list(train_keys)
            losses_epoch: List[float] = []
            for key in sampled_keys:
                samples = train_groups.get(key, [])
                if len(samples) < self.min_cross_section:
                    continue
                if int(self.batch_size) > 0 and len(samples) > int(self.batch_size):
                    idx = rng.choice(len(samples), size=int(self.batch_size), replace=False)
                    batch = [samples[int(i)] for i in idx]
                else:
                    batch = samples
                beta, alpha, theta, state, y = self._stack_batch(batch)
                x_beta = torch.tensor(beta, dtype=torch.float32, device=device)
                x_alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
                x_theta = torch.tensor(theta, dtype=torch.float32, device=device)
                x_state = torch.tensor(state, dtype=torch.float32, device=device)
                y_true = torch.tensor(y, dtype=torch.float32, device=device)

                out = net.forward(x_beta, x_alpha, x_theta, x_state)
                preds = [out["beta_pred"], out["alpha_pred"], out["theta_pred"], out["linear_pred"], out["fused_pred"]]
                ic_loss = torch.stack([self._ic_loss(p, y_true, torch=torch) for p in preds]).mean()
                mse_loss = torch.stack([F.mse_loss(p, y_true) for p in preds]).mean()
                contrast = (
                    self._contrastive_loss(out["beta_z"], y_true, torch=torch, F=F)
                    + self._contrastive_loss(out["alpha_z"], y_true, torch=torch, F=F)
                    + self._contrastive_loss(out["theta_z"], y_true, torch=torch, F=F)
                ) / 3.0
                orth = (
                    self._orthogonal_loss(out["beta_z"], out["alpha_z"], torch=torch)
                    + self._orthogonal_loss(out["beta_z"], out["theta_z"], torch=torch)
                    + self._orthogonal_loss(out["alpha_z"], out["theta_z"], torch=torch)
                ) / 3.0
                loss = (
                    float(self.ic_loss_weight) * ic_loss
                    + float(self.mse_weight) * mse_loss
                    + float(self.contrastive_weight) * contrast
                    + float(self.orthogonal_weight) * orth
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                optimizer.step()
                losses_epoch.append(float(loss.detach().cpu().item()))

            last_loss = float(np.mean(losses_epoch)) if losses_epoch else float("nan")
            val_ic = self._evaluate_ic(net, groups=val_groups, torch=torch, device=device)
            improved = np.isfinite(val_ic) and val_ic > best_ic + 1e-8
            if improved:
                best_ic = float(val_ic)
                best_state = self._copy_state_dict(net.state_dict())
                best_epoch = epoch
                patience = 0
            else:
                patience += 1
            if patience >= int(self.early_stop):
                break

        if best_state is not None:
            net.load_state_dict(best_state)
        return {
            "best_val_rank_ic": float(best_ic) if np.isfinite(best_ic) else float("nan"),
            "pretrain_final_val_rank_ic": float(val_ic) if np.isfinite(val_ic) else float("nan"),
            "best_epoch": float(best_epoch),
            "pretrain_epochs_trained": float(epoch if "epoch" in locals() else 0),
            "pretrain_loss_last": float(last_loss),
        }

    def _collect_ppo_transitions(
        self,
        net: Any,
        groups: Dict[pd.Timestamp, List[_TrainSample]],
        *,
        torch: Any,
        device: str,
    ) -> List[_PPOTransition]:
        net.eval()
        transitions: List[_PPOTransition] = []
        prev_action = np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
        keys = sorted(groups.keys())
        with torch.no_grad():
            for idx_key, key in enumerate(keys):
                samples = groups[key]
                if len(samples) < self.min_cross_section:
                    continue
                beta, alpha, theta, state, y = self._stack_batch(samples)
                x_beta = torch.tensor(beta, dtype=torch.float32, device=device)
                x_alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
                x_theta = torch.tensor(theta, dtype=torch.float32, device=device)
                x_state = torch.tensor(state, dtype=torch.float32, device=device)
                z = net.encode(x_beta, x_alpha, x_theta)
                day_state = net.day_state_from_z(z, x_state)
                dist = net.actor.dist(day_state.unsqueeze(0))
                action = dist.sample().squeeze(0)
                action = torch.clamp(action, min=1e-5)
                action = action / action.sum()
                log_prob = dist.log_prob(action.unsqueeze(0)).squeeze(0)
                value = net.critic(day_state.unsqueeze(0)).squeeze(0)
                pred = net.predict_with_weights(z, action).detach().cpu().numpy()
                ic = self._rank_ic(y, pred)
                reward_ic = float(ic) if np.isfinite(ic) else 0.0
                action_np = action.detach().cpu().numpy().astype(np.float32)
                stable = -float(np.mean(np.abs(action_np - prev_action)))
                entropy = -float(np.sum(action_np * np.log(action_np + 1e-12)) / math.log(3.0))
                reward = reward_ic + float(self.stable_weight) * stable + float(self.diversity_weight) * entropy
                transitions.append(
                    _PPOTransition(
                        state=day_state.detach().cpu().numpy().astype(np.float32),
                        action=action_np,
                        reward=float(reward),
                        log_prob=float(log_prob.detach().cpu().item()),
                        value=float(value.detach().cpu().item()),
                        done=idx_key == len(keys) - 1,
                    )
                )
                prev_action = action_np
        return transitions

    def _train_ppo(
        self,
        net: Any,
        train_groups: Dict[pd.Timestamp, List[_TrainSample]],
        val_groups: Dict[pd.Timestamp, List[_TrainSample]],
        *,
        torch: Any,
        F: Any,
        device: str,
    ) -> Dict[str, float]:
        params = list(net.actor.parameters()) + list(net.critic.parameters())
        optimizer = torch.optim.Adam(params, lr=float(self.ppo_lr), weight_decay=float(self.weight_decay))
        rng = np.random.default_rng(int(self.random_state) + 17)
        best_ic = self._evaluate_ic(net, groups=val_groups, torch=torch, device=device)
        best_state = self._copy_state_dict(net.state_dict())
        last_reward = float("nan")
        last_actor_loss = float("nan")
        last_value_loss = float("nan")

        for _episode in range(1, int(self.ppo_epochs) + 1):
            transitions = self._collect_ppo_transitions(net, train_groups, torch=torch, device=device)
            if len(transitions) < 2:
                break
            rewards = np.asarray([tr.reward for tr in transitions], dtype=np.float32)
            values = np.asarray([tr.value for tr in transitions], dtype=np.float32)
            dones = np.asarray([tr.done for tr in transitions], dtype=np.float32)
            adv = np.zeros_like(rewards, dtype=np.float32)
            last_gae = 0.0
            for t in range(len(rewards) - 1, -1, -1):
                next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
                non_terminal = 1.0 - dones[t]
                delta = rewards[t] + float(self.gamma) * next_value * non_terminal - values[t]
                last_gae = delta + float(self.gamma) * float(self.gae_lambda) * non_terminal * last_gae
                adv[t] = last_gae
            returns = adv + values
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            states = torch.tensor(np.stack([tr.state for tr in transitions], axis=0), dtype=torch.float32, device=device)
            actions = torch.tensor(np.stack([tr.action for tr in transitions], axis=0), dtype=torch.float32, device=device)
            old_logp = torch.tensor([tr.log_prob for tr in transitions], dtype=torch.float32, device=device)
            adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

            n = int(states.shape[0])
            batch_size = min(max(1, int(self.ppo_batch_size)), n)
            for _ in range(max(1, int(self.ppo_update_epochs))):
                order = rng.permutation(n)
                for start in range(0, n, batch_size):
                    idx = order[start : start + batch_size]
                    idx_t = torch.tensor(idx, dtype=torch.long, device=device)
                    dist = net.actor.dist(states.index_select(0, idx_t))
                    new_logp = dist.log_prob(actions.index_select(0, idx_t))
                    ratio = torch.exp(new_logp - old_logp.index_select(0, idx_t))
                    a = adv_t.index_select(0, idx_t)
                    unclipped = ratio * a
                    clipped = torch.clamp(ratio, 1.0 - float(self.ppo_clip), 1.0 + float(self.ppo_clip)) * a
                    actor_loss = -torch.min(unclipped, clipped).mean()
                    value_pred = net.critic(states.index_select(0, idx_t))
                    value_loss = F.mse_loss(value_pred, returns_t.index_select(0, idx_t))
                    entropy = dist.entropy().mean()
                    loss = actor_loss + float(self.value_weight) * value_loss - float(self.entropy_weight) * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                    optimizer.step()
                    last_actor_loss = float(actor_loss.detach().cpu().item())
                    last_value_loss = float(value_loss.detach().cpu().item())

            last_reward = float(np.mean(rewards))
            val_ic = self._evaluate_ic(net, groups=val_groups, torch=torch, device=device)
            if np.isfinite(val_ic) and (not np.isfinite(best_ic) or val_ic > best_ic + 1e-8):
                best_ic = float(val_ic)
                best_state = self._copy_state_dict(net.state_dict())

        net.load_state_dict(best_state)
        final_ic = self._evaluate_ic(net, groups=val_groups, torch=torch, device=device)
        return {
            "ppo_best_val_rank_ic": float(best_ic) if np.isfinite(best_ic) else float("nan"),
            "ppo_final_val_rank_ic": float(final_ic) if np.isfinite(final_ic) else float("nan"),
            "ppo_reward_last": float(last_reward),
            "ppo_actor_loss_last": float(last_actor_loss),
            "ppo_value_loss_last": float(last_value_loss),
        }

    def fit(self, train_df: pd.DataFrame, factor_cols: list[str], target_col: str) -> "DTLCRLStockModel":
        torch, nn, F = self._require_torch()
        self._set_seed(torch)

        self._factor_cols = list(factor_cols)
        self._set_space_cols(self._resolve_space_cols(self._factor_cols))
        self._time_col = self._resolve_time_col(train_df)

        x = train_df[self._factor_cols].replace([np.inf, -np.inf], np.nan)
        self._fill_values = x.median(numeric_only=True).reindex(self._factor_cols).fillna(0.0)
        df = train_df.copy()
        df[self._factor_cols] = self._transform_feature_frame(x)
        anchor = self._time_anchor(df[self._time_col])
        target = self._build_target(df, target_col=target_col, anchor=anchor)

        state_source_cols = [
            c
            for c in [
                "ret_1d",
                "ret_1",
                "mom_5",
                "ret_3",
                "ret_6",
                "realized_vol_20",
                "rv_12",
                "intraday_range",
                "range_norm",
                "turn",
                "turn_ratio_5",
                "amount_ratio_20",
                "amount_ratio_12",
                "amount",
                "volume",
            ]
            if c in df.columns
        ]
        state_df = df[list(dict.fromkeys(["code", self._time_col, *self._factor_cols, *state_source_cols]))]
        state_lookup = self._fit_market_state(state_df)
        grouped = self._build_train_samples(df=df, factor_cols=self._factor_cols, target=target, state_lookup=state_lookup)
        grouped = {k: v for k, v in grouped.items() if len(v) >= int(self.min_cross_section)}
        if not grouped:
            raise RuntimeError("DTLC_RL valid training samples are empty after sequence construction.")

        keys = sorted(grouped.keys())
        split = max(1, int(len(keys) * 0.8))
        train_keys = keys[:split]
        val_keys = keys[split:] or keys[-1:]
        train_groups = {k: grouped[k] for k in train_keys}
        val_groups = {k: grouped[k] for k in val_keys}

        net = self._build_network(
            beta_dim=len(self._space_indices["beta"]),
            alpha_dim=len(self._space_indices["alpha"]),
            theta_dim=len(self._space_indices["theta"]),
            state_dim=len(self._state_cols),
            torch=torch,
            nn=nn,
            F=F,
        )
        device = self._choose_device(torch)
        self._device_used = device
        net.to(device)

        pretrain_summary = self._pretrain_supervised(net, train_groups, val_groups, torch=torch, F=F, device=device)
        ppo_summary = self._train_ppo(net, train_groups, val_groups, torch=torch, F=F, device=device)
        final_ic = self._evaluate_ic(net, groups=val_groups, torch=torch, device=device)
        self._score_sign = -1.0 if np.isfinite(final_ic) and final_ic < 0.0 else 1.0

        self._model = net
        self._train_summary = {
            **pretrain_summary,
            **ppo_summary,
            "final_val_rank_ic": float(final_ic) if np.isfinite(final_ic) else float("nan"),
            "score_sign": float(self._score_sign),
            "device": self._device_used,
            "factor_count": float(len(self._factor_cols)),
            "beta_factor_count": float(len(self._space_indices["beta"])),
            "alpha_factor_count": float(len(self._space_indices["alpha"])),
            "theta_factor_count": float(len(self._space_indices["theta"])),
        }
        return self

    def _build_predict_batches(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        state_lookup: Dict[pd.Timestamp, np.ndarray],
    ) -> Dict[pd.Timestamp, List[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
        keep_cols = list(dict.fromkeys(["code", self._time_col, *factor_cols]))
        out = df[keep_cols].copy()
        out["_model_time"] = pd.to_datetime(out[self._time_col], errors="coerce")
        out = out.dropna(subset=["code", "_model_time"]).copy()
        out["code"] = out["code"].astype(str)
        out = out.sort_values(["code", "_model_time"])

        beta_idx = self._space_indices["beta"]
        alpha_idx = self._space_indices["alpha"]
        theta_idx = self._space_indices["theta"]
        seq_len = int(self.seq_len)
        batches: Dict[pd.Timestamp, List[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)

        for code, g in out.groupby("code", sort=False, observed=True):
            c = str(code)
            feat = g[factor_cols].to_numpy(dtype=np.float32)
            hist = self._history_by_code.get(c)
            if hist is None:
                hist = np.zeros((0, feat.shape[1]), dtype=np.float32)
            merged = np.vstack([hist, feat]) if len(hist) > 0 else feat
            hlen = len(hist)
            idxs = g.index.to_numpy()
            times = pd.to_datetime(g["_model_time"], errors="coerce")
            for pos, row_idx in enumerate(idxs):
                end = hlen + pos
                if end >= seq_len - 1:
                    full_seq = merged[end - seq_len + 1 : end + 1]
                else:
                    head = merged[: end + 1]
                    if len(head) == 0:
                        head = np.zeros((1, feat.shape[1]), dtype=np.float32)
                    pad = np.repeat(head[:1], repeats=max(0, seq_len - len(head)), axis=0)
                    full_seq = np.vstack([pad, head])
                key = pd.Timestamp(times.loc[row_idx])
                if key == key.normalize():
                    key = key.normalize()
                batches[key].append(
                    (
                        int(row_idx),
                        full_seq[:, beta_idx].astype(np.float32),
                        full_seq[:, alpha_idx].astype(np.float32),
                        merged[end, theta_idx].astype(np.float32),
                        self._state_for_key(key, state_lookup),
                    )
                )
        return batches

    def predict_score(self, df: pd.DataFrame, factor_cols: list[str]) -> pd.Series:
        if self._model is None or self._fill_values is None or self._time_col is None:
            raise RuntimeError("DTLCRLStockModel is not fitted.")
        if list(factor_cols) != self._factor_cols:
            missing = [c for c in self._factor_cols if c not in factor_cols]
            if missing:
                raise ValueError(f"predict factor cols missing: {missing}")

        torch, _nn, _F = self._require_torch()
        self._model.eval()
        state_cols = [
            "ret_1d",
            "ret_1",
            "mom_5",
            "ret_3",
            "ret_6",
            "realized_vol_20",
            "rv_12",
            "intraday_range",
            "range_norm",
            "turn",
            "turn_ratio_5",
            "amount_ratio_20",
            "amount_ratio_12",
            "amount",
            "volume",
        ]
        keep_cols = list(dict.fromkeys(["code", self._time_col, *self._factor_cols, *[c for c in state_cols if c in df.columns]]))
        out = df[keep_cols].copy()
        out[self._factor_cols] = self._transform_feature_frame(out[self._factor_cols])

        pred_state = self._normalize_market_state_frame(self._build_market_state_frame(out, time_col=self._time_col))
        state_lookup = dict(self._market_state_lookup)
        for k in pred_state.index:
            state_lookup[pd.Timestamp(k)] = pred_state.loc[k, self._state_cols].to_numpy(dtype=np.float32)

        batches = self._build_predict_batches(out, factor_cols=self._factor_cols, state_lookup=state_lookup)
        raw_pred = pd.Series(np.nan, index=out.index, dtype=float)
        with torch.no_grad():
            for key in sorted(batches.keys()):
                rows = batches[key]
                if not rows:
                    continue
                row_ids = [r[0] for r in rows]
                beta_np = np.stack([r[1] for r in rows], axis=0).astype(np.float32)
                alpha_np = np.stack([r[2] for r in rows], axis=0).astype(np.float32)
                theta_np = np.stack([r[3] for r in rows], axis=0).astype(np.float32)
                state_np = np.stack([r[4] for r in rows], axis=0).astype(np.float32)
                pred, _weights = self._model.predict_raw(
                    torch.tensor(beta_np, dtype=torch.float32, device=self._device_used),
                    torch.tensor(alpha_np, dtype=torch.float32, device=self._device_used),
                    torch.tensor(theta_np, dtype=torch.float32, device=self._device_used),
                    torch.tensor(state_np, dtype=torch.float32, device=self._device_used),
                )
                raw_pred.loc[row_ids] = pred.detach().cpu().numpy()

        raw_pred = raw_pred * float(self._score_sign)
        anchor = self._time_anchor(out[self._time_col])
        score = raw_pred.groupby(anchor, sort=False, observed=True).rank(pct=True, method="average")
        return score.fillna(0.5).reindex(df.index).fillna(0.5).rename("pred_score")

    def fill_values(self) -> pd.Series:
        if self._fill_values is None:
            return pd.Series(dtype=float)
        return self._fill_values

    def save(self, folder: Path, run_tag: str) -> Dict[str, str]:
        if self._model is None:
            raise RuntimeError("DTLCRLStockModel is not fitted.")
        torch, _nn, _F = self._require_torch()
        folder.mkdir(parents=True, exist_ok=True)
        model_path = folder / f"stock_model_dtlc_rl_{run_tag}.pt"
        meta_path = folder / f"stock_model_dtlc_rl_{run_tag}.json"
        config = {
            "seq_len": int(self.seq_len),
            "hidden_size": int(self.hidden_size),
            "latent_size": int(self.latent_size),
            "num_heads": int(self.num_heads),
            "encoder_layers": int(self.encoder_layers),
            "grn_layers": int(self.grn_layers),
            "ffn_mult": int(self.ffn_mult),
            "tcn_kernel_size": int(self.tcn_kernel_size),
            "alpha_scales": self._alpha_scales(),
            "dropout": float(self.dropout),
            "pretrain_epochs": int(self.pretrain_epochs),
            "ppo_epochs": int(self.ppo_epochs),
            "lr": float(self.lr),
            "ppo_lr": float(self.ppo_lr),
            "weight_decay": float(self.weight_decay),
            "early_stop": int(self.early_stop),
            "per_epoch_batch": int(self.per_epoch_batch),
            "batch_size": int(self.batch_size),
            "label_transform": str(self.label_transform),
            "input_clip": float(self.input_clip),
            "mse_weight": float(self.mse_weight),
            "ic_loss_weight": float(self.ic_loss_weight),
            "contrastive_weight": float(self.contrastive_weight),
            "orthogonal_weight": float(self.orthogonal_weight),
            "contrastive_tau": float(self.contrastive_tau),
            "positive_rank_pct": float(self.positive_rank_pct),
            "ppo_clip": float(self.ppo_clip),
            "gae_lambda": float(self.gae_lambda),
            "gamma": float(self.gamma),
            "ppo_update_epochs": int(self.ppo_update_epochs),
            "ppo_batch_size": int(self.ppo_batch_size),
            "entropy_weight": float(self.entropy_weight),
            "value_weight": float(self.value_weight),
            "stable_weight": float(self.stable_weight),
            "diversity_weight": float(self.diversity_weight),
            "min_cross_section": int(self.min_cross_section),
            "random_state": int(self.random_state),
            "device_used": self._device_used,
            "target_col": self._target_col,
            "score_sign": float(self._score_sign),
        }
        checkpoint = {
            "state_dict": self._model.state_dict(),
            "factor_cols": self._factor_cols,
            "space_cols": self._space_cols,
            "fill_values": self._fill_values.to_dict() if self._fill_values is not None else {},
            "time_col": self._time_col,
            "market_state_mean": self._market_state_mean.to_dict() if self._market_state_mean is not None else {},
            "market_state_std": self._market_state_std.to_dict() if self._market_state_std is not None else {},
            "market_state_lookup": {str(k): v.tolist() for k, v in self._market_state_lookup.items()},
            "train_summary": self._train_summary,
            "config": config,
        }
        torch.save(checkpoint, model_path)
        dump_json(
            meta_path,
            {
                "model_type": "dtlc_rl",
                "target_col": self._target_col,
                "factor_count": len(self._factor_cols),
                "space_factor_count": {k: len(v) for k, v in self._space_cols.items()},
                "train_summary": self._train_summary,
                "config": config,
            },
        )
        return {"model_pt": str(model_path), "meta_json": str(meta_path)}
