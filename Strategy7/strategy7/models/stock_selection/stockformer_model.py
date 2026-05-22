"""StockFormer stock-selection model.

This module implements a Strategy7-compatible reproduction of the StockFormer
idea described in the Minsheng Securities report:

1) Three predictive-coding Transformer branches build relation, short-return,
   and long-return latent states.
2) Two cross-sectional attention fusion stages build the SAC state.
3) A Soft Actor-Critic policy with twin critics learns portfolio actions from
   excess-return, turnover-cost, and tracking-error rewards.

The original paper/report emits portfolio actions directly.  Strategy7's stock
selection interface consumes a cross-sectional score, so deterministic actor
weights are converted to per-timestamp rank scores in ``predict_score``.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from ...core.constants import EPS
from ...core.utils import dump_json
from ..base import StockSelectionModel


@dataclass
class _TrainSample:
    code: str
    time_key: pd.Timestamp
    past_seq: np.ndarray
    rel_seq: np.ndarray
    short_target: float
    long_target: float
    rel_target: float
    reward_return: float


@dataclass
class _DayState:
    key: pd.Timestamp
    codes: List[str]
    state: np.ndarray
    returns: np.ndarray


@dataclass
class _Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class StockFormerStockModel(StockSelectionModel):
    """StockFormer predictive-coding + SAC model adapted to Strategy7."""

    seq_len: int = 60
    rel_seq_len: int = 252
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 10
    ffn_mult: int = 4
    dropout: float = 0.10

    pretrain_epochs: int = 50
    sac_episodes: int = 50
    lr: float = 1e-3
    sac_lr: float = 3e-4
    weight_decay: float = 0.0
    gamma: float = 0.999
    tau: float = 0.005
    init_alpha: float = 0.5
    target_entropy_scale: float = 1.0
    early_stop: int = 20
    buffer_size: int = 100000
    learning_starts: int = 100
    batch_transitions: int = 16
    updates_per_step: int = 1
    per_epoch_batch: int = 100
    batch_size: int = -1
    label_transform: str = "csrank"
    input_clip: float = 3.0
    mse_weight: float = 1.0
    ic_loss_weight: float = 1.0
    reward_cost_bps: float = 30.0
    turnover_penalty: float = 1.0
    tracking_penalty: float = 0.05
    min_cross_section: int = 8

    random_state: int = 42
    device: str = "auto"

    _model: Any = field(default=None, init=False, repr=False)
    _factor_cols: List[str] = field(default_factory=list, init=False, repr=False)
    _fill_values: pd.Series | None = field(default=None, init=False, repr=False)
    _time_col: str | None = field(default=None, init=False, repr=False)
    _history_by_code: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _history_rel_by_code: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
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
            raise RuntimeError(
                "StockFormer requires PyTorch. Please install torch first, for example: pip install torch"
            ) from exc
        return torch, nn, F

    @staticmethod
    def _resolve_time_col(df: pd.DataFrame) -> str:
        for c in ["signal_ts", "datetime", "date"]:
            if c in df.columns:
                return c
        raise ValueError("StockFormer requires one of ['signal_ts', 'datetime', 'date'] columns.")

    @staticmethod
    def _zscore(x: pd.Series) -> pd.Series:
        v = pd.to_numeric(x, errors="coerce")
        std = float(v.std(ddof=0)) if v.notna().sum() > 1 else 0.0
        if std <= EPS:
            return pd.Series(np.zeros(len(v), dtype=float), index=v.index)
        return (v - float(v.mean())) / (std + EPS)

    @staticmethod
    def _time_anchor(ts: pd.Series) -> pd.Series:
        dt = pd.to_datetime(ts, errors="coerce")
        normalized = dt.dt.normalize()
        has_intraday = bool(((dt - normalized).dt.total_seconds().fillna(0.0) != 0.0).any())
        return dt if has_intraday else normalized

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
        if transform == "cszscore":
            return y_raw.groupby(anchor, sort=False, observed=True).transform(self._zscore)
        if transform == "csranknorm":
            ranked = y_raw.groupby(anchor, sort=False, observed=True).transform(lambda s: s.rank(pct=True, method="average"))
            ranked = ranked.replace([np.inf, -np.inf], np.nan)
            return ranked.groupby(anchor, sort=False, observed=True).transform(self._zscore)
        return y_raw.groupby(anchor, sort=False, observed=True).transform(lambda s: s.rank(pct=True, method="average"))

    def _build_short_target(self, df: pd.DataFrame, fallback: pd.Series) -> pd.Series:
        if "close" not in df.columns or self._time_col is None:
            return fallback
        out = df[["code", self._time_col, "close"]].copy()
        out["_pos"] = np.arange(len(out), dtype=int)
        out["_time"] = pd.to_datetime(out[self._time_col], errors="coerce")
        out = out.sort_values(["code", "_time"])
        g = out.groupby("code", sort=False, observed=True)
        entry = g["close"].shift(-1)
        exit_ = g["close"].shift(-2)
        ret = exit_ / (entry + EPS) - 1.0
        short = pd.Series(ret.to_numpy(dtype=float), index=out["_pos"].to_numpy(dtype=int))
        short = short.reindex(np.arange(len(df), dtype=int))
        short.index = df.index
        return short.where(short.notna(), fallback)

    def _transform_feature_frame(self, x: pd.DataFrame) -> pd.DataFrame:
        out = x.replace([np.inf, -np.inf], np.nan)
        if self._fill_values is not None:
            out = out.fillna(self._fill_values.reindex(out.columns))
        out = out.fillna(0.0)
        clip = float(self.input_clip)
        if clip > 0.0:
            out = out.clip(lower=-clip, upper=clip)
        return out

    def _build_relation_frame(self, df: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
        if self._time_col is None:
            return x.copy()
        anchor = self._time_anchor(df[self._time_col])
        market_mean = x.groupby(anchor, sort=False, observed=True).transform("mean")
        rel = x - market_mean
        mix = 0.70 * x + 0.30 * rel
        if "industry_bucket" in df.columns:
            keys = [anchor, df["industry_bucket"].astype(str).fillna("unknown")]
            ind_mean = x.groupby(keys, sort=False, observed=True).transform("mean")
            mix = 0.60 * mix + 0.40 * (x - ind_mean)
        if "board_type" in df.columns:
            keys = [anchor, df["board_type"].astype(str).fillna("unknown")]
            board_mean = x.groupby(keys, sort=False, observed=True).transform("mean")
            mix = 0.80 * mix + 0.20 * (x - board_mean)
        return mix.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _build_train_samples(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        rel_df: pd.DataFrame,
        short_target: pd.Series,
        long_target: pd.Series,
    ) -> Dict[pd.Timestamp, List[_TrainSample]]:
        keep_cols = list(dict.fromkeys(["code", self._time_col, *factor_cols]))
        out = df[keep_cols].copy()
        out["_row"] = np.arange(len(out), dtype=int)
        out["_model_time"] = pd.to_datetime(out[self._time_col], errors="coerce")
        out["_short_target"] = pd.to_numeric(short_target, errors="coerce").to_numpy(dtype=float)
        out["_long_target"] = pd.to_numeric(long_target, errors="coerce").to_numpy(dtype=float)
        raw_ret = pd.to_numeric(df.get("target_return", df.get("future_ret_n", long_target)), errors="coerce")
        out["_reward_return"] = raw_ret.to_numpy(dtype=float)
        out = out.dropna(subset=["code", "_model_time"]).sort_values(["code", "_model_time"])

        x_all = df[factor_cols].to_numpy(dtype=np.float32)
        rel_all = rel_df[factor_cols].to_numpy(dtype=np.float32)
        grouped: Dict[pd.Timestamp, List[_TrainSample]] = defaultdict(list)
        self._history_by_code = {}
        self._history_rel_by_code = {}

        seq_len = int(self.seq_len)
        rel_len = int(self.rel_seq_len)
        start_idx = max(seq_len, rel_len) - 1
        for code, g in out.groupby("code", sort=False, observed=True):
            rows = g["_row"].to_numpy(dtype=int)
            x = x_all[rows]
            rel_x = rel_all[rows]
            t = pd.to_datetime(g["_model_time"], errors="coerce")
            short_y = g["_short_target"].to_numpy(dtype=np.float32)
            long_y = g["_long_target"].to_numpy(dtype=np.float32)
            reward_ret = g["_reward_return"].to_numpy(dtype=np.float32)
            n = len(g)
            if n == 0:
                continue
            self._history_by_code[str(code)] = x[max(0, n - seq_len + 1) :].copy()
            self._history_rel_by_code[str(code)] = rel_x[max(0, n - rel_len + 1) :].copy()
            if n <= start_idx:
                continue

            for i in range(start_idx, n):
                if np.isnan(long_y[i]) or np.isnan(short_y[i]) or np.isnan(reward_ret[i]):
                    continue
                past = x[i - seq_len + 1 : i + 1]
                rel_past = rel_x[i - rel_len + 1 : i + 1]
                if past.shape[0] != seq_len or rel_past.shape[0] != rel_len:
                    continue
                if np.isnan(past).any() or np.isnan(rel_past).any():
                    continue
                key = pd.Timestamp(t.iloc[i])
                if key == key.normalize():
                    key = key.normalize()
                grouped[key].append(
                    _TrainSample(
                        code=str(code),
                        time_key=key,
                        past_seq=past.astype(np.float32),
                        rel_seq=rel_past.astype(np.float32),
                        short_target=float(short_y[i]),
                        long_target=float(long_y[i]),
                        rel_target=float(long_y[i]),
                        reward_return=float(reward_ret[i]),
                    )
                )
        return grouped

    @staticmethod
    def _stack_samples(samples: Iterable[_TrainSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ss = list(samples)
        past = np.stack([s.past_seq for s in ss], axis=0).astype(np.float32)
        rel = np.stack([s.rel_seq for s in ss], axis=0).astype(np.float32)
        y_short = np.asarray([s.short_target for s in ss], dtype=np.float32)
        y_long = np.asarray([s.long_target for s in ss], dtype=np.float32)
        y_rel = np.asarray([s.rel_target for s in ss], dtype=np.float32)
        return past, rel, y_short, y_long, y_rel

    def _build_network(self, input_dim: int, torch: Any, nn: Any, F: Any) -> Any:
        hidden_size = int(self.hidden_size)
        num_heads = int(self.num_heads)
        num_layers = int(self.num_layers)
        ffn_mult = int(self.ffn_mult)
        dropout = float(self.dropout)
        seq_len = int(self.seq_len)
        rel_seq_len = int(self.rel_seq_len)

        class DecoupledAttentionBlock(nn.Module):
            def __init__(self, d_model: int, heads: int, mult: int, drop: float) -> None:
                super().__init__()
                self.heads = max(1, heads)
                self.head_dim = int(math.ceil(d_model / float(self.heads)))
                self.inner_dim = self.heads * self.head_dim
                self.q_proj = nn.Linear(d_model, self.inner_dim)
                self.k_proj = nn.Linear(d_model, self.inner_dim)
                self.v_proj = nn.Linear(d_model, self.inner_dim)
                self.head_ffns = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(self.head_dim, max(self.head_dim, self.head_dim * max(1, mult))),
                            nn.GELU(),
                            nn.Dropout(drop),
                            nn.Linear(max(self.head_dim, self.head_dim * max(1, mult)), self.head_dim),
                        )
                        for _ in range(self.heads)
                    ]
                )
                self.out_proj = nn.Linear(self.inner_dim, d_model)
                self.drop = nn.Dropout(drop)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.mix = nn.Sequential(
                    nn.Linear(d_model, d_model * max(1, mult)),
                    nn.GELU(),
                    nn.Dropout(drop),
                    nn.Linear(d_model * max(1, mult), d_model),
                )

            def _shape(self, x: Any) -> Any:
                bsz, length, _ = x.shape
                return x.view(bsz, length, self.heads, self.head_dim).transpose(1, 2).contiguous()

            def forward(self, x: Any, key_value: Any | None = None) -> Any:
                kv = x if key_value is None else key_value
                q = self._shape(self.q_proj(x))
                k = self._shape(self.k_proj(kv))
                v = self._shape(self.v_proj(kv))
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
                attn = torch.softmax(scores, dim=-1)
                attn = self.drop(attn)
                head_out = torch.matmul(attn, v)
                per_head = []
                for h in range(self.heads):
                    per_head.append(self.head_ffns[h](head_out[:, h, :, :]))
                y = torch.cat(per_head, dim=-1)
                x = self.norm1(x + self.drop(self.out_proj(y)))
                x = self.norm2(x + self.drop(self.mix(x)))
                return x

        class PredictiveBranch(nn.Module):
            def __init__(self, length: int) -> None:
                super().__init__()
                self.length = int(length)
                self.in_proj = nn.Linear(input_dim, hidden_size)
                self.pos = nn.Parameter(torch.zeros(1, self.length, hidden_size))
                nn.init.normal_(self.pos, mean=0.0, std=0.02)
                self.blocks = nn.ModuleList(
                    [DecoupledAttentionBlock(hidden_size, num_heads, ffn_mult, dropout) for _ in range(max(1, num_layers))]
                )
                self.pred = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 1),
                )

            def forward(self, x: Any) -> tuple[Any, Any]:
                h = self.in_proj(x)
                pos = self.pos[:, -h.shape[1] :, :]
                h = h + pos
                for block in self.blocks:
                    h = block(h)
                state = h[:, -1, :]
                pred = self.pred(state).squeeze(-1)
                return state, pred

        class FusionModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.future_attn = DecoupledAttentionBlock(hidden_size, num_heads, ffn_mult, dropout)
                self.state_attn = DecoupledAttentionBlock(hidden_size, num_heads, ffn_mult, dropout)
                self.pred = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 1),
                )

            def forward(self, rel_state: Any, short_state: Any, long_state: Any) -> tuple[Any, Any]:
                long_tok = long_state.unsqueeze(0)
                short_tok = short_state.unsqueeze(0)
                rel_tok = rel_state.unsqueeze(0)
                future = self.future_attn(long_tok, key_value=short_tok)
                fused = self.state_attn(future, key_value=rel_tok).squeeze(0)
                return fused, self.pred(fused).squeeze(-1)

        class Actor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(
                    [DecoupledAttentionBlock(hidden_size, num_heads, ffn_mult, dropout) for _ in range(max(1, num_layers))]
                )
                self.mean = nn.Linear(hidden_size, 1)
                self.log_std = nn.Linear(hidden_size, 1)

            def forward(self, state: Any) -> tuple[Any, Any]:
                h = state.unsqueeze(0)
                for block in self.blocks:
                    h = block(h)
                h = h.squeeze(0)
                mean = self.mean(h).squeeze(-1)
                log_std = torch.clamp(self.log_std(h).squeeze(-1), min=-5.0, max=2.0)
                return mean, log_std

            def sample(self, state: Any, deterministic: bool = False) -> tuple[Any, Any, Any]:
                mean, log_std = self.forward(state)
                std = torch.exp(log_std)
                if deterministic:
                    raw = mean
                else:
                    raw = mean + std * torch.randn_like(std)
                positive = F.softplus(raw) + 1e-6
                weights = positive / (positive.sum() + 1e-12)
                log_prob_each = -0.5 * (((raw - mean) / (std + 1e-12)) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi))
                log_prob = log_prob_each.sum()
                return weights, log_prob, raw

        class Critic(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.action_proj = nn.Linear(1, hidden_size)
                self.blocks = nn.ModuleList(
                    [DecoupledAttentionBlock(hidden_size, num_heads, ffn_mult, dropout) for _ in range(max(1, num_layers))]
                )
                self.q = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, 1),
                )

            def forward(self, state: Any, action: Any) -> Any:
                h = state + self.action_proj(action.reshape(-1, 1))
                h = h.unsqueeze(0)
                for block in self.blocks:
                    h = block(h)
                pooled = h.squeeze(0).mean(dim=0)
                return self.q(pooled).squeeze(-1)

        class StockFormerNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rel_branch = PredictiveBranch(rel_seq_len)
                self.short_branch = PredictiveBranch(seq_len)
                self.long_branch = PredictiveBranch(seq_len)
                self.fusion = FusionModule()
                self.actor = Actor()
                self.critic1 = Critic()
                self.critic2 = Critic()
                self.target_critic1 = copy.deepcopy(self.critic1)
                self.target_critic2 = copy.deepcopy(self.critic2)
                for p in self.target_critic1.parameters():
                    p.requires_grad_(False)
                for p in self.target_critic2.parameters():
                    p.requires_grad_(False)

            def predictive(self, past: Any, rel: Any) -> Dict[str, Any]:
                rel_state, rel_pred = self.rel_branch(rel)
                short_state, short_pred = self.short_branch(past)
                long_state, long_pred = self.long_branch(past)
                fused_state, fused_pred = self.fusion(rel_state, short_state, long_state)
                return {
                    "rel_state": rel_state,
                    "short_state": short_state,
                    "long_state": long_state,
                    "state": fused_state,
                    "rel_pred": rel_pred,
                    "short_pred": short_pred,
                    "long_pred": long_pred,
                    "fused_pred": fused_pred,
                }

            def soft_update_targets(self, tau_value: float) -> None:
                for target, src in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                    target.data.mul_(1.0 - tau_value).add_(src.data, alpha=tau_value)
                for target, src in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                    target.data.mul_(1.0 - tau_value).add_(src.data, alpha=tau_value)

        return StockFormerNet()

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

    def _pretrain_predictive(
        self,
        net: Any,
        train_groups: Dict[pd.Timestamp, List[_TrainSample]],
        val_groups: Dict[pd.Timestamp, List[_TrainSample]],
        torch: Any,
        F: Any,
        device: str,
    ) -> Dict[str, float]:
        pred_params = list(net.rel_branch.parameters()) + list(net.short_branch.parameters()) + list(net.long_branch.parameters()) + list(net.fusion.parameters())
        optimizer = torch.optim.Adam(pred_params, lr=float(self.lr), weight_decay=float(self.weight_decay))
        rng = np.random.default_rng(int(self.random_state))
        train_keys = sorted(train_groups.keys())

        best_ic = -np.inf
        best_state: Dict[str, Any] | None = None
        best_epoch = 0
        patience = 0
        losses: List[float] = []

        for epoch in range(1, int(self.pretrain_epochs) + 1):
            net.train()
            epoch_keys = list(train_keys)
            if int(self.per_epoch_batch) > 0 and len(epoch_keys) > int(self.per_epoch_batch):
                epoch_keys = rng.choice(epoch_keys, size=int(self.per_epoch_batch), replace=False).tolist()
            epoch_losses: List[float] = []
            for key in epoch_keys:
                samples = train_groups.get(key, [])
                if len(samples) < int(self.min_cross_section):
                    continue
                if int(self.batch_size) > 0 and len(samples) > int(self.batch_size):
                    idx = rng.choice(len(samples), size=int(self.batch_size), replace=False)
                    batch = [samples[int(i)] for i in idx]
                else:
                    batch = samples
                past_np, rel_np, y_short_np, y_long_np, y_rel_np = self._stack_samples(batch)
                past = torch.tensor(past_np, dtype=torch.float32, device=device)
                rel = torch.tensor(rel_np, dtype=torch.float32, device=device)
                y_short = torch.tensor(y_short_np, dtype=torch.float32, device=device)
                y_long = torch.tensor(y_long_np, dtype=torch.float32, device=device)
                y_rel = torch.tensor(y_rel_np, dtype=torch.float32, device=device)
                out = net.predictive(past, rel)
                mse = (
                    F.mse_loss(out["short_pred"], y_short)
                    + F.mse_loss(out["long_pred"], y_long)
                    + F.mse_loss(out["rel_pred"], y_rel)
                    + F.mse_loss(out["fused_pred"], y_long)
                )
                ic = (
                    self._ic_loss(out["short_pred"], y_short, torch)
                    + self._ic_loss(out["long_pred"], y_long, torch)
                    + self._ic_loss(out["rel_pred"], y_rel, torch)
                    + self._ic_loss(out["fused_pred"], y_long, torch)
                )
                loss = float(self.mse_weight) * mse + float(self.ic_loss_weight) * ic
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pred_params, max_norm=5.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
            losses.append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))

            val_ic = self._evaluate_predictive_ic(net, val_groups, torch=torch, device=device)
            if np.isfinite(val_ic) and val_ic > best_ic + 1e-8:
                best_ic = float(val_ic)
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                best_epoch = epoch
                patience = 0
            else:
                patience += 1
            if patience >= int(self.early_stop):
                break

        if best_state is not None:
            net.load_state_dict(best_state)
        return {
            "pretrain_best_val_rank_ic": float(best_ic) if np.isfinite(best_ic) else float("nan"),
            "pretrain_best_epoch": float(best_epoch),
            "pretrain_epochs_trained": float(len(losses)),
            "pretrain_loss_last": float(losses[-1]) if losses else float("nan"),
        }

    def _evaluate_predictive_ic(self, net: Any, groups: Dict[pd.Timestamp, List[_TrainSample]], torch: Any, device: str) -> float:
        if not groups:
            return float("nan")
        net.eval()
        ics: List[float] = []
        with torch.no_grad():
            for key in sorted(groups.keys()):
                samples = groups[key]
                if len(samples) < int(self.min_cross_section):
                    continue
                past_np, rel_np, _ys, y_long_np, _yr = self._stack_samples(samples)
                past = torch.tensor(past_np, dtype=torch.float32, device=device)
                rel = torch.tensor(rel_np, dtype=torch.float32, device=device)
                pred = net.predictive(past, rel)["fused_pred"].detach().cpu().numpy()
                ic = self._rank_ic(y_long_np, pred)
                if np.isfinite(ic):
                    ics.append(float(ic))
        return float(np.mean(ics)) if ics else float("nan")

    def _build_day_states(
        self,
        net: Any,
        groups: Dict[pd.Timestamp, List[_TrainSample]],
        torch: Any,
        device: str,
    ) -> Dict[pd.Timestamp, _DayState]:
        out: Dict[pd.Timestamp, _DayState] = {}
        net.eval()
        with torch.no_grad():
            for key in sorted(groups.keys()):
                samples = groups[key]
                if len(samples) < int(self.min_cross_section):
                    continue
                past_np, rel_np, _ys, _yl, _yr = self._stack_samples(samples)
                past = torch.tensor(past_np, dtype=torch.float32, device=device)
                rel = torch.tensor(rel_np, dtype=torch.float32, device=device)
                state = net.predictive(past, rel)["state"].detach().cpu().numpy().astype(np.float32)
                returns = np.asarray([s.reward_return for s in samples], dtype=np.float32)
                codes = [str(s.code) for s in samples]
                valid = np.isfinite(returns)
                if int(valid.sum()) < int(self.min_cross_section):
                    continue
                out[key] = _DayState(
                    key=key,
                    codes=[c for c, ok in zip(codes, valid) if ok],
                    state=state[valid].astype(np.float32),
                    returns=returns[valid].astype(np.float32),
                )
        return out

    def _reward_from_weights(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        codes: List[str],
        prev_weights: Dict[str, float],
    ) -> tuple[float, Dict[str, float], Dict[str, float]]:
        w = np.asarray(weights, dtype=float)
        r = np.asarray(returns, dtype=float)
        if len(w) == 0 or len(r) == 0:
            return 0.0, {}, {"portfolio_ret": 0.0, "benchmark_ret": 0.0, "turnover": 0.0}
        w = np.clip(w, 0.0, None)
        w = w / (float(w.sum()) + EPS)
        n = len(w)
        bench_w = np.full(n, 1.0 / max(n, 1), dtype=float)
        port_ret = float(np.dot(w, r))
        bench_ret = float(np.dot(bench_w, r))
        prev_vec = np.asarray([float(prev_weights.get(c, 0.0)) for c in codes], dtype=float)
        dropped = float(sum(v for c, v in prev_weights.items() if c not in set(codes) and v > 0.0))
        turnover = float(np.abs(w - prev_vec).sum() + dropped)
        cost = float(self.reward_cost_bps) / 10000.0 * turnover
        tracking = float(np.sqrt(np.mean(np.square(w - bench_w)))) if n > 1 else 0.0
        reward = port_ret - bench_ret - float(self.turnover_penalty) * cost - float(self.tracking_penalty) * tracking
        new_weights = {str(c): float(x) for c, x in zip(codes, w) if x > 1e-8}
        diag = {
            "portfolio_ret": port_ret,
            "benchmark_ret": bench_ret,
            "turnover": turnover,
            "cost": cost,
            "tracking": tracking,
            "reward": reward,
        }
        return float(reward), new_weights, diag

    def _evaluate_policy(
        self,
        net: Any,
        day_states: Dict[pd.Timestamp, _DayState],
        torch: Any,
        device: str,
    ) -> Dict[str, float]:
        if not day_states:
            return {"val_reward_mean": float("nan"), "val_excess_net": float("nan"), "val_turnover_mean": float("nan")}
        net.eval()
        prev: Dict[str, float] = {}
        rewards: List[float] = []
        excess_rets: List[float] = []
        turnovers: List[float] = []
        with torch.no_grad():
            for key in sorted(day_states.keys()):
                ds = day_states[key]
                state = torch.tensor(ds.state, dtype=torch.float32, device=device)
                weights = net.actor.sample(state, deterministic=True)[0].detach().cpu().numpy()
                reward, prev, diag = self._reward_from_weights(weights, ds.returns, ds.codes, prev)
                rewards.append(reward)
                excess_rets.append(float(diag["portfolio_ret"] - diag["benchmark_ret"]))
                turnovers.append(float(diag["turnover"]))
        excess_net = float(np.prod(1.0 + np.asarray(excess_rets, dtype=float)) - 1.0) if excess_rets else float("nan")
        return {
            "val_reward_mean": float(np.mean(rewards)) if rewards else float("nan"),
            "val_excess_net": excess_net,
            "val_turnover_mean": float(np.mean(turnovers)) if turnovers else float("nan"),
        }

    def _train_sac(
        self,
        net: Any,
        train_days: Dict[pd.Timestamp, _DayState],
        val_days: Dict[pd.Timestamp, _DayState],
        torch: Any,
        F: Any,
        device: str,
    ) -> Dict[str, float]:
        keys = sorted(train_days.keys())
        if len(keys) < 2:
            return {
                "sac_episodes_trained": 0.0,
                "sac_best_val_excess_net": float("nan"),
                "sac_best_episode": 0.0,
            }

        actor_params = list(net.actor.parameters())
        critic_params = list(net.critic1.parameters()) + list(net.critic2.parameters())
        optimizer_actor = torch.optim.Adam(actor_params, lr=float(self.sac_lr), weight_decay=float(self.weight_decay))
        optimizer_critic = torch.optim.Adam(critic_params, lr=float(self.sac_lr), weight_decay=float(self.weight_decay))
        log_alpha = torch.tensor(math.log(max(float(self.init_alpha), 1e-6)), dtype=torch.float32, device=device, requires_grad=True)
        optimizer_alpha = torch.optim.Adam([log_alpha], lr=float(self.sac_lr))

        buffer: deque[_Transition] = deque(maxlen=max(1, int(self.buffer_size)))
        rng = np.random.default_rng(int(self.random_state) + 17)
        best_val = -np.inf
        best_state: Dict[str, Any] | None = None
        best_episode = 0
        patience = 0
        train_reward_last = float("nan")
        val_diag_last: Dict[str, float] = {}

        def _update_one(batch: List[_Transition]) -> tuple[float, float, float]:
            critic_losses = []
            for tr in batch:
                state = torch.tensor(tr.state, dtype=torch.float32, device=device)
                action = torch.tensor(tr.action, dtype=torch.float32, device=device)
                next_state = torch.tensor(tr.next_state, dtype=torch.float32, device=device)
                reward = torch.tensor(float(tr.reward), dtype=torch.float32, device=device)
                done = torch.tensor(1.0 if tr.done else 0.0, dtype=torch.float32, device=device)

                with torch.no_grad():
                    next_action, next_logp, _raw = net.actor.sample(next_state, deterministic=False)
                    tq1 = net.target_critic1(next_state, next_action)
                    tq2 = net.target_critic2(next_state, next_action)
                    alpha = log_alpha.exp()
                    target_q = reward + float(self.gamma) * (1.0 - done) * (torch.minimum(tq1, tq2) - alpha * next_logp)

                q1 = net.critic1(state, action)
                q2 = net.critic2(state, action)
                critic_losses.append(F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q))

            critic_loss = torch.stack(critic_losses).mean()
            optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_params, max_norm=5.0)
            optimizer_critic.step()

            for p in critic_params:
                p.requires_grad_(False)
            actor_losses = []
            alpha_losses = []
            for tr in batch:
                state = torch.tensor(tr.state, dtype=torch.float32, device=device)
                new_action, logp, _raw = net.actor.sample(state, deterministic=False)
                q_pi = torch.minimum(net.critic1(state, new_action), net.critic2(state, new_action))
                alpha = log_alpha.exp()
                actor_losses.append(alpha.detach() * logp - q_pi)
                target_entropy = -float(self.target_entropy_scale) * math.log(max(2, len(tr.action)))
                alpha_losses.append(-(log_alpha * (logp.detach() + target_entropy)))

            actor_loss = torch.stack(actor_losses).mean()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_params, max_norm=5.0)
            optimizer_actor.step()
            for p in critic_params:
                p.requires_grad_(True)

            alpha_loss = torch.stack(alpha_losses).mean()
            optimizer_alpha.zero_grad()
            alpha_loss.backward()
            optimizer_alpha.step()

            net.soft_update_targets(float(self.tau))
            return (
                float(critic_loss.detach().cpu().item()),
                float(actor_loss.detach().cpu().item()),
                float(alpha_loss.detach().cpu().item()),
            )

        for episode in range(1, int(self.sac_episodes) + 1):
            net.train()
            prev: Dict[str, float] = {}
            rewards_episode: List[float] = []
            for pos, key in enumerate(keys):
                ds = train_days[key]
                next_key = keys[min(pos + 1, len(keys) - 1)]
                next_ds = train_days[next_key]
                state_t = torch.tensor(ds.state, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action_t = net.actor.sample(state_t, deterministic=False)[0].detach().cpu().numpy()
                reward, prev, _diag = self._reward_from_weights(action_t, ds.returns, ds.codes, prev)
                done = bool(pos == len(keys) - 1)
                buffer.append(
                    _Transition(
                        state=ds.state,
                        action=np.asarray(action_t, dtype=np.float32),
                        reward=float(reward),
                        next_state=next_ds.state,
                        done=done,
                    )
                )
                rewards_episode.append(float(reward))
                if len(buffer) >= max(1, int(self.learning_starts)):
                    for _ in range(max(1, int(self.updates_per_step))):
                        sample_size = min(int(self.batch_transitions), len(buffer))
                        idx = rng.choice(len(buffer), size=sample_size, replace=False)
                        batch = [list(buffer)[int(i)] for i in idx]
                        _update_one(batch)
            train_reward_last = float(np.mean(rewards_episode)) if rewards_episode else float("nan")
            val_diag_last = self._evaluate_policy(net, val_days, torch=torch, device=device)
            val_ref = float(val_diag_last.get("val_excess_net", float("nan")))
            if np.isfinite(val_ref) and val_ref > best_val + 1e-8:
                best_val = val_ref
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                best_episode = episode
                patience = 0
            else:
                patience += 1
            if patience >= int(self.early_stop):
                break

        if best_state is not None:
            net.load_state_dict(best_state)
        return {
            "sac_episodes_trained": float(episode if "episode" in locals() else 0),
            "sac_best_val_excess_net": float(best_val) if np.isfinite(best_val) else float("nan"),
            "sac_best_episode": float(best_episode),
            "sac_train_reward_last": float(train_reward_last),
            "sac_val_reward_mean_last": float(val_diag_last.get("val_reward_mean", float("nan"))),
            "sac_val_turnover_mean_last": float(val_diag_last.get("val_turnover_mean", float("nan"))),
            "sac_alpha_last": float(log_alpha.exp().detach().cpu().item()),
            "replay_buffer_size": float(len(buffer)),
        }

    def fit(self, train_df: pd.DataFrame, factor_cols: list[str], target_col: str) -> "StockFormerStockModel":
        torch, nn, F = self._require_torch()
        self._set_seed(torch)
        self._factor_cols = list(factor_cols)
        self._time_col = self._resolve_time_col(train_df)

        x = train_df[self._factor_cols].replace([np.inf, -np.inf], np.nan)
        self._fill_values = x.median(numeric_only=True).reindex(self._factor_cols).fillna(0.0)
        df = train_df.copy()
        x_clean = self._transform_feature_frame(x)
        df[self._factor_cols] = x_clean
        rel_df = self._build_relation_frame(df=df, x=x_clean)
        anchor = self._time_anchor(df[self._time_col])
        long_target = self._build_target(df, target_col=target_col, anchor=anchor)
        short_target_raw = self._build_short_target(df, fallback=pd.to_numeric(df.get("target_return", long_target), errors="coerce"))
        transform = str(self.label_transform).lower().strip()
        if transform == "raw":
            short_target = short_target_raw
        elif transform in {"cszscore", "csranknorm"}:
            short_target = short_target_raw.groupby(anchor, sort=False, observed=True).transform(self._zscore)
        else:
            short_target = short_target_raw.groupby(anchor, sort=False, observed=True).transform(
                lambda s: s.rank(pct=True, method="average")
            )

        grouped = self._build_train_samples(
            df=df,
            factor_cols=self._factor_cols,
            rel_df=rel_df,
            short_target=short_target,
            long_target=long_target,
        )
        grouped = {k: v for k, v in grouped.items() if len(v) >= int(self.min_cross_section)}
        if not grouped:
            raise RuntimeError("StockFormer valid training samples are empty after sequence construction.")

        keys = sorted(grouped.keys())
        split = max(1, int(len(keys) * 0.8))
        train_keys = keys[:split]
        val_keys = keys[split:] or keys[-1:]
        train_groups = {k: grouped[k] for k in train_keys}
        val_groups = {k: grouped[k] for k in val_keys}

        net = self._build_network(input_dim=len(self._factor_cols), torch=torch, nn=nn, F=F)
        device = self._choose_device(torch)
        self._device_used = device
        net.to(device)

        pretrain_summary = self._pretrain_predictive(net, train_groups, val_groups, torch=torch, F=F, device=device)
        train_day_states = self._build_day_states(net, train_groups, torch=torch, device=device)
        val_day_states = self._build_day_states(net, val_groups, torch=torch, device=device)
        sac_summary = self._train_sac(net, train_day_states, val_day_states, torch=torch, F=F, device=device)
        final_policy = self._evaluate_policy(net, val_day_states, torch=torch, device=device)
        final_ic = self._evaluate_predictive_ic(net, val_groups, torch=torch, device=device)
        sign_ref_ic = final_ic if np.isfinite(final_ic) else pretrain_summary.get("pretrain_best_val_rank_ic", float("nan"))
        self._score_sign = -1.0 if np.isfinite(sign_ref_ic) and float(sign_ref_ic) < 0.0 else 1.0

        self._model = net
        self._train_summary = {
            **pretrain_summary,
            **sac_summary,
            "final_val_rank_ic": float(final_ic) if np.isfinite(final_ic) else float("nan"),
            "final_val_excess_net": float(final_policy.get("val_excess_net", float("nan"))),
            "final_val_reward_mean": float(final_policy.get("val_reward_mean", float("nan"))),
            "score_sign": float(self._score_sign),
            "score_sign_ref_ic": float(sign_ref_ic) if np.isfinite(sign_ref_ic) else float("nan"),
            "device": self._device_used,
            "train_day_count": float(len(train_day_states)),
            "val_day_count": float(len(val_day_states)),
        }
        return self

    def _build_predict_batches(
        self,
        df: pd.DataFrame,
        rel_df: pd.DataFrame,
        factor_cols: List[str],
    ) -> Dict[pd.Timestamp, List[tuple[int, np.ndarray, np.ndarray]]]:
        keep_cols = list(dict.fromkeys(["code", self._time_col, *factor_cols]))
        out = df[keep_cols].copy()
        out["_row"] = np.arange(len(out), dtype=int)
        out["_model_time"] = pd.to_datetime(out[self._time_col], errors="coerce")
        out = out.dropna(subset=["code", "_model_time"]).copy()
        out["code"] = out["code"].astype(str)
        out = out.sort_values(["code", "_model_time"])

        x_all = df[factor_cols].to_numpy(dtype=np.float32)
        rel_all = rel_df[factor_cols].to_numpy(dtype=np.float32)
        batches: Dict[pd.Timestamp, List[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)

        seq_len = int(self.seq_len)
        rel_len = int(self.rel_seq_len)
        for code, g in out.groupby("code", sort=False, observed=True):
            c = str(code)
            idxs = g.index.to_numpy()
            rows = g["_row"].to_numpy(dtype=int)
            feat = x_all[rows]
            rel_feat = rel_all[rows]
            hist = self._history_by_code.get(c)
            rel_hist = self._history_rel_by_code.get(c)
            if hist is None:
                hist = np.zeros((0, feat.shape[1]), dtype=np.float32)
            if rel_hist is None:
                rel_hist = np.zeros((0, rel_feat.shape[1]), dtype=np.float32)
            merged = np.vstack([hist, feat]) if len(hist) > 0 else feat
            rel_merged = np.vstack([rel_hist, rel_feat]) if len(rel_hist) > 0 else rel_feat
            hlen = len(hist)
            rhlen = len(rel_hist)
            times = pd.to_datetime(g["_model_time"], errors="coerce")
            for pos, row_idx in enumerate(idxs):
                end = hlen + pos
                if end >= seq_len - 1:
                    past = merged[end - seq_len + 1 : end + 1]
                else:
                    head = merged[: end + 1]
                    if len(head) == 0:
                        head = np.zeros((1, feat.shape[1]), dtype=np.float32)
                    pad = np.repeat(head[:1], repeats=max(0, seq_len - len(head)), axis=0)
                    past = np.vstack([pad, head])
                rend = rhlen + pos
                if rend >= rel_len - 1:
                    rel_past = rel_merged[rend - rel_len + 1 : rend + 1]
                else:
                    rel_head = rel_merged[: rend + 1]
                    if len(rel_head) == 0:
                        rel_head = np.zeros((1, rel_feat.shape[1]), dtype=np.float32)
                    rel_pad = np.repeat(rel_head[:1], repeats=max(0, rel_len - len(rel_head)), axis=0)
                    rel_past = np.vstack([rel_pad, rel_head])
                key = pd.Timestamp(times.loc[row_idx])
                if key == key.normalize():
                    key = key.normalize()
                batches[key].append((int(row_idx), past.astype(np.float32), rel_past.astype(np.float32)))
        return batches

    def predict_score(self, df: pd.DataFrame, factor_cols: list[str]) -> pd.Series:
        if self._model is None or self._fill_values is None or self._time_col is None:
            raise RuntimeError("StockFormerStockModel is not fitted.")
        if list(factor_cols) != self._factor_cols:
            missing = [c for c in self._factor_cols if c not in factor_cols]
            if missing:
                raise ValueError(f"predict factor cols missing: {missing}")

        torch, _nn, _F = self._require_torch()
        self._model.eval()
        keep_cols = list(dict.fromkeys(["code", self._time_col, *self._factor_cols, "industry_bucket", "board_type"]))
        out = df[[c for c in keep_cols if c in df.columns]].copy()
        out[self._factor_cols] = self._transform_feature_frame(out[self._factor_cols])
        rel_df = self._build_relation_frame(df=out, x=out[self._factor_cols])
        batches = self._build_predict_batches(out, rel_df=rel_df, factor_cols=self._factor_cols)
        raw_pred = pd.Series(np.nan, index=out.index, dtype=float)

        with torch.no_grad():
            for key in sorted(batches.keys()):
                rows = batches[key]
                if not rows:
                    continue
                row_ids = [r[0] for r in rows]
                past_np = np.stack([r[1] for r in rows], axis=0).astype(np.float32)
                rel_np = np.stack([r[2] for r in rows], axis=0).astype(np.float32)
                past = torch.tensor(past_np, dtype=torch.float32, device=self._device_used)
                rel = torch.tensor(rel_np, dtype=torch.float32, device=self._device_used)
                state = self._model.predictive(past, rel)["state"]
                weights = self._model.actor.sample(state, deterministic=True)[0].detach().cpu().numpy()
                raw_pred.loc[row_ids] = weights

        raw_pred = raw_pred * float(self._score_sign)
        anchor = self._time_anchor(out[self._time_col])
        score = raw_pred.groupby(anchor, sort=False, observed=True).rank(pct=True, method="average")
        score = score.fillna(0.5)
        return score.reindex(df.index).fillna(0.5).rename("pred_score")

    def fill_values(self) -> pd.Series:
        if self._fill_values is None:
            return pd.Series(dtype=float)
        return self._fill_values

    def save(self, folder: Path, run_tag: str) -> Dict[str, str]:
        if self._model is None:
            raise RuntimeError("StockFormerStockModel is not fitted.")
        torch, _nn, _F = self._require_torch()
        folder.mkdir(parents=True, exist_ok=True)
        model_path = folder / f"stock_model_stockformer_{run_tag}.pt"
        meta_path = folder / f"stock_model_stockformer_{run_tag}.json"
        config = {
            "seq_len": int(self.seq_len),
            "rel_seq_len": int(self.rel_seq_len),
            "hidden_size": int(self.hidden_size),
            "num_layers": int(self.num_layers),
            "num_heads": int(self.num_heads),
            "ffn_mult": int(self.ffn_mult),
            "dropout": float(self.dropout),
            "pretrain_epochs": int(self.pretrain_epochs),
            "sac_episodes": int(self.sac_episodes),
            "lr": float(self.lr),
            "sac_lr": float(self.sac_lr),
            "weight_decay": float(self.weight_decay),
            "gamma": float(self.gamma),
            "tau": float(self.tau),
            "init_alpha": float(self.init_alpha),
            "target_entropy_scale": float(self.target_entropy_scale),
            "early_stop": int(self.early_stop),
            "buffer_size": int(self.buffer_size),
            "learning_starts": int(self.learning_starts),
            "batch_transitions": int(self.batch_transitions),
            "updates_per_step": int(self.updates_per_step),
            "per_epoch_batch": int(self.per_epoch_batch),
            "batch_size": int(self.batch_size),
            "label_transform": str(self.label_transform),
            "input_clip": float(self.input_clip),
            "mse_weight": float(self.mse_weight),
            "ic_loss_weight": float(self.ic_loss_weight),
            "reward_cost_bps": float(self.reward_cost_bps),
            "turnover_penalty": float(self.turnover_penalty),
            "tracking_penalty": float(self.tracking_penalty),
            "min_cross_section": int(self.min_cross_section),
            "random_state": int(self.random_state),
            "device_used": self._device_used,
            "target_col": self._target_col,
            "score_sign": float(self._score_sign),
        }
        checkpoint = {
            "state_dict": self._model.state_dict(),
            "factor_cols": self._factor_cols,
            "fill_values": self._fill_values.to_dict() if self._fill_values is not None else {},
            "time_col": self._time_col,
            "train_summary": self._train_summary,
            "config": config,
        }
        torch.save(checkpoint, model_path)
        dump_json(
            meta_path,
            {
                "model_type": "stockformer",
                "target_col": self._target_col,
                "factor_count": len(self._factor_cols),
                "train_summary": self._train_summary,
                "config": config,
            },
        )
        return {"model_pt": str(model_path), "meta_json": str(meta_path)}
