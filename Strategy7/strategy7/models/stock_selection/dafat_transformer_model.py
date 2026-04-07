"""DAFAT stock-selection model.

DAFAT (Dynamic Adaptive Fusion Attention Transformer) reproduces the report's
three adaptive modules and integrates them into Strategy7:

1) Double-Gate dynamic positional encoding (DPE)
2) Triple sparse attention (volatility gating + local window + top-k)
3) Multi-scale feature fusion (micro/meso/macro)

This module is PyTorch-based and lazily imported, so non-DL stock models still
work without torch installed.
"""

from __future__ import annotations

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
    cycle_seq: np.ndarray
    state_seq: np.ndarray
    target: float


@dataclass
class DAFATStockModel(StockSelectionModel):
    """Transformer stock model with DPE + sparse attention + multi-scale fusion."""

    seq_len: int = 40
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.10

    local_window: int = 20
    topk_ratio: float = 0.30
    vol_quantile: float = 0.40

    meso_scale: int = 5
    macro_scale: int = 20

    n_epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-4
    early_stop: int = 20
    per_epoch_batch: int = 120
    batch_size: int = -1
    label_transform: str = "csranknorm"
    mse_weight: float = 0.05

    use_dpe: bool = True
    use_sparse_attn: bool = True
    use_multiscale: bool = True

    random_state: int = 42
    device: str = "auto"

    _model: Any = field(default=None, init=False, repr=False)
    _factor_cols: List[str] = field(default_factory=list, init=False, repr=False)
    _fill_values: pd.Series | None = field(default=None, init=False, repr=False)
    _time_col: str | None = field(default=None, init=False, repr=False)
    _history_by_code: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _history_time_by_code: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _market_state_lookup: Dict[pd.Timestamp, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _market_state_mean: pd.Series | None = field(default=None, init=False, repr=False)
    _market_state_std: pd.Series | None = field(default=None, init=False, repr=False)
    _state_default: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32), init=False, repr=False)
    _state_cols: List[str] = field(
        default_factory=lambda: ["market_vol", "industry_rotation", "market_liquidity"], init=False, repr=False
    )
    _device_used: str = field(default="cpu", init=False, repr=False)
    _target_col: str = field(default="target_return", init=False, repr=False)
    _train_summary: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _score_sign: float = field(default=1.0, init=False, repr=False)
    _input_dim: int = field(default=0, init=False, repr=False)

    @staticmethod
    def _require_torch() -> tuple[Any, Any, Any]:
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except Exception as exc:
            raise RuntimeError(
                "DAFAT requires PyTorch. Please install torch first, for example: pip install torch"
            ) from exc
        return torch, nn, F

    @staticmethod
    def _resolve_time_col(df: pd.DataFrame) -> str:
        for c in ["signal_ts", "datetime", "date"]:
            if c in df.columns:
                return c
        raise ValueError("DAFAT requires one of ['signal_ts', 'datetime', 'date'] columns.")

    @staticmethod
    def _time_anchor(ts: pd.Series) -> pd.Series:
        dt = pd.to_datetime(ts, errors="coerce")
        return dt.dt.normalize()

    @staticmethod
    def _zscore(x: pd.Series) -> pd.Series:
        v = pd.to_numeric(x, errors="coerce")
        std = float(v.std(ddof=0)) if v.notna().sum() > 1 else 0.0
        if std <= EPS:
            return pd.Series(np.zeros(len(v), dtype=float), index=v.index)
        return (v - float(v.mean())) / (std + EPS)

    def _build_target(self, df: pd.DataFrame, target_col: str, anchor: pd.Series) -> pd.Series:
        base_col = target_col
        y_raw = pd.to_numeric(df[target_col], errors="coerce") if target_col in df.columns else pd.Series(np.nan, index=df.index)
        uniq = set(pd.Series(y_raw.dropna().unique()).tolist())
        if uniq.issubset({0, 1}) and "target_return" in df.columns:
            base_col = "target_return"
            y_raw = pd.to_numeric(df[base_col], errors="coerce")

        self._target_col = base_col
        if self.label_transform == "raw":
            return y_raw
        if self.label_transform == "csrank":
            return y_raw.groupby(anchor).transform(lambda s: s.rank(pct=True, method="average"))
        if self.label_transform == "cszscore":
            return y_raw.groupby(anchor).transform(self._zscore)

        ranked = y_raw.groupby(anchor).transform(lambda s: s.rank(pct=True, method="average"))
        ranked = ranked.replace([np.inf, -np.inf], np.nan)
        return ranked.groupby(anchor).transform(self._zscore)

    @staticmethod
    def _pick_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _build_market_state_frame(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        out = df.copy()
        out["_state_time"] = pd.to_datetime(out[time_col], errors="coerce").dt.normalize()
        out = out.dropna(subset=["_state_time"]).copy()
        if out.empty:
            return pd.DataFrame(columns=self._state_cols, dtype=float)

        idx = pd.DatetimeIndex(sorted(out["_state_time"].drop_duplicates().tolist()))
        state = pd.DataFrame(index=idx, columns=self._state_cols, dtype=float)

        vol_col = self._pick_first_existing_column(
            out,
            ["realized_vol_20", "rv_12", "target_volatility", "ret_1d", "target_return", "future_ret_n"],
        )
        if vol_col is not None:
            vol_raw = pd.to_numeric(out[vol_col], errors="coerce").abs()
            state["market_vol"] = vol_raw.groupby(out["_state_time"]).median().reindex(idx)
        else:
            state["market_vol"] = 0.0

        liq_col = self._pick_first_existing_column(out, ["turn", "turn_ratio_5", "amount_ratio_20", "amount", "volume"])
        if liq_col is not None:
            liq_raw = pd.to_numeric(out[liq_col], errors="coerce")
            state["market_liquidity"] = liq_raw.groupby(out["_state_time"]).median().reindex(idx)
        else:
            state["market_liquidity"] = 0.0

        ret_col = self._pick_first_existing_column(out, ["target_return", "future_ret_n", "ret_1d"])
        if "industry_bucket" in out.columns and ret_col is not None:
            tmp = out[["_state_time", "industry_bucket", ret_col]].copy()
            tmp[ret_col] = pd.to_numeric(tmp[ret_col], errors="coerce")
            tmp = tmp.dropna(subset=["_state_time", "industry_bucket", ret_col])
            if not tmp.empty:
                ind = tmp.groupby(["_state_time", "industry_bucket"], as_index=False)[ret_col].mean()
                ind["ind_rank"] = ind.groupby("_state_time")[ret_col].rank(pct=True, method="average")
                pivot = ind.pivot(index="_state_time", columns="industry_bucket", values="ind_rank").sort_index()
                rot = pivot.diff().abs().mean(axis=1)
                state["industry_rotation"] = rot.reindex(idx)
            else:
                state["industry_rotation"] = np.nan
        elif ret_col is not None:
            ret_raw = pd.to_numeric(out[ret_col], errors="coerce")
            state["industry_rotation"] = ret_raw.groupby(out["_state_time"]).std().reindex(idx)
        else:
            state["industry_rotation"] = 0.0

        state = state.replace([np.inf, -np.inf], np.nan).sort_index()
        state = state.ffill().bfill().fillna(0.0)
        return state

    def _fit_market_state(self, train_df: pd.DataFrame) -> Dict[pd.Timestamp, np.ndarray]:
        state = self._build_market_state_frame(train_df, time_col=self._time_col)
        if state.empty:
            self._market_state_mean = pd.Series([0.0, 0.0, 0.0], index=self._state_cols, dtype=float)
            self._market_state_std = pd.Series([1.0, 1.0, 1.0], index=self._state_cols, dtype=float)
            self._market_state_lookup = {}
            self._state_default = np.zeros(3, dtype=np.float32)
            return {}

        mean = state.mean(axis=0)
        std = state.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        norm = (state - mean) / std
        norm = norm.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self._market_state_mean = mean
        self._market_state_std = std
        self._market_state_lookup = {
            pd.Timestamp(k): norm.loc[k, self._state_cols].to_numpy(dtype=np.float32) for k in norm.index
        }
        self._state_default = np.zeros(3, dtype=np.float32)
        return dict(self._market_state_lookup)

    def _normalize_state_frame(self, state: pd.DataFrame) -> pd.DataFrame:
        if state.empty:
            return state
        if self._market_state_mean is None or self._market_state_std is None:
            return state.fillna(0.0)
        mean = self._market_state_mean.reindex(self._state_cols).fillna(0.0)
        std = self._market_state_std.reindex(self._state_cols).replace(0.0, 1.0).fillna(1.0)
        out = (state.reindex(columns=self._state_cols) - mean) / std
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _calendar_features_from_times(times: np.ndarray) -> np.ndarray:
        ts = pd.to_datetime(pd.Series(times), errors="coerce")
        weekday = ts.dt.weekday.fillna(0).to_numpy(dtype=np.float32)
        month = ts.dt.month.fillna(1).to_numpy(dtype=np.float32) - 1.0
        quarter = ts.dt.quarter.fillna(1).to_numpy(dtype=np.float32) - 1.0

        week_sin = np.sin(2.0 * np.pi * weekday / 5.0)
        week_cos = np.cos(2.0 * np.pi * weekday / 5.0)
        month_sin = np.sin(2.0 * np.pi * month / 12.0)
        month_cos = np.cos(2.0 * np.pi * month / 12.0)
        quarter_sin = np.sin(2.0 * np.pi * quarter / 4.0)
        quarter_cos = np.cos(2.0 * np.pi * quarter / 4.0)

        feat = np.stack([week_sin, week_cos, month_sin, month_cos, quarter_sin, quarter_cos], axis=1)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        return feat.astype(np.float32)

    def _state_features_from_times(self, times: np.ndarray, state_lookup: Dict[pd.Timestamp, np.ndarray]) -> np.ndarray:
        anchors = pd.to_datetime(pd.Series(times), errors="coerce").dt.normalize()
        out = np.zeros((len(times), len(self._state_cols)), dtype=np.float32)
        for i, t in enumerate(anchors):
            if pd.isna(t):
                out[i] = self._state_default
                continue
            out[i] = state_lookup.get(pd.Timestamp(t), self._state_default)
        return out

    def _build_train_samples(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        target: pd.Series,
        state_lookup: Dict[pd.Timestamp, np.ndarray],
    ) -> Dict[pd.Timestamp, List[_TrainSample]]:
        out = df.copy()
        out["_model_time"] = pd.to_datetime(out[self._time_col], errors="coerce")
        out["_target"] = pd.to_numeric(target, errors="coerce")
        out = out.dropna(subset=["code", "_model_time"]).sort_values(["code", "_model_time"])

        grouped: Dict[pd.Timestamp, List[_TrainSample]] = defaultdict(list)
        self._history_by_code = {}
        self._history_time_by_code = {}

        for code, g in out.groupby("code"):
            g = g.sort_values("_model_time")
            x = g[factor_cols].to_numpy(dtype=np.float32)
            y = g["_target"].to_numpy(dtype=np.float32)
            t = pd.to_datetime(g["_model_time"], errors="coerce")
            n = len(g)
            if n == 0:
                continue

            hist_start = max(0, n - (int(self.seq_len) - 1))
            self._history_by_code[str(code)] = x[hist_start:].copy()
            self._history_time_by_code[str(code)] = t.iloc[hist_start:].to_numpy(dtype="datetime64[ns]")

            start = int(self.seq_len) - 1
            if n <= start:
                continue
            for i in range(start, n):
                if np.isnan(y[i]):
                    continue
                past = x[i - int(self.seq_len) + 1 : i + 1]
                if past.shape[0] != int(self.seq_len):
                    continue
                if np.isnan(past).any():
                    continue

                seq_times = t.iloc[i - int(self.seq_len) + 1 : i + 1].to_numpy(dtype="datetime64[ns]")
                cycle_seq = self._calendar_features_from_times(seq_times)
                state_seq = self._state_features_from_times(seq_times, state_lookup=state_lookup)

                key = pd.Timestamp(t.iloc[i]).normalize()
                grouped[key].append(
                    _TrainSample(
                        code=str(code),
                        time_key=key,
                        past_seq=past,
                        cycle_seq=cycle_seq,
                        state_seq=state_seq,
                        target=float(y[i]),
                    )
                )
        return grouped

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

    @staticmethod
    def _stack_batch(samples: Iterable[_TrainSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ss = list(samples)
        past = np.stack([s.past_seq for s in ss], axis=0).astype(np.float32)
        cycle = np.stack([s.cycle_seq for s in ss], axis=0).astype(np.float32)
        state = np.stack([s.state_seq for s in ss], axis=0).astype(np.float32)
        y = np.asarray([s.target for s in ss], dtype=np.float32)
        return past, cycle, state, y

    def _build_network(self, input_dim: int, torch: Any, nn: Any, F: Any) -> Any:
        hidden_size = int(self.hidden_size)
        num_layers = int(self.num_layers)
        num_heads = int(self.num_heads)
        ffn_mult = int(self.ffn_mult)
        dropout = float(self.dropout)
        local_window = int(self.local_window)
        topk_ratio = float(self.topk_ratio)
        vol_quantile = float(self.vol_quantile)
        meso_scale = int(self.meso_scale)
        macro_scale = int(self.macro_scale)
        use_sparse_attn = bool(self.use_sparse_attn)
        use_multiscale = bool(self.use_multiscale)
        use_dpe = bool(self.use_dpe)

        class DynamicPositionalEncoding(nn.Module):
            def __init__(self, hidden_dim: int) -> None:
                super().__init__()
                self.cycle_mlp = nn.Sequential(
                    nn.Linear(6, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.state_gru = nn.GRU(input_size=3, hidden_size=hidden_dim, batch_first=True)
                self.state_mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                )
                self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

            def forward(self, cycle_feat: Any, state_feat: Any) -> Any:
                cyc = self.cycle_mlp(cycle_feat)
                state_h, _ = self.state_gru(state_feat)
                st = self.state_mlp(state_h)
                g = torch.sigmoid(self.gate(torch.cat([cyc, st], dim=-1)))
                return g * cyc + (1.0 - g) * st

        class SparseSelfAttention(nn.Module):
            def __init__(
                self,
                hidden_dim: int,
                n_heads: int,
                attn_dropout: float,
                lw: int,
                tk_ratio: float,
                vq: float,
            ) -> None:
                super().__init__()
                if hidden_dim % n_heads != 0:
                    raise ValueError(f"hidden_size={hidden_dim} must be divisible by num_heads={n_heads}.")
                self.num_heads = n_heads
                self.head_dim = hidden_dim // n_heads
                self.scale = float(self.head_dim) ** 0.5
                self.local_window = max(1, lw)
                self.topk_ratio = float(np.clip(tk_ratio, 0.05, 1.0))
                self.vol_quantile = float(np.clip(vq, 0.0, 1.0))

                self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
                self.attn_drop = nn.Dropout(attn_dropout)
                self.out_proj = nn.Linear(hidden_dim, hidden_dim)
                self.out_drop = nn.Dropout(attn_dropout)

            def _quantile(self, vol: Any) -> Any:
                if hasattr(torch, "quantile"):
                    return torch.quantile(vol, q=self.vol_quantile, dim=1, keepdim=True)
                sorted_v, _ = torch.sort(vol, dim=1)
                idx = int(round((vol.shape[1] - 1) * self.vol_quantile))
                idx = max(0, min(vol.shape[1] - 1, idx))
                return sorted_v[:, idx : idx + 1]

            def forward(self, x: Any, vol_seq: Any) -> Any:
                # x: [B, L, H], vol_seq: [B, L]
                bsz, seqlen, _ = x.shape
                qkv = self.qkv(x).reshape(bsz, seqlen, 3, self.num_heads, self.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

                # (1) Local window mask (short-term dependency prior).
                idx = torch.arange(seqlen, device=x.device)
                local_mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= self.local_window
                scores = scores.masked_fill(~local_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

                # (2) Volatility gate: keep only high-information periods.
                vol_seq = torch.nan_to_num(vol_seq, nan=0.0)
                vol_threshold = self._quantile(vol_seq)
                key_keep = vol_seq >= vol_threshold

                min_keep = max(1, int(np.ceil(seqlen * self.topk_ratio)))
                top_vol_idx = torch.topk(vol_seq, k=min(min_keep, seqlen), dim=1).indices
                key_keep = key_keep.clone()
                key_keep.scatter_(1, top_vol_idx, True)
                scores = scores.masked_fill(~key_keep[:, None, None, :], float("-inf"))

                # (3) Top-k sparse selection per query row.
                k_keep = max(1, int(np.ceil(seqlen * self.topk_ratio)))
                k_keep = min(k_keep, seqlen)
                top_vals, top_idx = torch.topk(scores, k=k_keep, dim=-1)
                sparse_scores = torch.full_like(scores, float("-inf"))
                sparse_scores.scatter_(-1, top_idx, top_vals)
                has_valid = torch.isfinite(scores).any(dim=-1, keepdim=True)
                scores = torch.where(has_valid, sparse_scores, scores)

                attn = torch.softmax(scores, dim=-1)
                attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
                attn = self.attn_drop(attn)

                out = torch.matmul(attn, v)
                out = out.permute(0, 2, 1, 3).reshape(bsz, seqlen, -1)
                out = self.out_proj(out)
                return self.out_drop(out)

        class DenseSelfAttention(nn.Module):
            def __init__(self, hidden_dim: int, n_heads: int, attn_dropout: float) -> None:
                super().__init__()
                self.attn = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=n_heads,
                    dropout=attn_dropout,
                    batch_first=True,
                )
                self.out_drop = nn.Dropout(attn_dropout)

            def forward(self, x: Any, vol_seq: Any) -> Any:
                out, _ = self.attn(x, x, x, need_weights=False)
                return self.out_drop(out)

        class TransformerBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm1 = nn.LayerNorm(hidden_size)
                if use_sparse_attn:
                    self.attn = SparseSelfAttention(
                        hidden_dim=hidden_size,
                        n_heads=num_heads,
                        attn_dropout=dropout,
                        lw=local_window,
                        tk_ratio=topk_ratio,
                        vq=vol_quantile,
                    )
                else:
                    self.attn = DenseSelfAttention(hidden_dim=hidden_size, n_heads=num_heads, attn_dropout=dropout)
                self.norm2 = nn.LayerNorm(hidden_size)
                self.ffn = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * ffn_mult),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size * ffn_mult, hidden_size),
                )
                self.drop = nn.Dropout(dropout)

            def forward(self, x: Any, vol_seq: Any) -> Any:
                x = x + self.drop(self.attn(self.norm1(x), vol_seq))
                x = x + self.drop(self.ffn(self.norm2(x)))
                return x

        class MultiScaleFusion(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.meso_scale = max(1, meso_scale)
                self.macro_scale = max(1, macro_scale)
                self.attn_micro = nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self.attn_meso = nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self.attn_macro = nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self.gate_net = nn.Sequential(
                    nn.Linear(hidden_size * 3, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, 3),
                )
                self.out_proj = nn.Linear(hidden_size, hidden_size)

            @staticmethod
            def _pool_and_upsample(x: Any, scale: int) -> Any:
                if scale <= 1 or x.shape[1] <= 1:
                    return x
                bsz, seqlen, hdim = x.shape
                x_t = x.transpose(1, 2)
                pooled = F.avg_pool1d(x_t, kernel_size=scale, stride=scale, ceil_mode=True)
                up = F.interpolate(pooled, size=seqlen, mode="linear", align_corners=False)
                return up.transpose(1, 2).reshape(bsz, seqlen, hdim)

            def forward(self, x: Any) -> tuple[Any, Any]:
                micro = x
                meso = self._pool_and_upsample(x, self.meso_scale)
                macro = self._pool_and_upsample(x, self.macro_scale)

                ctx_micro, _ = self.attn_micro(micro, micro, micro, need_weights=False)
                ctx_meso, _ = self.attn_meso(micro, meso, meso, need_weights=False)
                ctx_macro, _ = self.attn_macro(micro, macro, macro, need_weights=False)

                # Gating is generated from per-sample global context.
                gate_input = torch.cat(
                    [ctx_micro.mean(dim=1), ctx_meso.mean(dim=1), ctx_macro.mean(dim=1)],
                    dim=-1,
                )
                gate = torch.softmax(self.gate_net(gate_input), dim=-1)

                fused = (
                    gate[:, 0:1, None] * ctx_micro
                    + gate[:, 1:2, None] * ctx_meso
                    + gate[:, 2:3, None] * ctx_macro
                )
                return self.out_proj(fused), gate

        class DAFATNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.use_dpe = use_dpe
                self.use_multiscale = use_multiscale
                self.input_proj = nn.Linear(input_dim, hidden_size)
                self.input_norm = nn.LayerNorm(hidden_size)
                self.drop = nn.Dropout(dropout)

                self.pos_cycle_only = nn.Linear(6, hidden_size)
                self.dpe = DynamicPositionalEncoding(hidden_dim=hidden_size)
                self.blocks = nn.ModuleList([TransformerBlock() for _ in range(max(1, num_layers))])
                self.multi_scale = MultiScaleFusion()
                self.final_norm = nn.LayerNorm(hidden_size)
                self.head = nn.Linear(hidden_size, 1)

            def forward_train(self, x: Any, cycle_feat: Any, state_feat: Any, vol_seq: Any) -> Any:
                h = self.input_norm(self.input_proj(x))
                if self.use_dpe:
                    h = h + self.dpe(cycle_feat, state_feat)
                else:
                    h = h + self.pos_cycle_only(cycle_feat)
                h = self.drop(h)

                if self.use_multiscale:
                    ms, _ = self.multi_scale(h)
                    h = h + self.drop(ms)

                for blk in self.blocks:
                    h = blk(h, vol_seq)

                if self.use_multiscale and len(self.blocks) > 1:
                    ms, _ = self.multi_scale(h)
                    h = h + self.drop(ms)

                h = self.final_norm(h)
                pooled = 0.5 * (h[:, -1, :] + h.mean(dim=1))
                pred = self.head(pooled).squeeze(-1)
                return pred

            def predict_raw(self, x: Any, cycle_feat: Any, state_feat: Any, vol_seq: Any) -> Any:
                return self.forward_train(x, cycle_feat, state_feat, vol_seq)

        return DAFATNet()

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

    def _evaluate_ic(self, net: Any, groups: Dict[pd.Timestamp, List[_TrainSample]], torch: Any, device: str) -> float:
        if not groups:
            return float("nan")
        net.eval()
        ics: List[float] = []
        with torch.no_grad():
            for key in sorted(groups.keys()):
                samples = groups[key]
                if len(samples) < 6:
                    continue
                past, cycle, state, target = self._stack_batch(samples)
                x = torch.tensor(past, dtype=torch.float32, device=device)
                cyc = torch.tensor(cycle, dtype=torch.float32, device=device)
                st = torch.tensor(state, dtype=torch.float32, device=device)
                vol = st[:, :, 0]
                pred = net.predict_raw(x, cyc, st, vol).detach().cpu().numpy()
                ic = self._rank_ic(target, pred)
                if np.isfinite(ic):
                    ics.append(float(ic))
        if not ics:
            return float("nan")
        return float(np.mean(ics))

    @staticmethod
    def _copy_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v.detach().cpu().clone() for k, v in state_dict.items()}

    @staticmethod
    def _average_state_dicts(state_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not state_dicts:
            raise ValueError("state_dicts cannot be empty")
        avg: Dict[str, Any] = {}
        keys = list(state_dicts[0].keys())
        for k in keys:
            tensors = [sd[k] for sd in state_dicts]
            stacked = tensors[0].clone()
            for t in tensors[1:]:
                stacked = stacked + t
            avg[k] = stacked / float(len(tensors))
        return avg

    def fit(self, train_df: pd.DataFrame, factor_cols: list[str], target_col: str) -> "DAFATStockModel":
        torch, nn, F = self._require_torch()
        self._set_seed(torch)

        self._factor_cols = list(factor_cols)
        self._time_col = self._resolve_time_col(train_df)

        x = train_df[self._factor_cols].replace([np.inf, -np.inf], np.nan)
        self._fill_values = x.median(numeric_only=True)

        df = train_df.copy()
        df[self._factor_cols] = x.fillna(self._fill_values)
        time_anchor = self._time_anchor(df[self._time_col])
        target = self._build_target(df, target_col=target_col, anchor=time_anchor)

        # Fit market-state normalization and build date -> state lookup.
        state_lookup = self._fit_market_state(df)

        grouped = self._build_train_samples(df=df, factor_cols=self._factor_cols, target=target, state_lookup=state_lookup)
        grouped = {k: v for k, v in grouped.items() if len(v) >= 8}
        if not grouped:
            raise RuntimeError("DAFAT valid training samples are empty after sequence construction.")

        keys = sorted(grouped.keys())
        split = max(1, int(len(keys) * 0.8))
        train_keys = keys[:split]
        val_keys = keys[split:]
        if not val_keys:
            val_keys = keys[-1:]

        train_groups = {k: grouped[k] for k in train_keys}
        val_groups = {k: grouped[k] for k in val_keys}

        input_dim = len(self._factor_cols)
        self._input_dim = input_dim
        net = self._build_network(input_dim=input_dim, torch=torch, nn=nn, F=F)

        device = self._choose_device(torch)
        self._device_used = device
        net.to(device)

        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )

        best_ic = -np.inf
        best_state: Dict[str, Any] | None = None
        best_epoch = -1
        best_states: deque[Dict[str, Any]] = deque(maxlen=5)
        patience = 0

        rng = np.random.default_rng(int(self.random_state))
        train_losses: List[float] = []
        train_ic_losses: List[float] = []
        train_mse_losses: List[float] = []
        val_ics: List[float] = []

        for epoch in range(1, int(self.n_epochs) + 1):
            net.train()
            losses_epoch: List[float] = []
            ic_losses_epoch: List[float] = []
            mse_losses_epoch: List[float] = []

            epoch_keys = list(train_keys)
            if int(self.per_epoch_batch) > 0 and len(epoch_keys) > 0:
                n_pick = int(self.per_epoch_batch)
                if len(epoch_keys) <= n_pick:
                    sampled_keys = list(epoch_keys)
                else:
                    sampled_keys = rng.choice(epoch_keys, size=n_pick, replace=False).tolist()
            else:
                sampled_keys = epoch_keys

            # Training is done by date-batches (cross-sectional slices) to optimize IC.
            for key in sampled_keys:
                samples = train_groups.get(key, [])
                if len(samples) < 8:
                    continue

                if int(self.batch_size) > 0 and len(samples) > int(self.batch_size):
                    idx = rng.choice(len(samples), size=int(self.batch_size), replace=False)
                    batch = [samples[int(i)] for i in idx]
                else:
                    batch = samples

                past_np, cycle_np, state_np, y_np = self._stack_batch(batch)
                x_past = torch.tensor(past_np, dtype=torch.float32, device=device)
                x_cycle = torch.tensor(cycle_np, dtype=torch.float32, device=device)
                x_state = torch.tensor(state_np, dtype=torch.float32, device=device)
                y_true = torch.tensor(y_np, dtype=torch.float32, device=device)
                vol_seq = x_state[:, :, 0]

                pred = net.forward_train(x_past, x_cycle, x_state, vol_seq)
                loss_ic = self._ic_loss(pred, y_true, torch=torch)
                loss_mse = F.mse_loss(pred, y_true)
                loss = loss_ic + float(self.mse_weight) * loss_mse

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                optimizer.step()

                losses_epoch.append(float(loss.detach().cpu().item()))
                ic_losses_epoch.append(float(loss_ic.detach().cpu().item()))
                mse_losses_epoch.append(float(loss_mse.detach().cpu().item()))

            train_losses.append(float(np.mean(losses_epoch)) if losses_epoch else float("nan"))
            train_ic_losses.append(float(np.mean(ic_losses_epoch)) if ic_losses_epoch else float("nan"))
            train_mse_losses.append(float(np.mean(mse_losses_epoch)) if mse_losses_epoch else float("nan"))

            val_ic = self._evaluate_ic(net, groups=val_groups, torch=torch, device=device)
            val_ics.append(val_ic)

            improved = np.isfinite(val_ic) and (val_ic > best_ic + 1e-8)
            if improved:
                best_ic = float(val_ic)
                best_state = self._copy_state_dict(net.state_dict())
                best_states.append(best_state)
                best_epoch = epoch
                patience = 0
            else:
                patience += 1

            if patience >= int(self.early_stop):
                break

        if best_state is None:
            best_state = self._copy_state_dict(net.state_dict())
            best_states.append(best_state)
            best_epoch = len(train_losses)
            best_ic = self._evaluate_ic(net, groups=val_groups, torch=torch, device=device)

        final_state = self._average_state_dicts(list(best_states)) if len(best_states) > 1 else best_state
        net.load_state_dict(final_state)
        final_val_ic = self._evaluate_ic(net, groups=val_groups, torch=torch, device=device)
        sign_ref_ic = final_val_ic if np.isfinite(final_val_ic) else best_ic
        self._score_sign = -1.0 if np.isfinite(sign_ref_ic) and float(sign_ref_ic) < 0.0 else 1.0

        self._model = net
        self._train_summary = {
            "best_val_rank_ic": float(best_ic) if np.isfinite(best_ic) else float("nan"),
            "final_val_rank_ic": float(final_val_ic) if np.isfinite(final_val_ic) else float("nan"),
            "best_epoch": float(best_epoch),
            "epochs_trained": float(len(train_losses)),
            "avg_train_loss_last": float(train_losses[-1]) if train_losses else float("nan"),
            "avg_train_ic_loss_last": float(train_ic_losses[-1]) if train_ic_losses else float("nan"),
            "avg_train_mse_loss_last": float(train_mse_losses[-1]) if train_mse_losses else float("nan"),
            "val_rank_ic_last": float(val_ics[-1]) if val_ics else float("nan"),
            "score_sign": float(self._score_sign),
            "score_sign_ref_ic": float(sign_ref_ic) if np.isfinite(sign_ref_ic) else float("nan"),
            "device": self._device_used,
            "module_dpe_enabled": float(1.0 if self.use_dpe else 0.0),
            "module_sparse_enabled": float(1.0 if self.use_sparse_attn else 0.0),
            "module_multiscale_enabled": float(1.0 if self.use_multiscale else 0.0),
        }
        return self

    def _build_predict_batches(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
    ) -> Dict[pd.Timestamp, List[tuple[int, np.ndarray, np.ndarray]]]:
        out = df.copy()
        out["_model_time"] = pd.to_datetime(out[self._time_col], errors="coerce")
        out = out.dropna(subset=["code", "_model_time"]).copy()
        out["code"] = out["code"].astype(str)
        out = out.sort_values(["code", "_model_time"])

        batches: Dict[pd.Timestamp, List[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)

        for code, g in out.groupby("code"):
            c = str(code)
            feat = g[factor_cols].to_numpy(dtype=np.float32)
            times = pd.to_datetime(g["_model_time"], errors="coerce").to_numpy(dtype="datetime64[ns]")

            hist_feat = self._history_by_code.get(c)
            if hist_feat is None:
                hist_feat = np.zeros((0, feat.shape[1]), dtype=np.float32)
            hist_time = self._history_time_by_code.get(c)
            if hist_time is None:
                hist_time = np.array([], dtype="datetime64[ns]")

            merged_feat = np.vstack([hist_feat, feat]) if len(hist_feat) > 0 else feat
            merged_time = np.concatenate([hist_time, times]) if len(hist_time) > 0 else times
            hlen = len(hist_feat)
            idxs = g.index.to_numpy()

            for pos, row_idx in enumerate(idxs):
                end = hlen + pos
                if end >= int(self.seq_len) - 1:
                    past_feat = merged_feat[end - int(self.seq_len) + 1 : end + 1]
                    past_time = merged_time[end - int(self.seq_len) + 1 : end + 1]
                else:
                    head_feat = merged_feat[: end + 1]
                    head_time = merged_time[: end + 1]
                    if len(head_feat) == 0:
                        head_feat = np.zeros((1, feat.shape[1]), dtype=np.float32)
                        head_time = np.array([np.datetime64("1970-01-01")], dtype="datetime64[ns]")
                    need = int(self.seq_len) - len(head_feat)
                    pad_feat = np.repeat(head_feat[:1], repeats=max(0, need), axis=0)
                    pad_time = np.repeat(head_time[:1], repeats=max(0, need), axis=0)
                    past_feat = np.vstack([pad_feat, head_feat])
                    past_time = np.concatenate([pad_time, head_time])

                cur_time = pd.Timestamp(times[pos]).normalize()
                batches[cur_time].append((int(row_idx), past_feat.astype(np.float32), past_time.astype("datetime64[ns]")))
        return batches

    def _build_aux_inputs_for_batch(
        self,
        past_times_batch: List[np.ndarray],
        state_lookup: Dict[pd.Timestamp, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        cycle_list: List[np.ndarray] = []
        state_list: List[np.ndarray] = []
        for t in past_times_batch:
            cycle_list.append(self._calendar_features_from_times(t))
            state_list.append(self._state_features_from_times(t, state_lookup=state_lookup))
        return (
            np.stack(cycle_list, axis=0).astype(np.float32),
            np.stack(state_list, axis=0).astype(np.float32),
        )

    def predict_score(self, df: pd.DataFrame, factor_cols: list[str]) -> pd.Series:
        if self._model is None or self._fill_values is None or self._time_col is None:
            raise RuntimeError("DAFATStockModel is not fitted.")
        if list(factor_cols) != self._factor_cols:
            missing = [c for c in self._factor_cols if c not in factor_cols]
            if missing:
                raise ValueError(f"predict factor cols missing: {missing}")

        torch, _nn, _F = self._require_torch()
        self._model.eval()

        out = df.copy()
        out[self._factor_cols] = (
            out[self._factor_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(self._fill_values.reindex(self._factor_cols))
            .fillna(0.0)
        )

        # Prediction-time market state uses current panel statistics, then applies
        # train-fitted normalization to stay on the same scale as training.
        pred_state_raw = self._build_market_state_frame(out, time_col=self._time_col)
        pred_state_norm = self._normalize_state_frame(pred_state_raw)
        state_lookup = dict(self._market_state_lookup)
        for k in pred_state_norm.index:
            state_lookup[pd.Timestamp(k)] = pred_state_norm.loc[k, self._state_cols].to_numpy(dtype=np.float32)

        batches = self._build_predict_batches(out, factor_cols=self._factor_cols)
        raw_pred = pd.Series(np.nan, index=out.index, dtype=float)

        with torch.no_grad():
            for key in sorted(batches.keys()):
                rows = batches[key]
                if not rows:
                    continue
                row_ids = [r[0] for r in rows]
                past_np = np.stack([r[1] for r in rows], axis=0).astype(np.float32)
                past_times = [r[2] for r in rows]
                cycle_np, state_np = self._build_aux_inputs_for_batch(past_times, state_lookup=state_lookup)

                x_past = torch.tensor(past_np, dtype=torch.float32, device=self._device_used)
                x_cycle = torch.tensor(cycle_np, dtype=torch.float32, device=self._device_used)
                x_state = torch.tensor(state_np, dtype=torch.float32, device=self._device_used)
                vol_seq = x_state[:, :, 0]
                pred = self._model.predict_raw(x_past, x_cycle, x_state, vol_seq).detach().cpu().numpy()
                raw_pred.loc[row_ids] = pred

        raw_pred = raw_pred * float(self._score_sign)
        anchor = self._time_anchor(out[self._time_col])
        score = raw_pred.groupby(anchor).rank(pct=True, method="average")
        score = score.fillna(0.5)
        score = score.reindex(df.index).fillna(0.5)
        return score.rename("pred_score")

    def fill_values(self) -> pd.Series:
        if self._fill_values is None:
            return pd.Series(dtype=float)
        return self._fill_values

    def save(self, folder: Path, run_tag: str) -> Dict[str, str]:
        if self._model is None:
            raise RuntimeError("DAFATStockModel is not fitted.")
        torch, _nn, _F = self._require_torch()

        folder.mkdir(parents=True, exist_ok=True)
        model_path = folder / f"stock_model_dafat_{run_tag}.pt"
        meta_path = folder / f"stock_model_dafat_{run_tag}.json"

        checkpoint = {
            "state_dict": self._model.state_dict(),
            "factor_cols": self._factor_cols,
            "fill_values": self._fill_values.to_dict() if self._fill_values is not None else {},
            "time_col": self._time_col,
            "train_summary": self._train_summary,
            "market_state_lookup": {str(k.date()): v.tolist() for k, v in self._market_state_lookup.items()},
            "market_state_mean": self._market_state_mean.to_dict() if self._market_state_mean is not None else {},
            "market_state_std": self._market_state_std.to_dict() if self._market_state_std is not None else {},
            "config": {
                "seq_len": int(self.seq_len),
                "hidden_size": int(self.hidden_size),
                "num_layers": int(self.num_layers),
                "num_heads": int(self.num_heads),
                "ffn_mult": int(self.ffn_mult),
                "dropout": float(self.dropout),
                "local_window": int(self.local_window),
                "topk_ratio": float(self.topk_ratio),
                "vol_quantile": float(self.vol_quantile),
                "meso_scale": int(self.meso_scale),
                "macro_scale": int(self.macro_scale),
                "n_epochs": int(self.n_epochs),
                "lr": float(self.lr),
                "weight_decay": float(self.weight_decay),
                "early_stop": int(self.early_stop),
                "per_epoch_batch": int(self.per_epoch_batch),
                "batch_size": int(self.batch_size),
                "label_transform": str(self.label_transform),
                "mse_weight": float(self.mse_weight),
                "use_dpe": bool(self.use_dpe),
                "use_sparse_attn": bool(self.use_sparse_attn),
                "use_multiscale": bool(self.use_multiscale),
                "random_state": int(self.random_state),
                "device_used": self._device_used,
                "target_col": self._target_col,
                "score_sign": float(self._score_sign),
            },
        }
        torch.save(checkpoint, model_path)

        dump_json(
            meta_path,
            {
                "model_type": "dafat",
                "target_col": self._target_col,
                "factor_count": len(self._factor_cols),
                "state_dim": len(self._state_cols),
                "train_summary": self._train_summary,
                "config": checkpoint["config"],
            },
        )
        return {"model_pt": str(model_path), "meta_json": str(meta_path)}
