"""DFQ-TimesNet stock-selection model.

This implementation follows the Oriental Securities DFQ-TimesNet report:
TokenEmbedding only, fixed multi-period folding, two Inception blocks, direct
mean period fusion, residual connection, and last-step projection.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

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
    target: float


def _parse_periods(raw: str | Sequence[int]) -> List[int]:
    if isinstance(raw, str):
        text = raw.replace("，", ",").replace(";", ",")
        vals = [x.strip() for x in text.split(",") if x.strip()]
        periods = [int(float(x)) for x in vals]
    else:
        periods = [int(x) for x in raw]
    out: List[int] = []
    for p in periods:
        if p <= 0:
            continue
        if p not in out:
            out.append(p)
    return out or [5, 60]


@dataclass
class DFQTimesNetStockModel(StockSelectionModel):
    """TimesNet-style two-dimensional period folding model for stock ranking."""

    seq_len: int = 60
    hidden_size: int = 128
    e_layers: int = 1
    hidden_size2: int = 128
    periods: str | Sequence[int] = "5,60"
    num_kernels: int = 3
    dropout: float = 0.0

    n_epochs: int = 200
    lr: float = 9e-5
    weight_decay: float = 0.0
    early_stop: int = 20
    smooth_steps: int = 5
    per_epoch_batch: int = 100
    batch_size: int = -1
    label_transform: str = "cszscore"
    input_clip: float = 3.0
    mse_weight: float = 1.0
    ic_loss_weight: float = 0.0

    random_state: int = 1000
    device: str = "auto"

    _model: Any = field(default=None, init=False, repr=False)
    _factor_cols: List[str] = field(default_factory=list, init=False, repr=False)
    _fill_values: pd.Series | None = field(default=None, init=False, repr=False)
    _time_col: str | None = field(default=None, init=False, repr=False)
    _history_by_code: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
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
                "DFQ-TimesNet requires PyTorch. Please install torch first, for example: pip install torch"
            ) from exc
        return torch, nn, F

    @staticmethod
    def _resolve_time_col(df: pd.DataFrame) -> str:
        for c in ["signal_ts", "datetime", "date"]:
            if c in df.columns:
                return c
        raise ValueError("DFQ-TimesNet requires one of ['signal_ts', 'datetime', 'date'] columns.")

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

    def _period_list(self) -> List[int]:
        return _parse_periods(self.periods)

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
        if self.label_transform == "csranknorm":
            ranked = y_raw.groupby(anchor).transform(lambda s: s.rank(pct=True, method="average"))
            ranked = ranked.replace([np.inf, -np.inf], np.nan)
            return ranked.groupby(anchor).transform(self._zscore)
        return y_raw.groupby(anchor).transform(self._zscore)

    def _transform_feature_frame(self, x: pd.DataFrame) -> pd.DataFrame:
        out = x.replace([np.inf, -np.inf], np.nan)
        if self._fill_values is not None:
            out = out.fillna(self._fill_values.reindex(out.columns))
        out = out.fillna(0.0)
        clip = float(self.input_clip)
        if clip > 0.0:
            out = out.clip(lower=-clip, upper=clip)
        return out

    def _build_train_samples(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        target: pd.Series,
    ) -> Dict[pd.Timestamp, List[_TrainSample]]:
        out = df.copy()
        out["_model_time"] = pd.to_datetime(out[self._time_col], errors="coerce")
        out["_target"] = pd.to_numeric(target, errors="coerce")
        out = out.dropna(subset=["code", "_model_time"]).sort_values(["code", "_model_time"])

        grouped: Dict[pd.Timestamp, List[_TrainSample]] = defaultdict(list)
        self._history_by_code = {}

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
                key = pd.Timestamp(t.iloc[i])
                if key == key.normalize():
                    key = key.normalize()
                grouped[key].append(
                    _TrainSample(
                        code=str(code),
                        time_key=key,
                        past_seq=past.astype(np.float32),
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
    def _stack_batch(samples: Iterable[_TrainSample]) -> tuple[np.ndarray, np.ndarray]:
        ss = list(samples)
        past = np.stack([s.past_seq for s in ss], axis=0).astype(np.float32)
        y = np.asarray([s.target for s in ss], dtype=np.float32)
        return past, y

    def _build_network(self, input_dim: int, torch: Any, nn: Any, F: Any) -> Any:
        hidden_size = int(self.hidden_size)
        e_layers = int(self.e_layers)
        hidden_size2 = int(self.hidden_size2)
        periods = self._period_list()
        num_kernels = int(self.num_kernels)
        dropout = float(self.dropout)

        class TokenEmbedding(nn.Module):
            def __init__(self, c_in: int, d_model: int) -> None:
                super().__init__()
                self.token_conv = nn.Conv1d(
                    in_channels=c_in,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    padding_mode="circular",
                    bias=False,
                )
                nn.init.kaiming_normal_(self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu")

            def forward(self, x: Any) -> Any:
                return self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)

        class InceptionBlock(nn.Module):
            def __init__(self, in_channels: int, out_channels: int, kernels: int) -> None:
                super().__init__()
                self.convs = nn.ModuleList()
                for i in range(max(1, kernels)):
                    k = 2 * i + 1
                    conv = nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=k,
                        padding=i,
                    )
                    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
                    if conv.bias is not None:
                        nn.init.constant_(conv.bias, 0.0)
                    self.convs.append(conv)

            def forward(self, x: Any) -> Any:
                outs = [conv(x) for conv in self.convs]
                return torch.stack(outs, dim=-1).mean(dim=-1)

        class TimesBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Sequential(
                    InceptionBlock(hidden_size, hidden_size2, num_kernels),
                    nn.GELU(),
                    InceptionBlock(hidden_size2, hidden_size, num_kernels),
                )
                self.drop = nn.Dropout(dropout)

            def _fold_period(self, x: Any, period: int) -> Any:
                bsz, seq_len, channels = x.shape
                pad_len = int(np.ceil(seq_len / float(period)) * period)
                if pad_len > seq_len:
                    pad = torch.zeros(
                        bsz,
                        pad_len - seq_len,
                        channels,
                        dtype=x.dtype,
                        device=x.device,
                    )
                    x_pad = torch.cat([x, pad], dim=1)
                else:
                    x_pad = x
                out = x_pad.reshape(bsz, pad_len // period, period, channels)
                out = out.permute(0, 3, 1, 2).contiguous()
                out = self.conv(out)
                out = out.permute(0, 2, 3, 1).contiguous().reshape(bsz, pad_len, channels)
                return out[:, :seq_len, :]

            def forward(self, x: Any) -> Any:
                period_outputs = [self._fold_period(x, p) for p in periods]
                y = torch.stack(period_outputs, dim=-1).mean(dim=-1)
                return x + self.drop(y)

        class DFQTimesNetNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = TokenEmbedding(input_dim, hidden_size)
                self.blocks = nn.ModuleList([TimesBlock() for _ in range(max(1, e_layers))])
                self.projection = nn.Linear(hidden_size, 1)

            def forward(self, x: Any) -> Any:
                enc = self.embedding(x)
                for block in self.blocks:
                    enc = block(enc)
                return self.projection(enc[:, -1, :]).squeeze(-1)

            def predict_raw(self, x: Any) -> Any:
                return self.forward(x)

        return DFQTimesNetNet()

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
                past, target = self._stack_batch(samples)
                x = torch.tensor(past, dtype=torch.float32, device=device)
                pred = net.predict_raw(x).detach().cpu().numpy()
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

    def fit(self, train_df: pd.DataFrame, factor_cols: list[str], target_col: str) -> "DFQTimesNetStockModel":
        torch, nn, F = self._require_torch()
        self._set_seed(torch)

        self._factor_cols = list(factor_cols)
        self._time_col = self._resolve_time_col(train_df)

        x = train_df[self._factor_cols].replace([np.inf, -np.inf], np.nan)
        self._fill_values = x.median(numeric_only=True).reindex(self._factor_cols).fillna(0.0)

        df = train_df.copy()
        df[self._factor_cols] = self._transform_feature_frame(x)
        time_anchor = self._time_anchor(df[self._time_col])
        target = self._build_target(df, target_col=target_col, anchor=time_anchor)

        grouped = self._build_train_samples(df=df, factor_cols=self._factor_cols, target=target)
        grouped = {k: v for k, v in grouped.items() if len(v) >= 8}
        if not grouped:
            raise RuntimeError("DFQ-TimesNet valid training samples are empty after sequence construction.")

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

        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )

        best_ic = -np.inf
        best_state: Dict[str, Any] | None = None
        best_epoch = -1
        best_states: deque[Dict[str, Any]] = deque(maxlen=max(1, int(self.smooth_steps)))
        patience = 0

        rng = np.random.default_rng(int(self.random_state))
        train_losses: List[float] = []
        train_mse_losses: List[float] = []
        train_ic_losses: List[float] = []
        val_ics: List[float] = []

        for epoch in range(1, int(self.n_epochs) + 1):
            net.train()
            losses_epoch: List[float] = []
            mse_losses_epoch: List[float] = []
            ic_losses_epoch: List[float] = []

            epoch_keys = list(train_keys)
            if int(self.per_epoch_batch) > 0 and len(epoch_keys) > 0:
                n_pick = int(self.per_epoch_batch)
                if len(epoch_keys) <= n_pick:
                    sampled_keys = list(epoch_keys)
                else:
                    sampled_keys = rng.choice(epoch_keys, size=n_pick, replace=False).tolist()
            else:
                sampled_keys = epoch_keys

            for key in sampled_keys:
                samples = train_groups.get(key, [])
                if len(samples) < 8:
                    continue

                if int(self.batch_size) > 0 and len(samples) > int(self.batch_size):
                    idx = rng.choice(len(samples), size=int(self.batch_size), replace=False)
                    batch = [samples[int(i)] for i in idx]
                else:
                    batch = samples

                past_np, y_np = self._stack_batch(batch)
                x_past = torch.tensor(past_np, dtype=torch.float32, device=device)
                y_true = torch.tensor(y_np, dtype=torch.float32, device=device)

                pred = net.forward(x_past)
                loss_mse = F.mse_loss(pred, y_true)
                if float(self.ic_loss_weight) > 0.0:
                    loss_ic = self._ic_loss(pred, y_true, torch=torch)
                else:
                    loss_ic = torch.tensor(0.0, dtype=torch.float32, device=device)
                loss = float(self.mse_weight) * loss_mse + float(self.ic_loss_weight) * loss_ic

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                optimizer.step()

                losses_epoch.append(float(loss.detach().cpu().item()))
                mse_losses_epoch.append(float(loss_mse.detach().cpu().item()))
                ic_losses_epoch.append(float(loss_ic.detach().cpu().item()))

            train_losses.append(float(np.mean(losses_epoch)) if losses_epoch else float("nan"))
            train_mse_losses.append(float(np.mean(mse_losses_epoch)) if mse_losses_epoch else float("nan"))
            train_ic_losses.append(float(np.mean(ic_losses_epoch)) if ic_losses_epoch else float("nan"))

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
            "avg_train_mse_loss_last": float(train_mse_losses[-1]) if train_mse_losses else float("nan"),
            "avg_train_ic_loss_last": float(train_ic_losses[-1]) if train_ic_losses else float("nan"),
            "val_rank_ic_last": float(val_ics[-1]) if val_ics else float("nan"),
            "score_sign": float(self._score_sign),
            "score_sign_ref_ic": float(sign_ref_ic) if np.isfinite(sign_ref_ic) else float("nan"),
            "device": self._device_used,
            "periods": [float(x) for x in self._period_list()],
        }
        return self

    def _build_predict_batches(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
    ) -> Dict[pd.Timestamp, List[tuple[int, np.ndarray]]]:
        out = df.copy()
        out["_model_time"] = pd.to_datetime(out[self._time_col], errors="coerce")
        out = out.dropna(subset=["code", "_model_time"]).copy()
        out["code"] = out["code"].astype(str)
        out = out.sort_values(["code", "_model_time"])

        batches: Dict[pd.Timestamp, List[tuple[int, np.ndarray]]] = defaultdict(list)

        for code, g in out.groupby("code"):
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
                if end >= int(self.seq_len) - 1:
                    past = merged[end - int(self.seq_len) + 1 : end + 1]
                else:
                    head = merged[: end + 1]
                    if len(head) == 0:
                        head = np.zeros((1, feat.shape[1]), dtype=np.float32)
                    need = int(self.seq_len) - len(head)
                    pad = np.repeat(head[:1], repeats=max(0, need), axis=0)
                    past = np.vstack([pad, head])
                key = pd.Timestamp(times.loc[row_idx])
                if key == key.normalize():
                    key = key.normalize()
                batches[key].append((int(row_idx), past.astype(np.float32)))

        return batches

    def predict_score(self, df: pd.DataFrame, factor_cols: list[str]) -> pd.Series:
        if self._model is None or self._fill_values is None or self._time_col is None:
            raise RuntimeError("DFQTimesNetStockModel is not fitted.")
        if list(factor_cols) != self._factor_cols:
            missing = [c for c in self._factor_cols if c not in factor_cols]
            if missing:
                raise ValueError(f"predict factor cols missing: {missing}")

        torch, _nn, _F = self._require_torch()
        self._model.eval()

        out = df.copy()
        out[self._factor_cols] = self._transform_feature_frame(out[self._factor_cols])

        batches = self._build_predict_batches(out, factor_cols=self._factor_cols)
        raw_pred = pd.Series(np.nan, index=out.index, dtype=float)

        with torch.no_grad():
            for key in sorted(batches.keys()):
                rows = batches[key]
                if not rows:
                    continue
                row_ids = [r[0] for r in rows]
                past_np = np.stack([r[1] for r in rows], axis=0).astype(np.float32)
                x_past = torch.tensor(past_np, dtype=torch.float32, device=self._device_used)
                pred = self._model.predict_raw(x_past).detach().cpu().numpy()
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
            raise RuntimeError("DFQTimesNetStockModel is not fitted.")
        torch, _nn, _F = self._require_torch()

        folder.mkdir(parents=True, exist_ok=True)
        model_path = folder / f"stock_model_dfq_timesnet_{run_tag}.pt"
        meta_path = folder / f"stock_model_dfq_timesnet_{run_tag}.json"

        config = {
            "seq_len": int(self.seq_len),
            "hidden_size": int(self.hidden_size),
            "e_layers": int(self.e_layers),
            "hidden_size2": int(self.hidden_size2),
            "periods": self._period_list(),
            "num_kernels": int(self.num_kernels),
            "dropout": float(self.dropout),
            "n_epochs": int(self.n_epochs),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "early_stop": int(self.early_stop),
            "smooth_steps": int(self.smooth_steps),
            "per_epoch_batch": int(self.per_epoch_batch),
            "batch_size": int(self.batch_size),
            "label_transform": str(self.label_transform),
            "input_clip": float(self.input_clip),
            "mse_weight": float(self.mse_weight),
            "ic_loss_weight": float(self.ic_loss_weight),
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
                "model_type": "dfq_timesnet",
                "target_col": self._target_col,
                "factor_count": len(self._factor_cols),
                "train_summary": self._train_summary,
                "config": config,
            },
        )
        return {"model_pt": str(model_path), "meta_json": str(meta_path)}
