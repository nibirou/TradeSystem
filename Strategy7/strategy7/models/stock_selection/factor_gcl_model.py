"""DFQ-FactorGCL stock-selection model.

This implementation follows the report's main design:
1) HyperGCN prior/hidden beta residual stack
2) Dual-path (past/future) TRCL training
3) Improved temporal InfoNCE + cross-sectional InfoNCE
4) Total loss = MSE + gamma * (temporal + cross_section)

The implementation is PyTorch-based and lazily imported, so existing non-DL models
remain usable even when torch is not installed.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from ...core.constants import EPS
from ...core.utils import dump_json, infer_board_type, infer_industry_bucket
from ..base import StockSelectionModel


@dataclass
class _TrainSample:
    code: str
    time_key: pd.Timestamp
    past_seq: np.ndarray
    future_seq: np.ndarray
    target: float


@dataclass
class FactorGCLStockModel(StockSelectionModel):
    """HyperGCN + TRCL stock selection model (DFQ-FactorGCL style)."""

    seq_len: int = 30
    future_look: int = 20
    hidden_size: int = 128
    num_layers: int = 2
    num_factor: int = 48
    gamma: float = 1.0
    tau: float = 0.25
    n_epochs: int = 200
    lr: float = 9e-5
    early_stop: int = 20
    smooth_steps: int = 5
    per_epoch_batch: int = 100
    batch_size: int = -1
    label_transform: str = "csranknorm"
    weight_decay: float = 1e-4
    dropout: float = 0.0
    random_state: int = 42
    device: str = "auto"

    _model: Any = field(default=None, init=False, repr=False)
    _factor_cols: List[str] = field(default_factory=list, init=False, repr=False)
    _fill_values: pd.Series | None = field(default=None, init=False, repr=False)
    _time_col: str | None = field(default=None, init=False, repr=False)
    _history_by_code: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _concepts: List[str] = field(default_factory=list, init=False, repr=False)
    _code_concept_idx: Dict[str, List[int]] = field(default_factory=dict, init=False, repr=False)
    _all_concept_idx: int = field(default=0, init=False, repr=False)
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
                "FactorGCL requires PyTorch. Please install torch first, for example: "
                "pip install torch"
            ) from exc
        return torch, nn, F

    @staticmethod
    def _resolve_time_col(df: pd.DataFrame) -> str:
        for c in ["signal_ts", "datetime", "date"]:
            if c in df.columns:
                return c
        raise ValueError("FactorGCL requires one of ['signal_ts', 'datetime', 'date'] columns.")

    @staticmethod
    def _zscore(x: pd.Series) -> pd.Series:
        v = pd.to_numeric(x, errors="coerce")
        std = float(v.std(ddof=0)) if v.notna().sum() > 1 else 0.0
        if std <= EPS:
            return pd.Series(np.zeros(len(v), dtype=float), index=v.index)
        return (v - float(v.mean())) / (std + EPS)

    def _time_anchor(self, ts: pd.Series) -> pd.Series:
        dt = pd.to_datetime(ts, errors="coerce")
        return dt.dt.normalize()

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

        # default: csranknorm = cross-sectional rank percentile then zscore
        ranked = y_raw.groupby(anchor).transform(lambda s: s.rank(pct=True, method="average"))
        ranked = ranked.replace([np.inf, -np.inf], np.nan)
        return ranked.groupby(anchor).transform(self._zscore)

    def _fit_concepts(self, df: pd.DataFrame) -> None:
        base = df.copy()
        base["code"] = base["code"].astype(str)

        if "industry_bucket" in base.columns:
            industry = base[["code", "industry_bucket"]].dropna().drop_duplicates("code", keep="last")
            ind_map = {str(r.code): str(r.industry_bucket) for r in industry.itertuples(index=False)}
        else:
            ind_map = {}

        if "board_type" in base.columns:
            board = base[["code", "board_type"]].dropna().drop_duplicates("code", keep="last")
            board_map = {str(r.code): str(r.board_type) for r in board.itertuples(index=False)}
        else:
            board_map = {}

        concepts: set[str] = {"all:all"}
        code_to_tags: Dict[str, List[str]] = {}
        for code in sorted(base["code"].dropna().unique().tolist()):
            c = str(code)
            ind = ind_map.get(c, infer_industry_bucket(c))
            brd = board_map.get(c, infer_board_type(c))
            tags = [f"industry:{ind}", f"board:{brd}", "all:all"]
            code_to_tags[c] = tags
            concepts.update(tags)

        self._concepts = sorted(concepts)
        cidx = {c: i for i, c in enumerate(self._concepts)}
        self._all_concept_idx = cidx.get("all:all", 0)
        self._code_concept_idx = {
            code: sorted({cidx[t] for t in tags if t in cidx}) for code, tags in code_to_tags.items()
        }

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
            self._history_by_code[str(code)] = x[max(0, n - self.seq_len + 1) :].copy()

            start = self.seq_len - 1
            end = n - self.future_look
            if end <= start:
                continue

            for i in range(start, end):
                if np.isnan(y[i]):
                    continue
                past = x[i - self.seq_len + 1 : i + 1]
                future = x[i + 1 : i + 1 + self.future_look]
                if past.shape[0] != self.seq_len or future.shape[0] != self.future_look:
                    continue
                if np.isnan(past).any() or np.isnan(future).any():
                    continue
                key = pd.Timestamp(t.iloc[i]).normalize()
                grouped[key].append(
                    _TrainSample(
                        code=str(code),
                        time_key=key,
                        past_seq=past,
                        future_seq=future,
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
    def _stack_batch(samples: Iterable[_TrainSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        ss = list(samples)
        past = np.stack([s.past_seq for s in ss], axis=0).astype(np.float32)
        future = np.stack([s.future_seq for s in ss], axis=0).astype(np.float32)
        y = np.asarray([s.target for s in ss], dtype=np.float32)
        codes = [s.code for s in ss]
        return past, future, y, codes

    def _prior_beta_tensor(self, codes: List[str], torch: Any, device: str) -> Any:
        h = torch.zeros((len(codes), len(self._concepts)), dtype=torch.float32, device=device)
        for i, c in enumerate(codes):
            idxs = self._code_concept_idx.get(str(c), [self._all_concept_idx])
            if not idxs:
                idxs = [self._all_concept_idx]
            h[i, idxs] = 1.0
        return h

    def _build_network(self, input_dim: int, num_prior_concepts: int, torch: Any, nn: Any) -> Any:
        hidden_size = int(self.hidden_size)
        num_layers = int(self.num_layers)
        num_factor = int(self.num_factor)
        dropout = float(self.dropout)

        class HyperGCNLayer(nn.Module):
            def __init__(self, hidden_dim: int, num_concepts: int) -> None:
                super().__init__()
                self.theta = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.edge_weight = nn.Parameter(torch.ones(num_concepts, dtype=torch.float32))
                self.act = nn.LeakyReLU(0.1)

            def forward(self, x: Any, h: Any) -> Any:
                # x: [N, H], h: [N, M]
                h = h.float()
                dv = torch.clamp(h.sum(dim=1), min=1.0)
                de = torch.clamp(h.sum(dim=0), min=1.0)
                we = torch.clamp(self.edge_weight, min=1e-4)

                x_theta = self.theta(x)
                x1 = x_theta / torch.sqrt(dv).unsqueeze(-1)
                edge_feat = torch.matmul(h.transpose(0, 1), x1)
                edge_feat = edge_feat * (we / de).unsqueeze(-1)
                out = torch.matmul(h, edge_feat)
                out = out / torch.sqrt(dv).unsqueeze(-1)
                return self.act(out)

        class FactorGCLNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.past_gru = nn.GRU(
                    input_size=input_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.future_gru = nn.GRU(
                    input_size=input_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.prior_layer = HyperGCNLayer(hidden_size, num_prior_concepts)
                self.hidden_layer = HyperGCNLayer(hidden_size, num_factor)
                self.prototypes = nn.Parameter(torch.randn(num_factor, hidden_size) * 0.02)

                self.alpha_layer = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LeakyReLU(0.1),
                )

                self.out0 = nn.Linear(hidden_size, 1, bias=False)
                self.out1 = nn.Linear(hidden_size, 1, bias=False)
                self.out2 = nn.Linear(hidden_size, 1, bias=False)
                self.out_bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

                self.proj_head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LeakyReLU(0.1),
                    nn.Linear(hidden_size, hidden_size),
                )

            def _encode_hist(self, past_seq: Any, beta_prior: Any) -> Dict[str, Any]:
                _, h_last = self.past_gru(past_seq)
                x0 = h_last[-1]

                x_hat0 = self.prior_layer(x0, beta_prior)
                x1 = x0 - x_hat0

                beta_hidden = torch.sigmoid(torch.matmul(x1, self.prototypes.transpose(0, 1)))
                x_hat1 = self.hidden_layer(x1, beta_hidden)
                x2 = x1 - x_hat1

                alpha_past = self.alpha_layer(x2)
                y_hat = self.out0(x_hat0) + self.out1(x_hat1) + self.out2(alpha_past) + self.out_bias
                return {
                    "y_hat": y_hat.squeeze(-1),
                    "alpha": alpha_past,
                    "beta_hidden": beta_hidden,
                }

            def _encode_future(self, future_seq: Any, beta_prior: Any, beta_hidden_shared: Any) -> Any:
                _, h_last = self.future_gru(future_seq)
                x0 = h_last[-1]
                x_hat0 = self.prior_layer(x0, beta_prior)
                x1 = x0 - x_hat0
                x_hat1 = self.hidden_layer(x1, beta_hidden_shared)
                x2 = x1 - x_hat1
                return self.alpha_layer(x2)

            def forward_train(self, past_seq: Any, future_seq: Any, beta_prior: Any) -> Dict[str, Any]:
                hist = self._encode_hist(past_seq, beta_prior)
                alpha_future = self._encode_future(future_seq, beta_prior, hist["beta_hidden"].detach())
                z_past = self.proj_head(hist["alpha"])
                z_future = self.proj_head(alpha_future)
                return {
                    "y_hat": hist["y_hat"],
                    "z_past": z_past,
                    "z_future": z_future,
                    "alpha_past": hist["alpha"],
                }

            def predict_raw(self, past_seq: Any, beta_prior: Any) -> Any:
                hist = self._encode_hist(past_seq, beta_prior)
                return hist["y_hat"]

        return FactorGCLNet()

    @staticmethod
    def _cosine_matrix(z1: Any, z2: Any, torch: Any) -> Any:
        z1n = z1 / (torch.norm(z1, p=2, dim=1, keepdim=True) + 1e-12)
        z2n = z2 / (torch.norm(z2, p=2, dim=1, keepdim=True) + 1e-12)
        return torch.matmul(z1n, z2n.transpose(0, 1))

    def _temporal_infonce(self, z_past: Any, z_future: Any, torch: Any) -> Any:
        sim = self._cosine_matrix(z_past, z_future, torch)
        pos = torch.diag(sim)
        pos_mod = torch.sign(pos) * torch.pow(pos, 2)
        numerator = torch.exp(pos_mod / float(self.tau))
        denominator = torch.exp(torch.pow(sim, 2) / float(self.tau)).sum(dim=1) + 1e-12
        return -torch.log((numerator / denominator).clamp(min=1e-12)).mean()

    def _cross_infonce(self, z_past: Any, torch: Any) -> Any:
        sim = self._cosine_matrix(z_past, z_past, torch)
        numerator = torch.exp(torch.diag(sim) / float(self.tau))
        denominator = torch.exp(sim / float(self.tau)).sum(dim=1) + 1e-12
        return -torch.log((numerator / denominator).clamp(min=1e-12)).mean()

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
                past, _future, target, codes = self._stack_batch(samples)
                x_past = torch.tensor(past, dtype=torch.float32, device=device)
                beta = self._prior_beta_tensor(codes, torch=torch, device=device)
                pred = net.predict_raw(x_past, beta).detach().cpu().numpy()
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

    def fit(self, train_df: pd.DataFrame, factor_cols: list[str], target_col: str) -> "FactorGCLStockModel":
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

        self._fit_concepts(df)
        grouped = self._build_train_samples(df=df, factor_cols=self._factor_cols, target=target)
        grouped = {k: v for k, v in grouped.items() if len(v) >= 8}
        if not grouped:
            raise RuntimeError("FactorGCL valid training samples are empty after sequence construction.")

        keys = sorted(grouped.keys())
        split = max(1, int(len(keys) * 0.8))
        train_keys = keys[:split]
        val_keys = keys[split:]
        if not val_keys:
            val_keys = keys[-1:]

        train_groups = {k: grouped[k] for k in train_keys}
        val_groups = {k: grouped[k] for k in val_keys}

        input_dim = len(self._factor_cols)
        net = self._build_network(input_dim=input_dim, num_prior_concepts=len(self._concepts), torch=torch, nn=nn)

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
        val_ics: List[float] = []

        for epoch in range(1, int(self.n_epochs) + 1):
            net.train()
            losses_epoch: List[float] = []

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

                past_np, future_np, y_np, codes = self._stack_batch(batch)
                x_past = torch.tensor(past_np, dtype=torch.float32, device=device)
                x_future = torch.tensor(future_np, dtype=torch.float32, device=device)
                y_true = torch.tensor(y_np, dtype=torch.float32, device=device)
                beta_prior = self._prior_beta_tensor(codes, torch=torch, device=device)

                out = net.forward_train(x_past, x_future, beta_prior)
                mse = F.mse_loss(out["y_hat"], y_true)
                loss_temporal = self._temporal_infonce(out["z_past"], out["z_future"], torch=torch)
                loss_cross = self._cross_infonce(out["z_past"], torch=torch)
                loss = mse + float(self.gamma) * (loss_temporal + loss_cross)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                optimizer.step()

                losses_epoch.append(float(loss.detach().cpu().item()))

            avg_loss = float(np.mean(losses_epoch)) if losses_epoch else float("nan")
            train_losses.append(avg_loss)

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
            "val_rank_ic_last": float(val_ics[-1]) if val_ics else float("nan"),
            "score_sign": float(self._score_sign),
            "score_sign_ref_ic": float(sign_ref_ic) if np.isfinite(sign_ref_ic) else float("nan"),
            "device": self._device_used,
        }
        return self

    def _build_predict_batches(self, df: pd.DataFrame, factor_cols: List[str]) -> Dict[pd.Timestamp, List[tuple[int, str, np.ndarray]]]:
        out = df.copy()
        out["_model_time"] = pd.to_datetime(out[self._time_col], errors="coerce")
        out = out.dropna(subset=["code", "_model_time"]).copy()
        out["code"] = out["code"].astype(str)
        out = out.sort_values(["code", "_model_time"])

        batches: Dict[pd.Timestamp, List[tuple[int, str, np.ndarray]]] = defaultdict(list)

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
                if end >= self.seq_len - 1:
                    past = merged[end - self.seq_len + 1 : end + 1]
                else:
                    head = merged[: end + 1]
                    if len(head) == 0:
                        head = np.zeros((1, feat.shape[1]), dtype=np.float32)
                    need = self.seq_len - len(head)
                    pad = np.repeat(head[:1], repeats=max(0, need), axis=0)
                    past = np.vstack([pad, head])
                key = pd.Timestamp(times.loc[row_idx]).normalize()
                batches[key].append((int(row_idx), c, past.astype(np.float32)))

        return batches

    def predict_score(self, df: pd.DataFrame, factor_cols: list[str]) -> pd.Series:
        if self._model is None or self._fill_values is None or self._time_col is None:
            raise RuntimeError("FactorGCLStockModel is not fitted.")
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

        batches = self._build_predict_batches(out, factor_cols=self._factor_cols)
        raw_pred = pd.Series(np.nan, index=out.index, dtype=float)

        with torch.no_grad():
            for key in sorted(batches.keys()):
                rows = batches[key]
                if not rows:
                    continue
                row_ids = [r[0] for r in rows]
                codes = [r[1] for r in rows]
                past_np = np.stack([r[2] for r in rows], axis=0).astype(np.float32)

                x_past = torch.tensor(past_np, dtype=torch.float32, device=self._device_used)
                beta_prior = self._prior_beta_tensor(codes, torch=torch, device=self._device_used)
                pred = self._model.predict_raw(x_past, beta_prior).detach().cpu().numpy()
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
            raise RuntimeError("FactorGCLStockModel is not fitted.")
        torch, _nn, _F = self._require_torch()

        folder.mkdir(parents=True, exist_ok=True)
        model_path = folder / f"stock_model_factor_gcl_{run_tag}.pt"
        meta_path = folder / f"stock_model_factor_gcl_{run_tag}.json"

        checkpoint = {
            "state_dict": self._model.state_dict(),
            "factor_cols": self._factor_cols,
            "concepts": self._concepts,
            "code_concept_idx": self._code_concept_idx,
            "fill_values": self._fill_values.to_dict() if self._fill_values is not None else {},
            "time_col": self._time_col,
            "train_summary": self._train_summary,
            "config": {
                "seq_len": int(self.seq_len),
                "future_look": int(self.future_look),
                "hidden_size": int(self.hidden_size),
                "num_layers": int(self.num_layers),
                "num_factor": int(self.num_factor),
                "gamma": float(self.gamma),
                "tau": float(self.tau),
                "n_epochs": int(self.n_epochs),
                "lr": float(self.lr),
                "early_stop": int(self.early_stop),
                "smooth_steps": int(self.smooth_steps),
                "per_epoch_batch": int(self.per_epoch_batch),
                "batch_size": int(self.batch_size),
                "label_transform": str(self.label_transform),
                "weight_decay": float(self.weight_decay),
                "dropout": float(self.dropout),
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
                "model_type": "factor_gcl",
                "target_col": self._target_col,
                "factor_count": len(self._factor_cols),
                "concept_count": len(self._concepts),
                "train_summary": self._train_summary,
                "config": checkpoint["config"],
            },
        )
        return {"model_pt": str(model_path), "meta_json": str(meta_path)}
