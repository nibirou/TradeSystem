"""LSTM + MADL timing model.

This module implements the timing model described in the Huafu Securities
report "LSTM neural-network timing fused with multi-factor stock selection".
The report uses an index-level sequence model:

    input -> [LSTM + BatchNorm] x N -> Linear -> Tanh

and optimizes a directional MADL loss.  Strategy7 receives a stock panel rather
than a native index panel, so the model first builds a market proxy by
aggregating each signal timestamp's cross section.  This keeps the component
compatible with every research frequency and with price-volume, fundamental,
text, catalog, and custom factors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ...core.constants import EPS
from ...core.utils import dump_json
from ..base import TimingModel


REPORT_TECHNICAL_FEATURES: List[str] = [
    "returns_1",
    "returns_5",
    "returns_15",
    "returns_30",
    "returns_60",
    "sma_5",
    "sma_10",
    "sma_20",
    "sma_30",
    "sma_60",
    "ema_5",
    "ema_12",
    "ema_26",
    "sma_cross_5_10",
    "sma_cross_10_20",
    "ema_cross_12_26",
    "price_position_sma20",
    "price_position_sma60",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "volatility_30",
    "bb_position",
    "bb_width",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "volume_ratio_10",
    "volume_ratio_20",
    "vpt_sma",
    "high_low_ratio",
    "close_open_ratio",
    "high_close_ratio",
    "low_close_ratio",
    "atr_14",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "williams_r",
    "k_percent",
    "d_percent",
    "price_acceleration",
    "vwap_ratio",
]

BASE_BAR_FEATURES: List[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "trade_count",
]

META_AND_LABEL_COLUMNS = {
    "code",
    "date",
    "datetime",
    "signal_ts",
    "entry_date",
    "exit_date",
    "entry_ts",
    "exit_ts",
    "target_date",
    "time_freq",
    "board_type",
    "industry_bucket",
    "pred_score",
    "pred_up",
    "weight_target",
    "executed_weight",
    "fill_ratio",
    "realized_trade_ret",
    "gross_trade_ret",
    "net_trade_ret",
    "future_ret_n",
    "target_return",
    "target_up",
    "target_volatility",
    "fwd_vol_label",
    "entry_price",
    "exit_price",
    "tradestatus",
}


def _parse_int_list(raw: str | Sequence[int]) -> List[int]:
    if isinstance(raw, str):
        parts = [x.strip() for x in raw.replace("，", ",").replace(";", ",").split(",") if x.strip()]
        vals = [int(float(x)) for x in parts]
    else:
        vals = [int(x) for x in raw]
    out: List[int] = []
    for v in vals:
        if v > 0:
            out.append(v)
    return out or [512, 256, 128]


def _parse_float_list(raw: str | Sequence[float]) -> List[float]:
    if isinstance(raw, str):
        parts = [x.strip() for x in raw.replace("，", ",").replace(";", ",").split(",") if x.strip()]
        return [float(x) for x in parts]
    return [float(x) for x in raw]


def _numeric(s: pd.Series | np.ndarray | float, index: pd.Index | None = None) -> pd.Series:
    if isinstance(s, pd.Series):
        out = pd.to_numeric(s, errors="coerce")
    else:
        out = pd.Series(s, index=index, dtype=float)
    return out.replace([np.inf, -np.inf], np.nan)


def _rolling_mean(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    return s.rolling(window, min_periods=min_periods or max(2, min(window, 5))).mean()


def _rolling_std(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    return s.rolling(window, min_periods=min_periods or max(2, min(window, 5))).std(ddof=0)


class _LSTMMADLNet:
    """Factory wrapper.  The actual torch module is created lazily."""

    @staticmethod
    def build(torch: Any, nn: Any, input_dim: int, hidden_sizes: Sequence[int], dropout: float) -> Any:
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList()
                self.norms = nn.ModuleList()
                in_dim = int(input_dim)
                for h in hidden_sizes:
                    h_int = int(h)
                    self.layers.append(nn.LSTM(input_size=in_dim, hidden_size=h_int, num_layers=1, batch_first=True))
                    self.norms.append(nn.BatchNorm1d(h_int))
                    in_dim = h_int
                self.dropout = nn.Dropout(float(dropout))
                self.head = nn.Linear(in_dim, 1)

            def forward(self, x: Any) -> Any:
                out = x
                for lstm, norm in zip(self.layers, self.norms):
                    out, _state = lstm(out)
                    bsz, steps, hidden = out.shape
                    flat = out.reshape(bsz * steps, hidden)
                    flat = norm(flat)
                    out = flat.reshape(bsz, steps, hidden)
                    out = self.dropout(out)
                last = out[:, -1, :]
                return torch.tanh(self.head(last)).squeeze(-1)

        return Net()


@dataclass
class LSTMMADLTimingModel(TimingModel):
    seq_len: int = 20
    intraday_seq_len: int = 48
    hidden_sizes: str | Sequence[int] = "512,256,128"
    dropout: float = 0.2
    n_epochs: int = 120
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop: int = 15
    batch_size: int = 128
    min_train_samples: int = 80
    feature_mode: str = "auto"
    max_features: int = 96
    input_clip: float = 5.0
    target_clip: float = 0.20
    loss_mode: str = "madl_mse"
    mse_weight: float = 0.05
    exposure_mode: str = "long_only_bands"
    long_threshold: float = -0.3
    band_thresholds: str | Sequence[float] = "-0.1,0.1,0.6,0.999999"
    band_exposures: str | Sequence[float] = "0.0,0.3,0.5,0.8,1.0"
    signal_scale: float = 1.0
    calibrate_sign: bool = True
    market_agg: str = "amount_weighted"
    extra_feature_limit: int = 48
    random_state: int = 1000
    device: str = "auto"

    _model: Any = field(default=None, init=False, repr=False)
    _time_col: str | None = field(default=None, init=False, repr=False)
    _is_intraday: bool = field(default=False, init=False, repr=False)
    _effective_seq_len: int = field(default=20, init=False, repr=False)
    _effective_feature_mode: str = field(default="daily_bar", init=False, repr=False)
    _feature_cols: List[str] = field(default_factory=list, init=False, repr=False)
    _extra_cols: List[str] = field(default_factory=list, init=False, repr=False)
    _feature_fill: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), init=False, repr=False)
    _feature_mean: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), init=False, repr=False)
    _feature_std: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), init=False, repr=False)
    _history_market_raw: pd.DataFrame = field(default_factory=pd.DataFrame, init=False, repr=False)
    _fit_history_market_raw: pd.DataFrame = field(default_factory=pd.DataFrame, init=False, repr=False)
    _device_used: str = field(default="cpu", init=False, repr=False)
    _target_scale: float = field(default=0.01, init=False, repr=False)
    _score_sign: float = field(default=1.0, init=False, repr=False)
    _train_summary: Dict[str, float] = field(default_factory=dict, init=False, repr=False)

    @staticmethod
    def _require_torch() -> tuple[Any, Any, Any]:
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except Exception as exc:
            raise RuntimeError(
                "lstm_madl timing requires PyTorch. Use env_quant or install torch in the active environment."
            ) from exc
        return torch, nn, F

    @staticmethod
    def _resolve_time_col(df: pd.DataFrame) -> str:
        for col in ("signal_ts", "datetime", "date"):
            if col in df.columns:
                return col
        raise ValueError("LSTMMADLTimingModel requires one of signal_ts/datetime/date.")

    @staticmethod
    def _time_key(series: pd.Series, *, force_intraday: bool | None = None) -> tuple[pd.Series, bool]:
        dt = pd.to_datetime(series, errors="coerce")
        normalized = dt.dt.normalize()
        inferred_intraday = bool(((dt - normalized).dt.total_seconds().fillna(0.0) != 0.0).any())
        use_intraday = inferred_intraday if force_intraday is None else bool(force_intraday)
        return (dt if use_intraday else normalized), use_intraday

    @staticmethod
    def _choose_device(torch: Any, requested: str) -> str:
        req = str(requested).strip().lower()
        if req == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if req == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return req

    def _hidden_sizes(self) -> List[int]:
        return _parse_int_list(self.hidden_sizes)

    def _set_seed(self, torch: Any) -> None:
        seed = int(self.random_state)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _mode_for_panel(self) -> str:
        mode = str(self.feature_mode).strip().lower()
        if mode == "auto":
            return "technical" if self._is_intraday else "daily_bar"
        if mode in {"daily", "report_daily"}:
            return "daily_bar"
        if mode in {"intraday", "report_intraday", "minute"}:
            return "technical"
        if mode not in {"daily_bar", "technical", "hybrid", "all_numeric"}:
            return "technical" if self._is_intraday else "daily_bar"
        return mode

    def _select_extra_cols(self, df: pd.DataFrame, budget: int) -> List[str]:
        mode = str(self.feature_mode).strip().lower()
        if mode not in {"hybrid", "all_numeric"}:
            return []
        budget = int(max(0, min(budget, self.extra_feature_limit)))
        if budget <= 0:
            return []
        target = _numeric(df["future_ret_n"]) if "future_ret_n" in df.columns else pd.Series(np.nan, index=df.index)
        base_exclude = set(META_AND_LABEL_COLUMNS) | set(BASE_BAR_FEATURES)
        rows: List[tuple[float, str]] = []
        for col in df.columns:
            name = str(col)
            if name in base_exclude or name.startswith("_") or name.startswith("px_"):
                continue
            s = pd.to_numeric(df[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
            valid = s.notna()
            if int(valid.sum()) < max(20, int(0.05 * len(s))):
                continue
            std = float(s.loc[valid].std(ddof=0)) if int(valid.sum()) > 1 else 0.0
            if not np.isfinite(std) or std <= EPS:
                continue
            coverage = float(valid.mean())
            score = coverage + 0.10 * float(np.log1p(abs(std)))
            if target.notna().any():
                both = valid & target.notna()
                if int(both.sum()) >= 50:
                    corr = s.loc[both].corr(target.loc[both])
                    if np.isfinite(corr):
                        score += 0.75 * abs(float(corr))
            rows.append((float(score), name))
        rows.sort(key=lambda x: (-x[0], x[1]))
        return [c for _score, c in rows[:budget]]

    def _aggregate_market_raw(self, df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        time_col = self._resolve_time_col(df) if fit or not self._time_col else str(self._time_col)
        keys, is_intraday = self._time_key(df[time_col], force_intraday=None if fit else self._is_intraday)
        if fit:
            self._time_col = time_col
            self._is_intraday = bool(is_intraday)

        work = pd.DataFrame({"_time_key": keys})
        for col in ("open", "high", "low", "close", "volume", "amount", "turn", "trade_count"):
            if col in df.columns:
                work[col] = _numeric(df[col])
            else:
                work[col] = np.nan
        if work["trade_count"].notna().sum() == 0:
            work["trade_count"] = work["turn"].where(work["turn"].notna(), work["volume"])
        if "future_ret_n" in df.columns:
            work["_target"] = _numeric(df["future_ret_n"])
        for col in self._extra_cols:
            work[f"extra__{col}"] = _numeric(df[col]) if col in df.columns else np.nan
        work = work.dropna(subset=["_time_key"]).copy()
        if work.empty:
            return pd.DataFrame()

        grouped = work.groupby("_time_key", sort=True, observed=True)
        idx = grouped.size().index
        out = pd.DataFrame({"_time_key": pd.to_datetime(idx)})
        weight = work["amount"].clip(lower=0.0).fillna(0.0)
        wsum = weight.groupby(work["_time_key"], sort=True).sum().reindex(idx)

        def agg_center(col: str) -> pd.Series:
            center = grouped[col].median() if str(self.market_agg).lower() == "median" else grouped[col].mean()
            if str(self.market_agg).lower() != "amount_weighted" or col not in work.columns:
                return center.reindex(idx)
            weighted = (work[col].fillna(0.0) * weight).groupby(work["_time_key"], sort=True).sum().reindex(idx)
            weighted = weighted / (wsum + EPS)
            return weighted.where(wsum > EPS, center.reindex(idx))

        for col in ("open", "high", "low", "close"):
            out[col] = agg_center(col).to_numpy(dtype=float)
        for col in ("volume", "amount", "trade_count"):
            out[col] = grouped[col].sum(min_count=1).reindex(idx).to_numpy(dtype=float)
        if "_target" in work.columns:
            if str(self.market_agg).lower() == "amount_weighted":
                target_center = grouped["_target"].mean().reindex(idx)
                target_weighted = (work["_target"].fillna(0.0) * weight).groupby(work["_time_key"], sort=True).sum().reindex(idx)
                target_weighted = target_weighted / (wsum + EPS)
                out["_target"] = target_weighted.where(wsum > EPS, target_center).to_numpy(dtype=float)
            else:
                out["_target"] = grouped["_target"].mean().reindex(idx).to_numpy(dtype=float)
        for col in self._extra_cols:
            raw_name = f"extra__{col}"
            if raw_name in work.columns:
                out[raw_name] = grouped[raw_name].mean().reindex(idx).to_numpy(dtype=float)
        return out.sort_values("_time_key").reset_index(drop=True)

    def _add_report_technical_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = raw.copy().sort_values("_time_key").reset_index(drop=True)
        for col in BASE_BAR_FEATURES:
            if col not in out.columns:
                out[col] = np.nan
            out[col] = _numeric(out[col], index=out.index)
        close = out["close"].ffill()
        open_ = out["open"].where(out["open"].notna(), close).ffill()
        high = out["high"].where(out["high"].notna(), close).ffill()
        low = out["low"].where(out["low"].notna(), close).ffill()
        volume = out["volume"].fillna(0.0)
        amount = out["amount"].fillna(0.0)

        for p in (1, 5, 15, 30, 60):
            out[f"returns_{p}"] = close.pct_change(p)
        for p in (5, 10, 20, 30, 60):
            out[f"sma_{p}"] = _rolling_mean(close, p)
        for p in (5, 12, 26):
            out[f"ema_{p}"] = close.ewm(span=p, adjust=False, min_periods=max(2, min(p, 5))).mean()
        out["sma_cross_5_10"] = (out["sma_5"] > out["sma_10"]).astype(float)
        out["sma_cross_10_20"] = (out["sma_10"] > out["sma_20"]).astype(float)
        out["ema_cross_12_26"] = (out["ema_12"] > out["ema_26"]).astype(float)
        out["price_position_sma20"] = close / (out["sma_20"] + EPS) - 1.0
        out["price_position_sma60"] = close / (out["sma_60"] + EPS) - 1.0
        ret1 = close.pct_change(1)
        for p in (5, 10, 20, 30):
            out[f"volatility_{p}"] = _rolling_std(ret1, p)

        mid = out["sma_20"]
        band_std = _rolling_std(close, 20)
        upper = mid + 2.0 * band_std
        lower = mid - 2.0 * band_std
        out["bb_position"] = (close - lower) / (upper - lower + EPS)
        out["bb_width"] = (upper - lower) / (mid.abs() + EPS)

        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        rs = _rolling_mean(gain, 14) / (_rolling_mean(loss, 14) + EPS)
        out["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)

        out["macd"] = out["ema_12"] - out["ema_26"]
        out["macd_signal"] = out["macd"].ewm(span=9, adjust=False, min_periods=3).mean()
        out["macd_histogram"] = out["macd"] - out["macd_signal"]
        out["volume_ratio_10"] = volume / (_rolling_mean(volume, 10) + EPS)
        out["volume_ratio_20"] = volume / (_rolling_mean(volume, 20) + EPS)

        vpt = (ret1.fillna(0.0) * volume.fillna(0.0)).cumsum()
        out["vpt_sma"] = _rolling_mean(vpt, 20)
        out["high_low_ratio"] = high / (low.abs() + EPS)
        out["close_open_ratio"] = close / (open_.abs() + EPS)
        out["high_close_ratio"] = high / (close.abs() + EPS)
        out["low_close_ratio"] = low / (close.abs() + EPS)

        prev_close = close.shift(1)
        true_range = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        out["atr_14"] = _rolling_mean(true_range, 14)
        for p in (5, 10, 20):
            out[f"momentum_{p}"] = close / (close.shift(p).abs() + EPS)
        high14 = high.rolling(14, min_periods=5).max()
        low14 = low.rolling(14, min_periods=5).min()
        out["williams_r"] = -100.0 * (high14 - close) / (high14 - low14 + EPS)
        out["k_percent"] = 100.0 * (close - low14) / (high14 - low14 + EPS)
        out["d_percent"] = _rolling_mean(out["k_percent"], 3)
        out["price_acceleration"] = ret1.diff()
        vwap = amount / (volume + EPS)
        out["vwap_ratio"] = close / (vwap.abs() + EPS)
        return out.replace([np.inf, -np.inf], np.nan)

    def _feature_candidates(self, feature_df: pd.DataFrame) -> List[str]:
        mode = self._effective_feature_mode
        base = [c for c in BASE_BAR_FEATURES if c in feature_df.columns]
        tech = [c for c in REPORT_TECHNICAL_FEATURES if c in feature_df.columns]
        extra = [f"extra__{c}" for c in self._extra_cols if f"extra__{c}" in feature_df.columns]
        if mode == "daily_bar":
            return base
        if mode == "technical":
            return list(dict.fromkeys([*base, *tech]))
        if mode == "hybrid":
            return list(dict.fromkeys([*base, *tech, *extra]))
        cols = [
            c
            for c in feature_df.columns
            if c not in {"_time_key", "_target"} and pd.api.types.is_numeric_dtype(feature_df[c])
        ]
        return cols

    def _select_feature_cols(self, feature_df: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
        cols = [c for c in candidates if c in feature_df.columns]
        max_features = int(max(1, self.max_features))
        if len(cols) <= max_features:
            return cols
        target = _numeric(feature_df["_target"]) if "_target" in feature_df.columns else pd.Series(np.nan, index=feature_df.index)
        scored: List[tuple[float, str]] = []
        for col in cols:
            s = _numeric(feature_df[col])
            valid = s.notna()
            if int(valid.sum()) < 5:
                continue
            std = float(s.loc[valid].std(ddof=0)) if int(valid.sum()) > 1 else 0.0
            if not np.isfinite(std) or std <= EPS:
                continue
            score = float(valid.mean()) + 0.05 * float(np.log1p(abs(std)))
            both = valid & target.notna()
            if int(both.sum()) >= 20:
                corr = s.loc[both].corr(target.loc[both])
                if np.isfinite(corr):
                    score += 0.75 * abs(float(corr))
            scored.append((score, col))
        if not scored:
            return cols[:max_features]
        scored.sort(key=lambda x: (-x[0], x[1]))
        selected = [c for _score, c in scored[:max_features]]
        preferred = [c for c in cols if c in selected]
        return preferred

    def _prepare_feature_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        return self._add_report_technical_features(raw)

    def _fit_transformer(self, feature_df: pd.DataFrame) -> np.ndarray:
        x = feature_df[self._feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        self._feature_fill = x.median(numeric_only=True).reindex(self._feature_cols).fillna(0.0)
        filled = x.fillna(self._feature_fill).fillna(0.0)
        self._feature_mean = filled.mean().reindex(self._feature_cols).fillna(0.0)
        self._feature_std = filled.std(ddof=0).reindex(self._feature_cols).replace(0.0, np.nan).fillna(1.0)
        return self._transform_feature_frame(feature_df)

    def _transform_feature_frame(self, feature_df: pd.DataFrame) -> np.ndarray:
        x = feature_df[self._feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        x = x.fillna(self._feature_fill.reindex(self._feature_cols)).fillna(0.0)
        x = (x - self._feature_mean.reindex(self._feature_cols).fillna(0.0)) / (
            self._feature_std.reindex(self._feature_cols).replace(0.0, np.nan).fillna(1.0) + EPS
        )
        clip = float(self.input_clip)
        if clip > 0.0:
            x = x.clip(lower=-clip, upper=clip)
        return x.to_numpy(dtype=np.float32)

    def _build_sequences(self, x: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        seq_len = int(max(2, self._effective_seq_len))
        xs: List[np.ndarray] = []
        ys: List[float] = []
        for i in range(seq_len - 1, len(x)):
            y = float(target[i])
            if not np.isfinite(y):
                continue
            seq = x[i - seq_len + 1 : i + 1]
            if not np.isfinite(seq).all():
                continue
            xs.append(seq.astype(np.float32))
            ys.append(y)
        if not xs:
            return np.zeros((0, seq_len, x.shape[1] if x.ndim == 2 else 0), dtype=np.float32), np.zeros(0, dtype=np.float32)
        yarr = np.asarray(ys, dtype=np.float32)
        clip = float(self.target_clip)
        if clip > 0.0:
            yarr = np.clip(yarr, -clip, clip)
        return np.stack(xs, axis=0).astype(np.float32), yarr

    def _loss(self, pred: Any, y: Any, torch: Any, F: Any) -> Any:
        scale = max(float(self._target_scale), EPS)
        directional_weight = torch.abs(y) / scale
        madl = torch.mean(-torch.sign(y) * pred * directional_weight)
        mode = str(self.loss_mode).strip().lower()
        if mode == "madl":
            return madl
        y_scaled = torch.tanh(y / scale)
        mse = F.mse_loss(pred, y_scaled)
        if mode == "mse":
            return mse
        return madl + float(self.mse_weight) * mse

    def _fallback_exposure(self, day_df: pd.DataFrame) -> tuple[float, Dict[str, float]]:
        raw = self._aggregate_market_raw(day_df, fit=False)
        signal = 0.0
        if not raw.empty and "close" in raw.columns:
            hist = pd.concat([self._history_market_raw, raw], ignore_index=True)
            close = _numeric(hist["close"]).ffill()
            if len(close) >= 2:
                short_ret = close.iloc[-1] / (close.iloc[max(0, len(close) - 6)] + EPS) - 1.0
                long_ret = close.iloc[-1] / (close.iloc[max(0, len(close) - 21)] + EPS) - 1.0
                signal = float(np.tanh(8.0 * short_ret + 4.0 * long_ret))
            self._append_history(raw)
        exposure = self._signal_to_exposure(signal)
        scaled_signal = float(np.clip(signal * float(self.signal_scale), -1.0, 1.0))
        return exposure, {
            "timing_enabled": 1.0,
            "timing_model": 2.0,
            "timing_lstm_fallback": 1.0,
            "timing_signal": scaled_signal,
            "timing_exposure": float(exposure),
        }

    def _append_history(self, raw: pd.DataFrame) -> None:
        if raw.empty:
            return
        hist = pd.concat([self._history_market_raw, raw], ignore_index=True)
        hist["_time_key"] = pd.to_datetime(hist["_time_key"], errors="coerce")
        hist = hist.dropna(subset=["_time_key"]).drop_duplicates(subset=["_time_key"], keep="last")
        hist = hist.sort_values("_time_key").tail(max(300, int(self._effective_seq_len) + 80)).reset_index(drop=True)
        self._history_market_raw = hist

    def _signal_to_exposure(self, signal: float) -> float:
        s = float(np.clip(signal, -1.0, 1.0)) * float(self.signal_scale)
        mode = str(self.exposure_mode).strip().lower()
        if mode in {"long_only_threshold", "report_daily_long", "threshold"}:
            return float(1.0 if s > float(self.long_threshold) else 0.0)
        if mode in {"raw", "raw_clip", "linear"}:
            return float(np.clip(max(s, 0.0), 0.0, 1.0))
        thresholds = _parse_float_list(self.band_thresholds)
        exposures = _parse_float_list(self.band_exposures)
        thresholds = sorted(thresholds)
        if len(exposures) != len(thresholds) + 1:
            exposures = [0.0, 0.3, 0.5, 0.8, 1.0]
            thresholds = [-0.1, 0.1, 0.6, 0.999999]
        for threshold, exposure in zip(thresholds, exposures):
            if s < threshold:
                return float(np.clip(exposure, 0.0, 1.0))
        return float(np.clip(exposures[-1], 0.0, 1.0))

    def _build_network(self, input_dim: int, torch: Any, nn: Any) -> Any:
        return _LSTMMADLNet.build(
            torch=torch,
            nn=nn,
            input_dim=int(input_dim),
            hidden_sizes=self._hidden_sizes(),
            dropout=float(self.dropout),
        )

    def fit(self, train_df: pd.DataFrame) -> "LSTMMADLTimingModel":
        torch, nn, F = self._require_torch()
        self._set_seed(torch)
        self._device_used = self._choose_device(torch, self.device)

        self._time_col = self._resolve_time_col(train_df)
        keys, inferred_intraday = self._time_key(train_df[self._time_col])
        self._is_intraday = bool(inferred_intraday)
        self._effective_seq_len = int(self.intraday_seq_len if self._is_intraday else self.seq_len)
        self._effective_seq_len = max(2, self._effective_seq_len)
        self._effective_feature_mode = self._mode_for_panel()

        mode_base_size = 7 if self._effective_feature_mode == "daily_bar" else 51
        self._extra_cols = self._select_extra_cols(train_df, budget=max(0, int(self.max_features) - mode_base_size))
        raw = self._aggregate_market_raw(train_df.assign(_timing_key_hint=keys), fit=True)
        raw = raw.dropna(subset=["_target"]).reset_index(drop=True) if "_target" in raw.columns else raw
        if raw.empty or "_target" not in raw.columns:
            self._train_summary = {"fallback": 1.0, "reason": 1.0, "train_market_rows": float(len(raw))}
            return self

        feature_df = self._prepare_feature_frame(raw)
        candidates = self._feature_candidates(feature_df)
        self._feature_cols = self._select_feature_cols(feature_df, candidates)
        if not self._feature_cols:
            self._train_summary = {"fallback": 1.0, "reason": 2.0, "train_market_rows": float(len(raw))}
            self._fit_history_market_raw = raw.tail(max(300, self._effective_seq_len + 80)).reset_index(drop=True)
            self._history_market_raw = self._fit_history_market_raw.copy()
            return self

        x = self._fit_transformer(feature_df)
        y = pd.to_numeric(feature_df["_target"], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        valid_abs = np.abs(y[np.isfinite(y)])
        self._target_scale = float(max(np.nanmedian(valid_abs) * 3.0 if len(valid_abs) else 0.0, 0.005))
        x_seq, y_seq = self._build_sequences(x, y)

        self._fit_history_market_raw = raw.tail(max(300, self._effective_seq_len + 80)).reset_index(drop=True)
        self._history_market_raw = self._fit_history_market_raw.copy()
        if len(y_seq) < int(self.min_train_samples):
            self._train_summary = {
                "fallback": 1.0,
                "reason": 3.0,
                "train_market_rows": float(len(raw)),
                "train_samples": float(len(y_seq)),
                "feature_count": float(len(self._feature_cols)),
            }
            return self

        self._model = self._build_network(input_dim=len(self._feature_cols), torch=torch, nn=nn)
        self._model.to(self._device_used)
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )

        n = len(y_seq)
        val_n = max(1, int(0.20 * n)) if n >= 50 else max(1, int(0.10 * n))
        train_n = max(1, n - val_n)
        x_train = torch.tensor(x_seq[:train_n], dtype=torch.float32)
        y_train = torch.tensor(y_seq[:train_n], dtype=torch.float32)
        x_val = torch.tensor(x_seq[train_n:], dtype=torch.float32, device=self._device_used)
        y_val = torch.tensor(y_seq[train_n:], dtype=torch.float32, device=self._device_used)
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=max(1, min(int(self.batch_size), train_n)),
            shuffle=True,
            drop_last=False,
        )

        best_loss = float("inf")
        best_state = None
        stale = 0
        for _epoch in range(max(1, int(self.n_epochs))):
            self._model.train()
            for xb, yb in loader:
                xb = xb.to(self._device_used)
                yb = yb.to(self._device_used)
                optimizer.zero_grad(set_to_none=True)
                pred = self._model(xb)
                loss = self._loss(pred, yb, torch, F)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5.0)
                optimizer.step()
            self._model.eval()
            with torch.no_grad():
                if len(x_val) > 0:
                    val_pred = self._model(x_val)
                    val_loss = float(self._loss(val_pred, y_val, torch, F).detach().cpu().item())
                else:
                    val_loss = 0.0
            if val_loss + 1e-7 < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= int(self.early_stop):
                    break
        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.to(self._device_used)
        self._model.eval()

        with torch.no_grad():
            pred_all = self._model(torch.tensor(x_seq, dtype=torch.float32, device=self._device_used)).detach().cpu().numpy()
        corr = float(np.corrcoef(pred_all, y_seq)[0, 1]) if len(y_seq) > 2 and np.std(pred_all) > EPS and np.std(y_seq) > EPS else 0.0
        dir_acc = float((np.sign(pred_all) == np.sign(y_seq)).mean()) if len(y_seq) else 0.0
        if bool(self.calibrate_sign) and np.isfinite(corr) and corr < 0.0:
            self._score_sign = -1.0
            pred_all = -pred_all
            corr = -corr
            dir_acc = float((np.sign(pred_all) == np.sign(y_seq)).mean()) if len(y_seq) else dir_acc
        else:
            self._score_sign = 1.0

        self._train_summary = {
            "fallback": 0.0,
            "train_market_rows": float(len(raw)),
            "train_samples": float(len(y_seq)),
            "feature_count": float(len(self._feature_cols)),
            "effective_seq_len": float(self._effective_seq_len),
            "is_intraday": float(self._is_intraday),
            "best_val_loss": float(best_loss),
            "train_pred_corr": float(corr if np.isfinite(corr) else 0.0),
            "train_direction_accuracy": float(dir_acc if np.isfinite(dir_acc) else 0.0),
            "score_sign": float(self._score_sign),
        }
        return self

    def predict_exposure(self, day_df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        if day_df.empty:
            return 0.0, {"timing_enabled": 1.0, "timing_model": 2.0, "timing_exposure": 0.0}
        if self._model is None or not self._feature_cols:
            return self._fallback_exposure(day_df)

        torch, _nn, _F = self._require_torch()
        raw = self._aggregate_market_raw(day_df, fit=False)
        if raw.empty:
            return 0.0, {"timing_enabled": 1.0, "timing_model": 2.0, "timing_exposure": 0.0, "timing_missing_row": 1.0}
        hist = pd.concat([self._history_market_raw, raw], ignore_index=True)
        hist["_time_key"] = pd.to_datetime(hist["_time_key"], errors="coerce")
        hist = hist.dropna(subset=["_time_key"]).drop_duplicates(subset=["_time_key"], keep="last")
        hist = hist.sort_values("_time_key").tail(max(300, self._effective_seq_len + 80)).reset_index(drop=True)
        feature_df = self._prepare_feature_frame(hist)
        x = self._transform_feature_frame(feature_df)
        if len(x) < int(self._effective_seq_len):
            first = x[:1] if len(x) else np.zeros((1, len(self._feature_cols)), dtype=np.float32)
            pad = np.repeat(first, repeats=int(self._effective_seq_len) - len(x), axis=0)
            seq = np.vstack([pad, x]) if len(x) else np.repeat(first, repeats=int(self._effective_seq_len), axis=0)
        else:
            seq = x[-int(self._effective_seq_len) :]
        self._model.eval()
        with torch.no_grad():
            pred = self._model(torch.tensor(seq[None, :, :], dtype=torch.float32, device=self._device_used))
            raw_signal = float(pred.detach().cpu().numpy().reshape(-1)[0])
        signal = float(np.clip(raw_signal * float(self._score_sign), -1.0, 1.0))
        exposure = self._signal_to_exposure(signal)
        scaled_signal = float(np.clip(signal * float(self.signal_scale), -1.0, 1.0))
        self._append_history(raw)
        diag = {
            "timing_enabled": 1.0,
            "timing_model": 2.0,
            "timing_lstm_fallback": 0.0,
            "timing_raw_signal": raw_signal,
            "timing_signal": scaled_signal,
            "timing_exposure": float(exposure),
            "timing_lstm_feature_count": float(len(self._feature_cols)),
            "timing_lstm_seq_len": float(self._effective_seq_len),
            "timing_lstm_score_sign": float(self._score_sign),
        }
        return float(exposure), diag

    def _config_dict(self) -> Dict[str, object]:
        return {
            "seq_len": int(self.seq_len),
            "intraday_seq_len": int(self.intraday_seq_len),
            "hidden_sizes": self._hidden_sizes(),
            "dropout": float(self.dropout),
            "n_epochs": int(self.n_epochs),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "early_stop": int(self.early_stop),
            "batch_size": int(self.batch_size),
            "min_train_samples": int(self.min_train_samples),
            "feature_mode": str(self.feature_mode),
            "max_features": int(self.max_features),
            "input_clip": float(self.input_clip),
            "target_clip": float(self.target_clip),
            "loss_mode": str(self.loss_mode),
            "mse_weight": float(self.mse_weight),
            "exposure_mode": str(self.exposure_mode),
            "long_threshold": float(self.long_threshold),
            "band_thresholds": _parse_float_list(self.band_thresholds),
            "band_exposures": _parse_float_list(self.band_exposures),
            "signal_scale": float(self.signal_scale),
            "calibrate_sign": bool(self.calibrate_sign),
            "market_agg": str(self.market_agg),
            "extra_feature_limit": int(self.extra_feature_limit),
            "random_state": int(self.random_state),
            "device_used": self._device_used,
        }

    def save(self, folder: Path, run_tag: str) -> Dict[str, str]:
        torch, _nn, _F = self._require_torch()
        folder.mkdir(parents=True, exist_ok=True)
        model_path = folder / f"timing_lstm_madl_{run_tag}.pt"
        meta_path = folder / f"timing_lstm_madl_{run_tag}.json"
        history = self._fit_history_market_raw.copy()
        if "_time_key" in history.columns:
            history["_time_key"] = pd.to_datetime(history["_time_key"], errors="coerce").astype(str)
        checkpoint = {
            "state_dict": self._model.state_dict() if self._model is not None else None,
            "config": self._config_dict(),
            "time_col": self._time_col,
            "is_intraday": bool(self._is_intraday),
            "effective_seq_len": int(self._effective_seq_len),
            "effective_feature_mode": self._effective_feature_mode,
            "feature_cols": list(self._feature_cols),
            "extra_cols": list(self._extra_cols),
            "feature_fill": self._feature_fill.to_dict(),
            "feature_mean": self._feature_mean.to_dict(),
            "feature_std": self._feature_std.to_dict(),
            "target_scale": float(self._target_scale),
            "score_sign": float(self._score_sign),
            "train_summary": dict(self._train_summary),
            "fit_history_market_raw": history.to_dict(orient="list"),
        }
        torch.save(checkpoint, model_path)
        dump_json(
            meta_path,
            {
                "model_type": "lstm_madl",
                "model_pt": str(model_path),
                "feature_count": len(self._feature_cols),
                "train_summary": self._train_summary,
                "config": self._config_dict(),
            },
        )
        return {"model_pt": str(model_path), "meta_json": str(meta_path)}

    @classmethod
    def load(cls, model_path: str | Path, cfg_obj: object | None = None) -> "LSTMMADLTimingModel":
        path = Path(model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"timing LSTM model file not found: {path}")
        if path.suffix.lower() == ".json":
            meta = json.loads(path.read_text(encoding="utf-8"))
            pt = meta.get("model_pt")
            if not pt:
                raise ValueError("timing_lstm_madl JSON metadata does not contain model_pt.")
            pt_path = Path(str(pt))
            if not pt_path.is_absolute():
                pt_path = path.parent / pt_path
            return cls.load(pt_path, cfg_obj=cfg_obj)

        torch, nn, _F = cls._require_torch()
        try:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(path), map_location="cpu")
        conf = dict(ckpt.get("config", {}) or {})

        def get_cfg(name: str, default: Any) -> Any:
            return getattr(cfg_obj, name, conf.get(name, default)) if cfg_obj is not None else conf.get(name, default)

        model = cls(
            seq_len=int(conf.get("seq_len", get_cfg("lstm_seq_len", 20))),
            intraday_seq_len=int(conf.get("intraday_seq_len", get_cfg("lstm_intraday_seq_len", 48))),
            hidden_sizes=conf.get("hidden_sizes", get_cfg("lstm_hidden_sizes", "512,256,128")),
            dropout=float(conf.get("dropout", get_cfg("lstm_dropout", 0.2))),
            n_epochs=int(conf.get("n_epochs", get_cfg("lstm_epochs", 120))),
            lr=float(conf.get("lr", get_cfg("lstm_lr", 1e-3))),
            weight_decay=float(conf.get("weight_decay", get_cfg("lstm_weight_decay", 1e-4))),
            early_stop=int(conf.get("early_stop", get_cfg("lstm_early_stop", 15))),
            batch_size=int(conf.get("batch_size", get_cfg("lstm_batch_size", 128))),
            min_train_samples=int(conf.get("min_train_samples", get_cfg("lstm_min_train_samples", 80))),
            feature_mode=str(conf.get("feature_mode", get_cfg("lstm_feature_mode", "auto"))),
            max_features=int(conf.get("max_features", get_cfg("lstm_max_features", 96))),
            input_clip=float(conf.get("input_clip", get_cfg("lstm_input_clip", 5.0))),
            target_clip=float(conf.get("target_clip", get_cfg("lstm_target_clip", 0.20))),
            loss_mode=str(conf.get("loss_mode", get_cfg("lstm_loss_mode", "madl_mse"))),
            mse_weight=float(conf.get("mse_weight", get_cfg("lstm_mse_weight", 0.05))),
            exposure_mode=str(conf.get("exposure_mode", get_cfg("lstm_exposure_mode", "long_only_bands"))),
            long_threshold=float(conf.get("long_threshold", get_cfg("lstm_long_threshold", -0.3))),
            band_thresholds=conf.get("band_thresholds", get_cfg("lstm_band_thresholds", "-0.1,0.1,0.6,0.999999")),
            band_exposures=conf.get("band_exposures", get_cfg("lstm_band_exposures", "0.0,0.3,0.5,0.8,1.0")),
            signal_scale=float(conf.get("signal_scale", get_cfg("lstm_signal_scale", 1.0))),
            calibrate_sign=bool(conf.get("calibrate_sign", get_cfg("lstm_calibrate_sign", True))),
            market_agg=str(conf.get("market_agg", get_cfg("lstm_market_agg", "amount_weighted"))),
            extra_feature_limit=int(conf.get("extra_feature_limit", get_cfg("lstm_extra_feature_limit", 48))),
            random_state=int(conf.get("random_state", get_cfg("random_state", 1000))),
            device=str(get_cfg("lstm_device", conf.get("device_used", "auto"))),
        )
        model._time_col = ckpt.get("time_col")
        model._is_intraday = bool(ckpt.get("is_intraday", False))
        model._effective_seq_len = int(ckpt.get("effective_seq_len", model.intraday_seq_len if model._is_intraday else model.seq_len))
        model._effective_feature_mode = str(ckpt.get("effective_feature_mode", model._mode_for_panel()))
        model._feature_cols = [str(x) for x in ckpt.get("feature_cols", [])]
        model._extra_cols = [str(x) for x in ckpt.get("extra_cols", [])]
        model._feature_fill = pd.Series(ckpt.get("feature_fill", {}) or {}, dtype=float)
        model._feature_mean = pd.Series(ckpt.get("feature_mean", {}) or {}, dtype=float)
        model._feature_std = pd.Series(ckpt.get("feature_std", {}) or {}, dtype=float).replace(0.0, 1.0)
        model._target_scale = float(ckpt.get("target_scale", 0.01))
        model._score_sign = float(ckpt.get("score_sign", 1.0))
        model._train_summary = dict(ckpt.get("train_summary", {}) or {})
        hist_payload = ckpt.get("fit_history_market_raw", {}) or {}
        hist = pd.DataFrame(hist_payload)
        if "_time_key" in hist.columns:
            hist["_time_key"] = pd.to_datetime(hist["_time_key"], errors="coerce")
            hist = hist.dropna(subset=["_time_key"]).sort_values("_time_key").reset_index(drop=True)
        model._fit_history_market_raw = hist
        model._history_market_raw = hist.copy()
        model._device_used = model._choose_device(torch, model.device)
        state = ckpt.get("state_dict")
        if state is not None and model._feature_cols:
            net = model._build_network(input_dim=len(model._feature_cols), torch=torch, nn=nn)
            net.load_state_dict(state)
            net.to(model._device_used)
            net.eval()
            model._model = net
        return model
