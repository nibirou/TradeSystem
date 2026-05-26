"""Timing model factory."""

from __future__ import annotations

from ...config import TimingModelConfig
from ...core.utils import import_module_from_file
from ..base import TimingModel
from .lstm_madl import LSTMMADLTimingModel
from .models import NoTimingModel, VolatilityRegimeTimingModel


def _canonical_timing_type(model_type: str) -> str:
    t = str(model_type).strip().lower()
    if t in {"lstm", "madl_lstm", "lstm-madl", "lstm_madl_timing"}:
        return "lstm_madl"
    return t


def build_timing_model(cfg: TimingModelConfig) -> TimingModel:
    if cfg.custom_model_py:
        mod = import_module_from_file(cfg.custom_model_py, module_name="strategy7_custom_timing_model")
        if not hasattr(mod, "build_model"):
            raise RuntimeError("custom timing model module must provide build_model(cfg).")
        model = mod.build_model(cfg)
        if not isinstance(model, TimingModel):
            raise TypeError("custom timing model must inherit TimingModel.")
        return model

    model_type = _canonical_timing_type(cfg.model_type)
    if model_type == "none":
        return NoTimingModel()
    if model_type == "volatility_regime":
        return VolatilityRegimeTimingModel(
            vol_threshold=cfg.vol_threshold,
            momentum_threshold=cfg.momentum_threshold,
        )
    if model_type == "lstm_madl":
        return LSTMMADLTimingModel(
            seq_len=cfg.lstm_seq_len,
            intraday_seq_len=cfg.lstm_intraday_seq_len,
            hidden_sizes=cfg.lstm_hidden_sizes,
            dropout=cfg.lstm_dropout,
            n_epochs=cfg.lstm_epochs,
            lr=cfg.lstm_lr,
            weight_decay=cfg.lstm_weight_decay,
            early_stop=cfg.lstm_early_stop,
            batch_size=cfg.lstm_batch_size,
            min_train_samples=cfg.lstm_min_train_samples,
            feature_mode=cfg.lstm_feature_mode,
            max_features=cfg.lstm_max_features,
            input_clip=cfg.lstm_input_clip,
            target_clip=cfg.lstm_target_clip,
            loss_mode=cfg.lstm_loss_mode,
            mse_weight=cfg.lstm_mse_weight,
            exposure_mode=cfg.lstm_exposure_mode,
            long_threshold=cfg.lstm_long_threshold,
            band_thresholds=cfg.lstm_band_thresholds,
            band_exposures=cfg.lstm_band_exposures,
            signal_scale=cfg.lstm_signal_scale,
            calibrate_sign=cfg.lstm_calibrate_sign,
            market_agg=cfg.lstm_market_agg,
            extra_feature_limit=cfg.lstm_extra_feature_limit,
            random_state=cfg.random_state,
            device=cfg.lstm_device,
        )
    raise ValueError(f"unsupported timing model type: {cfg.model_type}")
