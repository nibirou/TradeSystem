"""Factory for timing models."""

from __future__ import annotations

from ...config import TimingModelConfig
from ...core.utils import import_module_from_file
from ..base import TimingModel
from .models import NoTimingModel, VolatilityRegimeTimingModel


def build_timing_model(cfg: TimingModelConfig) -> TimingModel:
    if cfg.custom_model_py:
        mod = import_module_from_file(cfg.custom_model_py, module_name="strategy7_custom_timing_model")
        if not hasattr(mod, "build_model"):
            raise RuntimeError("custom timing model module must provide build_model(cfg).")
        model = mod.build_model(cfg)
        if not isinstance(model, TimingModel):
            raise TypeError("custom timing model must inherit TimingModel.")
        return model
    if cfg.model_type == "none":
        return NoTimingModel()
    if cfg.model_type == "volatility_regime":
        return VolatilityRegimeTimingModel(
            vol_threshold=cfg.vol_threshold,
            momentum_threshold=cfg.momentum_threshold,
        )
    raise ValueError(f"unsupported timing model type: {cfg.model_type}")
