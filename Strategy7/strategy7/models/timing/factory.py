"""择时模型工厂。

说明：
1. 负责将 `TimingModelConfig` 映射为具体择时模型；
2. 支持内置模型与自定义插件模型两种路径。
"""

from __future__ import annotations

from ...config import TimingModelConfig
from ...core.utils import import_module_from_file
from ..base import TimingModel
from .models import NoTimingModel, VolatilityRegimeTimingModel


def build_timing_model(cfg: TimingModelConfig) -> TimingModel:
    # 自定义模型优先：便于外部扩展状态机/宏观择时等策略。
    if cfg.custom_model_py:
        mod = import_module_from_file(cfg.custom_model_py, module_name="strategy7_custom_timing_model")
        if not hasattr(mod, "build_model"):
            raise RuntimeError("custom timing model module must provide build_model(cfg).")
        model = mod.build_model(cfg)
        if not isinstance(model, TimingModel):
            raise TypeError("custom timing model must inherit TimingModel.")
        return model

    # 不做择时：始终满仓暴露。
    if cfg.model_type == "none":
        return NoTimingModel()
    # 波动-动量状态择时：通过阈值控制风险暴露。
    if cfg.model_type == "volatility_regime":
        return VolatilityRegimeTimingModel(
            vol_threshold=cfg.vol_threshold,
            momentum_threshold=cfg.momentum_threshold,
        )
    raise ValueError(f"unsupported timing model type: {cfg.model_type}")
