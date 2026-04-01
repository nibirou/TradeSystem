"""Factory for execution engine models."""

from __future__ import annotations

from ...config import ExecutionModelConfig
from ...core.utils import import_module_from_file
from ..base import ExecutionModel
from .engines import IdealFillExecutionModel, RealisticFillExecutionModel


def build_execution_model(cfg: ExecutionModelConfig) -> ExecutionModel:
    if cfg.custom_model_py:
        mod = import_module_from_file(cfg.custom_model_py, module_name="strategy7_custom_execution_model")
        if not hasattr(mod, "build_model"):
            raise RuntimeError("custom execution model module must provide build_model(cfg).")
        model = mod.build_model(cfg)
        if not isinstance(model, ExecutionModel):
            raise TypeError("custom execution model must inherit ExecutionModel.")
        return model
    if cfg.model_type == "ideal_fill":
        return IdealFillExecutionModel()
    if cfg.model_type == "realistic_fill":
        return RealisticFillExecutionModel(cfg=cfg)
    raise ValueError(f"unsupported execution model type: {cfg.model_type}")
