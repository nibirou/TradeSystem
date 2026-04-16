"""执行模型工厂。

作用：
1. 将 `ExecutionModelConfig` 映射到成交仿真模型；
2. 提供理想成交与现实成交两套实现；
3. 支持用户自定义执行插件。
"""

from __future__ import annotations

from ...config import ExecutionModelConfig
from ...core.utils import import_module_from_file
from ..base import ExecutionModel
from .engines import IdealFillExecutionModel, RealisticFillExecutionModel


def build_execution_model(cfg: ExecutionModelConfig) -> ExecutionModel:
    # 自定义执行模型优先。
    if cfg.custom_model_py:
        mod = import_module_from_file(cfg.custom_model_py, module_name="strategy7_custom_execution_model")
        if not hasattr(mod, "build_model"):
            raise RuntimeError("custom execution model module must provide build_model(cfg).")
        model = mod.build_model(cfg)
        if not isinstance(model, ExecutionModel):
            raise TypeError("custom execution model must inherit ExecutionModel.")
        return model
    # 理想成交：不受流动性与延迟约束。
    if cfg.model_type == "ideal_fill":
        return IdealFillExecutionModel()
    # 现实成交：受参与率、流动性状态、延迟惩罚影响。
    if cfg.model_type == "realistic_fill":
        return RealisticFillExecutionModel(cfg=cfg)
    raise ValueError(f"unsupported execution model type: {cfg.model_type}")
