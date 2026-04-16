"""组合模型工厂。

职责：
1. 根据 `PortfolioOptConfig.mode` 选择等权或动态优化模型；
2. 保证自定义模型满足统一的 `PortfolioModel` 接口。
"""

from __future__ import annotations

from ...config import PortfolioOptConfig
from ...core.utils import import_module_from_file
from ..base import PortfolioModel
from .weighting import DynamicOptimizationPortfolioModel, EqualWeightPortfolioModel


def build_portfolio_model(cfg: PortfolioOptConfig) -> PortfolioModel:
    # 自定义组合模型优先：用于接入外部优化器或约束体系。
    if cfg.custom_model_py:
        mod = import_module_from_file(cfg.custom_model_py, module_name="strategy7_custom_portfolio_model")
        if not hasattr(mod, "build_model"):
            raise RuntimeError("custom portfolio model module must provide build_model(cfg).")
        model = mod.build_model(cfg)
        if not isinstance(model, PortfolioModel):
            raise TypeError("custom portfolio model must inherit PortfolioModel.")
        return model
    # 等权组合：简单、稳定、可作为基准。
    if cfg.mode == "equal_weight":
        return EqualWeightPortfolioModel()
    # 动态优化组合：显式纳入风险、风格、行业、拥挤与成本惩罚。
    if cfg.mode == "dynamic_opt":
        return DynamicOptimizationPortfolioModel(cfg=cfg)
    raise ValueError(f"unsupported portfolio mode: {cfg.mode}")
