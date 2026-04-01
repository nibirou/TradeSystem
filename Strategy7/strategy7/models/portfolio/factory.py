"""Factory for portfolio weighting models."""

from __future__ import annotations

from ...config import PortfolioOptConfig
from ...core.utils import import_module_from_file
from ..base import PortfolioModel
from .weighting import DynamicOptimizationPortfolioModel, EqualWeightPortfolioModel


def build_portfolio_model(cfg: PortfolioOptConfig) -> PortfolioModel:
    if cfg.custom_model_py:
        mod = import_module_from_file(cfg.custom_model_py, module_name="strategy7_custom_portfolio_model")
        if not hasattr(mod, "build_model"):
            raise RuntimeError("custom portfolio model module must provide build_model(cfg).")
        model = mod.build_model(cfg)
        if not isinstance(model, PortfolioModel):
            raise TypeError("custom portfolio model must inherit PortfolioModel.")
        return model
    if cfg.mode == "equal_weight":
        return EqualWeightPortfolioModel()
    if cfg.mode == "dynamic_opt":
        return DynamicOptimizationPortfolioModel(cfg=cfg)
    raise ValueError(f"unsupported portfolio mode: {cfg.mode}")
