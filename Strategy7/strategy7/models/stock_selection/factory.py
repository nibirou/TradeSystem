"""Factory for stock selection models."""

from __future__ import annotations

from ...config import StockModelConfig
from ...core.utils import import_module_from_file
from ..base import StockSelectionModel
from .tree_model import TreeStockModel


def build_stock_model(cfg: StockModelConfig) -> StockSelectionModel:
    if cfg.custom_model_py:
        mod = import_module_from_file(cfg.custom_model_py, module_name="strategy7_custom_stock_model")
        if not hasattr(mod, "build_model"):
            raise RuntimeError("custom stock model module must provide build_model(cfg).")
        model = mod.build_model(cfg)
        if not isinstance(model, StockSelectionModel):
            raise TypeError("custom stock model must inherit StockSelectionModel.")
        return model
    if cfg.model_type == "decision_tree":
        return TreeStockModel(
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.random_state,
        )
    raise ValueError(f"unsupported stock model type: {cfg.model_type}")
