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
    if cfg.model_type in {"factor_gcl", "factorgcl", "dfq_factorgcl"}:
        from .factor_gcl_model import FactorGCLStockModel

        return FactorGCLStockModel(
            seq_len=cfg.fgcl_seq_len,
            future_look=cfg.fgcl_future_look,
            hidden_size=cfg.fgcl_hidden_size,
            num_layers=cfg.fgcl_num_layers,
            num_factor=cfg.fgcl_num_factor,
            gamma=cfg.fgcl_gamma,
            tau=cfg.fgcl_tau,
            n_epochs=cfg.fgcl_epochs,
            lr=cfg.fgcl_lr,
            early_stop=cfg.fgcl_early_stop,
            smooth_steps=cfg.fgcl_smooth_steps,
            per_epoch_batch=cfg.fgcl_per_epoch_batch,
            batch_size=cfg.fgcl_batch_size,
            label_transform=cfg.fgcl_label_transform,
            weight_decay=cfg.fgcl_weight_decay,
            dropout=cfg.fgcl_dropout,
            random_state=cfg.random_state,
            device=cfg.fgcl_device,
        )
    raise ValueError(f"unsupported stock model type: {cfg.model_type}")
