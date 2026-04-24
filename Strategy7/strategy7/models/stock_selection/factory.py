"""Stock-selection model factory."""

from __future__ import annotations

from ...config import StockModelConfig
from ...core.utils import import_module_from_file
from ..base import StockSelectionModel
from .launch_boost_model import LaunchBoostStockModel
from .tree_model import TreeStockModel


def build_stock_model(cfg: StockModelConfig) -> StockSelectionModel:
    # 1) Custom model takes highest priority.
    if cfg.custom_model_py:
        mod = import_module_from_file(cfg.custom_model_py, module_name="strategy7_custom_stock_model")
        if not hasattr(mod, "build_model"):
            raise RuntimeError("custom stock model module must provide build_model(cfg).")
        model = mod.build_model(cfg)
        if not isinstance(model, StockSelectionModel):
            raise TypeError("custom stock model must inherit StockSelectionModel.")
        return model

    model_type = str(cfg.model_type).strip().lower()

    # 2) Tree baseline.
    if model_type == "decision_tree":
        return TreeStockModel(
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.random_state,
        )

    # 3) Launch-boost model for bottom-launch stock selection.
    if model_type in {"launch_boost", "bottom_launch_boost", "low_start_boost", "launch10_boost"}:
        return LaunchBoostStockModel(
            max_depth=cfg.launch_boost_max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            learning_rate=cfg.launch_boost_learning_rate,
            max_iter=cfg.launch_boost_max_iter,
            l2_regularization=cfg.launch_boost_l2,
            return_head_weight=cfg.launch_boost_return_head_weight,
            random_state=cfg.random_state,
        )

    # 4) FactorGCL.
    if model_type in {"factor_gcl", "factorgcl", "dfq_factorgcl"}:
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

    # 5) DAFAT.
    if model_type in {"dafat", "dafat_transformer", "transformer_dafat"}:
        from .dafat_transformer_model import DAFATStockModel

        return DAFATStockModel(
            seq_len=cfg.dafat_seq_len,
            hidden_size=cfg.dafat_hidden_size,
            num_layers=cfg.dafat_num_layers,
            num_heads=cfg.dafat_num_heads,
            ffn_mult=cfg.dafat_ffn_mult,
            dropout=cfg.dafat_dropout,
            local_window=cfg.dafat_local_window,
            topk_ratio=cfg.dafat_topk_ratio,
            vol_quantile=cfg.dafat_vol_quantile,
            meso_scale=cfg.dafat_meso_scale,
            macro_scale=cfg.dafat_macro_scale,
            n_epochs=cfg.dafat_epochs,
            lr=cfg.dafat_lr,
            weight_decay=cfg.dafat_weight_decay,
            early_stop=cfg.dafat_early_stop,
            per_epoch_batch=cfg.dafat_per_epoch_batch,
            batch_size=cfg.dafat_batch_size,
            label_transform=cfg.dafat_label_transform,
            mse_weight=cfg.dafat_mse_weight,
            use_dpe=cfg.dafat_use_dpe,
            use_sparse_attn=cfg.dafat_use_sparse_attn,
            use_multiscale=cfg.dafat_use_multiscale,
            random_state=cfg.random_state,
            device=cfg.dafat_device,
        )

    raise ValueError(f"unsupported stock model type: {cfg.model_type}")
