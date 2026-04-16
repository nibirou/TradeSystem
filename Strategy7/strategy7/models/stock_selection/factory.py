"""选股模型工厂。

职责：
1. 根据 `StockModelConfig.model_type` 选择具体实现；
2. 将配置参数一一映射到模型构造函数；
3. 支持自定义插件模型（`custom_model_py`）。
"""

from __future__ import annotations

from ...config import StockModelConfig
from ...core.utils import import_module_from_file
from ..base import StockSelectionModel
from .tree_model import TreeStockModel


def build_stock_model(cfg: StockModelConfig) -> StockSelectionModel:
    # 1) 自定义模型优先：用户可以完全接管选股模型实现。
    if cfg.custom_model_py:
        mod = import_module_from_file(cfg.custom_model_py, module_name="strategy7_custom_stock_model")
        if not hasattr(mod, "build_model"):
            raise RuntimeError("custom stock model module must provide build_model(cfg).")
        model = mod.build_model(cfg)
        if not isinstance(model, StockSelectionModel):
            raise TypeError("custom stock model must inherit StockSelectionModel.")
        return model

    # 2) 经典决策树基线模型。
    if cfg.model_type == "decision_tree":
        return TreeStockModel(
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.random_state,
        )

    # 3) FactorGCL：研报复现的超图卷积 + 时间对比学习模型。
    # 参数映射规则：cfg.fgcl_* -> FactorGCLStockModel 同名构造参数。
    if cfg.model_type in {"factor_gcl", "factorgcl", "dfq_factorgcl"}:
        from .factor_gcl_model import FactorGCLStockModel

        return FactorGCLStockModel(
            # 时序建模窗口
            seq_len=cfg.fgcl_seq_len,
            future_look=cfg.fgcl_future_look,
            # 网络结构
            hidden_size=cfg.fgcl_hidden_size,
            num_layers=cfg.fgcl_num_layers,
            num_factor=cfg.fgcl_num_factor,
            # 损失函数超参数
            gamma=cfg.fgcl_gamma,
            tau=cfg.fgcl_tau,
            # 训练超参数
            n_epochs=cfg.fgcl_epochs,
            lr=cfg.fgcl_lr,
            early_stop=cfg.fgcl_early_stop,
            smooth_steps=cfg.fgcl_smooth_steps,
            per_epoch_batch=cfg.fgcl_per_epoch_batch,
            batch_size=cfg.fgcl_batch_size,
            label_transform=cfg.fgcl_label_transform,
            weight_decay=cfg.fgcl_weight_decay,
            dropout=cfg.fgcl_dropout,
            # 运行控制
            random_state=cfg.random_state,
            device=cfg.fgcl_device,
        )

    # 4) DAFAT：增强 Transformer 多尺度结构。
    # 参数映射规则：cfg.dafat_* -> DAFATStockModel 同名构造参数。
    if cfg.model_type in {"dafat", "dafat_transformer", "transformer_dafat"}:
        from .dafat_transformer_model import DAFATStockModel

        return DAFATStockModel(
            # 序列与网络结构
            seq_len=cfg.dafat_seq_len,
            hidden_size=cfg.dafat_hidden_size,
            num_layers=cfg.dafat_num_layers,
            num_heads=cfg.dafat_num_heads,
            ffn_mult=cfg.dafat_ffn_mult,
            dropout=cfg.dafat_dropout,
            # 稀疏注意力与多尺度模块
            local_window=cfg.dafat_local_window,
            topk_ratio=cfg.dafat_topk_ratio,
            vol_quantile=cfg.dafat_vol_quantile,
            meso_scale=cfg.dafat_meso_scale,
            macro_scale=cfg.dafat_macro_scale,
            # 训练超参数
            n_epochs=cfg.dafat_epochs,
            lr=cfg.dafat_lr,
            weight_decay=cfg.dafat_weight_decay,
            early_stop=cfg.dafat_early_stop,
            per_epoch_batch=cfg.dafat_per_epoch_batch,
            batch_size=cfg.dafat_batch_size,
            label_transform=cfg.dafat_label_transform,
            mse_weight=cfg.dafat_mse_weight,
            # 模块开关
            use_dpe=cfg.dafat_use_dpe,
            use_sparse_attn=cfg.dafat_use_sparse_attn,
            use_multiscale=cfg.dafat_use_multiscale,
            # 运行控制
            random_state=cfg.random_state,
            device=cfg.dafat_device,
        )
    raise ValueError(f"unsupported stock model type: {cfg.model_type}")
