"""Model artifact resolution and loading helpers."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import (
    ExecutionModelConfig,
    PortfolioOptConfig,
    RunConfig,
    StockModelConfig,
    TimingModelConfig,
)
from ..core.utils import import_module_from_file
from .base import ExecutionModel, PortfolioModel, StockSelectionModel, TimingModel
from .execution.engines import IdealFillExecutionModel, RealisticFillExecutionModel
from .portfolio.weighting import DynamicOptimizationPortfolioModel, EqualWeightPortfolioModel
from .stock_selection.dafat_transformer_model import DAFATStockModel
from .stock_selection.factor_gcl_model import FactorGCLStockModel
from .stock_selection.launch_boost_model import LaunchBoostStockModel
from .stock_selection.tree_model import TreeStockModel
from .timing.models import NoTimingModel, VolatilityRegimeTimingModel


@dataclass
class ResolvedModelPaths:
    stock_model: str | None
    timing_model: str | None
    portfolio_model: str | None
    execution_model: str | None
    source: Dict[str, str]


def _normalize_stock_model_type(model_type: str) -> str:
    t = str(model_type).strip().lower()
    if t in {"factor_gcl", "factorgcl", "dfq_factorgcl"}:
        return "factor_gcl"
    if t in {"dafat", "dafat_transformer", "transformer_dafat"}:
        return "dafat"
    if t in {"launch_boost", "bottom_launch_boost", "low_start_boost", "launch10_boost"}:
        return "launch_boost"
    return t


def _normalize_timing_model_type(model_type: str) -> str:
    return str(model_type).strip().lower()


def _normalize_portfolio_mode(mode: str) -> str:
    return str(mode).strip().lower()


def _normalize_execution_model_type(model_type: str) -> str:
    return str(model_type).strip().lower()


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def _pick_component_file(entry: object) -> str | None:
    if not isinstance(entry, dict):
        return None
    for key in ("model_pt", "model_pkl", "model_file", "meta_json"):
        v = entry.get(key)
        if v:
            return str(v)
    return None


def _resolve_path_from_candidates(raw_path: str, *, base_dir: Path | None = None) -> Path | None:
    if not str(raw_path).strip():
        return None
    raw = Path(str(raw_path).strip()).expanduser()
    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        if base_dir is not None:
            candidates.append((base_dir / raw).expanduser())
        candidates.append(raw)
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def _coerce_dataclass_kwargs(dataclass_type, payload: Dict[str, Any], fallback_obj: object) -> Dict[str, Any]:
    valid = {f.name for f in fields(dataclass_type)}
    base = {k: getattr(fallback_obj, k) for k in valid}
    for k, v in (payload or {}).items():
        if k in valid:
            base[k] = v
    return base


def _candidate_name(component: str, model_type: str, run_tag: str | None) -> str | None:
    tag = str(run_tag).strip() if run_tag else ""
    suffix = f"_{tag}" if tag else ""
    if component == "stock_model":
        m = _normalize_stock_model_type(model_type)
        if m == "decision_tree":
            return f"stock_model_tree{suffix}.pkl"
        if m == "launch_boost":
            return f"stock_model_launch_boost{suffix}.pkl"
        if m == "factor_gcl":
            return f"stock_model_factor_gcl{suffix}.pt"
        if m == "dafat":
            return f"stock_model_dafat{suffix}.pt"
    if component == "timing_model":
        m = _normalize_timing_model_type(model_type)
        if m == "none":
            return f"timing_none{suffix}.json"
        if m == "volatility_regime":
            return f"timing_vol_regime{suffix}.pkl"
    if component == "portfolio_model":
        m = _normalize_portfolio_mode(model_type)
        if m == "equal_weight":
            return f"portfolio_equal{suffix}.json"
        if m == "dynamic_opt":
            return f"portfolio_dynamic{suffix}.pkl"
    if component == "execution_model":
        m = _normalize_execution_model_type(model_type)
        if m == "ideal_fill":
            return f"execution_ideal{suffix}.json"
        if m == "realistic_fill":
            return f"execution_realistic{suffix}.pkl"
    return None


def _latest_matching_file(models_dir: Path, component: str, model_type: str) -> str | None:
    filename = _candidate_name(component=component, model_type=model_type, run_tag=None)
    if not filename:
        return None
    exact = models_dir / filename
    if exact.exists() and exact.is_file():
        return str(exact.resolve())
    stem, ext = Path(filename).stem, Path(filename).suffix
    pattern = f"{stem}_*{ext}"
    matches = [p for p in models_dir.glob(pattern) if p.is_file()]
    if not matches:
        broad_pattern = f"{stem}*{ext}"
        matches = [p for p in models_dir.glob(broad_pattern) if p.is_file()]
    if not matches:
        return None
    latest = max(matches, key=lambda p: p.stat().st_mtime)
    return str(latest.resolve())


def _resolve_from_dir(models_dir: Path, component: str, model_type: str, run_tag: str | None) -> str | None:
    if run_tag:
        filename = _candidate_name(component=component, model_type=model_type, run_tag=run_tag)
        if filename:
            p = models_dir / filename
            if p.exists():
                return str(p.resolve())
            # fallback: allow legacy names with "_run_tag" or minor naming changes
            prefix = f"{Path(filename).stem}_"
            ext = Path(filename).suffix
            fallback = [x for x in models_dir.glob(f"{prefix}*{ext}") if x.is_file()]
            if fallback:
                latest = max(fallback, key=lambda q: q.stat().st_mtime)
                return str(latest.resolve())
            fallback_broad = [x for x in models_dir.glob(f"{Path(filename).stem}*{ext}") if x.is_file()]
            if fallback_broad:
                latest = max(fallback_broad, key=lambda q: q.stat().st_mtime)
                return str(latest.resolve())
    return _latest_matching_file(models_dir=models_dir, component=component, model_type=model_type)


def resolve_model_artifact_paths(cfg: RunConfig) -> ResolvedModelPaths:
    model_run = cfg.model_run
    summary_entries: Dict[str, object] = {}
    summary_base_dir: Path | None = None
    if model_run.model_summary_json:
        summary_path = Path(model_run.model_summary_json).expanduser()
        if not summary_path.exists():
            raise FileNotFoundError(f"model_summary_json not found: {summary_path}")
        summary = _safe_read_json(summary_path)
        summary_entries = dict(
            summary.get("outputs", {}).get("model_files", {})
            or summary.get("model_files", {})
            or {}
        )
        summary_base_dir = summary_path.parent

    models_dir = Path(model_run.models_load_dir).expanduser() if model_run.models_load_dir else None
    if models_dir is not None and not models_dir.exists():
        raise FileNotFoundError(f"models_load_dir not found: {models_dir}")

    run_tag = model_run.models_load_run_tag
    source: Dict[str, str] = {
        "stock_model": "none",
        "timing_model": "none",
        "portfolio_model": "none",
        "execution_model": "none",
    }
    # Components that are deterministic by config default do not require artifact files.
    # Custom plugins may still consume model_path, so custom_* keeps path resolution enabled.
    timing_needs_artifact = bool(cfg.timing_model.custom_model_py) or (
        _normalize_timing_model_type(cfg.timing_model.model_type) != "none"
    )
    portfolio_needs_artifact = bool(cfg.portfolio_opt.custom_model_py) or (
        _normalize_portfolio_mode(cfg.portfolio_opt.mode) != "equal_weight"
    )
    execution_needs_artifact = bool(cfg.execution_model.custom_model_py) or (
        _normalize_execution_model_type(cfg.execution_model.model_type) != "ideal_fill"
    )

    def _resolve(
        *,
        component: str,
        explicit_path: str | None,
        model_type: str,
    ) -> str | None:
        if explicit_path:
            p = Path(explicit_path).expanduser()
            if not p.exists():
                raise FileNotFoundError(f"{component} path not found: {p}")
            source[component] = "explicit_path"
            return str(p.resolve())

        summary_path = _pick_component_file(summary_entries.get(component))
        if summary_path:
            p = _resolve_path_from_candidates(summary_path, base_dir=summary_base_dir)
            if p is not None:
                source[component] = "summary_json"
                return str(p)

        if models_dir is not None:
            p = _resolve_from_dir(
                models_dir=models_dir,
                component=component,
                model_type=model_type,
                run_tag=run_tag,
            )
            if p:
                source[component] = "models_load_dir"
                return p
        return None

    stock_path = _resolve(
        component="stock_model",
        explicit_path=model_run.stock_model_path,
        model_type=cfg.stock_model.model_type,
    )
    if timing_needs_artifact:
        timing_path = _resolve(
            component="timing_model",
            explicit_path=model_run.timing_model_path,
            model_type=cfg.timing_model.model_type,
        )
    else:
        timing_path = None
        source["timing_model"] = "not_required"

    if portfolio_needs_artifact:
        portfolio_path = _resolve(
            component="portfolio_model",
            explicit_path=model_run.portfolio_model_path,
            model_type=cfg.portfolio_opt.mode,
        )
    else:
        portfolio_path = None
        source["portfolio_model"] = "not_required"

    if execution_needs_artifact:
        execution_path = _resolve(
            component="execution_model",
            explicit_path=model_run.execution_model_path,
            model_type=cfg.execution_model.model_type,
        )
    else:
        execution_path = None
        source["execution_model"] = "not_required"

    return ResolvedModelPaths(
        stock_model=stock_path,
        timing_model=timing_path,
        portfolio_model=portfolio_path,
        execution_model=execution_path,
        source=source,
    )


def peek_stock_model_factor_cols(cfg: StockModelConfig, model_path: str | None) -> List[str]:
    if not model_path:
        return []
    # Custom stock plugins may persist non-pickle artifacts (json/yaml/etc.).
    # Allow plugin-level factor-col probing and avoid forcing built-in deserializers.
    if cfg.custom_model_py:
        mod = import_module_from_file(cfg.custom_model_py, module_name="strategy7_custom_stock_model")
        peek_fn = getattr(mod, "peek_factor_cols", None)
        if callable(peek_fn):
            cols = peek_fn(cfg, model_path)
            if isinstance(cols, (list, tuple)):
                return [str(x) for x in cols if str(x).strip()]
        return []
    canonical = _normalize_stock_model_type(cfg.model_type)
    p = Path(model_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"stock model file not found: {p}")

    if canonical == "decision_tree":
        with open(p, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            cols = payload.get("factor_cols", [])
            if isinstance(cols, list):
                resolved = [str(x) for x in cols if str(x).strip()]
                if resolved:
                    return resolved
            raw_model = payload.get("model")
            if raw_model is not None and hasattr(raw_model, "feature_names_in_"):
                return [str(x) for x in list(raw_model.feature_names_in_)]
        return []

    if canonical == "launch_boost":
        with open(p, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            cols = payload.get("factor_cols", [])
            if isinstance(cols, list):
                return [str(x) for x in cols if str(x).strip()]
        return []

    if canonical in {"factor_gcl", "dafat"}:
        torch, _nn, _F = (
            FactorGCLStockModel._require_torch() if canonical == "factor_gcl" else DAFATStockModel._require_torch()
        )
        ckpt = torch.load(str(p), map_location="cpu")
        cols = ckpt.get("factor_cols", [])
        if isinstance(cols, list):
            return [str(x) for x in cols if str(x).strip()]
        return []

    return []


def load_stock_model(cfg: StockModelConfig, model_path: str | None) -> Tuple[StockSelectionModel, str]:
    if cfg.custom_model_py:
        mod = import_module_from_file(cfg.custom_model_py, module_name="strategy7_custom_stock_model")
        if hasattr(mod, "load_model"):
            model = mod.load_model(cfg, model_path)
            if not isinstance(model, StockSelectionModel):
                raise TypeError("custom stock load_model must return StockSelectionModel.")
            return model, "custom_load_hook"
        raise RuntimeError("custom stock model in load mode requires load_model(cfg, model_path).")

    canonical = _normalize_stock_model_type(cfg.model_type)
    if not model_path:
        raise ValueError("stock model path is required in load mode.")
    p = Path(model_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"stock model file not found: {p}")

    if canonical == "decision_tree":
        if p.suffix.lower() not in {".pkl", ".pickle"}:
            raise ValueError(f"decision_tree load mode expects .pkl/.pickle file, got: {p.suffix}")
        with open(p, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, TreeStockModel):
            return payload, "pickle_tree_object"
        if not isinstance(payload, dict):
            raise TypeError("unexpected decision_tree model payload type.")
        model = TreeStockModel(
            max_depth=int(cfg.max_depth),
            min_samples_leaf=int(cfg.min_samples_leaf),
            random_state=int(cfg.random_state),
        )
        model._model = payload.get("model")
        fill_values = payload.get("fill_values")
        model._fill_values = fill_values if isinstance(fill_values, pd.Series) else pd.Series(fill_values or {}, dtype=float)
        model._is_classifier = bool(payload.get("is_classifier", True))
        model._target_col = str(payload.get("target_col", "target_up"))
        factor_cols = payload.get("factor_cols", [])
        model._factor_cols = [str(x) for x in factor_cols if str(x).strip()] if isinstance(factor_cols, list) else []
        if not model._factor_cols and model._model is not None and hasattr(model._model, "feature_names_in_"):
            model._factor_cols = [str(x) for x in list(model._model.feature_names_in_)]
        if model._model is None:
            raise RuntimeError("decision_tree payload missing `model`.")
        return model, "artifact_file"

    if canonical == "launch_boost":
        if p.suffix.lower() not in {".pkl", ".pickle"}:
            raise ValueError(f"launch_boost load mode expects .pkl/.pickle file, got: {p.suffix}")
        with open(p, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, LaunchBoostStockModel):
            return payload, "pickle_launch_boost_object"
        if not isinstance(payload, dict):
            raise TypeError("unexpected launch_boost model payload type.")
        model = LaunchBoostStockModel(
            max_depth=int(cfg.launch_boost_max_depth),
            min_samples_leaf=int(cfg.min_samples_leaf),
            learning_rate=float(cfg.launch_boost_learning_rate),
            max_iter=int(cfg.launch_boost_max_iter),
            l2_regularization=float(cfg.launch_boost_l2),
            return_head_weight=float(cfg.launch_boost_return_head_weight),
            random_state=int(cfg.random_state),
        )
        model._cls_model = payload.get("cls_model")
        model._reg_model = payload.get("reg_model")
        model._has_return_head = bool(payload.get("has_return_head", False))
        const_prob = payload.get("constant_cls_prob")
        model._constant_cls_prob = float(const_prob) if const_prob is not None else None
        fill_values = payload.get("fill_values")
        model._fill_values = fill_values if isinstance(fill_values, pd.Series) else pd.Series(fill_values or {}, dtype=float)
        model._target_col = str(payload.get("target_col", "target_return"))
        factor_cols = payload.get("factor_cols", [])
        model._factor_cols = [str(x) for x in factor_cols if str(x).strip()] if isinstance(factor_cols, list) else []
        model._train_summary = dict(payload.get("train_summary", {}) or {})
        if model._cls_model is None and model._constant_cls_prob is None:
            raise RuntimeError("launch_boost payload missing classifier head and constant probability.")
        return model, "artifact_file"

    if canonical == "factor_gcl":
        if p.suffix.lower() not in {".pt", ".pth"}:
            raise ValueError(f"factor_gcl load mode expects .pt/.pth checkpoint, got: {p.suffix}")
        torch, nn, _F = FactorGCLStockModel._require_torch()
        ckpt = torch.load(str(p), map_location="cpu")
        conf = dict(ckpt.get("config", {}) or {})
        model = FactorGCLStockModel(
            seq_len=int(conf.get("seq_len", cfg.fgcl_seq_len)),
            future_look=int(conf.get("future_look", cfg.fgcl_future_look)),
            hidden_size=int(conf.get("hidden_size", cfg.fgcl_hidden_size)),
            num_layers=int(conf.get("num_layers", cfg.fgcl_num_layers)),
            num_factor=int(conf.get("num_factor", cfg.fgcl_num_factor)),
            gamma=float(conf.get("gamma", cfg.fgcl_gamma)),
            tau=float(conf.get("tau", cfg.fgcl_tau)),
            n_epochs=int(conf.get("n_epochs", cfg.fgcl_epochs)),
            lr=float(conf.get("lr", cfg.fgcl_lr)),
            early_stop=int(conf.get("early_stop", cfg.fgcl_early_stop)),
            smooth_steps=int(conf.get("smooth_steps", cfg.fgcl_smooth_steps)),
            per_epoch_batch=int(conf.get("per_epoch_batch", cfg.fgcl_per_epoch_batch)),
            batch_size=int(conf.get("batch_size", cfg.fgcl_batch_size)),
            label_transform=str(conf.get("label_transform", cfg.fgcl_label_transform)),
            weight_decay=float(conf.get("weight_decay", cfg.fgcl_weight_decay)),
            dropout=float(conf.get("dropout", cfg.fgcl_dropout)),
            random_state=int(conf.get("random_state", cfg.random_state)),
            device=str(conf.get("device_used", cfg.fgcl_device)),
        )
        model._factor_cols = [str(x) for x in ckpt.get("factor_cols", []) if str(x).strip()]
        if not model._factor_cols:
            raise RuntimeError("factor_gcl checkpoint missing factor_cols.")
        model._concepts = [str(x) for x in ckpt.get("concepts", []) if str(x).strip()]
        if not model._concepts:
            model._concepts = ["all:all"]
        concept_idx = {c: i for i, c in enumerate(model._concepts)}
        model._all_concept_idx = int(concept_idx.get("all:all", 0))
        raw_map = ckpt.get("code_concept_idx", {}) or {}
        code_map: Dict[str, List[int]] = {}
        for code, idxs in raw_map.items():
            if isinstance(idxs, list):
                code_map[str(code)] = [int(x) for x in idxs]
        model._code_concept_idx = code_map
        model._fill_values = pd.Series(ckpt.get("fill_values", {}) or {}, dtype=float)
        model._time_col = str(ckpt.get("time_col", "signal_ts"))
        model._train_summary = dict(ckpt.get("train_summary", {}) or {})
        model._target_col = str(conf.get("target_col", "target_return"))
        model._score_sign = float(conf.get("score_sign", 1.0))
        model._device_used = model._choose_device(torch)
        net = model._build_network(
            input_dim=len(model._factor_cols),
            num_prior_concepts=len(model._concepts),
            torch=torch,
            nn=nn,
        )
        net.load_state_dict(ckpt["state_dict"])
        net.to(model._device_used)
        net.eval()
        model._model = net
        return model, "artifact_file"

    if canonical == "dafat":
        if p.suffix.lower() not in {".pt", ".pth"}:
            raise ValueError(f"dafat load mode expects .pt/.pth checkpoint, got: {p.suffix}")
        torch, nn, F = DAFATStockModel._require_torch()
        ckpt = torch.load(str(p), map_location="cpu")
        conf = dict(ckpt.get("config", {}) or {})
        model = DAFATStockModel(
            seq_len=int(conf.get("seq_len", cfg.dafat_seq_len)),
            hidden_size=int(conf.get("hidden_size", cfg.dafat_hidden_size)),
            num_layers=int(conf.get("num_layers", cfg.dafat_num_layers)),
            num_heads=int(conf.get("num_heads", cfg.dafat_num_heads)),
            ffn_mult=int(conf.get("ffn_mult", cfg.dafat_ffn_mult)),
            dropout=float(conf.get("dropout", cfg.dafat_dropout)),
            local_window=int(conf.get("local_window", cfg.dafat_local_window)),
            topk_ratio=float(conf.get("topk_ratio", cfg.dafat_topk_ratio)),
            vol_quantile=float(conf.get("vol_quantile", cfg.dafat_vol_quantile)),
            meso_scale=int(conf.get("meso_scale", cfg.dafat_meso_scale)),
            macro_scale=int(conf.get("macro_scale", cfg.dafat_macro_scale)),
            n_epochs=int(conf.get("n_epochs", cfg.dafat_epochs)),
            lr=float(conf.get("lr", cfg.dafat_lr)),
            weight_decay=float(conf.get("weight_decay", cfg.dafat_weight_decay)),
            early_stop=int(conf.get("early_stop", cfg.dafat_early_stop)),
            per_epoch_batch=int(conf.get("per_epoch_batch", cfg.dafat_per_epoch_batch)),
            batch_size=int(conf.get("batch_size", cfg.dafat_batch_size)),
            label_transform=str(conf.get("label_transform", cfg.dafat_label_transform)),
            mse_weight=float(conf.get("mse_weight", cfg.dafat_mse_weight)),
            use_dpe=bool(conf.get("use_dpe", cfg.dafat_use_dpe)),
            use_sparse_attn=bool(conf.get("use_sparse_attn", cfg.dafat_use_sparse_attn)),
            use_multiscale=bool(conf.get("use_multiscale", cfg.dafat_use_multiscale)),
            random_state=int(conf.get("random_state", cfg.random_state)),
            device=str(conf.get("device_used", cfg.dafat_device)),
        )
        model._factor_cols = [str(x) for x in ckpt.get("factor_cols", []) if str(x).strip()]
        if not model._factor_cols:
            raise RuntimeError("dafat checkpoint missing factor_cols.")
        model._fill_values = pd.Series(ckpt.get("fill_values", {}) or {}, dtype=float)
        model._time_col = str(ckpt.get("time_col", "signal_ts"))
        model._train_summary = dict(ckpt.get("train_summary", {}) or {})
        model._target_col = str(conf.get("target_col", "target_return"))
        model._score_sign = float(conf.get("score_sign", 1.0))
        model._market_state_lookup = {}
        for k, v in (ckpt.get("market_state_lookup", {}) or {}).items():
            try:
                ts = pd.to_datetime(k).normalize()
            except Exception:
                continue
            arr = np.asarray(v, dtype=np.float32)
            model._market_state_lookup[pd.Timestamp(ts)] = arr
        ms_mean = ckpt.get("market_state_mean", {}) or {}
        ms_std = ckpt.get("market_state_std", {}) or {}
        model._market_state_mean = pd.Series(ms_mean, dtype=float).reindex(model._state_cols).fillna(0.0)
        model._market_state_std = pd.Series(ms_std, dtype=float).reindex(model._state_cols).replace(0.0, 1.0).fillna(1.0)
        model._state_default = np.zeros(len(model._state_cols), dtype=np.float32)
        model._input_dim = len(model._factor_cols)
        model._device_used = model._choose_device(torch)
        net = model._build_network(input_dim=len(model._factor_cols), torch=torch, nn=nn, F=F)
        net.load_state_dict(ckpt["state_dict"])
        net.to(model._device_used)
        net.eval()
        model._model = net
        return model, "artifact_file"

    raise ValueError(f"unsupported stock model type in load mode: {cfg.model_type}")


def load_timing_model(cfg: TimingModelConfig, model_path: str | None) -> Tuple[TimingModel, str]:
    if cfg.custom_model_py:
        mod = import_module_from_file(cfg.custom_model_py, module_name="strategy7_custom_timing_model")
        if hasattr(mod, "load_model"):
            model = mod.load_model(cfg, model_path)
            if not isinstance(model, TimingModel):
                raise TypeError("custom timing load_model must return TimingModel.")
            return model, "custom_load_hook"
        raise RuntimeError("custom timing model in load mode requires load_model(cfg, model_path).")

    canonical = _normalize_timing_model_type(cfg.model_type)
    if canonical == "none":
        return NoTimingModel(), "config_default"

    if canonical == "volatility_regime":
        if model_path:
            p = Path(model_path).expanduser()
            if not p.exists():
                raise FileNotFoundError(f"timing model file not found: {p}")
            if p.suffix.lower() == ".pkl":
                with open(p, "rb") as f:
                    model = pickle.load(f)
                if not isinstance(model, TimingModel):
                    raise TypeError("timing model pickle does not contain TimingModel.")
                return model, "artifact_file"
            if p.suffix.lower() == ".json":
                meta = _safe_read_json(p)
                return (
                    VolatilityRegimeTimingModel(
                        vol_threshold=float(meta.get("vol_threshold", cfg.vol_threshold)),
                        momentum_threshold=float(meta.get("momentum_threshold", cfg.momentum_threshold)),
                    ),
                    "artifact_meta",
                )
            raise ValueError(f"volatility_regime supports .pkl/.json model files, got: {p.suffix}")
        return (
            VolatilityRegimeTimingModel(
                vol_threshold=float(cfg.vol_threshold),
                momentum_threshold=float(cfg.momentum_threshold),
            ),
            "config_default",
        )

    raise ValueError(f"unsupported timing model type in load mode: {cfg.model_type}")


def load_portfolio_model(cfg: PortfolioOptConfig, model_path: str | None) -> Tuple[PortfolioModel, str]:
    if cfg.custom_model_py:
        mod = import_module_from_file(cfg.custom_model_py, module_name="strategy7_custom_portfolio_model")
        if hasattr(mod, "load_model"):
            model = mod.load_model(cfg, model_path)
            if not isinstance(model, PortfolioModel):
                raise TypeError("custom portfolio load_model must return PortfolioModel.")
            return model, "custom_load_hook"
        raise RuntimeError("custom portfolio model in load mode requires load_model(cfg, model_path).")

    canonical = _normalize_portfolio_mode(cfg.mode)
    if canonical == "equal_weight":
        return EqualWeightPortfolioModel(), "config_default"

    if canonical == "dynamic_opt":
        if model_path:
            p = Path(model_path).expanduser()
            if not p.exists():
                raise FileNotFoundError(f"portfolio model file not found: {p}")
            if p.suffix.lower() == ".pkl":
                with open(p, "rb") as f:
                    model = pickle.load(f)
                if not isinstance(model, PortfolioModel):
                    raise TypeError("portfolio model pickle does not contain PortfolioModel.")
                return model, "artifact_file"
            if p.suffix.lower() == ".json":
                meta = _safe_read_json(p)
                merged = _coerce_dataclass_kwargs(PortfolioOptConfig, dict(meta.get("config", {}) or {}), cfg)
                return DynamicOptimizationPortfolioModel(cfg=PortfolioOptConfig(**merged)), "artifact_meta"
            raise ValueError(f"dynamic_opt supports .pkl/.json model files, got: {p.suffix}")
        return DynamicOptimizationPortfolioModel(cfg=cfg), "config_default"

    raise ValueError(f"unsupported portfolio mode in load mode: {cfg.mode}")


def load_execution_model(cfg: ExecutionModelConfig, model_path: str | None) -> Tuple[ExecutionModel, str]:
    if cfg.custom_model_py:
        mod = import_module_from_file(cfg.custom_model_py, module_name="strategy7_custom_execution_model")
        if hasattr(mod, "load_model"):
            model = mod.load_model(cfg, model_path)
            if not isinstance(model, ExecutionModel):
                raise TypeError("custom execution load_model must return ExecutionModel.")
            return model, "custom_load_hook"
        raise RuntimeError("custom execution model in load mode requires load_model(cfg, model_path).")

    canonical = _normalize_execution_model_type(cfg.model_type)
    if canonical == "ideal_fill":
        return IdealFillExecutionModel(), "config_default"

    if canonical == "realistic_fill":
        if model_path:
            p = Path(model_path).expanduser()
            if not p.exists():
                raise FileNotFoundError(f"execution model file not found: {p}")
            if p.suffix.lower() == ".pkl":
                with open(p, "rb") as f:
                    model = pickle.load(f)
                if not isinstance(model, ExecutionModel):
                    raise TypeError("execution model pickle does not contain ExecutionModel.")
                return model, "artifact_file"
            if p.suffix.lower() == ".json":
                meta = _safe_read_json(p)
                merged = _coerce_dataclass_kwargs(ExecutionModelConfig, dict(meta.get("config", {}) or {}), cfg)
                return RealisticFillExecutionModel(cfg=ExecutionModelConfig(**merged)), "artifact_meta"
            raise ValueError(f"realistic_fill supports .pkl/.json model files, got: {p.suffix}")
        return RealisticFillExecutionModel(cfg=cfg), "config_default"

    raise ValueError(f"unsupported execution model type in load mode: {cfg.model_type}")


def stock_model_factor_cols(model: StockSelectionModel, fallback: List[str]) -> List[str]:
    cols = getattr(model, "_factor_cols", None)
    if isinstance(cols, list) and cols:
        return [str(x) for x in cols if str(x).strip()]
    return list(fallback)


def bootstrap_stock_model_history(
    model: StockSelectionModel,
    history_df: pd.DataFrame,
    factor_cols: List[str],
) -> None:
    if isinstance(model, FactorGCLStockModel):
        if not factor_cols:
            return
        time_col = model._time_col or model._resolve_time_col(history_df)
        work = history_df.copy()
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        work = work.dropna(subset=["code", time_col]).copy()
        work["code"] = work["code"].astype(str)
        missing = [c for c in factor_cols if c not in work.columns]
        if missing:
            return
        fv = model.fill_values().reindex(factor_cols).fillna(0.0)
        work[factor_cols] = (
            work[factor_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(fv)
            .fillna(0.0)
        )
        work = work.sort_values(["code", time_col])
        model._history_by_code = {}
        for code, g in work.groupby("code"):
            arr = g[factor_cols].to_numpy(dtype=np.float32)
            n = len(arr)
            if n == 0:
                continue
            model._history_by_code[str(code)] = arr[max(0, n - int(model.seq_len) + 1) :].copy()
        return

    if isinstance(model, DAFATStockModel):
        if not factor_cols:
            return
        time_col = model._time_col or model._resolve_time_col(history_df)
        work = history_df.copy()
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        work = work.dropna(subset=["code", time_col]).copy()
        work["code"] = work["code"].astype(str)
        missing = [c for c in factor_cols if c not in work.columns]
        if missing:
            return
        fv = model.fill_values().reindex(factor_cols).fillna(0.0)
        work[factor_cols] = (
            work[factor_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(fv)
            .fillna(0.0)
        )
        work = work.sort_values(["code", time_col])
        model._history_by_code = {}
        model._history_time_by_code = {}
        for code, g in work.groupby("code"):
            arr = g[factor_cols].to_numpy(dtype=np.float32)
            ts_arr = pd.to_datetime(g[time_col], errors="coerce").to_numpy(dtype="datetime64[ns]")
            n = len(arr)
            if n == 0:
                continue
            start = max(0, n - int(model.seq_len) + 1)
            model._history_by_code[str(code)] = arr[start:].copy()
            model._history_time_by_code[str(code)] = ts_arr[start:].copy()
        return
