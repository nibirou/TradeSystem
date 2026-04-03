"""Configuration models and CLI parser."""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .core.constants import EXECUTION_SCHEMES, SUPPORTED_FREQS
from .core.utils import parse_date


@dataclass
class DataConfig:
    data_root: str
    hs300_list_path: str
    index_root: str
    file_format: str
    max_files: int | None
    main_board_only: bool
    extra_factor_paths: List[str]
    extra_source_module: str | None


@dataclass
class FactorConfig:
    factor_freq: str
    factor_list: str
    custom_factor_py: str | None
    list_factors: bool
    lookback_days: int
    min_ic_cross_section: int
    label_task: str


@dataclass
class StockModelConfig:
    model_type: str
    max_depth: int
    min_samples_leaf: int
    random_state: int
    custom_model_py: str | None
    fgcl_seq_len: int
    fgcl_future_look: int
    fgcl_hidden_size: int
    fgcl_num_layers: int
    fgcl_num_factor: int
    fgcl_gamma: float
    fgcl_tau: float
    fgcl_epochs: int
    fgcl_lr: float
    fgcl_early_stop: int
    fgcl_smooth_steps: int
    fgcl_per_epoch_batch: int
    fgcl_batch_size: int
    fgcl_label_transform: str
    fgcl_weight_decay: float
    fgcl_dropout: float
    fgcl_device: str


@dataclass
class TimingModelConfig:
    model_type: str
    vol_threshold: float
    momentum_threshold: float
    custom_model_py: str | None


@dataclass
class PortfolioOptConfig:
    mode: str
    max_weight: float
    max_turnover: float
    liquidity_scale: float
    expected_return_weight: float
    risk_aversion: float
    style_penalty: float
    industry_penalty: float
    crowding_penalty: float
    transaction_cost_penalty: float
    max_iter: int
    step_size: float
    tolerance: float
    custom_model_py: str | None


@dataclass
class ExecutionModelConfig:
    model_type: str
    max_participation_rate: float
    base_fill_rate: float
    latency_bars: int
    custom_model_py: str | None


@dataclass
class BacktestConfig:
    horizon: int
    top_k: int
    long_threshold: float
    execution_scheme: str
    fee_bps: float
    slippage_bps: float
    portfolio_mode: str
    rebalance_stride: int


@dataclass
class DateConfig:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class RunConfig:
    data: DataConfig
    factors: FactorConfig
    stock_model: StockModelConfig
    timing_model: TimingModelConfig
    portfolio_opt: PortfolioOptConfig
    execution_model: ExecutionModelConfig
    backtest: BacktestConfig
    dates: DateConfig
    output_dir: Path
    save_models: bool

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["dates"] = {
            "train_start": str(self.dates.train_start.date()),
            "train_end": str(self.dates.train_end.date()),
            "test_start": str(self.dates.test_start.date()),
            "test_end": str(self.dates.test_end.date()),
        }
        d["output_dir"] = str(self.output_dir)
        return d


def _candidate_quant_roots() -> List[Path]:
    roots: List[Path] = []
    for env_var in ("STRATEGY7_QUANT_ROOT", "QUANT_ROOT"):
        env_value = os.getenv(env_var)
        if env_value:
            roots.append(Path(env_value).expanduser())
    here = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    roots.extend(
        [
            cwd,
            cwd.parent,
            cwd.parent.parent,
            here.parent.parent.parent,  # .../TradeSystem
            here.parent.parent.parent.parent,  # .../Quant
            Path("/workspace/Quant"),
            Path("D:/PythonProject/Quant"),
        ]
    )
    deduped: List[Path] = []
    seen: set[str] = set()
    for p in roots:
        key = str(p).replace("\\", "/").lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def _autodetect_default_path(relative_parts: List[str], fallback: str, env_var: str) -> str:
    env_value = os.getenv(env_var)
    if env_value:
        return str(Path(env_value).expanduser())
    for root in _candidate_quant_roots():
        cand = root.joinpath(*relative_parts)
        if cand.exists():
            return str(cand)
    return fallback


def _resolve_path(value: str) -> str:
    return str(Path(value).expanduser())


def _short_alnum_token(text: str, max_len: int) -> str:
    raw = "".join(ch for ch in str(text).strip().lower() if ch.isalnum())
    if not raw:
        raw = "x"
    if len(raw) <= max_len:
        return raw
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:6]
    keep = max(1, max_len - 7)
    return f"{raw[:keep]}_{digest}"


def _compact_output_leaf(
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    factor_freq: str,
    horizon: int,
    top_k: int,
    execution_scheme: str,
    board_tag: str,
    portfolio_mode: str,
) -> str:
    tr_s = str(train_start).replace("-", "")
    tr_e = str(train_end).replace("-", "")
    te_s = str(test_start).replace("-", "")
    te_e = str(test_end).replace("-", "")

    portfolio_map = {"equal_weight": "eqw", "dynamic_opt": "dyn"}
    port_tag = portfolio_map.get(str(portfolio_mode), _short_alnum_token(str(portfolio_mode), 8))
    scheme_tag = _short_alnum_token(str(execution_scheme), 14)
    leaf = (
        f"tr{tr_s[-6:]}{tr_e[-6:]}"
        f"_te{te_s[-6:]}{te_e[-6:]}"
        f"_{factor_freq}_h{horizon}_k{top_k}_{scheme_tag}_{board_tag}_{port_tag}"
    )
    if len(leaf) <= 72:
        return leaf

    digest = hashlib.sha1(leaf.encode("utf-8")).hexdigest()[:10]
    return f"{factor_freq}_h{horizon}_k{top_k}_{board_tag}_{port_tag}_{digest}"


DEFAULT_DATA_ROOT = _autodetect_default_path(
    ["data_baostock", "stock_hist", "hs300"],
    fallback=r"/workspace/Quant/data_baostock/stock_hist/hs300",
    env_var="STRATEGY7_DATA_ROOT",
)
DEFAULT_HS300_LIST_PATH = _autodetect_default_path(
    ["data_baostock", "metadata", "stock_list_hs300.csv"],
    fallback=r"/workspace/Quant/data_baostock/metadata/stock_list_hs300.csv",
    env_var="STRATEGY7_HS300_LIST_PATH",
)
DEFAULT_INDEX_ROOT = _autodetect_default_path(
    ["data_baostock", "ak_index"],
    fallback=r"/workspace/Quant/data_baostock/ak_index",
    env_var="STRATEGY7_INDEX_ROOT",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strategy7: modular quant research engine with pluggable data/factor/model/backtest components."
        )
    )
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--hs300-list-path", type=str, default=DEFAULT_HS300_LIST_PATH)
    parser.add_argument("--index-root", type=str, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--file-format", type=str, choices=["auto", "csv", "parquet"], default="auto")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--main-board-only", action="store_true")
    parser.add_argument("--extra-factor-paths", type=str, default="")
    parser.add_argument("--extra-source-module", type=str, default=None, help="custom source module with register_sources(registry)")

    parser.add_argument("--train-start", type=str, default="2024-01-01")
    parser.add_argument("--train-end", type=str, default="2024-12-31")
    parser.add_argument("--test-start", type=str, default="2025-01-01")
    parser.add_argument("--test-end", type=str, default="2025-12-31")

    parser.add_argument("--factor-freq", type=str, choices=SUPPORTED_FREQS, default="D")
    parser.add_argument("--factor-list", type=str, default="")
    parser.add_argument("--custom-factor-py", type=str, default=None)
    parser.add_argument("--list-factors", action="store_true")
    parser.add_argument("--lookback-days", type=int, default=160)
    parser.add_argument("--min-ic-cross-section", type=int, default=30)
    parser.add_argument(
        "--label-task",
        type=str,
        choices=["direction", "return", "volatility", "multi_task"],
        default="direction",
        help="prediction target type for training labels",
    )

    parser.add_argument("--stock-model-type", type=str, default="decision_tree")
    parser.add_argument("--custom-stock-model-py", type=str, default=None)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-samples-leaf", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--fgcl-seq-len", type=int, default=30)
    parser.add_argument("--fgcl-future-look", type=int, default=20)
    parser.add_argument("--fgcl-hidden-size", type=int, default=128)
    parser.add_argument("--fgcl-num-layers", type=int, default=2)
    parser.add_argument("--fgcl-num-factor", type=int, default=48)
    parser.add_argument("--fgcl-gamma", type=float, default=1.0)
    parser.add_argument("--fgcl-tau", type=float, default=0.25)
    parser.add_argument("--fgcl-epochs", type=int, default=200)
    parser.add_argument("--fgcl-lr", type=float, default=9e-5)
    parser.add_argument("--fgcl-early-stop", type=int, default=20)
    parser.add_argument("--fgcl-smooth-steps", type=int, default=5)
    parser.add_argument("--fgcl-per-epoch-batch", type=int, default=100)
    parser.add_argument("--fgcl-batch-size", type=int, default=-1, help="-1 -> one whole cross section per trading day")
    parser.add_argument(
        "--fgcl-label-transform",
        type=str,
        choices=["raw", "csrank", "cszscore", "csranknorm"],
        default="csranknorm",
        help="target transform used by FactorGCL training",
    )
    parser.add_argument("--fgcl-weight-decay", type=float, default=1e-4)
    parser.add_argument("--fgcl-dropout", type=float, default=0.0)
    parser.add_argument("--fgcl-device", type=str, choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--timing-model-type", type=str, choices=["none", "volatility_regime"], default="none")
    parser.add_argument("--custom-timing-model-py", type=str, default=None)
    parser.add_argument("--timing-vol-threshold", type=float, default=0.0)
    parser.add_argument("--timing-momentum-threshold", type=float, default=0.0)

    parser.add_argument("--execution-model-type", type=str, choices=["ideal_fill", "realistic_fill"], default="ideal_fill")
    parser.add_argument("--custom-execution-model-py", type=str, default=None)
    parser.add_argument("--max-participation-rate", type=float, default=0.15)
    parser.add_argument("--base-fill-rate", type=float, default=0.95)
    parser.add_argument("--latency-bars", type=int, default=0)

    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--long-threshold", type=float, default=0.50)
    parser.add_argument("--execution-scheme", type=str, choices=sorted(EXECUTION_SCHEMES.keys()), default="vwap30_vwap30")
    parser.add_argument("--fee-bps", type=float, default=1.5)
    parser.add_argument("--slippage-bps", type=float, default=1.5)
    parser.add_argument("--rebalance-stride", type=int, default=0, help="0 -> use horizon")

    parser.add_argument(
        "--portfolio-model-type",
        type=str,
        choices=["equal_weight", "dynamic_opt"],
        default="equal_weight",
    )
    parser.add_argument(
        "--portfolio-weighting",
        type=str,
        choices=["equal_weight", "dynamic_opt"],
        default=None,
        help="backward-compatible alias of --portfolio-model-type",
    )
    parser.add_argument("--opt-max-weight", type=float, default=0.25)
    parser.add_argument("--custom-portfolio-model-py", type=str, default=None)
    parser.add_argument("--opt-max-turnover", type=float, default=1.20)
    parser.add_argument("--opt-liquidity-scale", type=float, default=3.0)
    parser.add_argument("--opt-expected-return-weight", type=float, default=1.0)
    parser.add_argument("--opt-risk-aversion", type=float, default=1.2)
    parser.add_argument("--opt-style-penalty", type=float, default=0.8)
    parser.add_argument("--opt-industry-penalty", type=float, default=0.7)
    parser.add_argument("--opt-crowding-penalty", type=float, default=0.5)
    parser.add_argument("--opt-transaction-cost-penalty", type=float, default=0.6)
    parser.add_argument("--opt-max-iter", type=int, default=220)
    parser.add_argument("--opt-step-size", type=float, default=0.08)
    parser.add_argument("--opt-tolerance", type=float, default=1e-6)

    parser.add_argument("--output-dir", type=str, default="auto")
    parser.add_argument("--save-models", default=True)

    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if str(args.output_dir).strip().lower() not in {"", "auto"}:
        return Path(args.output_dir).expanduser()
    now_tag = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d_%H%M%S")
    board_tag = "mb" if bool(args.main_board_only) else "all"
    portfolio_mode = args.portfolio_weighting or args.portfolio_model_type
    leaf = _compact_output_leaf(
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        factor_freq=args.factor_freq,
        horizon=int(args.horizon),
        top_k=int(args.top_k),
        execution_scheme=args.execution_scheme,
        board_tag=board_tag,
        portfolio_mode=str(portfolio_mode),
    )
    return (
        Path(__file__).resolve().parent.parent
        / "outputs"
        / now_tag
        / leaf
    )


def build_run_config(args: argparse.Namespace) -> RunConfig:
    dates = DateConfig(
        train_start=parse_date(args.train_start),
        train_end=parse_date(args.train_end),
        test_start=parse_date(args.test_start),
        test_end=parse_date(args.test_end),
    )
    if not (dates.train_start <= dates.train_end < dates.test_start <= dates.test_end):
        raise ValueError("Date constraint: train_start <= train_end < test_start <= test_end.")
    if args.horizon <= 0:
        raise ValueError("horizon must be positive.")
    if args.top_k <= 0:
        raise ValueError("top_k must be positive.")
    if not (0.0 <= args.long_threshold <= 1.0):
        raise ValueError("long_threshold must be in [0,1].")
    if int(args.lookback_days) <= 0:
        raise ValueError("lookback_days must be positive.")
    if int(args.min_ic_cross_section) <= 1:
        raise ValueError("min_ic_cross_section must be greater than 1.")
    if args.max_files is not None and int(args.max_files) <= 0:
        raise ValueError("max_files must be positive when provided.")
    if int(args.max_depth) <= 0:
        raise ValueError("max_depth must be positive.")
    if int(args.min_samples_leaf) <= 0:
        raise ValueError("min_samples_leaf must be positive.")
    if int(args.fgcl_seq_len) <= 1:
        raise ValueError("fgcl_seq_len must be greater than 1.")
    if int(args.fgcl_future_look) <= 0:
        raise ValueError("fgcl_future_look must be positive.")
    if int(args.fgcl_hidden_size) <= 0:
        raise ValueError("fgcl_hidden_size must be positive.")
    if int(args.fgcl_num_layers) <= 0:
        raise ValueError("fgcl_num_layers must be positive.")
    if int(args.fgcl_num_factor) <= 0:
        raise ValueError("fgcl_num_factor must be positive.")
    if float(args.fgcl_gamma) < 0.0:
        raise ValueError("fgcl_gamma must be non-negative.")
    if float(args.fgcl_tau) <= 0.0:
        raise ValueError("fgcl_tau must be positive.")
    if int(args.fgcl_epochs) <= 0:
        raise ValueError("fgcl_epochs must be positive.")
    if float(args.fgcl_lr) <= 0.0:
        raise ValueError("fgcl_lr must be positive.")
    if int(args.fgcl_early_stop) <= 0:
        raise ValueError("fgcl_early_stop must be positive.")
    if int(args.fgcl_smooth_steps) <= 0:
        raise ValueError("fgcl_smooth_steps must be positive.")
    if int(args.fgcl_per_epoch_batch) <= 0:
        raise ValueError("fgcl_per_epoch_batch must be positive.")
    if int(args.fgcl_batch_size) == 0 or int(args.fgcl_batch_size) < -1:
        raise ValueError("fgcl_batch_size must be -1 or positive.")
    if float(args.fgcl_weight_decay) < 0.0:
        raise ValueError("fgcl_weight_decay must be non-negative.")
    if not (0.0 <= float(args.fgcl_dropout) < 1.0):
        raise ValueError("fgcl_dropout must be in [0, 1).")
    if not (0.0 < float(args.max_participation_rate) <= 1.0):
        raise ValueError("max_participation_rate must be in (0,1].")
    if not (0.0 <= float(args.base_fill_rate) <= 1.0):
        raise ValueError("base_fill_rate must be in [0,1].")
    if int(args.latency_bars) < 0:
        raise ValueError("latency_bars must be non-negative.")
    if float(args.fee_bps) < 0.0 or float(args.slippage_bps) < 0.0:
        raise ValueError("fee_bps and slippage_bps must be non-negative.")
    if int(args.rebalance_stride) < 0:
        raise ValueError("rebalance_stride must be non-negative.")
    if not (0.0 < float(args.opt_max_weight) <= 1.0):
        raise ValueError("opt_max_weight must be in (0,1].")
    if float(args.opt_max_turnover) < 0.0:
        raise ValueError("opt_max_turnover must be non-negative.")
    if float(args.opt_liquidity_scale) <= 0.0:
        raise ValueError("opt_liquidity_scale must be positive.")
    if int(args.opt_max_iter) <= 0:
        raise ValueError("opt_max_iter must be positive.")
    if float(args.opt_step_size) <= 0.0:
        raise ValueError("opt_step_size must be positive.")
    if float(args.opt_tolerance) <= 0.0:
        raise ValueError("opt_tolerance must be positive.")
    if args.portfolio_weighting and args.portfolio_weighting != args.portfolio_model_type:
        raise ValueError(
            "portfolio_weighting and portfolio_model_type are inconsistent. "
            "Please set one or keep both identical."
        )

    portfolio_mode = args.portfolio_weighting or args.portfolio_model_type
    extra_paths = [_resolve_path(x.strip()) for x in str(args.extra_factor_paths).split(",") if x.strip()]

    data = DataConfig(
        data_root=_resolve_path(args.data_root),
        hs300_list_path=_resolve_path(args.hs300_list_path),
        index_root=_resolve_path(args.index_root),
        file_format=args.file_format,
        max_files=args.max_files,
        main_board_only=bool(args.main_board_only),
        extra_factor_paths=extra_paths,
        extra_source_module=_resolve_path(args.extra_source_module) if args.extra_source_module else None,
    )
    factors = FactorConfig(
        factor_freq=args.factor_freq,
        factor_list=args.factor_list,
        custom_factor_py=_resolve_path(args.custom_factor_py) if args.custom_factor_py else None,
        list_factors=bool(args.list_factors),
        lookback_days=int(args.lookback_days),
        min_ic_cross_section=int(args.min_ic_cross_section),
        label_task=args.label_task,
    )
    stock = StockModelConfig(
        model_type=args.stock_model_type,
        max_depth=int(args.max_depth),
        min_samples_leaf=int(args.min_samples_leaf),
        random_state=int(args.random_state),
        custom_model_py=_resolve_path(args.custom_stock_model_py) if args.custom_stock_model_py else None,
        fgcl_seq_len=int(args.fgcl_seq_len),
        fgcl_future_look=int(args.fgcl_future_look),
        fgcl_hidden_size=int(args.fgcl_hidden_size),
        fgcl_num_layers=int(args.fgcl_num_layers),
        fgcl_num_factor=int(args.fgcl_num_factor),
        fgcl_gamma=float(args.fgcl_gamma),
        fgcl_tau=float(args.fgcl_tau),
        fgcl_epochs=int(args.fgcl_epochs),
        fgcl_lr=float(args.fgcl_lr),
        fgcl_early_stop=int(args.fgcl_early_stop),
        fgcl_smooth_steps=int(args.fgcl_smooth_steps),
        fgcl_per_epoch_batch=int(args.fgcl_per_epoch_batch),
        fgcl_batch_size=int(args.fgcl_batch_size),
        fgcl_label_transform=str(args.fgcl_label_transform),
        fgcl_weight_decay=float(args.fgcl_weight_decay),
        fgcl_dropout=float(args.fgcl_dropout),
        fgcl_device=str(args.fgcl_device),
    )
    timing = TimingModelConfig(
        model_type=args.timing_model_type,
        vol_threshold=float(args.timing_vol_threshold),
        momentum_threshold=float(args.timing_momentum_threshold),
        custom_model_py=_resolve_path(args.custom_timing_model_py) if args.custom_timing_model_py else None,
    )
    port = PortfolioOptConfig(
        mode=portfolio_mode,
        max_weight=float(args.opt_max_weight),
        max_turnover=float(args.opt_max_turnover),
        liquidity_scale=float(args.opt_liquidity_scale),
        expected_return_weight=float(args.opt_expected_return_weight),
        risk_aversion=float(args.opt_risk_aversion),
        style_penalty=float(args.opt_style_penalty),
        industry_penalty=float(args.opt_industry_penalty),
        crowding_penalty=float(args.opt_crowding_penalty),
        transaction_cost_penalty=float(args.opt_transaction_cost_penalty),
        max_iter=int(args.opt_max_iter),
        step_size=float(args.opt_step_size),
        tolerance=float(args.opt_tolerance),
        custom_model_py=_resolve_path(args.custom_portfolio_model_py) if args.custom_portfolio_model_py else None,
    )
    exec_model = ExecutionModelConfig(
        model_type=args.execution_model_type,
        max_participation_rate=float(args.max_participation_rate),
        base_fill_rate=float(args.base_fill_rate),
        latency_bars=int(args.latency_bars),
        custom_model_py=_resolve_path(args.custom_execution_model_py) if args.custom_execution_model_py else None,
    )
    backtest = BacktestConfig(
        horizon=int(args.horizon),
        top_k=int(args.top_k),
        long_threshold=float(args.long_threshold),
        execution_scheme=args.execution_scheme,
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        portfolio_mode=portfolio_mode,
        rebalance_stride=int(args.rebalance_stride) if int(args.rebalance_stride) > 0 else int(args.horizon),
    )
    return RunConfig(
        data=data,
        factors=factors,
        stock_model=stock,
        timing_model=timing,
        portfolio_opt=port,
        execution_model=exec_model,
        backtest=backtest,
        dates=dates,
        output_dir=resolve_output_dir(args),
        save_models=bool(args.save_models),
    )
