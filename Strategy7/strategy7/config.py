"""Configuration models and CLI parser."""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strategy7: modular quant research engine with pluggable data/factor/model/backtest components."
        )
    )
    parser.add_argument("--data-root", type=str, default=r"D:/PythonProject/Quant/data_baostock/stock_hist/hs300")
    parser.add_argument("--hs300-list-path", type=str, default=r"D:/PythonProject/Quant/data_baostock/metadata/stock_list_hs300.csv")
    parser.add_argument("--index-root", type=str, default=r"D:/PythonProject/Quant/data_baostock/ak_index")
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
    parser.add_argument("--save-models", action="store_true")

    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if str(args.output_dir).strip().lower() not in {"", "auto"}:
        return Path(args.output_dir)
    now_tag = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d_%H%M%S")
    board_tag = "mainboard" if bool(args.main_board_only) else "allboards"
    portfolio_mode = args.portfolio_weighting or args.portfolio_model_type
    return (
        Path(__file__).resolve().parent.parent
        / f"outputs_{now_tag}"
        / f"tr{args.train_start.replace('-', '')}-{args.train_end.replace('-', '')}"
        / f"te{args.test_start.replace('-', '')}-{args.test_end.replace('-', '')}"
        / f"{args.factor_freq}_h{args.horizon}_k{args.top_k}_{args.execution_scheme}_{board_tag}_{portfolio_mode}"
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

    portfolio_mode = args.portfolio_weighting or args.portfolio_model_type
    extra_paths = [x.strip() for x in str(args.extra_factor_paths).split(",") if x.strip()]

    data = DataConfig(
        data_root=args.data_root,
        hs300_list_path=args.hs300_list_path,
        index_root=args.index_root,
        file_format=args.file_format,
        max_files=args.max_files,
        main_board_only=bool(args.main_board_only),
        extra_factor_paths=extra_paths,
        extra_source_module=args.extra_source_module,
    )
    factors = FactorConfig(
        factor_freq=args.factor_freq,
        factor_list=args.factor_list,
        custom_factor_py=args.custom_factor_py,
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
        custom_model_py=args.custom_stock_model_py,
    )
    timing = TimingModelConfig(
        model_type=args.timing_model_type,
        vol_threshold=float(args.timing_vol_threshold),
        momentum_threshold=float(args.timing_momentum_threshold),
        custom_model_py=args.custom_timing_model_py,
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
        custom_model_py=args.custom_portfolio_model_py,
    )
    exec_model = ExecutionModelConfig(
        model_type=args.execution_model_type,
        max_participation_rate=float(args.max_participation_rate),
        base_fill_rate=float(args.base_fill_rate),
        latency_bars=int(args.latency_bars),
        custom_model_py=args.custom_execution_model_py,
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
