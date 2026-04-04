"""CLI entry for Strategy7 factor mining framework."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from strategy7.config import DataConfig, DateConfig
from strategy7.data.loaders import HS300MarketDataLoader, build_feature_bundle
from strategy7.factors.labeling import add_labels
from strategy7.mining.custom import load_custom_specs
from strategy7.mining.runner import FactorMiningConfig, run_factor_mining


def _parse_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).normalize()


def _infer_data_baostock_root(data_root: str) -> Path:
    p = Path(data_root).expanduser()
    for cand in [p, *p.parents]:
        if cand.name.lower() == "data_baostock":
            return cand
    if p.name.lower() in {"hs300", "all", "zz500", "zz1000", "sz50"} and p.parent.name.lower() == "stock_hist":
        return p.parent.parent
    if p.name.lower() == "stock_hist":
        return p.parent
    return p


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strategy7 factor mining framework (fundamental/minute/custom)")
    parser.add_argument("--framework", type=str, choices=["fundamental_multiobj", "minute_parametric", "custom"], default="fundamental_multiobj")

    parser.add_argument("--data-root", type=str, default=r"D:/PythonProject/Quant/data_baostock/stock_hist/hs300")
    parser.add_argument("--hs300-list-path", type=str, default=r"D:/PythonProject/Quant/data_baostock/metadata/stock_list_hs300.csv")
    parser.add_argument("--index-root", type=str, default=r"D:/PythonProject/Quant/data_baostock/ak_index")
    parser.add_argument("--file-format", type=str, choices=["auto", "csv", "parquet"], default="auto")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--main-board-only", action="store_true")

    parser.add_argument("--train-start", type=str, default="2021-01-01")
    parser.add_argument("--train-end", type=str, default="2023-12-31")
    parser.add_argument("--valid-start", type=str, default="2024-01-01")
    parser.add_argument("--valid-end", type=str, default="2024-12-31")

    parser.add_argument("--factor-freq", type=str, default="D", help="factor frequency tag for catalog registration")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--execution-scheme", type=str, default="vwap30_vwap30")

    parser.add_argument("--population-size", type=int, default=128)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--elite-size", type=int, default=12)
    parser.add_argument("--mutation-rate", type=float, default=0.25)
    parser.add_argument("--crossover-rate", type=float, default=0.70)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--corr-threshold", type=float, default=0.60)
    parser.add_argument("--min-cross-section", type=int, default=30)
    parser.add_argument("--top-frac", type=float, default=0.10)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--custom-spec-json", type=str, default="", help="json list for custom factor specs when framework=custom")

    parser.add_argument("--factor-store-root", type=str, default="auto")
    parser.add_argument("--catalog-path", type=str, default="auto")
    parser.add_argument("--save-format", type=str, choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--min-abs-ic-mean", type=float, default=None)
    parser.add_argument("--min-ic-win-rate", type=float, default=None)
    parser.add_argument("--min-ic-ir", type=float, default=None)
    parser.add_argument("--min-long-excess-annualized", type=float, default=None)
    parser.add_argument("--min-long-sharpe", type=float, default=None)
    parser.add_argument("--min-long-win-rate", type=float, default=None)
    parser.add_argument("--min-coverage", type=float, default=None)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    train_start = _parse_date(args.train_start)
    train_end = _parse_date(args.train_end)
    valid_start = _parse_date(args.valid_start)
    valid_end = _parse_date(args.valid_end)

    if not (train_start <= train_end < valid_start <= valid_end):
        raise ValueError("date constraint: train_start <= train_end < valid_start <= valid_end")

    data_root = str(Path(args.data_root).expanduser())
    data_cfg = DataConfig(
        data_root=data_root,
        hs300_list_path=str(Path(args.hs300_list_path).expanduser()),
        index_root=str(Path(args.index_root).expanduser()),
        file_format=str(args.file_format),
        max_files=args.max_files,
        main_board_only=bool(args.main_board_only),
        extra_factor_paths=[],
        extra_source_module=None,
        factor_catalog_path=None,
        auto_load_catalog_factors=False,
    )

    date_cfg = DateConfig(
        train_start=train_start,
        train_end=train_end,
        test_start=valid_start,
        test_end=valid_end,
    )

    loader = HS300MarketDataLoader(
        data_cfg=data_cfg,
        date_cfg=date_cfg,
        lookback_days=160,
        horizon=int(args.horizon),
    )
    market_bundle = loader.load()
    feat_bundle = build_feature_bundle(market_bundle)

    daily_panel = feat_bundle.by_freq["D"].copy()
    daily_panel = add_labels(
        panel=daily_panel,
        horizon=int(args.horizon),
        execution_scheme=str(args.execution_scheme),
        price_table_daily=feat_bundle.price_table_daily,
        factor_freq="D",
    )

    custom_specs = []
    if str(args.framework).lower() == "custom":
        if not str(args.custom_spec_json).strip():
            raise RuntimeError("framework=custom requires --custom-spec-json")
        custom_specs = load_custom_specs(args.custom_spec_json)
        if not custom_specs:
            raise RuntimeError(f"no valid custom specs loaded from: {args.custom_spec_json}")

    store_root = str(args.factor_store_root).strip()
    if store_root.lower() in {"", "auto"}:
        store_root = str(_infer_data_baostock_root(data_root))

    catalog_path = str(args.catalog_path).strip()
    if catalog_path.lower() in {"", "auto"}:
        catalog_path = str((Path(store_root) / "factor_mining" / "factor_catalog.json").resolve())

    mine_cfg = FactorMiningConfig(
        framework=str(args.framework),
        factor_freq=str(args.factor_freq),
        horizon=int(args.horizon),
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        population_size=int(args.population_size),
        generations=int(args.generations),
        elite_size=int(args.elite_size),
        mutation_rate=float(args.mutation_rate),
        crossover_rate=float(args.crossover_rate),
        top_n=int(args.top_n),
        corr_threshold=float(args.corr_threshold),
        min_cross_section=int(args.min_cross_section),
        top_frac=float(args.top_frac),
        random_state=int(args.random_state),
        factor_store_root=store_root,
        catalog_path=catalog_path,
        save_format=str(args.save_format),
        min_abs_ic_mean=args.min_abs_ic_mean,
        min_ic_win_rate=args.min_ic_win_rate,
        min_ic_ir=args.min_ic_ir,
        min_long_excess_annualized=args.min_long_excess_annualized,
        min_long_sharpe=args.min_long_sharpe,
        min_long_win_rate=args.min_long_win_rate,
        min_coverage=args.min_coverage,
    )

    summary = run_factor_mining(
        cfg=mine_cfg,
        daily_panel_with_label=daily_panel,
        minute_df=market_bundle.minute5,
        custom_specs=custom_specs,
    )

    print("=== Factor Mining Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
