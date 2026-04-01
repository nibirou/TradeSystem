"""Strategy7 end-to-end pipeline runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..backtest.engine import run_backtest
from ..backtest.metrics import (
    calc_trade_return,
    calc_ic_for_column,
    compute_factor_ic_statistics,
    compute_score_spread,
    evaluate_selection_model,
    summarize_ic,
)
from ..backtest.plotting import plot_backtest_curves
from ..config import RunConfig
from ..core.constants import INTRADAY_FREQS
from ..core.utils import dump_json, ensure_dir
from ..data.loaders import HS300MarketDataLoader, build_feature_bundle, load_index_benchmark_data
from ..data.preprocess import PreprocessOptions, apply_cross_section_pipeline, fill_feature_na
from ..data.sources import DataSourceRegistry, TableFileSource, load_custom_source_module, merge_external_sources
from ..factors.base import FactorLibrary, compute_factor_panel, load_custom_factor_module, resolve_selected_factors
from ..factors.defaults import DEFAULT_FACTOR_SET_BY_FREQ, register_default_factors
from ..factors.labeling import add_labels, pick_target_column, split_train_test
from ..models import build_execution_model, build_portfolio_model, build_stock_model, build_timing_model
from .artifacts import build_run_tag, save_common_artifacts


def _safe_trade_dates_from_panel(panel: pd.DataFrame, factor_freq: str) -> pd.DatetimeIndex:
    if panel.empty:
        return pd.DatetimeIndex([])
    if factor_freq in {"D", "W", "M"} and "date" in panel.columns:
        return pd.DatetimeIndex(pd.to_datetime(panel["date"], errors="coerce").dropna().unique()).sort_values()
    if factor_freq in INTRADAY_FREQS and "datetime" in panel.columns:
        dt = pd.to_datetime(panel["datetime"], errors="coerce").dropna()
        return pd.DatetimeIndex(dt.dt.normalize().unique()).sort_values()
    if "date" in panel.columns:
        return pd.DatetimeIndex(pd.to_datetime(panel["date"], errors="coerce").dropna().unique()).sort_values()
    return pd.DatetimeIndex([])


def _build_external_source_registry(cfg: RunConfig) -> DataSourceRegistry:
    reg = DataSourceRegistry()
    for i, p in enumerate(cfg.data.extra_factor_paths):
        name = f"extra_table_{i+1}"
        reg.register(name, TableFileSource(name=name, path=p, date_col="date", code_col="code", prefix=name))
    if cfg.data.extra_source_module:
        load_custom_source_module(reg, cfg.data.extra_source_module)
    return reg


def run_pipeline(cfg: RunConfig) -> Dict[str, object]:
    output_dir = ensure_dir(cfg.output_dir)
    model_dir = ensure_dir(output_dir / "models")

    loader = HS300MarketDataLoader(
        data_cfg=cfg.data,
        date_cfg=cfg.dates,
        lookback_days=cfg.factors.lookback_days,
        horizon=cfg.backtest.horizon,
    )
    market_bundle = loader.load()
    feat_bundle = build_feature_bundle(market_bundle)

    factor_freq = cfg.factors.factor_freq
    if factor_freq not in feat_bundle.by_freq:
        raise ValueError(f"feature bundle missing freq={factor_freq}")
    base_df = feat_bundle.by_freq[factor_freq].copy()
    if base_df.empty:
        raise RuntimeError(f"base feature frame is empty for freq={factor_freq}")

    # Merge external sources (fundamental / NLP / mined factors)
    source_registry = _build_external_source_registry(cfg)
    external_rows: Dict[str, int] = {}
    if source_registry.keys():
        trade_dates = _safe_trade_dates_from_panel(base_df, factor_freq=factor_freq)
        code_universe = sorted(base_df["code"].astype(str).drop_duplicates().tolist())
        base_df, external_rows = merge_external_sources(
            base_panel=base_df,
            registry=source_registry,
            trade_dates=trade_dates,
            code_universe=code_universe,
        )

    factor_lib = FactorLibrary()
    register_default_factors(factor_lib)
    if cfg.factors.custom_factor_py:
        load_custom_factor_module(factor_lib, cfg.factors.custom_factor_py)

    if cfg.factors.list_factors:
        print(factor_lib.metadata(freq=factor_freq).to_string(index=False))
        return {"status": "listed_factors_only", "factor_freq": factor_freq}

    default_set = DEFAULT_FACTOR_SET_BY_FREQ.get(factor_freq, [])
    selected_factors = resolve_selected_factors(
        library=factor_lib,
        freq=factor_freq,
        factor_list_arg=cfg.factors.factor_list,
        default_set=default_set,
    )

    panel = compute_factor_panel(base_df=base_df, library=factor_lib, freq=factor_freq, selected_factors=selected_factors)

    # feature cross-sectional pipeline
    pp_opt = PreprocessOptions(winsorize_limit=0.01, do_zscore=True, neutralize=False, fill_method="median")
    group_col = "date" if factor_freq in {"D", "W", "M"} else "signal_date_proxy"
    if factor_freq in INTRADAY_FREQS:
        panel[group_col] = pd.to_datetime(panel["datetime"], errors="coerce").dt.normalize()
    panel = apply_cross_section_pipeline(panel, selected_factors, pp_opt, group_col=group_col)
    panel = fill_feature_na(panel, selected_factors, method=pp_opt.fill_method)

    panel = add_labels(
        panel=panel,
        horizon=cfg.backtest.horizon,
        execution_scheme=cfg.backtest.execution_scheme,
        price_table_daily=feat_bundle.price_table_daily,
        factor_freq=factor_freq,
    )

    train_df, test_df = split_train_test(
        panel=panel,
        train_start=cfg.dates.train_start,
        train_end=cfg.dates.train_end,
        test_start=cfg.dates.test_start,
        test_end=cfg.dates.test_end,
        factor_freq=factor_freq,
        label_task=cfg.factors.label_task,
    )
    if train_df.empty:
        raise RuntimeError("training set is empty.")
    if test_df.empty:
        raise RuntimeError("test set is empty.")

    target_col = pick_target_column(cfg.factors.label_task)

    stock_model = build_stock_model(cfg.stock_model)
    stock_model.fit(train_df=train_df, factor_cols=selected_factors, target_col=target_col)
    timing_model = build_timing_model(cfg.timing_model).fit(train_df)
    portfolio_model = build_portfolio_model(cfg.portfolio_opt)
    execution_model = build_execution_model(cfg.execution_model)

    test_df = test_df.copy()
    test_df["pred_score"] = stock_model.predict_score(test_df, selected_factors)
    test_df["pred_up"] = (test_df["pred_score"] >= cfg.backtest.long_threshold).astype(int)
    if "entry_price" in test_df.columns and "exit_price" in test_df.columns:
        test_df["gross_trade_ret"] = test_df["exit_price"] / (test_df["entry_price"] + 1e-12) - 1.0
        test_df["net_trade_ret"] = calc_trade_return(
            test_df["entry_price"],
            test_df["exit_price"],
            fee_bps=cfg.backtest.fee_bps,
            slippage_bps=cfg.backtest.slippage_bps,
        )
    model_metrics = evaluate_selection_model(
        target=test_df[target_col],
        pred_score=test_df["pred_score"],
        threshold=cfg.backtest.long_threshold,
    )

    index_benchmarks = load_index_benchmark_data(
        index_root=Path(cfg.data.index_root),
        start_date=market_bundle.start_date,
        end_date=market_bundle.end_date,
        file_format=cfg.data.file_format,
    )

    trades_df, positions_df, curve_df, bt_summary = run_backtest(
        pred_df=test_df,
        backtest_cfg=cfg.backtest,
        factor_freq=factor_freq,
        timing_model=timing_model,
        portfolio_model=portfolio_model,
        execution_model=execution_model,
        index_benchmarks=index_benchmarks,
    )

    ic_group_col = "signal_ts" if "signal_ts" in test_df.columns else ("date" if "date" in test_df.columns else "signal_ts")
    factor_ic_summary_df, factor_ic_series_df = compute_factor_ic_statistics(
        pred_df=test_df,
        factor_cols=selected_factors,
        ret_col="future_ret_n",
        min_cross_section=cfg.factors.min_ic_cross_section,
        group_col=ic_group_col,
    )
    model_ic_series_df = calc_ic_for_column(
        test_df,
        score_col="pred_score",
        ret_col="future_ret_n",
        min_cross_section=cfg.factors.min_ic_cross_section,
        group_col=ic_group_col,
    )
    model_ic_summary = summarize_ic(model_ic_series_df)
    score_spread = compute_score_spread(
        test_df,
        score_col="pred_score",
        ret_col="future_ret_n",
        quantiles=5,
        group_col=ic_group_col,
    )

    board_tag = "mainboard" if cfg.data.main_board_only else "allboards"
    run_tag = build_run_tag(
        train_start=cfg.dates.train_start.strftime("%Y%m%d"),
        train_end=cfg.dates.train_end.strftime("%Y%m%d"),
        test_start=cfg.dates.test_start.strftime("%Y%m%d"),
        test_end=cfg.dates.test_end.strftime("%Y%m%d"),
        horizon=cfg.backtest.horizon,
        execution_scheme=cfg.backtest.execution_scheme,
        factor_freq=factor_freq,
        board_tag=board_tag,
        portfolio_mode=cfg.backtest.portfolio_mode,
    )

    pred_cols = [
        "signal_ts",
        "code",
        "entry_ts",
        "exit_ts",
        "pred_score",
        "pred_up",
        "target_up",
        "target_return",
        "target_volatility",
        "future_ret_n",
        "entry_price",
        "exit_price",
        "gross_trade_ret",
        "net_trade_ret",
    ] + selected_factors
    pred_cols = [c for c in pred_cols if c in test_df.columns]
    pred_to_save = test_df[pred_cols].copy()

    files = save_common_artifacts(
        output_dir=output_dir,
        run_tag=run_tag,
        pred_df=pred_to_save,
        trades_df=trades_df,
        positions_df=positions_df,
        curve_df=curve_df,
        factor_ic_summary_df=factor_ic_summary_df,
        factor_ic_series_df=factor_ic_series_df,
        model_ic_series_df=model_ic_series_df,
        factor_meta_df=factor_lib.metadata(freq=factor_freq),
    )

    plot_main_path = output_dir / f"backtest_curve_main_{run_tag}.png"
    plot_excess_path = output_dir / f"backtest_curve_excess_{run_tag}.png"
    plot_status = plot_backtest_curves(
        curve_df=curve_df,
        output_main_png=plot_main_path,
        output_excess_png=plot_excess_path,
        title_prefix=f"Strategy7 ({factor_freq}, {cfg.backtest.execution_scheme})",
    )
    files["backtest_main_plot_png"] = plot_main_path
    files["backtest_excess_plot_png"] = plot_excess_path

    model_files: Dict[str, Dict[str, str]] = {}
    if cfg.save_models:
        model_files["stock_model"] = stock_model.save(model_dir, run_tag)
        model_files["timing_model"] = timing_model.save(model_dir, run_tag)
        model_files["portfolio_model"] = portfolio_model.save(model_dir, run_tag)
        model_files["execution_model"] = execution_model.save(model_dir, run_tag)

    top_factors = factor_ic_summary_df.head(10).copy() if not factor_ic_summary_df.empty else pd.DataFrame()
    summary = {
        "config": cfg.to_dict(),
        "sample_count": {
            "daily_rows": int(len(market_bundle.daily)),
            "minute_rows": int(len(market_bundle.minute5)),
            "base_rows": int(len(base_df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "trade_points": int(len(trades_df)),
            "position_count": int(len(positions_df)),
            "external_rows": external_rows,
        },
        "selected_factors": selected_factors,
        "model_metrics": model_metrics,
        "backtest_metrics": bt_summary,
        "model_ic_summary": model_ic_summary,
        "model_score_spread": score_spread,
        "factor_ic_top10": top_factors.to_dict(orient="records") if not top_factors.empty else [],
        "outputs": {
            **{k: str(v) for k, v in files.items()},
            "plot_main_generated": bool(plot_status.get("main", False)),
            "plot_excess_generated": bool(plot_status.get("excess", False)),
            "model_files": model_files,
        },
        "notes": {
            "no_future_leakage": (
                "factor features only use current/past information; labels use shifted future window;"
                "train/test split constrained by signal timestamps and target end timestamps."
            ),
            "timing_model_enabled": cfg.timing_model.model_type != "none",
            "portfolio_dynamic_enabled": cfg.portfolio_opt.mode == "dynamic_opt",
            "realistic_execution_enabled": cfg.execution_model.model_type == "realistic_fill",
        },
    }
    summary_path = output_dir / f"summary_{run_tag}.json"
    dump_json(summary_path, summary)
    return summary
