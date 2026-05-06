"""Strategy7 end-to-end pipeline runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
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
from ..core.time_utils import infer_periods_per_year
from ..core.utils import dump_json, ensure_dir, log_progress
from ..data.loaders import MarketUniverseDataLoader, build_feature_bundle, load_index_benchmark_data
from ..data.feature_engineering import FactorEngineeringOptions, apply_factor_engineering
from ..data.preprocess import (
    PreprocessOptions,
    apply_cross_section_pipeline,
    fill_feature_na_with_reference,
    fit_feature_fill_values,
)
from ..data.sources import DataSourceRegistry, TableFileSource, load_custom_source_module, merge_external_sources
from ..factors.base import (
    FactorLibrary,
    compute_factor_panel,
    enrich_factor_metadata_for_display,
    load_custom_factor_module,
    register_passthrough_panel_factors,
    resolve_selected_factors,
)
from ..factors.defaults import (
    DEFAULT_FACTOR_PACKS_BY_FREQ,
    build_factor_package_index,
    list_default_factor_packages,
    register_default_factors,
    resolve_primary_factor_package,
    resolve_default_factor_set,
)
from ..factors.labeling import add_labels, pick_target_column, split_train_test, validate_label_frequency_alignment
from ..factors.reporting import export_factor_snapshot, normalize_factor_package_alias
from ..factors.store import (
    build_factor_store_for_full_list,
    hydrate_factor_panel_with_store,
    resolve_factor_store_root,
)
from ..models import build_execution_model, build_portfolio_model, build_stock_model, build_timing_model
from ..models.loading import (
    bootstrap_stock_model_history,
    load_execution_model,
    load_portfolio_model,
    load_stock_model,
    load_timing_model,
    peek_stock_model_factor_cols,
    resolve_model_artifact_paths,
    stock_model_factor_cols,
)
from ..mining.catalog import (
    list_catalog_factor_packages,
    load_active_catalog_entries,
    merge_catalog_factors,
    register_catalog_factors,
)
from .artifacts import build_run_tag, save_common_artifacts, save_dataframe


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


def _profile_market_frame(
    frame: pd.DataFrame,
    *,
    time_col: str,
    required_cols: List[str],
) -> Dict[str, object]:
    missing_required = [c for c in required_cols if c not in frame.columns]
    out: Dict[str, object] = {
        "rows": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "missing_required_cols": missing_required,
    }
    if frame.empty:
        out["code_count"] = 0
        out["time_min"] = ""
        out["time_max"] = ""
        out["all_na_required_numeric"] = []
        return out

    if "code" in frame.columns:
        code_s = frame["code"].astype(str).str.strip()
        non_empty_code = code_s[code_s.ne("")]
        out["code_count"] = int(non_empty_code.nunique())
        out["sample_codes"] = non_empty_code.drop_duplicates().head(12).tolist()
    else:
        out["code_count"] = 0
        out["sample_codes"] = []

    if time_col in frame.columns:
        ts = pd.to_datetime(frame[time_col], errors="coerce")
        valid_ts = ts.dropna()
        out["valid_time_count"] = int(len(valid_ts))
        if valid_ts.empty:
            out["time_min"] = ""
            out["time_max"] = ""
        else:
            out["time_min"] = str(valid_ts.min())
            out["time_max"] = str(valid_ts.max())
    else:
        out["valid_time_count"] = 0
        out["time_min"] = ""
        out["time_max"] = ""

    numeric_req = [c for c in required_cols if c in frame.columns and c not in {"date", "datetime", "code", "time"}]
    na_ratio: Dict[str, float] = {}
    all_na: List[str] = []
    for c in numeric_req:
        s = pd.to_numeric(frame[c], errors="coerce")
        ratio = float(s.isna().mean()) if len(s) else 1.0
        na_ratio[str(c)] = ratio
        if ratio >= 0.999999:
            all_na.append(str(c))
    out["required_numeric_na_ratio"] = na_ratio
    out["all_na_required_numeric"] = all_na
    return out


def _build_market_data_health_report(market_bundle, *, factor_freq: str) -> Dict[str, object]:
    daily_required = ["date", "code", "open", "high", "low", "close", "volume", "amount"]
    minute_required = ["date", "datetime", "code", "open", "high", "low", "close", "volume", "amount"]
    daily_profile = _profile_market_frame(
        market_bundle.daily,
        time_col="date",
        required_cols=daily_required,
    )
    minute_profile = _profile_market_frame(
        market_bundle.minute5,
        time_col="datetime",
        required_cols=minute_required,
    )
    return {
        "factor_freq": str(factor_freq),
        "load_window": {
            "start_date": str(pd.to_datetime(market_bundle.start_date).date()),
            "end_date": str(pd.to_datetime(market_bundle.end_date).date()),
        },
        "summary": {
            "daily_rows": int(len(market_bundle.daily)),
            "minute_rows": int(len(market_bundle.minute5)),
            "codes": int(len(market_bundle.codes)),
            "daily_code_count": int(daily_profile.get("code_count", 0)),
            "minute_code_count": int(minute_profile.get("code_count", 0)),
        },
        "daily": daily_profile,
        "minute5": minute_profile,
        "source_notes": dict(market_bundle.source_notes),
    }


def _safe_read_json_dict(path: Path) -> Dict[str, object]:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        raw = path.read_text(encoding="utf-8-sig")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError(f"summary json must be a dict object: {path}")
    return payload


def _parse_factor_names_expr(expr: str) -> List[str]:
    return [x.strip() for x in str(expr).split(",") if x.strip()]


def _resolve_used_factors_for_snapshot(
    *,
    library: FactorLibrary,
    freq: str,
    factor_list_arg: str,
    default_set: List[str],
) -> List[str]:
    if str(factor_list_arg).strip():
        requested = [x.strip() for x in str(factor_list_arg).split(",") if x.strip()]
    else:
        requested = list(default_set)
    available = set(library.names(freq))
    return [x for x in requested if x in available]


def _split_package_tokens_for_default_and_catalog(freq: str, package_expr: str) -> tuple[List[str], str]:
    tokens = [x.strip() for x in str(package_expr).split(",") if x.strip()]
    if not tokens:
        return [], "all"
    tokens_lower = {x.lower() for x in tokens}
    if "all" in tokens_lower:
        return [], "all"

    avail_defaults = {x.lower(): x for x in list_default_factor_packages(freq)}
    default_tokens: List[str] = []
    catalog_tokens: List[str] = []
    for tok in tokens:
        k = tok.lower()
        if k in avail_defaults:
            default_tokens.append(avail_defaults[k])
        else:
            catalog_tokens.append(tok)
    return default_tokens, ",".join(catalog_tokens)


def _resolve_default_factors_by_packages(freq: str, package_expr: str) -> List[str]:
    tokens = [x.strip() for x in str(package_expr).split(",") if x.strip()]
    if not tokens:
        return resolve_default_factor_set(freq=freq, package_expr="all")
    if "all" in {x.lower() for x in tokens}:
        return resolve_default_factor_set(freq=freq, package_expr="all")
    avail_defaults = {x.lower(): x for x in list_default_factor_packages(freq)}
    selected = [avail_defaults[x.lower()] for x in tokens if x.lower() in avail_defaults]
    if not selected:
        return []
    return resolve_default_factor_set(freq=freq, package_expr=",".join(selected))


def _summarize_factor_categories(meta_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Build factor-package count summary for list-factors output."""
    group_col = "factor_package" if "factor_package" in meta_df.columns else "category"
    if meta_df.empty or group_col not in meta_df.columns:
        return {"all": {}, "price_volume": {}, "fundamental": {}, "text": {}, "mined": {}}

    counts = (
        meta_df.assign(_pkg_norm=meta_df[group_col].map(normalize_factor_package_alias))
        .groupby("_pkg_norm")
        .size()
        .sort_values(ascending=False)
        .astype(int)
    )
    all_counts = {str(k): int(v) for k, v in counts.items()}
    fund_counts = {k: v for k, v in all_counts.items() if str(k).startswith("fund_") or str(k).startswith("fundamental_")}
    text_counts = {k: v for k, v in all_counts.items() if str(k).startswith("text_")}
    mined_counts = {k: v for k, v in all_counts.items() if str(k).startswith("mined_")}
    pv_counts = {
        k: v
        for k, v in all_counts.items()
        if not (
            str(k).startswith("fund_")
            or str(k).startswith("fundamental_")
            or str(k).startswith("text_")
            or str(k).startswith("mined_")
        )
    }
    return {
        "all": all_counts,
        "price_volume": pv_counts,
        "fundamental": fund_counts,
        "text": text_counts,
        "mined": mined_counts,
    }


def _print_factor_category_summary(meta_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    summary = _summarize_factor_categories(meta_df)
    pv_counts = summary["price_volume"]
    fund_counts = summary["fundamental"]
    text_counts = summary["text"]
    mined_counts = summary["mined"]

    print("\n=== Factor Package Summary ===")
    if pv_counts:
        print("[Price-Volume / Non-Fundamental Packages]")
        print(pd.Series(pv_counts, dtype=int).sort_values(ascending=False).to_string())
    else:
        print("[Price-Volume / Non-Fundamental Packages]\n(none)")

    if fund_counts:
        print("\n[Fundamental Packages]")
        print(pd.Series(fund_counts, dtype=int).sort_values(ascending=False).to_string())
    else:
        print("\n[Fundamental Packages]\n(none)")
    if text_counts:
        print("\n[Text Packages]")
        print(pd.Series(text_counts, dtype=int).sort_values(ascending=False).to_string())
    else:
        print("\n[Text Packages]\n(none)")
    if mined_counts:
        print("\n[Mined Packages]")
        print(pd.Series(mined_counts, dtype=int).sort_values(ascending=False).to_string())
    else:
        print("\n[Mined Packages]\n(none)")
    return summary


def _attach_factor_package_columns(meta_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if meta_df.empty or "factor" not in meta_df.columns:
        out = meta_df.copy()
        out["factor_package"] = pd.Series(dtype=str)
        out["factor_packages"] = pd.Series(dtype=str)
        return out

    out = meta_df.copy()
    package_index = build_factor_package_index(freq=str(freq))
    categories = out["category"].astype(str).tolist() if "category" in out.columns else [""] * len(out)
    primary_packages: List[str] = []
    all_packages: List[str] = []
    for fac, cat in zip(out["factor"].astype(str).tolist(), categories):
        pri, members = resolve_primary_factor_package(
            freq=str(freq),
            factor=fac,
            category=cat,
            package_index=package_index,
        )
        primary_packages.append(normalize_factor_package_alias(pri))
        all_packages.append(",".join([normalize_factor_package_alias(x) for x in members]))
    out["factor_package"] = primary_packages
    out["factor_packages"] = all_packages
    return out


def _markdown_escape(value: object) -> str:
    s = str(value)
    return s.replace("|", "\\|").replace("\n", "<br>").replace("\r", "")


def _to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)\n"
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for row in df.itertuples(index=False):
        vals = [_markdown_escape(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _export_factor_list_file(
    *,
    meta_df_view: pd.DataFrame,
    cfg: RunConfig,
    package_summary: Dict[str, Dict[str, int]],
    fund_coverage: Dict[str, object],
) -> Path:
    fmt = str(getattr(cfg.factors, "factor_list_export_format", "csv")).strip().lower()
    if fmt not in {"csv", "json", "markdown"}:
        raise ValueError(f"unsupported factor list export format: {fmt}")

    path_cfg = getattr(cfg.factors, "factor_list_export_path", None)
    if path_cfg:
        out_path = Path(str(path_cfg)).expanduser()
    else:
        export_dir = ensure_dir(cfg.output_dir / "factor_lists")
        ts = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d_%H%M%S")
        ext = {"csv": "csv", "json": "json", "markdown": "md"}[fmt]
        out_path = export_dir / f"factor_list_{cfg.factors.factor_freq}_{ts}.{ext}"
    ensure_dir(out_path.parent)

    if fmt == "csv":
        meta_df_view.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path

    if fmt == "json":
        dump_json(
            out_path,
            {
                "factor_freq": str(cfg.factors.factor_freq),
                "factor_count": int(len(meta_df_view)),
                "fundamental_coverage": fund_coverage,
                "factor_package_summary": package_summary,
                "items": meta_df_view.to_dict(orient="records"),
            },
        )
        return out_path

    # markdown
    md = []
    md.append(f"# Factor List ({cfg.factors.factor_freq})")
    md.append("")
    md.append(
        f"- factor_count: {int(len(meta_df_view))}\n"
        f"- fundamental_expected: {int(fund_coverage.get('expected', 0))}\n"
        f"- fundamental_listed: {int(fund_coverage.get('listed', 0))}\n"
        f"- fundamental_missing: {int(len(fund_coverage.get('missing', [])))}"
    )
    md.append("")
    md.append("## Factor Package Summary (Raw)")
    md.append("")
    raw_df = pd.DataFrame(
        [{"factor_package": k, "count": int(v)} for k, v in package_summary.get("all", {}).items()]
    )
    raw_df = raw_df.sort_values("count", ascending=False).reset_index(drop=True) if not raw_df.empty else raw_df
    md.append(_to_markdown_table(raw_df))
    md.append("## Factor Table")
    md.append("")
    md.append(_to_markdown_table(meta_df_view))
    out_path.write_text("\n".join(md), encoding="utf-8")
    return out_path


def _expected_fundamental_factor_names(freq: str) -> List[str]:
    packs = DEFAULT_FACTOR_PACKS_BY_FREQ.get(str(freq), {})
    names: List[str] = []
    for pack, factors in packs.items():
        if str(pack).startswith("fund_"):
            names.extend([str(x) for x in factors])
    return sorted(set(names))


def _ensure_fundamental_factor_coverage(
    *,
    meta_df: pd.DataFrame,
    factor_lib: FactorLibrary,
    freq: str,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    """Ensure list-factors metadata includes all default fundamental factors for the target freq."""
    expected = _expected_fundamental_factor_names(freq)
    if not expected:
        return meta_df, {"expected": 0, "listed": 0, "missing": []}

    listed_df = meta_df[meta_df["category"].astype(str).str.startswith("fundamental_")] if not meta_df.empty else pd.DataFrame()
    listed = set(listed_df.get("factor", pd.Series(dtype=str)).astype(str).tolist())
    missing = [x for x in expected if x not in listed]

    # Safety net: if metadata somehow misses registered factors, append from registry.
    if missing:
        rows: List[Dict[str, object]] = []
        for fac in missing:
            if not factor_lib.has(freq, fac):
                continue
            fd = factor_lib.get(freq, fac)
            rows.append(
                {
                    "factor": fd.name,
                    "freq": fd.freq,
                    "category": fd.category,
                    "description": fd.description,
                }
            )
        if rows:
            meta_df = pd.concat([meta_df, pd.DataFrame(rows)], ignore_index=True)
            meta_df = (
                meta_df.drop_duplicates(subset=["freq", "factor"], keep="first")
                .sort_values(["freq", "factor"])
                .reset_index(drop=True)
            )
            listed = set(meta_df[meta_df["category"].astype(str).str.startswith("fundamental_")]["factor"].astype(str).tolist())
            missing = [x for x in expected if x not in listed]

    return meta_df, {"expected": len(expected), "listed": len(listed), "missing": missing}


def _build_external_source_registry(cfg: RunConfig) -> DataSourceRegistry:
    reg = DataSourceRegistry()
    if cfg.data.extra_factor_paths:
        log_progress(
            "检测到 --extra-factor-paths（兼容模式）。建议迁移到 --custom-factor-py 并使用 register_external_factor_table。",
            module="pipeline",
        )
    for i, p in enumerate(cfg.data.extra_factor_paths):
        name = f"extra_table_{i+1}"
        reg.register(name, TableFileSource(name=name, path=p, date_col="date", code_col="code", prefix=name))
    if cfg.data.extra_source_module:
        log_progress(
            "检测到 --extra-source-module（兼容模式）。建议迁移到 --custom-factor-py 注册外部因子。",
            module="pipeline",
        )
        load_custom_source_module(reg, cfg.data.extra_source_module)
    return reg


def _build_next_bar_inference(
    *,
    panel_df: pd.DataFrame,
    factor_freq: str,
    factor_cols: List[str],
    stock_model,
    timing_model,
    portfolio_model,
    execution_model,
    cfg: RunConfig,
    prev_weights: Dict[str, float],
) -> tuple[pd.DataFrame, Dict[str, object]]:
    if panel_df.empty:
        return pd.DataFrame(), {"enabled": True, "status": "empty_panel"}

    time_col = "date" if factor_freq in {"D", "W", "M"} else "datetime"
    if time_col not in panel_df.columns:
        return pd.DataFrame(), {"enabled": True, "status": "missing_time_col", "time_col": time_col}

    scoped = panel_df.copy()
    scoped["_signal_time"] = pd.to_datetime(scoped[time_col], errors="coerce")
    scoped = scoped.dropna(subset=["_signal_time", "code"]).copy()
    if scoped.empty:
        return pd.DataFrame(), {"enabled": True, "status": "no_valid_signal_rows"}

    latest_signal = pd.Timestamp(scoped["_signal_time"].max())
    latest_df = scoped.loc[scoped["_signal_time"] == latest_signal].copy()
    if latest_df.empty:
        return pd.DataFrame(), {"enabled": True, "status": "latest_slice_empty"}

    missing = [c for c in factor_cols if c not in latest_df.columns]
    if missing:
        return pd.DataFrame(), {"enabled": True, "status": "missing_factor_cols", "missing_factor_cols": missing}

    latest_df["pred_score"] = stock_model.predict_score(latest_df, factor_cols)
    latest_df["pred_up"] = (latest_df["pred_score"] >= cfg.backtest.long_threshold).astype(int)

    latest_df = latest_df.sort_values("pred_score", ascending=False).reset_index(drop=True)
    rule_pick = latest_df.loc[latest_df["pred_score"] >= float(cfg.backtest.long_threshold)].copy()
    rule_pick = rule_pick.head(int(cfg.backtest.top_k)).copy()
    pick_codes = set(rule_pick["code"].astype(str).tolist())

    timing_exposure, timing_diag = timing_model.predict_exposure(latest_df)
    portfolio_diag: Dict[str, float] = {}
    execution_diag: Dict[str, float] = {}

    executed_df = pd.DataFrame()
    if not rule_pick.empty and float(timing_exposure) > 0.0:
        target_w, portfolio_diag = portfolio_model.compute_weights(
            day_pick=rule_pick,
            day_universe=latest_df,
            prev_weights=prev_weights,
            fee_bps=float(cfg.backtest.fee_bps),
            slippage_bps=float(cfg.backtest.slippage_bps),
        )
        if target_w.empty or float(target_w.sum()) <= 1e-12:
            target_w = pd.Series(
                np.full(len(rule_pick), 1.0 / max(len(rule_pick), 1), dtype=float),
                index=rule_pick["code"].astype(str),
            )
            portfolio_diag["optimizer_fallback"] = 1.0

        executed_df = rule_pick.copy()
        executed_df["weight_target"] = executed_df["code"].astype(str).map(target_w).fillna(0.0)
        sw = float(executed_df["weight_target"].sum())
        if sw > 1e-12:
            executed_df["weight_target"] = executed_df["weight_target"] / sw
        executed_df["weight_target"] = executed_df["weight_target"] * float(timing_exposure)
        if "net_trade_ret" not in executed_df.columns:
            executed_df["net_trade_ret"] = 0.0
        executed_df, execution_diag = execution_model.apply_execution(
            day_pick=executed_df,
            weight_col="weight_target",
            fee_bps=float(cfg.backtest.fee_bps),
            slippage_bps=float(cfg.backtest.slippage_bps),
        )

    merge_cols = ["code", "pred_score", "pred_up"]
    if not executed_df.empty:
        for c in ["weight_target", "executed_weight", "fill_ratio", "extra_cost_bps"]:
            if c in executed_df.columns and c not in merge_cols:
                merge_cols.append(c)
        latest_df = latest_df.merge(executed_df[merge_cols], on=["code", "pred_score", "pred_up"], how="left")
    latest_df["selected_by_rule"] = latest_df["code"].astype(str).isin(pick_codes).astype(int)
    if "weight_target" not in latest_df.columns:
        latest_df["weight_target"] = np.nan
    if "executed_weight" not in latest_df.columns:
        latest_df["executed_weight"] = np.nan
    if "fill_ratio" not in latest_df.columns:
        latest_df["fill_ratio"] = np.nan

    top_n = int(getattr(cfg.model_run, "inference_top_k", cfg.backtest.top_k))
    latest_df = latest_df.head(max(top_n, 1)).copy()
    latest_df["signal_ts"] = latest_signal

    summary = {
        "enabled": True,
        "status": "ok",
        "signal_ts": str(latest_signal),
        "candidate_count": int(len(latest_df)),
        "selected_count": int(len(pick_codes)),
        "timing_exposure": float(timing_exposure),
        "timing_diag": timing_diag,
        "portfolio_diag": portfolio_diag,
        "execution_diag": execution_diag,
    }
    return latest_df, summary


def run_pipeline(cfg: RunConfig) -> Dict[str, object]:
    factor_freq = cfg.factors.factor_freq
    model_run_mode = str(getattr(cfg.model_run, "mode", "train")).strip().lower()
    load_fe_mode = str(getattr(cfg.model_run, "load_fe_mode", "refit")).strip().lower()
    if load_fe_mode not in {"strict", "refit", "off"}:
        load_fe_mode = "refit"
    if model_run_mode == "load" and bool(getattr(cfg.factors, "enable_factor_engineering", False)):
        if load_fe_mode == "strict":
            log_progress(
                "load+FE 策略：strict（将尝试严格按 summary 中 FE 结果回放）。",
                module="pipeline",
            )
        elif load_fe_mode == "refit":
            log_progress(
                "load+FE 策略：refit（将按当前样本重新拟合 FE）。",
                module="pipeline",
            )
        else:
            log_progress(
                "load+FE 策略：off（将跳过 FE，即使 enable_factor_engineering=true）。",
                module="pipeline",
            )
    if model_run_mode != "load":
        ignored_load_args: List[str] = []
        if getattr(cfg.model_run, "model_summary_json", None):
            ignored_load_args.append("model_summary_json")
        if getattr(cfg.model_run, "models_load_dir", None):
            ignored_load_args.append("models_load_dir")
        if getattr(cfg.model_run, "models_load_run_tag", None):
            ignored_load_args.append("models_load_run_tag")
        if getattr(cfg.model_run, "stock_model_path", None):
            ignored_load_args.append("stock_model_path")
        if getattr(cfg.model_run, "timing_model_path", None):
            ignored_load_args.append("timing_model_path")
        if getattr(cfg.model_run, "portfolio_model_path", None):
            ignored_load_args.append("portfolio_model_path")
        if getattr(cfg.model_run, "execution_model_path", None):
            ignored_load_args.append("execution_model_path")
        if ignored_load_args:
            log_progress(
                "train 模式下检测到 load 专用参数，将忽略："
                + ", ".join(ignored_load_args),
                module="pipeline",
            )
    explicit_factor_names = _parse_factor_names_expr(cfg.factors.factor_list)
    default_pkg_tokens, catalog_pkg_expr = _split_package_tokens_for_default_and_catalog(
        factor_freq,
        str(getattr(cfg.factors, "factor_packages", "")),
    )
    model_paths = None
    model_component_source: Dict[str, str] = {}
    load_hint_factor_cols: List[str] = []
    if model_run_mode == "load":
        if cfg.timing_model.model_type == "none" and getattr(cfg.model_run, "timing_model_path", None):
            log_progress(
                "timing_model_type=none 时 timing_model_path 不参与加载，将按配置构建 NoTimingModel。",
                module="pipeline",
            )
        if cfg.portfolio_opt.mode == "equal_weight" and getattr(cfg.model_run, "portfolio_model_path", None):
            log_progress(
                "portfolio_model_type=equal_weight 时 portfolio_model_path 不参与加载，将按配置构建 EqualWeightPortfolioModel。",
                module="pipeline",
            )
        if cfg.execution_model.model_type == "ideal_fill" and getattr(cfg.model_run, "execution_model_path", None):
            log_progress(
                "execution_model_type=ideal_fill 时 execution_model_path 不参与加载，将按配置构建 IdealFillExecutionModel。",
                module="pipeline",
            )
        model_paths = resolve_model_artifact_paths(cfg)
        model_component_source = dict(model_paths.source)
        load_hint_factor_cols = peek_stock_model_factor_cols(cfg.stock_model, model_paths.stock_model)
        log_progress(
            "检测到 load 模式：将跳过模型训练，直接加载已有模型文件。",
            module="pipeline",
        )
        if model_paths.stock_model:
            log_progress(f"load stock model path: {model_paths.stock_model}", module="pipeline")
    log_progress(
        f"主流水线启动：factor_freq={cfg.factors.factor_freq}, "
        f"train={cfg.dates.train_start.date()}~{cfg.dates.train_end.date()}, "
        f"test={cfg.dates.test_start.date()}~{cfg.dates.test_end.date()}。",
        module="pipeline",
    )
    # Fast path: listing factors should not require loading market data.
    # This makes `--list-factors` usable even when local data folders are unavailable.
    if cfg.factors.list_factors and not (
        bool(getattr(cfg.factors, "enable_factor_value_store", False))
        and bool(getattr(cfg.factors, "factor_value_store_build_all", False))
    ):
        log_progress("进入仅列出因子模式。", module="pipeline")
        factor_lib = FactorLibrary()
        register_default_factors(factor_lib)
        avail_packages = list_default_factor_packages(cfg.factors.factor_freq)
        if avail_packages:
            log_progress(
                f"default factor packages@{cfg.factors.factor_freq}: {avail_packages}",
                module="pipeline",
                level="debug",
            )
        avail_catalog_packages: List[str] = []
        if cfg.data.auto_load_catalog_factors and cfg.data.factor_catalog_path:
            avail_catalog_packages = list_catalog_factor_packages(
                catalog_path=cfg.data.factor_catalog_path,
                freq=cfg.factors.factor_freq,
            )
            if avail_catalog_packages:
                log_progress(
                    f"catalog factor packages@{cfg.factors.factor_freq}: {avail_catalog_packages}",
                    module="pipeline",
                    level="debug",
                )
        catalog_count = 0
        entries: List[Dict[str, object]] = []
        should_load_catalog_list = bool(cfg.data.auto_load_catalog_factors and cfg.data.factor_catalog_path)
        if should_load_catalog_list and str(getattr(cfg.factors, "factor_packages", "")).strip() and not catalog_pkg_expr and not explicit_factor_names:
            should_load_catalog_list = False
        if should_load_catalog_list:
            catalog_pkg_for_list = "all" if not str(getattr(cfg.factors, "factor_packages", "")).strip() else catalog_pkg_expr
            entries = load_active_catalog_entries(
                cfg.data.factor_catalog_path,
                freq=cfg.factors.factor_freq,
                factor_names=explicit_factor_names or None,
                package_expr=catalog_pkg_for_list,
            )
            if entries:
                register_catalog_factors(factor_lib, entries)
                catalog_count = int(len(entries))
        if cfg.factors.custom_factor_py:
            load_custom_factor_module(factor_lib, cfg.factors.custom_factor_py)
        if str(getattr(cfg.factors, "factor_packages", "")).strip():
            selected_default = _resolve_default_factors_by_packages(
                freq=cfg.factors.factor_freq,
                package_expr=str(cfg.factors.factor_packages),
            )
            log_progress(
                f"factor-packages filter active: {cfg.factors.factor_packages}, "
                f"default_selected_count={len(selected_default)}",
                module="pipeline",
            )
        meta_df = factor_lib.metadata(freq=cfg.factors.factor_freq)
        meta_df, fund_coverage = _ensure_fundamental_factor_coverage(
            meta_df=meta_df,
            factor_lib=factor_lib,
            freq=cfg.factors.factor_freq,
        )
        meta_df = _attach_factor_package_columns(meta_df, freq=cfg.factors.factor_freq)
        meta_df_view = enrich_factor_metadata_for_display(meta_df)
        if "category" in meta_df_view.columns:
            meta_df_view = meta_df_view.drop(columns=["category"])
        for c in ["category_l1", "category_l1_cn"]:
            if c in meta_df_view.columns:
                meta_df_view = meta_df_view.drop(columns=[c])
        preferred_cols = [
            "factor",
            "freq",
            "factor_package",
            "factor_packages",
            "name_cn",
            "meaning_cn",
            "formula_cn",
            "description",
        ]
        ordered_cols = [c for c in preferred_cols if c in meta_df_view.columns] + [c for c in meta_df_view.columns if c not in preferred_cols]
        meta_df_view = meta_df_view[ordered_cols]
        print(meta_df_view.to_string(index=False))
        print(
            "\n=== Fundamental Factor Coverage ===\n"
            f"expected={fund_coverage['expected']}, "
            f"listed={fund_coverage['listed']}, "
            f"missing={len(fund_coverage['missing'])}"
        )
        if fund_coverage["missing"]:
            print("missing factors:")
            print(pd.Series(fund_coverage["missing"], dtype=str).to_string(index=False))
        snapshot_default_set = _resolve_default_factors_by_packages(
            freq=cfg.factors.factor_freq,
            package_expr=str(getattr(cfg.factors, "factor_packages", "")),
        )
        snapshot_catalog_set: List[str] = []
        if not explicit_factor_names:
            if not str(getattr(cfg.factors, "factor_packages", "")).strip():
                snapshot_catalog_set = [str(e.get("name", "")).strip() for e in entries if str(e.get("name", "")).strip()]
            elif catalog_pkg_expr:
                snapshot_catalog_set = [str(e.get("name", "")).strip() for e in entries if str(e.get("name", "")).strip()]
        snapshot_default_set = sorted(set(snapshot_default_set) | set(snapshot_catalog_set))
        selected_for_snapshot = _resolve_used_factors_for_snapshot(
            library=factor_lib,
            freq=cfg.factors.factor_freq,
            factor_list_arg=str(getattr(cfg.factors, "factor_list", "")),
            default_set=snapshot_default_set,
        )
        snapshot_paths: Dict[str, object] = {}
        if bool(getattr(cfg.factors, "auto_export_factor_snapshot", False)):
            snapshot_paths = export_factor_snapshot(
                meta_df_view=meta_df_view,
                used_factors=selected_for_snapshot,
                entrypoint="run_strategy7",
                factor_freq=cfg.factors.factor_freq,
                output_root=Path(__file__).resolve().parents[2] / "outputs" / "factor_snapshots",
                run_tag="list_factors",
                extra_summary={
                    "mode": "list_factors",
                    "catalog_factor_count": int(catalog_count),
                    "selected_factor_count": int(len(selected_for_snapshot)),
                },
            )
            log_progress(
                f"自动因子快照导出完成：snapshot_dir={snapshot_paths.get('snapshot_dir', '')}",
                module="pipeline",
            )
        package_summary = _print_factor_category_summary(meta_df)
        pv_total = int(sum(package_summary["price_volume"].values()))
        fund_total = int(sum(package_summary["fundamental"].values()))
        text_total = int(sum(package_summary["text"].values()))
        mined_total = int(sum(package_summary["mined"].values()))
        export_path: Path | None = None
        if bool(getattr(cfg.factors, "export_factor_list", False)):
            export_path = _export_factor_list_file(
                meta_df_view=meta_df_view,
                cfg=cfg,
                package_summary=package_summary,
                fund_coverage=fund_coverage,
            )
            log_progress(f"因子清单导出完成：format={cfg.factors.factor_list_export_format}, path={export_path}", module="pipeline")
        log_progress(
            (
                f"因子清单输出完成：freq={cfg.factors.factor_freq}, "
                f"total_factor_count={len(meta_df)}, "
                f"price_volume_factor_count={pv_total}, "
                f"fundamental_factor_count={fund_total}, "
                f"text_factor_count={text_total}, "
                f"mined_factor_count={mined_total}, "
                f"fundamental_expected_count={fund_coverage['expected']}, "
                f"fundamental_missing_count={len(fund_coverage['missing'])}, "
                f"factor_package_count={len(package_summary['all'])}, "
                f"catalog_factor_count={catalog_count}。"
            ),
            module="pipeline",
        )
        return {
            "status": "listed_factors_only",
            "factor_freq": cfg.factors.factor_freq,
            "catalog_factor_count": catalog_count,
            "total_factor_count": int(len(meta_df)),
            "price_volume_factor_count": pv_total,
            "fundamental_factor_count": fund_total,
            "text_factor_count": text_total,
            "mined_factor_count": mined_total,
            "fundamental_expected_count": int(fund_coverage["expected"]),
            "fundamental_missing_count": int(len(fund_coverage["missing"])),
            "fundamental_missing_factors": list(fund_coverage["missing"]),
            "factor_package_counts": package_summary["all"],
            "price_volume_package_counts": package_summary["price_volume"],
            "fundamental_package_counts": package_summary["fundamental"],
            "text_package_counts": package_summary["text"],
            "mined_package_counts": package_summary["mined"],
            "factor_list_export_format": str(getattr(cfg.factors, "factor_list_export_format", "csv")),
            "factor_list_export_path": str(export_path) if export_path is not None else "",
            "factor_list_exported": bool(export_path is not None),
            "factor_snapshot_dir": str(snapshot_paths.get("snapshot_dir", "")),
            "factor_snapshot_summary_path": str(snapshot_paths.get("summary_path", "")),
        }

    output_dir = ensure_dir(cfg.output_dir)
    log_progress(f"输出目录已就绪：{output_dir}", module="pipeline")
    model_dir: Path | None = None
    market_data_health: Dict[str, object] = {}
    market_data_health_path: Path | None = None
    if cfg.save_models:
        model_dir = ensure_dir(output_dir / "models")
        log_progress(f"模型输出目录已就绪：{model_dir}", module="pipeline")

    # 1) Load raw market data and build feature bundle for all supported frequencies.
    log_progress("步骤 1/13：开始加载市场数据。", module="pipeline")
    loader = MarketUniverseDataLoader(
        data_cfg=cfg.data,
        date_cfg=cfg.dates,
        lookback_days=cfg.factors.lookback_days,
        horizon=cfg.backtest.horizon,
        factor_freq=factor_freq,
    )
    market_bundle = loader.load()
    market_data_health = _build_market_data_health_report(market_bundle, factor_freq=factor_freq)
    market_data_health_path = output_dir / "market_data_health.json"
    dump_json(market_data_health_path, market_data_health)
    log_progress(f"市场数据健康报告已写出：{market_data_health_path}", module="pipeline")
    log_progress(
        f"市场数据加载完成：daily_rows={len(market_bundle.daily)}, minute_rows={len(market_bundle.minute5)}, "
        f"codes={len(market_bundle.codes)}。",
        module="pipeline",
    )
    log_progress("步骤 1/13：开始构建多频特征视图。", module="pipeline")
    feat_bundle = build_feature_bundle(market_bundle)
    log_progress(
        f"特征构建完成，可用频率={sorted(feat_bundle.by_freq.keys())}。",
        module="pipeline",
    )

    if factor_freq not in feat_bundle.by_freq:
        raise ValueError(f"feature bundle missing freq={factor_freq}")
    base_df = feat_bundle.by_freq[factor_freq].copy()
    if base_df.empty:
        raise RuntimeError(f"base feature frame is empty for freq={factor_freq}")
    log_progress(f"基准面板就绪：freq={factor_freq}, rows={len(base_df)}。", module="pipeline")

    # 2) Merge admitted mined/custom factors from catalog (if enabled).
    # These factors are materialized values produced by the mining subsystem.
    log_progress("步骤 2/13：开始合并 catalog 因子（若启用）。", module="pipeline")
    catalog_rows: Dict[str, int] = {}
    catalog_entries: List[Dict[str, object]] = []
    should_load_catalog = bool(cfg.data.auto_load_catalog_factors and cfg.data.factor_catalog_path)
    if should_load_catalog and str(getattr(cfg.factors, "factor_packages", "")).strip() and not catalog_pkg_expr and not explicit_factor_names:
        should_load_catalog = False
    if should_load_catalog:
        catalog_pkg_for_merge = "all" if not str(getattr(cfg.factors, "factor_packages", "")).strip() else catalog_pkg_expr
        base_df, catalog_rows, catalog_entries = merge_catalog_factors(
            base_panel=base_df,
            catalog_path=cfg.data.factor_catalog_path,
            factor_freq=factor_freq,
            factor_names=explicit_factor_names or None,
            package_expr=catalog_pkg_for_merge,
        )
    log_progress(
        f"catalog 合并完成：entries={len(catalog_entries)}, merged_rows_meta={catalog_rows}。",
        module="pipeline",
    )

    # 3) Merge optional external sources (fundamental/NLP/custom tables).
    log_progress("步骤 3/13：开始合并外部数据源（若配置）。", module="pipeline")
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
    log_progress(f"外部数据源合并完成：registry_size={len(source_registry.keys())}, rows_meta={external_rows}。", module="pipeline")

    # 4) Build factor registry (default + catalog + optional custom Python plugin).
    log_progress("步骤 4/13：开始构建因子库（默认+catalog+自定义）。", module="pipeline")
    factor_lib = FactorLibrary()
    register_default_factors(factor_lib)
    if catalog_entries:
        register_catalog_factors(factor_lib, catalog_entries)
    if cfg.factors.custom_factor_py:
        load_custom_factor_module(factor_lib, cfg.factors.custom_factor_py)
    auto_panel_factors = register_passthrough_panel_factors(
        factor_lib,
        base_df=base_df,
        freq=factor_freq,
    )
    if auto_panel_factors:
        log_progress(
            f"auto passthrough factors registered: {len(auto_panel_factors)}",
            module="pipeline",
            level="debug",
        )
    log_progress("因子库构建完成。", module="pipeline")

    # 5) Resolve selected factors and compute the factor panel.
    log_progress("步骤 5/13：解析因子清单并计算因子面板。", module="pipeline")
    default_set = _resolve_default_factors_by_packages(
        freq=factor_freq,
        package_expr=str(getattr(cfg.factors, "factor_packages", "")),
    )
    catalog_default_set: List[str] = []
    if not explicit_factor_names:
        if not str(getattr(cfg.factors, "factor_packages", "")).strip():
            catalog_default_set = [str(e.get("name", "")).strip() for e in catalog_entries if str(e.get("name", "")).strip()]
        elif catalog_pkg_expr:
            catalog_default_set = [str(e.get("name", "")).strip() for e in catalog_entries if str(e.get("name", "")).strip()]
    default_set = sorted(set(default_set) | set(catalog_default_set))
    if str(getattr(cfg.factors, "factor_packages", "")).strip():
        log_progress(
            f"default factor package filter: {cfg.factors.factor_packages}; "
            f"default_count={len(default_set)}; "
            f"available_packages={list(DEFAULT_FACTOR_PACKS_BY_FREQ.get(factor_freq, {}).keys())}",
            module="pipeline",
            level="debug",
        )
    selected_factors = resolve_selected_factors(
        library=factor_lib,
        freq=factor_freq,
        factor_list_arg=cfg.factors.factor_list,
        default_set=default_set,
    )
    if model_run_mode == "load" and load_hint_factor_cols:
        missing_hint = [c for c in load_hint_factor_cols if c not in set(factor_lib.names(factor_freq))]
        if missing_hint:
            raise RuntimeError(
                "loaded stock model requires factors not available in current factor library: "
                + ", ".join(missing_hint[:20])
                + (" ..." if len(missing_hint) > 20 else "")
            )
        selected_factors = list(load_hint_factor_cols)
        log_progress(
            f"load 模式下按模型文件覆盖因子清单：factor_count={len(selected_factors)}。",
            module="pipeline",
        )
    log_progress(f"已选因子数量：{len(selected_factors)}。", module="pipeline")

    run_meta_df = factor_lib.metadata(freq=factor_freq)
    run_meta_df, _ = _ensure_fundamental_factor_coverage(
        meta_df=run_meta_df,
        factor_lib=factor_lib,
        freq=factor_freq,
    )
    run_meta_df = _attach_factor_package_columns(run_meta_df, freq=factor_freq)
    run_meta_view = enrich_factor_metadata_for_display(run_meta_df)
    if "category" in run_meta_view.columns:
        run_meta_view = run_meta_view.drop(columns=["category"])
    for c in ["category_l1", "category_l1_cn"]:
        if c in run_meta_view.columns:
            run_meta_view = run_meta_view.drop(columns=[c])
    snapshot_paths: Dict[str, object] = {}
    factor_package_map = {
        str(r["factor"]): str(r["factor_package"])
        for r in run_meta_df[["factor", "factor_package"]].to_dict(orient="records")
    }
    factor_store_enabled = bool(getattr(cfg.factors, "enable_factor_value_store", False))
    factor_store_root = resolve_factor_store_root(
        data_root=str(cfg.data.data_root),
        store_root_arg=str(getattr(cfg.factors, "factor_value_store_root", "auto")),
    )
    factor_store_fmt = str(getattr(cfg.factors, "factor_value_store_format", "parquet"))
    factor_store_build_report: Dict[str, object] = {}
    factor_store_hydrate_report: Dict[str, object] = {"cache_enabled": False}

    if factor_store_enabled and bool(getattr(cfg.factors, "factor_value_store_build_all", False)):
        all_factor_df = run_meta_df.copy()
        if "category" in all_factor_df.columns:
            all_factor_df = all_factor_df[all_factor_df["category"].astype(str) != "auto_panel"]
        all_factors = sorted(set(all_factor_df.get("factor", pd.Series(dtype=str)).astype(str).tolist()))
        log_progress(
            f"开始构建因子值缓存仓库（完整清单）：freq={factor_freq}, factor_count={len(all_factors)}。",
            module="pipeline",
        )
        factor_store_build_report = build_factor_store_for_full_list(
            base_df=base_df,
            library=factor_lib,
            freq=factor_freq,
            all_factors=all_factors,
            store_root=factor_store_root,
            file_format=factor_store_fmt,
            factor_package_map=factor_package_map,
            chunk_size=int(getattr(cfg.factors, "factor_value_store_chunk_size", 64)),
        )
        log_progress(
            "因子值缓存仓库构建完成："
            f"saved_factor_count={int(factor_store_build_report.get('saved_factor_count', 0))}, "
            f"saved_code_files={int(factor_store_build_report.get('saved_code_files', 0))}。",
            module="pipeline",
        )
        if bool(getattr(cfg.factors, "factor_value_store_build_only", False)):
            return {
                "status": "factor_value_store_built_only",
                "factor_freq": str(factor_freq),
                "factor_store_root": str(factor_store_root),
                "factor_store_format": str(factor_store_fmt),
                "factor_store_build_report": factor_store_build_report,
            }

    if factor_store_enabled:
        panel, factor_store_hydrate_report = hydrate_factor_panel_with_store(
            base_df=base_df,
            library=factor_lib,
            freq=factor_freq,
            selected_factors=selected_factors,
            store_root=factor_store_root,
            file_format=factor_store_fmt,
            factor_package_map=factor_package_map,
            coverage_threshold=0.999999,
            write_back=True,
        )
    else:
        panel = compute_factor_panel(base_df=base_df, library=factor_lib, freq=factor_freq, selected_factors=selected_factors)
    log_progress(f"因子面板计算完成：rows={len(panel)}, cols={len(panel.columns)}。", module="pipeline")

    # 6) Cross-sectional preprocessing (winsorize / zscore / optional neutralize).
    log_progress("步骤 6/13：执行截面预处理。", module="pipeline")
    pp_opt = PreprocessOptions(winsorize_limit=0.01, do_zscore=True, neutralize=False, fill_method="median")
    group_col = "date" if factor_freq in {"D", "W", "M"} else "datetime"
    if factor_freq in INTRADAY_FREQS and "datetime" in panel.columns:
        panel["datetime"] = pd.to_datetime(panel["datetime"], errors="coerce")
    panel = apply_cross_section_pipeline(panel, selected_factors, pp_opt, group_col=group_col)
    log_progress("截面预处理完成。", module="pipeline")

    # 7) Labeling and strict time-split.
    # split_train_test enforces target-date boundaries to avoid look-ahead leakage.
    log_progress("步骤 7/13：生成标签并按时间严格切分训练/测试集。", module="pipeline")
    panel = add_labels(
        panel=panel,
        horizon=cfg.backtest.horizon,
        execution_scheme=cfg.backtest.execution_scheme,
        price_table_daily=feat_bundle.price_table_daily,
        factor_freq=factor_freq,
    )
    label_align = validate_label_frequency_alignment(
        panel=panel,
        factor_freq=factor_freq,
        horizon=int(cfg.backtest.horizon),
        strict=True,
    )
    log_progress(
        (
            "标签频率/持有期对齐校验通过："
            f"bad_signal_time={int(label_align.get('bad_signal_time', 0))}, "
            f"bad_time_order={int(label_align.get('bad_time_order', 0))}, "
            f"bad_entry_shift={int(label_align.get('bad_entry_shift', 0))}, "
            f"bad_exit_shift={int(label_align.get('bad_exit_shift', 0))}。"
        ),
        module="pipeline",
        level="debug",
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
    log_progress(f"样本切分完成：train_rows={len(train_df)}, test_rows={len(test_df)}。", module="pipeline")
    target_col = pick_target_column(cfg.factors.label_task)
    raw_train_df = train_df.copy()
    panel_for_scoring = panel.copy()

    # 8) Train-fitted feature fill values are reused on test set.
    # This avoids using future-period statistics to fill missing values.
    log_progress("步骤 8/13：按训练集统计进行缺失值填充。", module="pipeline")
    fill_values = fit_feature_fill_values(train_df, selected_factors)
    train_df = fill_feature_na_with_reference(
        train_df,
        selected_factors,
        method=pp_opt.fill_method,
        reference_fill_values=fill_values,
    )
    test_df = fill_feature_na_with_reference(
        test_df,
        selected_factors,
        method=pp_opt.fill_method,
        reference_fill_values=fill_values,
    )
    panel_for_scoring = fill_feature_na_with_reference(
        panel_for_scoring,
        selected_factors,
        method=pp_opt.fill_method,
        reference_fill_values=fill_values,
    )
    log_progress("缺失值填充完成。", module="pipeline")
    selected_factors_before_fe = list(selected_factors)
    train_df_before_fe = train_df.copy()
    fe_requested = bool(getattr(cfg.factors, "enable_factor_engineering", False))
    fe_report: Dict[str, object] = {
        "enabled": False,
        "input_factor_count": int(len(selected_factors_before_fe)),
        "final_factor_count": int(len(selected_factors_before_fe)),
        "orth_method": "none",
        "mode": "disabled",
    }
    if fe_requested and model_run_mode == "load" and load_fe_mode == "off":
        log_progress("步骤 8.5/13：load-fe-mode=off，跳过因子特征工程。", module="pipeline")
        fe_report["mode"] = "skipped_by_load_fe_mode_off"
    elif fe_requested and model_run_mode == "load" and load_fe_mode == "strict":
        log_progress("步骤 8.5/13：执行 load+FE strict 回放。", module="pipeline")
        if not getattr(cfg.model_run, "model_summary_json", None):
            raise RuntimeError("load-fe-mode=strict requires --model-summary-json for FE replay.")
        summary_path = Path(str(cfg.model_run.model_summary_json)).expanduser()
        if not summary_path.exists():
            raise FileNotFoundError(f"model_summary_json not found for strict FE replay: {summary_path}")
        summary_payload = _safe_read_json_dict(summary_path)
        strict_fe = dict(summary_payload.get("notes", {}).get("feature_engineering_summary", {}) or {})
        if not strict_fe:
            raise RuntimeError("strict FE replay requires notes.feature_engineering_summary in model summary.")
        if not bool(strict_fe.get("enabled", False)):
            raise RuntimeError("strict FE replay requires source run with feature_engineering enabled.")
        strict_orth = str(strict_fe.get("orth_method", "none")).strip().lower()
        if strict_orth == "pca":
            raise RuntimeError(
                "strict FE replay does not support orth_method=pca yet (missing persisted PCA state). "
                "Use --load-fe-mode refit, or re-run source with fe-orth-method none."
            )
        strict_selected = [str(x) for x in strict_fe.get("selected_factors", []) if str(x).strip()]
        if not strict_selected:
            raise RuntimeError("strict FE replay requires non-empty selected_factors in source summary.")
        missing_strict = [c for c in strict_selected if c not in train_df.columns or c not in test_df.columns]
        if missing_strict:
            raise RuntimeError(
                "strict FE replay factors are not available in current dataset: "
                + ", ".join(missing_strict[:20])
                + (" ..." if len(missing_strict) > 20 else "")
            )
        missing_panel_strict = [c for c in strict_selected if c not in panel_for_scoring.columns]
        if missing_panel_strict:
            raise RuntimeError(
                "inference panel missing strict FE replay factors: "
                + ", ".join(missing_panel_strict[:20])
                + (" ..." if len(missing_panel_strict) > 20 else "")
            )
        selected_factors = list(strict_selected)
        fe_report = {
            **strict_fe,
            "enabled": True,
            "mode": "strict_replay_from_summary",
            "input_factor_count": int(len(selected_factors_before_fe)),
            "final_factor_count": int(len(selected_factors)),
        }
        log_progress(
            "load+FE strict 回放完成："
            f"final={int(len(selected_factors))}, orth={strict_orth}。",
            module="pipeline",
        )
    elif fe_requested:
        if model_run_mode == "load" and load_fe_mode == "refit":
            log_progress("步骤 8.5/13：执行 load+FE refit（按当前样本重新拟合）。", module="pipeline")
        else:
            log_progress("步骤 8.5/13：执行因子特征工程（覆盖率/相关性/正交化）。", module="pipeline")
        fe_opts = FactorEngineeringOptions(
            enabled=True,
            min_coverage=float(getattr(cfg.factors, "fe_min_coverage", 0.70)),
            min_std=float(getattr(cfg.factors, "fe_min_std", 1e-8)),
            corr_threshold=float(getattr(cfg.factors, "fe_corr_threshold", 0.92)),
            preselect_top_n=int(getattr(cfg.factors, "fe_preselect_top_n", 2000)),
            min_factors=int(getattr(cfg.factors, "fe_min_factors", 20)),
            max_factors=int(getattr(cfg.factors, "fe_max_factors", 600)),
            orth_method=str(getattr(cfg.factors, "fe_orth_method", "none")),
            pca_variance_ratio=float(getattr(cfg.factors, "fe_pca_variance_ratio", 0.95)),
            pca_max_components=int(getattr(cfg.factors, "fe_pca_max_components", 128)),
        )
        train_df, test_df, selected_factors, fe_report = apply_factor_engineering(
            train_df=train_df,
            test_df=test_df,
            factor_cols=selected_factors,
            options=fe_opts,
            target_col=target_col,
            raw_train_df=raw_train_df,
        )
        _train_ref_unused, panel_for_scoring, panel_fe_selected, _panel_fe_report = apply_factor_engineering(
            train_df=train_df_before_fe,
            test_df=panel_for_scoring,
            factor_cols=selected_factors_before_fe,
            options=fe_opts,
            target_col=target_col,
            raw_train_df=raw_train_df,
        )
        missing_panel_fe = [c for c in selected_factors if c not in panel_for_scoring.columns]
        if missing_panel_fe:
            raise RuntimeError(
                "inference panel missing feature-engineering columns required by model: "
                + ", ".join(missing_panel_fe[:20])
                + (" ..." if len(missing_panel_fe) > 20 else "")
            )
        if panel_fe_selected != selected_factors:
            log_progress(
                "feature engineering for inference panel produced different factor ordering; "
                "will align by trained selected_factors.",
                module="pipeline",
                level="debug",
            )
        fe_report["mode"] = "refit_current_run" if model_run_mode == "load" and load_fe_mode == "refit" else "train_fit"
        log_progress(
            "因子特征工程完成："
            f"input={int(fe_report.get('input_factor_count', len(selected_factors_before_fe)))}, "
            f"final={int(fe_report.get('final_factor_count', len(selected_factors)))}, "
            f"orth={fe_report.get('orth_method', 'none')}, "
            f"pca_components={int(fe_report.get('pca_components', 0))}。",
            module="pipeline",
        )
        if not selected_factors:
            raise RuntimeError("feature engineering removed all factors; please relax FE thresholds.")

    if bool(getattr(cfg.factors, "auto_export_factor_snapshot", False)):
        snapshot_paths = export_factor_snapshot(
            meta_df_view=run_meta_view,
            used_factors=selected_factors,
            entrypoint="run_strategy7",
            factor_freq=factor_freq,
            output_root=Path(__file__).resolve().parents[2] / "outputs" / "factor_snapshots",
            run_tag=Path(cfg.output_dir).name,
            extra_summary={
                "mode": "run",
                "catalog_factor_count": int(len(catalog_entries)),
                "selected_factor_count_before_fe": int(len(selected_factors_before_fe)),
                "selected_factor_count_after_fe": int(len(selected_factors)),
                "feature_engineering": fe_report,
                "factor_store_hydrate_report": factor_store_hydrate_report,
                "factor_store_build_report": factor_store_build_report,
                "output_dir": str(cfg.output_dir),
            },
        )
        log_progress(
            f"自动因子快照导出完成：snapshot_dir={snapshot_paths.get('snapshot_dir', '')}",
            module="pipeline",
        )

    # 9) Train models or load existing artifacts.
    if model_run_mode == "load":
        log_progress("步骤 9/13：加载模型（选股/择时/组合/执行）。", module="pipeline")
        if model_paths is None:
            model_paths = resolve_model_artifact_paths(cfg)
        stock_model, stock_source = load_stock_model(cfg.stock_model, model_paths.stock_model)
        model_component_source["stock_model"] = stock_source

        loaded_factor_cols = stock_model_factor_cols(stock_model, fallback=selected_factors)
        missing_loaded_factors = [c for c in loaded_factor_cols if c not in train_df.columns or c not in test_df.columns]
        if missing_loaded_factors:
            raise RuntimeError(
                "loaded stock model factor cols are not available in current dataset: "
                + ", ".join(missing_loaded_factors[:20])
                + (" ..." if len(missing_loaded_factors) > 20 else "")
            )
        selected_factors = list(loaded_factor_cols)

        # In load mode, prefer model-side fill values when available to keep inference behavior consistent.
        model_fill_values = stock_model.fill_values()
        if isinstance(model_fill_values, pd.Series) and not model_fill_values.empty:
            model_fill_values = model_fill_values.reindex(selected_factors).fillna(fill_values.reindex(selected_factors)).fillna(0.0)
            train_df = fill_feature_na_with_reference(
                train_df,
                selected_factors,
                method=pp_opt.fill_method,
                reference_fill_values=model_fill_values,
            )
            test_df = fill_feature_na_with_reference(
                test_df,
                selected_factors,
                method=pp_opt.fill_method,
                reference_fill_values=model_fill_values,
            )
            panel_for_scoring = fill_feature_na_with_reference(
                panel_for_scoring,
                selected_factors,
                method=pp_opt.fill_method,
                reference_fill_values=model_fill_values,
            )

        bootstrap_stock_model_history(stock_model, history_df=train_df, factor_cols=selected_factors)

        timing_model, timing_source = load_timing_model(cfg.timing_model, model_paths.timing_model)
        portfolio_model, portfolio_source = load_portfolio_model(cfg.portfolio_opt, model_paths.portfolio_model)
        execution_model, execution_source = load_execution_model(cfg.execution_model, model_paths.execution_model)
        model_component_source["timing_model"] = timing_source
        model_component_source["portfolio_model"] = portfolio_source
        model_component_source["execution_model"] = execution_source
        log_progress(
            "模型加载完成："
            f"stock={stock_source}, timing={timing_source}, "
            f"portfolio={portfolio_source}, execution={execution_source}。",
            module="pipeline",
        )
    else:
        log_progress("步骤 9/13：开始训练模型（选股/择时/组合/执行）。", module="pipeline")
        stock_model = build_stock_model(cfg.stock_model)
        stock_model.fit(train_df=train_df, factor_cols=selected_factors, target_col=target_col)
        timing_model = build_timing_model(cfg.timing_model).fit(train_df)
        portfolio_model = build_portfolio_model(cfg.portfolio_opt)
        execution_model = build_execution_model(cfg.execution_model)
        model_component_source = {
            "stock_model": "trained",
            "timing_model": "trained",
            "portfolio_model": "built_from_config",
            "execution_model": "built_from_config",
        }
        log_progress("模型训练与构建完成。", module="pipeline")

    # 10) Generate predictions and model-level metrics.
    log_progress("步骤 10/13：生成测试集预测并计算模型指标。", module="pipeline")
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
    log_progress("模型指标计算完成。", module="pipeline")
    next_bar_candidates_df = pd.DataFrame()
    next_bar_summary: Dict[str, object] = {"enabled": bool(getattr(cfg.model_run, "enable_next_bar_inference", False))}

    log_progress("加载指数基准数据。", module="pipeline")
    index_benchmarks = load_index_benchmark_data(
        index_root=Path(cfg.data.index_root),
        start_date=market_bundle.start_date,
        end_date=market_bundle.end_date,
        file_format=cfg.data.file_format,
    )
    log_progress("指数基准加载完成。", module="pipeline")

    # 11) Run backtest engine (timing + portfolio + execution).
    log_progress("步骤 11/13：执行回测引擎。", module="pipeline")
    trades_df, positions_df, curve_df, bt_summary = run_backtest(
        pred_df=test_df,
        backtest_cfg=cfg.backtest,
        factor_freq=factor_freq,
        timing_model=timing_model,
        portfolio_model=portfolio_model,
        execution_model=execution_model,
        index_benchmarks=index_benchmarks,
    )
    log_progress(
        f"回测完成：trades={len(trades_df)}, positions={len(positions_df)}, curve_rows={len(curve_df)}。",
        module="pipeline",
    )
    if bool(getattr(cfg.model_run, "enable_next_bar_inference", False)):
        log_progress("执行 next bar 快速推理。", module="pipeline")
        prev_weights_for_infer: Dict[str, float] = {}
        if not positions_df.empty and "signal_ts" in positions_df.columns and "executed_weight" in positions_df.columns:
            last_signal = pd.to_datetime(positions_df["signal_ts"], errors="coerce").max()
            if pd.notna(last_signal):
                tail_pos = positions_df.loc[pd.to_datetime(positions_df["signal_ts"], errors="coerce") == last_signal].copy()
                if not tail_pos.empty:
                    prev_weights_for_infer = {
                        str(code): float(w)
                        for code, w in tail_pos[["code", "executed_weight"]].itertuples(index=False, name=None)
                        if float(w) > 1e-8
                    }
        next_bar_candidates_df, next_bar_summary = _build_next_bar_inference(
            panel_df=panel_for_scoring,
            factor_freq=factor_freq,
            factor_cols=selected_factors,
            stock_model=stock_model,
            timing_model=timing_model,
            portfolio_model=portfolio_model,
            execution_model=execution_model,
            cfg=cfg,
            prev_weights=prev_weights_for_infer,
        )
        log_progress(
            "next bar 推理完成："
            f"status={next_bar_summary.get('status', '')}, "
            f"candidate_count={int(next_bar_summary.get('candidate_count', 0))}。",
            module="pipeline",
        )

    # 12) Compute IC diagnostics for model score and raw factors.
    log_progress("步骤 12/13：计算 IC 诊断与分层分位统计。", module="pipeline")
    ic_group_col = "signal_ts" if "signal_ts" in test_df.columns else ("date" if "date" in test_df.columns else "signal_ts")
    eval_mode = str(getattr(cfg.backtest, "ic_eval_mode", "strict_horizon")).strip().lower()
    if eval_mode not in {"strict_horizon", "per_bar"}:
        eval_mode = "strict_horizon"
    eval_stride = int(cfg.backtest.rebalance_stride) if eval_mode == "strict_horizon" else 1
    eval_stride = max(eval_stride, 1)
    ic_periods_per_year = infer_periods_per_year(factor_freq=factor_freq, stride=eval_stride)

    cs_counts = pd.Series(dtype=float)
    if ic_group_col in test_df.columns and "code" in test_df.columns and not test_df.empty:
        cs_counts = test_df.groupby(ic_group_col)["code"].nunique().astype(float)
    requested_min_cs = max(int(cfg.factors.min_ic_cross_section), 2)
    max_cs = int(cs_counts.max()) if not cs_counts.empty else 0
    effective_min_cs = max(2, min(requested_min_cs, max_cs)) if max_cs >= 2 else requested_min_cs

    factor_ic_summary_df, factor_ic_series_df = compute_factor_ic_statistics(
        pred_df=test_df,
        factor_cols=selected_factors,
        ret_col="future_ret_n",
        min_cross_section=effective_min_cs,
        group_col=ic_group_col,
        periods_per_year=ic_periods_per_year,
        eval_stride=eval_stride,
        constant_as_zero=True,
    )
    model_ic_series_df = calc_ic_for_column(
        test_df,
        score_col="pred_score",
        ret_col="future_ret_n",
        min_cross_section=effective_min_cs,
        group_col=ic_group_col,
        eval_stride=eval_stride,
        constant_as_zero=True,
    )
    model_ic_summary = summarize_ic(model_ic_series_df, periods_per_year=ic_periods_per_year)
    model_ic_summary["eval_mode"] = eval_mode
    model_ic_summary["eval_stride"] = float(eval_stride)
    model_ic_summary["min_cross_section_requested"] = float(requested_min_cs)
    model_ic_summary["min_cross_section_effective"] = float(effective_min_cs)
    model_ic_summary["group_count_total"] = float(len(cs_counts))
    model_ic_summary["group_count_used"] = float(len(model_ic_series_df))
    score_spread = compute_score_spread(
        test_df,
        score_col="pred_score",
        ret_col="future_ret_n",
        quantiles=5,
        group_col=ic_group_col,
        periods_per_year=ic_periods_per_year,
        eval_stride=eval_stride,
    )
    log_progress("IC 诊断计算完成。", module="pipeline")

    # 13) Persist artifacts and summarize run.
    log_progress("步骤 13/13：写出产物文件与 summary。", module="pipeline")
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
    if market_data_health_path is not None:
        files["market_data_health_json"] = market_data_health_path
    log_progress("核心 CSV 产物已写出。", module="pipeline")

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
    next_bar_candidates_path: Path | None = None
    if bool(getattr(cfg.model_run, "enable_next_bar_inference", False)) and not next_bar_candidates_df.empty:
        next_bar_candidates_path = output_dir / f"next_bar_candidates_{run_tag}.csv"
        save_dataframe(next_bar_candidates_path, next_bar_candidates_df)
        files["next_bar_candidates_csv"] = next_bar_candidates_path

    model_files: Dict[str, Dict[str, str]] = {}
    if cfg.save_models and model_dir is not None:
        log_progress("开始保存模型文件。", module="pipeline")
        model_files["stock_model"] = stock_model.save(model_dir, run_tag)
        model_files["timing_model"] = timing_model.save(model_dir, run_tag)
        model_files["portfolio_model"] = portfolio_model.save(model_dir, run_tag)
        model_files["execution_model"] = execution_model.save(model_dir, run_tag)
        log_progress(f"模型文件保存完成：{len(model_files)} 个组件。", module="pipeline")

    top_factors = factor_ic_summary_df.head(10).copy() if not factor_ic_summary_df.empty else pd.DataFrame()
    next_bar_summary_path: Path | None = None
    if bool(getattr(cfg.model_run, "enable_next_bar_inference", False)):
        next_bar_summary_path = output_dir / f"next_bar_summary_{run_tag}.json"
        dump_json(next_bar_summary_path, next_bar_summary)
        files["next_bar_summary_json"] = next_bar_summary_path
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
            "catalog_rows": catalog_rows,
            "factor_store_hydrate": factor_store_hydrate_report,
            "factor_store_build": factor_store_build_report,
        },
        "selected_factors": selected_factors,
        "model_metrics": model_metrics,
        "backtest_metrics": bt_summary,
        "model_ic_summary": model_ic_summary,
        "model_score_spread": score_spread,
        "next_bar_inference": next_bar_summary,
        "factor_ic_top10": top_factors.to_dict(orient="records") if not top_factors.empty else [],
        "outputs": {
            **{k: str(v) for k, v in files.items()},
            "plot_main_generated": bool(plot_status.get("main", False)),
            "plot_excess_generated": bool(plot_status.get("excess", False)),
            "save_models_enabled": bool(cfg.save_models),
            "model_files": model_files,
        },
        "notes": {
            "no_future_leakage": (
                "factor features only use current/past information; labels use shifted future window;"
                "train/test split constrained by signal timestamps and target end timestamps."
            ),
            "feature_fill_method": pp_opt.fill_method,
            "feature_fill_fit_scope": "train_only",
            "feature_engineering_requested": bool(getattr(cfg.factors, "enable_factor_engineering", False)),
            "feature_engineering_enabled": bool(fe_report.get("enabled", False)),
            "feature_engineering_summary": fe_report,
            "selected_factor_count_before_fe": int(len(selected_factors_before_fe)),
            "selected_factor_count_after_fe": int(len(selected_factors)),
            "load_fe_mode": load_fe_mode if model_run_mode == "load" else "n/a",
            "factor_value_store_enabled": bool(factor_store_enabled),
            "factor_value_store_root": str(factor_store_root) if factor_store_enabled else "",
            "factor_value_store_format": str(factor_store_fmt) if factor_store_enabled else "",
            "factor_value_store_build_all": bool(getattr(cfg.factors, "factor_value_store_build_all", False)),
            "factor_value_store_build_only": bool(getattr(cfg.factors, "factor_value_store_build_only", False)),
            "factor_value_store_build_report": factor_store_build_report,
            "factor_value_store_hydrate_report": factor_store_hydrate_report,
            "timing_model_enabled": cfg.timing_model.model_type != "none",
            "portfolio_dynamic_enabled": cfg.portfolio_opt.mode == "dynamic_opt",
            "realistic_execution_enabled": cfg.execution_model.model_type == "realistic_fill",
            "model_persistence": (
                "model files are saved only when --save-models is set;"
                " otherwise no models directory/files are created."
            ),
            "catalog_factors_enabled": bool(cfg.data.auto_load_catalog_factors),
            "catalog_factor_count": int(len(catalog_entries)),
            "factor_catalog_path": str(cfg.data.factor_catalog_path) if cfg.data.factor_catalog_path else "",
            "factor_snapshot_dir": str(snapshot_paths.get("snapshot_dir", "")),
            "factor_snapshot_summary_path": str(snapshot_paths.get("summary_path", "")),
            "model_run_mode": model_run_mode,
            "model_load_sources": model_component_source,
            "ic_eval_mode": eval_mode,
            "ic_eval_stride": int(eval_stride),
            "ic_periods_per_year": float(ic_periods_per_year),
            "ic_min_cross_section_requested": int(requested_min_cs),
            "ic_min_cross_section_effective": int(effective_min_cs),
            "ic_group_count_total": int(len(cs_counts)),
            "ic_group_count_used": int(len(model_ic_series_df)),
            "next_bar_inference_enabled": bool(getattr(cfg.model_run, "enable_next_bar_inference", False)),
            "next_bar_inference_summary_path": str(next_bar_summary_path) if next_bar_summary_path else "",
            "market_data_health_path": str(market_data_health_path) if market_data_health_path else "",
            "market_data_health_summary": dict(market_data_health.get("summary", {})),
            "market_source_notes": dict(market_bundle.source_notes),
        },
    }
    summary_path = output_dir / f"summary_{run_tag}.json"
    dump_json(summary_path, summary)
    summary["outputs"]["summary_json"] = str(summary_path)
    log_progress(f"流水线结束，summary 写入：{summary_path}", module="pipeline")
    return summary
