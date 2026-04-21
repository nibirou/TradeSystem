"""CLI entry for Strategy7 factor mining framework."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from strategy7.config import (
    DEFAULT_FUNDAMENTAL_ROOT_AK,
    DEFAULT_FUNDAMENTAL_ROOT_BSQ,
    DEFAULT_INDEX_ROOT,
    DEFAULT_TEXT_ROOT_NEWS,
    DEFAULT_TEXT_ROOT_NOTICE,
    DEFAULT_TEXT_ROOT_REPORT_EM,
    DEFAULT_TEXT_ROOT_REPORT_IWENCAI,
    DEFAULT_UNIVERSE,
    UNIVERSE_CHOICES,
    DataConfig,
    DateConfig,
    resolve_market_data_scope,
)
from strategy7.core.constants import INTRADAY_FREQS, SUPPORTED_FREQS
from strategy7.core.utils import log_progress, set_log_level
from strategy7.data.feature_engineering import FactorEngineeringOptions, apply_factor_engineering
from strategy7.data.loaders import MarketUniverseDataLoader, build_feature_bundle, load_index_benchmark_data
from strategy7.factors.base import (
    FactorLibrary,
    compute_factor_panel,
    enrich_factor_metadata_for_display,
    load_custom_factor_module,
    resolve_selected_factors,
)
from strategy7.factors.defaults import (
    build_factor_package_index,
    list_default_factor_packages,
    register_default_factors,
    resolve_primary_factor_package,
    resolve_default_factor_set,
)
from strategy7.factors.labeling import add_labels, validate_label_frequency_alignment
from strategy7.factors.reporting import export_factor_snapshot, normalize_factor_package_alias
from strategy7.mining.catalog import (
    list_catalog_factor_packages,
    load_active_catalog_entries,
    merge_catalog_factors,
    register_catalog_factors,
)
from strategy7.mining.custom import build_custom_specs_from_factor_names, load_custom_specs
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


def _build_hs300_context_features(index_hs300: pd.DataFrame) -> pd.DataFrame:
    """基于 HS300 指数构建可直接并入面板的市场上下文特征。"""
    if index_hs300.empty:
        return pd.DataFrame(columns=["date"])
    x = index_hs300.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    x["close"] = pd.to_numeric(x["close"], errors="coerce")
    x = x.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates(["date"], keep="last").reset_index(drop=True)
    if x.empty:
        return pd.DataFrame(columns=["date"])

    x["mkt_hs300_close"] = x["close"]
    x["mkt_hs300_ret_1d"] = x["close"].pct_change(1)
    x["mkt_hs300_ret_5d"] = x["close"].pct_change(5)
    x["mkt_hs300_ret_20d"] = x["close"].pct_change(20)
    x["mkt_hs300_vol_20d"] = x["mkt_hs300_ret_1d"].rolling(20, min_periods=10).std()
    x["mkt_hs300_ma_gap_20d"] = x["close"] / (x["close"].rolling(20, min_periods=10).mean() + 1e-12) - 1.0
    x["mkt_hs300_drawdown_60d"] = x["close"] / (x["close"].rolling(60, min_periods=20).max() + 1e-12) - 1.0

    keep = [
        "date",
        "mkt_hs300_close",
        "mkt_hs300_ret_1d",
        "mkt_hs300_ret_5d",
        "mkt_hs300_ret_20d",
        "mkt_hs300_vol_20d",
        "mkt_hs300_ma_gap_20d",
        "mkt_hs300_drawdown_60d",
    ]
    return x[keep].copy()


def _parse_factor_name_list(expr: str) -> List[str]:
    return [x.strip() for x in str(expr).split(",") if x.strip()]


def _split_mask_by_freq(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, *, time_col: str, freq: str) -> pd.Series:
    ts = pd.to_datetime(frame[time_col], errors="coerce")
    if str(freq).lower() in INTRADAY_FREQS:
        ts = ts.dt.normalize()
    return (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))


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


def _resolve_catalog_path(
    *,
    factor_catalog_path_arg: str,
    data_root: str,
    factor_store_root: str,
) -> str:
    factor_catalog_path_arg = str(factor_catalog_path_arg or "").strip()
    if factor_catalog_path_arg and factor_catalog_path_arg.lower() not in {"auto", "none", "null"}:
        return str(Path(factor_catalog_path_arg).expanduser().resolve())

    store_root = str(factor_store_root or "").strip()
    if store_root.lower() not in {"", "auto"}:
        return str((Path(store_root).expanduser() / "factor_mining" / "factor_catalog.json").resolve())
    root = _infer_data_baostock_root(data_root)
    return str((root / "factor_mining" / "factor_catalog.json").resolve())


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


def _resolve_default_materials_by_package(freq: str, package_expr: str) -> List[str]:
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


def _summarize_factor_packages(meta_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
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


def _print_factor_package_summary(meta_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    summary = _summarize_factor_packages(meta_df)
    print("\n=== Factor Package Summary ===")
    for title, key in [
        ("[Price-Volume / Non-Fundamental Packages]", "price_volume"),
        ("[Fundamental Packages]", "fundamental"),
        ("[Text Packages]", "text"),
        ("[Mined Packages]", "mined"),
    ]:
        print(title)
        part = summary[key]
        if part:
            print(pd.Series(part, dtype=int).sort_values(ascending=False).to_string())
        else:
            print("(none)")
        print("")
    return summary


def _export_factor_list(
    *,
    meta_df_view: pd.DataFrame,
    summary: Dict[str, Dict[str, int]],
    factor_freq: str,
    export_format: str,
    export_path_arg: str,
    default_dir: Path,
) -> Path:
    fmt = str(export_format).strip().lower()
    if fmt not in {"csv", "json", "markdown"}:
        raise ValueError(f"unsupported factor list export format: {fmt}")

    if str(export_path_arg).strip():
        out_path = Path(str(export_path_arg).strip()).expanduser().resolve()
    else:
        ts = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d_%H%M%S")
        ext = "md" if fmt == "markdown" else fmt
        out_path = (default_dir / f"factor_list_mining_{factor_freq}_{ts}.{ext}").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        meta_df_view.to_csv(out_path, index=False, encoding="utf-8")
        return out_path
    if fmt == "json":
        payload = {
            "factor_freq": str(factor_freq),
            "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            "summary": summary,
            "factors": meta_df_view.to_dict(orient="records"),
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return out_path

    # markdown
    raw_df = pd.DataFrame(
        [{"factor_package": k, "count": int(v)} for k, v in summary.get("all", {}).items()]
    )
    raw_df = raw_df.sort_values("count", ascending=False).reset_index(drop=True) if not raw_df.empty else raw_df
    md_lines: List[str] = []
    md_lines.append(f"# Factor List (Mining) @ {factor_freq}")
    md_lines.append("")
    md_lines.append("## Package Summary")
    md_lines.append("")
    if raw_df.empty:
        md_lines.append("(none)")
    else:
        md_lines.append(raw_df.to_markdown(index=False))
    md_lines.append("")
    md_lines.append("## Factor Table")
    md_lines.append("")
    if meta_df_view.empty:
        md_lines.append("(empty)")
    else:
        md_lines.append(meta_df_view.to_markdown(index=False))
    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strategy7 因子挖掘入口（基本面/分钟参数化/ML集成/GP符号遗传/自定义表达式）"
    )
    parser.add_argument(
        "--framework",
        type=str,
        choices=[
            "fundamental_multiobj",
            "minute_parametric",
            "minute_parametric_plus",
            "ml_ensemble_alpha",
            "gplearn_symbolic_alpha",
            "custom",
        ],
        default="fundamental_multiobj",
        help=(
            "挖掘框架类型："
            "fundamental_multiobj=基本面参数化+NSGA-II；"
            "minute_parametric=分钟参数化+NSGA-III；"
            "minute_parametric_plus=分钟增强参数化+NSGA-III；"
            "ml_ensemble_alpha=集成学习因子挖掘；"
            "gplearn_symbolic_alpha=基于 gplearn 的符号遗传规划挖掘；"
            "custom=用户自定义表达式。"
        ),
    )

    # =========================
    # 数据加载参数
    # =========================
    g_data = parser.add_argument_group("数据加载配置")
    g_data.add_argument(
        "--universe",
        type=str,
        choices=list(UNIVERSE_CHOICES),
        default=DEFAULT_UNIVERSE,
        help="股票池：hs300/sz50/zz500/all；默认 hs300",
    )
    g_data.add_argument(
        "--data-root",
        type=str,
        default="auto",
        help="行情数据根目录；auto 时自动使用 .../data_baostock/stock_hist/<universe>",
    )
    g_data.add_argument(
        "--stock-list-path",
        "--hs300-list-path",
        dest="stock_list_path",
        type=str,
        default="auto",
        help=(
            "可选股票列表路径（code 列），用于在 data-root 上二次过滤；"
            "auto 时 hs300/sz50/zz500 自动匹配 metadata 列表，all 默认不过滤。"
        ),
    )
    g_data.add_argument(
        "--fundamental-root-ak",
        type=str,
        default=DEFAULT_FUNDAMENTAL_ROOT_AK,
        help="AK 基本面数据根目录（financial_indicator_em / financial_indicator_sina / financial_abstract_sina）",
    )
    g_data.add_argument(
        "--fundamental-root-bsq",
        type=str,
        default=DEFAULT_FUNDAMENTAL_ROOT_BSQ,
        help="Baostock 季频财务数据根目录（balance/cash_flow/growth/profit 等）",
    )
    g_data.add_argument(
        "--fundamental-file-format",
        type=str,
        choices=["auto", "csv", "parquet"],
        default="auto",
        help="基本面文件格式：auto 自动识别，或强制 csv/parquet",
    )
    g_data.add_argument(
        "--disable-fundamental-data",
        action="store_true",
        help="禁用基本面/财务数据加载（仅使用量价数据）",
    )
    g_data.add_argument(
        "--text-root-news",
        type=str,
        default=DEFAULT_TEXT_ROOT_NEWS,
        help="金融文本-新闻数据根目录（data_em_news）",
    )
    g_data.add_argument(
        "--text-root-notice",
        type=str,
        default=DEFAULT_TEXT_ROOT_NOTICE,
        help="金融文本-公告数据根目录（data_em_notices）",
    )
    g_data.add_argument(
        "--text-root-report-em",
        type=str,
        default=DEFAULT_TEXT_ROOT_REPORT_EM,
        help="金融文本-东财研报数据根目录（data_em_reports）",
    )
    g_data.add_argument(
        "--text-root-report-iwencai",
        type=str,
        default=DEFAULT_TEXT_ROOT_REPORT_IWENCAI,
        help="金融文本-问财研报数据根目录（data_iwencai_reports）",
    )
    g_data.add_argument(
        "--text-file-format",
        type=str,
        choices=["auto", "csv", "parquet"],
        default="auto",
        help="金融文本文件格式：auto 自动识别，或强制 csv/parquet",
    )
    g_data.add_argument(
        "--disable-text-data",
        action="store_true",
        help="禁用金融文本数据加载（仅保留量价+基本面）",
    )
    g_data.add_argument(
        "--index-root",
        type=str,
        default=DEFAULT_INDEX_ROOT,
        help="指数目录（用于读取 HS300 指数并注入市场上下文特征）",
    )
    g_data.add_argument(
        "--file-format",
        type=str,
        choices=["auto", "csv", "parquet"],
        default="auto",
        help="行情文件格式：auto 自动识别，或强制 csv/parquet",
    )
    g_data.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="最多读取股票文件数量（调试提速用，默认全量）",
    )
    g_data.add_argument(
        "--main-board-only",
        action="store_true",
        help="仅使用主板股票参与挖掘",
    )

    # =========================
    # 时间与标签参数
    # =========================
    g_date = parser.add_argument_group("样本区间与标签配置")
    g_date.add_argument("--train-start", type=str, default="2021-01-01", help="挖掘训练期开始日期（含）")
    g_date.add_argument("--train-end", type=str, default="2023-12-31", help="挖掘训练期结束日期（含）")
    g_date.add_argument("--valid-start", type=str, default="2024-01-01", help="挖掘验证期开始日期（含）")
    g_date.add_argument("--valid-end", type=str, default="2024-12-31", help="挖掘验证期结束日期（含）")
    g_date.add_argument(
        "--factor-freq",
        type=str,
        choices=SUPPORTED_FREQS,
        default="D",
        help="研究主频率（标签生成、挖掘评估与因子落盘均使用该频率）",
    )
    g_date.add_argument(
        "--factor-packages",
        type=str,
        default="",
        help=(
            "默认因子库素材包过滤（逗号分隔；空=该频率全部默认包）。"
            "可选：trend,reversal,liquidity,volatility,structure,context,"
            "flow,crowding,price_action,intraday_signature,intraday_micro,period_signature,oscillator,overnight,"
            "multi_freq,bridge,multiscale,"
            "text_sentiment,text_attention,text_event,text_topic,text_fusion,"
            "fund_growth,fund_valuation,fund_profitability,fund_quality,fund_leverage,fund_cashflow,"
            "fund_efficiency,fund_expectation,fund_hf_fusion,"
            "mined_price_volume,mined_fundamental,mined_text,mined_fusion,mined_other,mined_custom,"
            "all。"
            "catalog 挖掘因子还支持维度标签：mined_fw_*, mined_universe_*, mined_freq_*, mined_materialpkg_*。"
        ),
    )
    g_date.add_argument(
        "--factor-list",
        type=str,
        default="",
        help="显式指定挖掘素材因子名列表（逗号分隔）；为空时按 --factor-packages 自动选择素材",
    )
    g_date.add_argument(
        "--custom-factor-py",
        type=str,
        default=None,
        help="自定义因子插件路径，模块需实现 register_factors(library)",
    )
    g_date.add_argument(
        "--list-factors",
        action="store_true",
        help="仅列出当前频率可用因子（默认+catalog+自定义）并退出，不加载市场数据",
    )
    g_date.add_argument(
        "--export-factor-list",
        action="store_true",
        help="在 --list-factors 模式下导出因子清单文件",
    )
    g_date.add_argument(
        "--factor-list-export-format",
        type=str,
        choices=["csv", "json", "markdown"],
        default="csv",
        help="因子清单导出格式（仅 --export-factor-list 生效）：csv/json/markdown",
    )
    g_date.add_argument(
        "--factor-list-export-path",
        type=str,
        default="",
        help="因子清单导出路径（可选）；为空时自动写入 factor_store_root/factor_mining/factor_lists/",
    )
    g_date.add_argument(
        "--disable-default-factor-materials",
        action="store_true",
        help="关闭默认因子库素材注入（默认开启）",
    )
    g_date.add_argument(
        "--enable-material-feature-engineering",
        action="store_true",
        help="是否在因子挖掘前对素材因子做特征工程筛选（训练期覆盖率+相关性去冗余，默认关闭）",
    )
    g_date.add_argument(
        "--material-fe-min-coverage",
        type=float,
        default=0.70,
        help="素材特征工程：训练期最小覆盖率阈值（0~1）",
    )
    g_date.add_argument(
        "--material-fe-min-std",
        type=float,
        default=1e-8,
        help="素材特征工程：训练期最小标准差阈值",
    )
    g_date.add_argument(
        "--material-fe-corr-threshold",
        type=float,
        default=0.95,
        help="素材特征工程：两两相关性上限（绝对值，Spearman）",
    )
    g_date.add_argument(
        "--material-fe-preselect-top-n",
        type=int,
        default=1500,
        help="素材特征工程：相关性去冗余前按质量评分预筛数量上限（0=不预筛）",
    )
    g_date.add_argument(
        "--material-fe-min-factors",
        type=int,
        default=20,
        help="素材特征工程：最终保留素材因子最小数量",
    )
    g_date.add_argument(
        "--material-fe-max-factors",
        type=int,
        default=800,
        help="素材特征工程：最终保留素材因子最大数量（0=不设上限）",
    )
    g_date.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="标签持有期长度（影响 future_ret_n 标签与评估年化换算）",
    )
    g_date.add_argument(
        "--execution-scheme",
        type=str,
        default="vwap30_vwap30",
        help="标签价格规则（与主回测执行方案保持一致）",
    )

    # =========================
    # 进化搜索参数
    # =========================
    g_evo = parser.add_argument_group("进化搜索配置")
    g_evo.add_argument("--population-size", type=int, default=128, help="每代种群规模")
    g_evo.add_argument("--generations", type=int, default=20, help="进化代数")
    g_evo.add_argument("--elite-size", type=int, default=12, help="每代精英保留数量")
    g_evo.add_argument("--mutation-rate", type=float, default=0.25, help="变异概率（0~1）")
    g_evo.add_argument("--crossover-rate", type=float, default=0.70, help="交叉概率（0~1）")
    g_evo.add_argument("--top-n", type=int, default=20, help="最终最多保留因子数（分散化筛选后）")
    g_evo.add_argument("--corr-threshold", type=float, default=0.60, help="候选因子两两相关性上限（绝对值）")
    g_evo.add_argument("--min-cross-section", type=int, default=30, help="评估时单期最小横截面样本数")
    g_evo.add_argument("--top-frac", type=float, default=0.10, help="多头分组比例（例如 0.1=前10%%）")
    g_evo.add_argument("--random-state", type=int, default=42, help="随机种子（保证可复现）")

    # =========================
    # ML 集成框架参数
    # =========================
    g_ml = parser.add_argument_group("ML 集成框架配置（framework=ml_ensemble_alpha 时）")
    g_ml.add_argument("--ml-population-size", type=int, default=48, help="ML 框架每代种群规模（建议小于基础框架）")
    g_ml.add_argument("--ml-generations", type=int, default=10, help="ML 框架进化代数")
    g_ml.add_argument(
        "--ml-model-pool",
        type=str,
        default="rf,et,hgbt",
        help="模型池（逗号分隔）：rf=随机森林，et=极端随机树，hgbt=直方图梯度提升树",
    )
    g_ml.add_argument("--ml-prefilter-topk", type=int, default=80, help="按训练期 IC 预筛后的特征数量上限")
    g_ml.add_argument("--ml-feature-min", type=int, default=10, help="单个候选模型最小特征数")
    g_ml.add_argument("--ml-feature-max", type=int, default=36, help="单个候选模型最大特征数")
    g_ml.add_argument("--ml-train-sample-frac", type=float, default=0.40, help="训练样本抽样比例（控制训练耗时）")
    g_ml.add_argument("--ml-max-train-rows", type=int, default=220000, help="训练样本最大行数上限")
    g_ml.add_argument("--ml-num-jobs", type=int, default=-1, help="并行线程数（RF/ET 生效，-1 表示尽量使用全部核）")

    # =========================
    # GP 库框架参数
    # =========================
    g_gp = parser.add_argument_group("GP-Library 框架配置（framework=gplearn_symbolic_alpha 时）")
    g_gp.add_argument("--gp-population-size", type=int, default=400, help="gplearn 每代种群规模")
    g_gp.add_argument("--gp-generations", type=int, default=12, help="gplearn 进化代数")
    g_gp.add_argument("--gp-num-runs", type=int, default=3, help="独立随机种子运行次数（结果合并）")
    g_gp.add_argument("--gp-n-components", type=int, default=24, help="每次运行输出候选表达式数量")
    g_gp.add_argument("--gp-hall-of-fame", type=int, default=64, help="每次运行保留精英表达式数量")
    g_gp.add_argument("--gp-tournament-size", type=int, default=20, help="锦标赛选择规模")
    g_gp.add_argument("--gp-parsimony", type=float, default=0.001, help="复杂度惩罚系数（越大越偏好简单表达式）")
    g_gp.add_argument(
        "--gp-metric",
        type=str,
        choices=["spearman", "pearson"],
        default="spearman",
        help="gplearn 内部适应度指标",
    )
    g_gp.add_argument(
        "--gp-function-set",
        type=str,
        default="add,sub,mul,div,sqrt,log,abs,neg,max,min",
        help="函数集合（逗号分隔，支持 add/sub/mul/div/sqrt/log/abs/neg/inv/max/min）",
    )
    g_gp.add_argument("--gp-prefilter-topk", type=int, default=80, help="训练期 IC 预筛后保留特征数")
    g_gp.add_argument("--gp-train-sample-frac", type=float, default=0.40, help="gplearn 训练样本抽样比例")
    g_gp.add_argument("--gp-max-train-rows", type=int, default=220000, help="gplearn 训练样本最大行数")
    g_gp.add_argument("--gp-max-depth", type=int, default=5, help="表达式初始最大深度")
    g_gp.add_argument("--gp-max-samples", type=float, default=0.90, help="每棵树可见样本比例")
    g_gp.add_argument("--gp-num-jobs", type=int, default=-1, help="并行线程数（-1 表示尽量使用全部核）")

    # =========================
    # 自定义表达式参数
    # =========================
    g_custom = parser.add_argument_group("自定义因子配置（framework=custom 时）")
    g_custom.add_argument(
        "--custom-spec-json",
        type=str,
        default="",
        help=(
            "自定义因子规格 JSON 文件路径（兼容模式，建议优先使用 --factor-list + --custom-factor-py）。"
            "当提供该参数时，framework=custom 会按 JSON 表达式评估；为空时将评估 --factor-list 指定的因子。"
        ),
    )

    # =========================
    # 因子落盘与入库参数
    # =========================
    g_out = parser.add_argument_group("因子落盘与 catalog 配置")
    g_out.add_argument(
        "--factor-store-root",
        type=str,
        default="auto",
        help="因子输出根目录；auto 时自动推断为 data_baostock 根目录",
    )
    g_out.add_argument(
        "--factor-catalog-path",
        "--catalog-path",
        dest="factor_catalog_path",
        type=str,
        default="auto",
        help=(
            "因子挖掘 catalog 路径（输入加载 + 输出更新）。"
            "auto 时使用 <factor-store-root>/factor_mining/factor_catalog.json。"
        ),
    )
    g_out.add_argument(
        "--disable-catalog-factors",
        action="store_true",
        help="禁用 catalog 因子加载（不影响本次挖掘结果写入 catalog）",
    )
    g_out.add_argument(
        "--save-format",
        type=str,
        choices=["parquet", "csv"],
        default="parquet",
        help="因子值表输出格式（parquet 优先，便于大数据读写）",
    )

    # =========================
    # 准入阈值覆盖参数
    # =========================
    g_adm = parser.add_argument_group("准入阈值覆盖（可选）")
    g_adm.add_argument("--min-abs-ic-mean", type=float, default=None, help="覆盖默认阈值：|RankIC|均值下限")
    g_adm.add_argument("--min-ic-win-rate", type=float, default=None, help="覆盖默认阈值：IC 胜率下限")
    g_adm.add_argument("--min-ic-ir", type=float, default=None, help="覆盖默认阈值：ICIR 下限")
    g_adm.add_argument(
        "--min-long-excess-annualized",
        type=float,
        default=None,
        help="覆盖默认阈值：多头相对全市场年化超额下限",
    )
    g_adm.add_argument("--min-long-sharpe", type=float, default=None, help="覆盖默认阈值：多头夏普下限")
    g_adm.add_argument("--min-long-win-rate", type=float, default=None, help="覆盖默认阈值：多头胜率下限")
    g_adm.add_argument("--min-coverage", type=float, default=None, help="覆盖默认阈值：因子覆盖率下限")

    g_out.add_argument(
        "--log-level",
        type=str,
        choices=["quiet", "normal", "verbose"],
        default="normal",
        help="运行过程日志级别：quiet / normal / verbose",
    )
    g_out.add_argument(
        "--quiet",
        action="store_true",
        help="等价于 --log-level quiet（优先级最高）",
    )
    g_out.add_argument(
        "--verbose",
        action="store_true",
        help="等价于 --log-level verbose（当未开启 --quiet 时生效）",
    )

    return parser.parse_args()


def main() -> None:
    # Step 1) 参数解析与日志级别初始化。
    args = _parse_args()
    effective_log_level = set_log_level(level=str(args.log_level), quiet=bool(args.quiet), verbose=bool(args.verbose))
    log_progress(f"日志级别：{effective_log_level}", module="run_factor_mining")
    log_progress("开始解析与校验挖掘参数。", module="run_factor_mining")

    train_start = _parse_date(args.train_start)
    train_end = _parse_date(args.train_end)
    valid_start = _parse_date(args.valid_start)
    valid_end = _parse_date(args.valid_end)

    if not (train_start <= train_end < valid_start <= valid_end):
        raise ValueError("date constraint: train_start <= train_end < valid_start <= valid_end")
    if int(args.ml_population_size) <= 0 or int(args.ml_generations) <= 0:
        raise ValueError("ml_population_size and ml_generations must be positive")
    if int(args.ml_feature_min) <= 0 or int(args.ml_feature_max) <= 0:
        raise ValueError("ml_feature_min and ml_feature_max must be positive")
    if int(args.ml_feature_min) > int(args.ml_feature_max):
        raise ValueError("ml_feature_min must be <= ml_feature_max")
    if not (0.0 < float(args.ml_train_sample_frac) <= 1.0):
        raise ValueError("ml_train_sample_frac must be in (0,1]")
    if int(args.ml_max_train_rows) < 2000:
        raise ValueError("ml_max_train_rows must be >= 2000")
    if int(args.gp_population_size) <= 0 or int(args.gp_generations) <= 0:
        raise ValueError("gp_population_size and gp_generations must be positive")
    if int(args.gp_num_runs) <= 0:
        raise ValueError("gp_num_runs must be positive")
    if int(args.gp_n_components) <= 0 or int(args.gp_hall_of_fame) <= 0:
        raise ValueError("gp_n_components and gp_hall_of_fame must be positive")
    if int(args.gp_tournament_size) <= 1:
        raise ValueError("gp_tournament_size must be > 1")
    if int(args.gp_prefilter_topk) <= 0:
        raise ValueError("gp_prefilter_topk must be positive")
    if float(args.gp_parsimony) < 0.0:
        raise ValueError("gp_parsimony must be >= 0")
    if int(args.gp_num_jobs) == 0:
        raise ValueError("gp_num_jobs cannot be 0; use -1 or a positive integer")
    if not (0.0 < float(args.gp_train_sample_frac) <= 1.0):
        raise ValueError("gp_train_sample_frac must be in (0,1]")
    if int(args.gp_max_train_rows) < 3000:
        raise ValueError("gp_max_train_rows must be >= 3000")
    if int(args.gp_max_depth) < 2:
        raise ValueError("gp_max_depth must be >= 2")
    if not (0.0 < float(args.gp_max_samples) <= 1.0):
        raise ValueError("gp_max_samples must be in (0,1]")
    if args.max_files is not None and int(args.max_files) <= 0:
        raise ValueError("max_files must be positive when provided")
    if not (0.0 <= float(args.material_fe_min_coverage) <= 1.0):
        raise ValueError("material_fe_min_coverage must be in [0,1]")
    if float(args.material_fe_min_std) < 0.0:
        raise ValueError("material_fe_min_std must be non-negative")
    if not (0.0 <= float(args.material_fe_corr_threshold) < 1.0):
        raise ValueError("material_fe_corr_threshold must be in [0,1)")
    if int(args.material_fe_preselect_top_n) < 0:
        raise ValueError("material_fe_preselect_top_n must be non-negative")
    if int(args.material_fe_min_factors) <= 0:
        raise ValueError("material_fe_min_factors must be positive")
    if int(args.material_fe_max_factors) < 0:
        raise ValueError("material_fe_max_factors must be non-negative")
    if int(args.material_fe_max_factors) > 0 and int(args.material_fe_max_factors) < int(args.material_fe_min_factors):
        raise ValueError("material_fe_max_factors must be >= material_fe_min_factors when positive")
    log_progress(
        f"参数校验通过：framework={args.framework}, train={train_start.date()}~{train_end.date()}, "
        f"valid={valid_start.date()}~{valid_end.date()}。",
        module="run_factor_mining",
    )

    # Step 2) 解析数据范围与 catalog 路径。
    data_root, universe, stock_list_path = resolve_market_data_scope(
        data_root=args.data_root,
        universe=args.universe,
        stock_list_path=getattr(args, "stock_list_path", "auto"),
    )
    store_root = str(args.factor_store_root).strip()
    if store_root.lower() in {"", "auto"}:
        store_root = str(_infer_data_baostock_root(data_root))
    catalog_path = _resolve_catalog_path(
        factor_catalog_path_arg=str(getattr(args, "factor_catalog_path", "")),
        data_root=data_root,
        factor_store_root=store_root,
    )
    factor_freq = str(args.factor_freq)
    explicit_factor_names = _parse_factor_name_list(str(args.factor_list))
    default_pkg_tokens, catalog_pkg_expr = _split_package_tokens_for_default_and_catalog(
        factor_freq,
        str(args.factor_packages),
    )

    # Fast path: --list-factors 不加载行情，仅输出清单。
    if bool(args.list_factors):
        log_progress("进入仅列出因子模式（run_factor_mining）。", module="run_factor_mining")
        fac_lib = FactorLibrary()
        register_default_factors(fac_lib)

        catalog_entries_for_list: List[Dict[str, object]] = []
        catalog_pkg_for_list = "all" if not str(args.factor_packages).strip() else catalog_pkg_expr
        should_load_catalog_list = not bool(args.disable_catalog_factors)
        if should_load_catalog_list and str(args.factor_packages).strip() and not catalog_pkg_expr and not explicit_factor_names:
            should_load_catalog_list = False
        if should_load_catalog_list:
            catalog_entries_for_list = load_active_catalog_entries(
                catalog_path=catalog_path,
                freq=factor_freq,
                factor_names=explicit_factor_names or None,
                package_expr=catalog_pkg_for_list,
            )
            if catalog_entries_for_list:
                register_catalog_factors(fac_lib, catalog_entries_for_list)
        if args.custom_factor_py:
            load_custom_factor_module(fac_lib, str(args.custom_factor_py))

        avail_default_packages = list_default_factor_packages(factor_freq)
        avail_catalog_packages = list_catalog_factor_packages(catalog_path=catalog_path, freq=factor_freq)
        if avail_default_packages:
            log_progress(
                f"default factor packages@{factor_freq}: {avail_default_packages}",
                module="run_factor_mining",
                level="debug",
            )
        if avail_catalog_packages:
            log_progress(
                f"catalog factor packages@{factor_freq}: {avail_catalog_packages}",
                module="run_factor_mining",
                level="debug",
            )

        meta_df = fac_lib.metadata(freq=factor_freq)
        meta_df = _attach_factor_package_columns(meta_df, freq=factor_freq)
        meta_df_view = enrich_factor_metadata_for_display(meta_df)
        if "category" in meta_df_view.columns:
            meta_df_view = meta_df_view.drop(columns=["category"])
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

        package_summary = _print_factor_package_summary(meta_df)
        snapshot_default_set = _resolve_default_materials_by_package(
            freq=factor_freq,
            package_expr=str(args.factor_packages),
        )
        snapshot_catalog_set: List[str] = []
        if not explicit_factor_names:
            if not str(args.factor_packages).strip():
                snapshot_catalog_set = [
                    str(e.get("name", "")).strip() for e in catalog_entries_for_list if str(e.get("name", "")).strip()
                ]
            elif catalog_pkg_expr:
                snapshot_catalog_set = [
                    str(e.get("name", "")).strip() for e in catalog_entries_for_list if str(e.get("name", "")).strip()
                ]
        snapshot_default_set = sorted(set(snapshot_default_set) | set(snapshot_catalog_set))
        selected_for_snapshot = _resolve_used_factors_for_snapshot(
            library=fac_lib,
            freq=factor_freq,
            factor_list_arg=str(args.factor_list),
            default_set=snapshot_default_set,
        )
        snapshot_paths = export_factor_snapshot(
            meta_df_view=meta_df_view,
            used_factors=selected_for_snapshot,
            entrypoint="run_factor_mining",
            factor_freq=factor_freq,
            output_root=Path(__file__).resolve().parent / "outputs" / "factor_snapshots",
            run_tag=f"list_factors_{str(args.framework)}",
            extra_summary={
                "mode": "list_factors",
                "catalog_factor_count": int(len(catalog_entries_for_list)),
                "selected_factor_count": int(len(selected_for_snapshot)),
            },
        )
        log_progress(
            f"自动因子快照导出完成：snapshot_dir={snapshot_paths.get('snapshot_dir', '')}",
            module="run_factor_mining",
        )
        pv_total = int(sum(package_summary["price_volume"].values()))
        fund_total = int(sum(package_summary["fundamental"].values()))
        text_total = int(sum(package_summary["text"].values()))
        mined_total = int(sum(package_summary["mined"].values()))
        export_path: Path | None = None
        if bool(args.export_factor_list):
            export_path = _export_factor_list(
                meta_df_view=meta_df_view,
                summary=package_summary,
                factor_freq=factor_freq,
                export_format=str(args.factor_list_export_format),
                export_path_arg=str(args.factor_list_export_path),
                default_dir=Path(__file__).resolve().parent / "outputs" / "factor_lists",
            )
            log_progress(
                f"因子清单导出完成：format={args.factor_list_export_format}, path={export_path}",
                module="run_factor_mining",
            )
        log_progress(
            (
                f"因子清单输出完成：freq={factor_freq}, total_factor_count={len(meta_df)}, "
                f"price_volume_factor_count={pv_total}, "
                f"fundamental_factor_count={fund_total}, "
                f"text_factor_count={text_total}, "
                f"mined_factor_count={mined_total}, "
                f"factor_package_count={len(package_summary['all'])}, "
                f"catalog_factor_count={len(catalog_entries_for_list)}。"
            ),
            module="run_factor_mining",
        )
        if export_path is not None:
            print(f"factor_list_export: {export_path}")
        print(f"factor_snapshot_dir: {snapshot_paths.get('snapshot_dir', '')}")
        return

    # Step 3) 构建数据配置并加载市场数据（含训练/验证所需回看窗口）。
    data_cfg = DataConfig(
        data_root=data_root,
        universe=universe,
        stock_list_path=stock_list_path,
        fundamental_root_ak=str(Path(args.fundamental_root_ak).expanduser()),
        fundamental_root_bsq=str(Path(args.fundamental_root_bsq).expanduser()),
        fundamental_file_format=str(args.fundamental_file_format),
        enable_fundamental_data=not bool(args.disable_fundamental_data),
        text_root_news=str(Path(args.text_root_news).expanduser()),
        text_root_notice=str(Path(args.text_root_notice).expanduser()),
        text_root_report_em=str(Path(args.text_root_report_em).expanduser()),
        text_root_report_iwencai=str(Path(args.text_root_report_iwencai).expanduser()),
        text_file_format=str(args.text_file_format),
        enable_text_data=not bool(args.disable_text_data),
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

    loader = MarketUniverseDataLoader(
        data_cfg=data_cfg,
        date_cfg=date_cfg,
        lookback_days=160,
        horizon=int(args.horizon),
        factor_freq=str(args.factor_freq),
    )
    log_progress(
        f"开始加载市场数据（日线+5分钟）：universe={universe}, stock_list_path={stock_list_path or 'None'}。",
        module="run_factor_mining",
    )
    market_bundle = loader.load()
    log_progress(
        f"市场数据加载完成：daily_rows={len(market_bundle.daily)}, minute_rows={len(market_bundle.minute5)}, "
        f"codes={len(market_bundle.codes)}。",
        module="run_factor_mining",
    )
    feat_bundle = build_feature_bundle(market_bundle)
    log_progress(
        f"特征构建完成，可用频率={sorted(feat_bundle.by_freq.keys())}。",
        module="run_factor_mining",
    )

    # Step 4) 选择主频面板。
    if factor_freq not in feat_bundle.by_freq:
        raise RuntimeError(f"feature bundle missing factor_freq={factor_freq}")
    panel = feat_bundle.by_freq[factor_freq].copy()
    if panel.empty:
        raise RuntimeError(f"base feature panel is empty for factor_freq={factor_freq}")
    time_col = "datetime" if factor_freq in INTRADAY_FREQS else "date"
    log_progress(
        f"选择研究主频率面板：freq={factor_freq}, rows={len(panel)}, time_col={time_col}。",
        module="run_factor_mining",
    )

    # 使用 --index-root 读取指数数据并构建市场上下文素材，确保该配置项在挖掘流程中实际生效。
    # Step 5) 注入指数上下文素材（来自 --index-root）。
    index_ctx_cols: list[str] = []
    try:
        idx_map = load_index_benchmark_data(
            index_root=Path(args.index_root).expanduser(),
            start_date=pd.Timestamp(market_bundle.start_date),
            end_date=pd.Timestamp(market_bundle.end_date),
            file_format=str(args.file_format),
        )
        hs300_ctx = _build_hs300_context_features(idx_map.get("hs300", pd.DataFrame()))
        if not hs300_ctx.empty:
            panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
            before_cols = set(panel.columns)
            panel = panel.merge(hs300_ctx, on="date", how="left")
            index_ctx_cols = [c for c in hs300_ctx.columns if c != "date" and c not in before_cols]
            log_progress(
                f"指数上下文合并完成：source=index_root, added={len(index_ctx_cols)}。",
                module="run_factor_mining",
            )
        else:
            log_progress("指数上下文为空：未找到可用 HS300 指数数据，跳过合并。", module="run_factor_mining", level="debug")
    except Exception as exc:
        log_progress(f"指数上下文加载失败，已跳过：{exc}", module="run_factor_mining", level="debug")

    # Step 6) 构建挖掘素材因子库（默认+catalog+自定义）并注入素材列。
    fac_lib = FactorLibrary()
    register_default_factors(fac_lib)

    catalog_rows: Dict[str, int] = {}
    catalog_entries: List[Dict[str, object]] = []
    should_load_catalog = not bool(args.disable_catalog_factors)
    if should_load_catalog and str(args.factor_packages).strip() and not catalog_pkg_expr and not explicit_factor_names:
        should_load_catalog = False
    if should_load_catalog:
        catalog_pkg_for_material = "all" if not str(args.factor_packages).strip() else catalog_pkg_expr
        panel, catalog_rows, catalog_entries = merge_catalog_factors(
            base_panel=panel,
            catalog_path=catalog_path,
            factor_freq=factor_freq,
            factor_names=explicit_factor_names or None,
            package_expr=catalog_pkg_for_material,
        )
        if catalog_entries:
            register_catalog_factors(fac_lib, catalog_entries)
    if args.custom_factor_py:
        load_custom_factor_module(fac_lib, str(args.custom_factor_py))

    default_material_cols: List[str] = []
    if not bool(args.disable_default_factor_materials):
        default_material_cols = _resolve_default_materials_by_package(
            freq=factor_freq,
            package_expr=str(args.factor_packages),
        )

    catalog_material_cols: List[str] = []
    if not explicit_factor_names:
        if not str(args.factor_packages).strip():
            catalog_material_cols = [str(e.get("name", "")).strip() for e in catalog_entries if str(e.get("name", "")).strip()]
        elif catalog_pkg_expr:
            catalog_material_cols = [str(e.get("name", "")).strip() for e in catalog_entries if str(e.get("name", "")).strip()]

    default_set = sorted(set(default_material_cols) | set(catalog_material_cols))
    selected_material_cols = resolve_selected_factors(
        library=fac_lib,
        freq=factor_freq,
        factor_list_arg=str(args.factor_list),
        default_set=default_set,
    )
    if selected_material_cols:
        log_progress(
            f"开始注入挖掘素材因子：freq={factor_freq}, selected={len(selected_material_cols)}。",
            module="run_factor_mining",
        )
        before_cols = set(panel.columns)
        panel = compute_factor_panel(
            base_df=panel,
            library=fac_lib,
            freq=factor_freq,
            selected_factors=selected_material_cols,
        )
        added_cols = [c for c in selected_material_cols if c not in before_cols]
        log_progress(
            f"挖掘素材因子注入完成：added={len(added_cols)}, panel_cols={len(panel.columns)}, "
            f"catalog_entries={len(catalog_entries)}, catalog_rows_meta={catalog_rows}。",
            module="run_factor_mining",
        )
    else:
        log_progress("未选择任何显式素材因子，继续使用原始特征面板挖掘。", module="run_factor_mining")

    # Step 7) 生成标签并做频率一致性检查。
    log_progress(f"开始生成挖掘标签 future_ret_n（freq={factor_freq}）。", module="run_factor_mining")
    panel = add_labels(
        panel=panel,
        horizon=int(args.horizon),
        execution_scheme=str(args.execution_scheme),
        price_table_daily=feat_bundle.price_table_daily,
        factor_freq=factor_freq,
    )
    label_align = validate_label_frequency_alignment(
        panel=panel,
        factor_freq=factor_freq,
        horizon=int(args.horizon),
        strict=True,
    )
    log_progress(
        (
            f"标签生成完成：panel_rows={len(panel)}, time_col={time_col}, "
            f"bad_entry_shift={int(label_align.get('bad_entry_shift', 0))}, "
            f"bad_exit_shift={int(label_align.get('bad_exit_shift', 0))}。"
        ),
        module="run_factor_mining",
    )

    selected_material_cols_before_fe = list(selected_material_cols)
    material_fe_report: Dict[str, object] = {
        "enabled": False,
        "input_factor_count": int(len(selected_material_cols_before_fe)),
        "final_factor_count": int(len(selected_material_cols_before_fe)),
        "orth_method": "none",
    }
    if bool(args.enable_material_feature_engineering) and selected_material_cols:
        train_mask_fe = _split_mask_by_freq(
            panel,
            train_start,
            train_end,
            time_col=time_col,
            freq=factor_freq,
        )
        valid_mask_fe = _split_mask_by_freq(
            panel,
            valid_start,
            valid_end,
            time_col=time_col,
            freq=factor_freq,
        )
        if int(train_mask_fe.sum()) > 0 and int(valid_mask_fe.sum()) > 0:
            fe_opts = FactorEngineeringOptions(
                enabled=True,
                min_coverage=float(args.material_fe_min_coverage),
                min_std=float(args.material_fe_min_std),
                corr_threshold=float(args.material_fe_corr_threshold),
                preselect_top_n=int(args.material_fe_preselect_top_n),
                min_factors=int(args.material_fe_min_factors),
                max_factors=int(args.material_fe_max_factors),
                orth_method="none",
                pca_variance_ratio=0.95,
                pca_max_components=128,
            )
            _, _, selected_after_fe, material_fe_report = apply_factor_engineering(
                train_df=panel.loc[train_mask_fe].copy(),
                test_df=panel.loc[valid_mask_fe].copy(),
                factor_cols=selected_material_cols,
                options=fe_opts,
                target_col="future_ret_n",
                raw_train_df=panel.loc[train_mask_fe].copy(),
            )
            if selected_after_fe:
                removed = sorted(set(selected_material_cols) - set(selected_after_fe))
                if removed:
                    panel = panel.drop(columns=[c for c in removed if c in panel.columns], errors="ignore")
                selected_material_cols = list(selected_after_fe)
                log_progress(
                    "素材特征工程完成："
                    f"input={int(material_fe_report.get('input_factor_count', len(selected_material_cols_before_fe)))}, "
                    f"final={int(material_fe_report.get('final_factor_count', len(selected_material_cols)))}, "
                    f"removed={len(removed)}。",
                    module="run_factor_mining",
                )
            else:
                log_progress(
                    "素材特征工程未保留任何因子，已自动回退到原始素材清单。",
                    module="run_factor_mining",
                )
                selected_material_cols = list(selected_material_cols_before_fe)
        else:
            log_progress(
                "素材特征工程跳过：训练/验证样本不足（无法拟合筛选规则）。",
                module="run_factor_mining",
                level="debug",
            )

    material_meta_df = fac_lib.metadata(freq=factor_freq)
    material_meta_df = _attach_factor_package_columns(material_meta_df, freq=factor_freq)
    material_meta_view = enrich_factor_metadata_for_display(material_meta_df)
    if "category" in material_meta_view.columns:
        material_meta_view = material_meta_view.drop(columns=["category"])
    material_snapshot = export_factor_snapshot(
        meta_df_view=material_meta_view,
        used_factors=selected_material_cols,
        entrypoint="run_factor_mining",
        factor_freq=factor_freq,
        output_root=Path(__file__).resolve().parent / "outputs" / "factor_snapshots",
        run_tag=f"run_{str(args.framework)}",
        extra_summary={
            "mode": "run",
            "framework": str(args.framework),
            "catalog_factor_count": int(len(catalog_entries)),
            "selected_factor_count_before_material_fe": int(len(selected_material_cols_before_fe)),
            "selected_factor_count_after_material_fe": int(len(selected_material_cols)),
            "material_feature_engineering": material_fe_report,
        },
    )
    log_progress(
        f"自动因子快照导出完成：snapshot_dir={material_snapshot.get('snapshot_dir', '')}",
        module="run_factor_mining",
    )

    # Step 8) 如使用 custom 框架，加载用户表达式规格。
    custom_specs = []
    if str(args.framework).lower() == "custom":
        custom_spec_json = str(args.custom_spec_json).strip()
        if custom_spec_json:
            log_progress(f"开始加载 custom 规格（兼容模式）：{custom_spec_json}", module="run_factor_mining")
            custom_specs = load_custom_specs(custom_spec_json)
            if not custom_specs:
                raise RuntimeError(f"no valid custom specs loaded from: {custom_spec_json}")
            log_progress(f"custom 规格加载完成：{len(custom_specs)} 条。", module="run_factor_mining")
        else:
            factor_names_for_custom = explicit_factor_names if explicit_factor_names else list(selected_material_cols)
            if not factor_names_for_custom:
                raise RuntimeError(
                    "framework=custom requires --factor-list (or provide --custom-spec-json in compatibility mode)"
                )
            custom_specs = build_custom_specs_from_factor_names(
                factor_names_for_custom,
                freq=str(args.factor_freq),
                category="mined_custom",
                name_prefix="custom_eval",
            )
            log_progress(
                f"custom 逐因子评估模式：source_factor_count={len(factor_names_for_custom)}, "
                f"generated_spec_count={len(custom_specs)}。",
                module="run_factor_mining",
            )

    # Step 9) 汇总挖掘配置并执行核心 runner。
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
        ml_population_size=int(args.ml_population_size),
        ml_generations=int(args.ml_generations),
        ml_model_pool=str(args.ml_model_pool),
        ml_prefilter_topk=int(args.ml_prefilter_topk),
        ml_feature_min=int(args.ml_feature_min),
        ml_feature_max=int(args.ml_feature_max),
        ml_train_sample_frac=float(args.ml_train_sample_frac),
        ml_max_train_rows=int(args.ml_max_train_rows),
        ml_num_jobs=int(args.ml_num_jobs),
        gp_population_size=int(args.gp_population_size),
        gp_generations=int(args.gp_generations),
        gp_num_runs=int(args.gp_num_runs),
        gp_n_components=int(args.gp_n_components),
        gp_hall_of_fame=int(args.gp_hall_of_fame),
        gp_tournament_size=int(args.gp_tournament_size),
        gp_parsimony=float(args.gp_parsimony),
        gp_metric=str(args.gp_metric),
        gp_function_set=str(args.gp_function_set),
        gp_prefilter_topk=int(args.gp_prefilter_topk),
        gp_train_sample_frac=float(args.gp_train_sample_frac),
        gp_max_train_rows=int(args.gp_max_train_rows),
        gp_max_depth=int(args.gp_max_depth),
        gp_max_samples=float(args.gp_max_samples),
        gp_num_jobs=int(args.gp_num_jobs),
        include_default_factor_materials=not bool(args.disable_default_factor_materials),
        factor_packages=str(args.factor_packages),
        factor_list=str(args.factor_list),
        material_factor_count=int(len(selected_material_cols)),
        material_factor_names=[str(x) for x in selected_material_cols],
        enable_material_feature_engineering=bool(args.enable_material_feature_engineering),
        material_fe_min_coverage=float(args.material_fe_min_coverage),
        material_fe_min_std=float(args.material_fe_min_std),
        material_fe_corr_threshold=float(args.material_fe_corr_threshold),
        material_fe_preselect_top_n=int(args.material_fe_preselect_top_n),
        material_fe_min_factors=int(args.material_fe_min_factors),
        material_fe_max_factors=int(args.material_fe_max_factors),
        universe=str(universe),
        index_context_cols=index_ctx_cols,
    )
    log_progress("开始执行因子挖掘核心流程。", module="run_factor_mining")

    summary = run_factor_mining(
        cfg=mine_cfg,
        panel_with_label=panel,
        minute_df=market_bundle.minute5,
        custom_specs=custom_specs,
    )
    summary["selected_material_factor_count_before_fe"] = int(len(selected_material_cols_before_fe))
    summary["selected_material_factor_count_after_fe"] = int(len(selected_material_cols))
    summary["material_feature_engineering"] = material_fe_report
    summary["factor_snapshot_dir"] = str(material_snapshot.get("snapshot_dir", ""))
    summary["factor_snapshot_summary_path"] = str(material_snapshot.get("summary_path", ""))
    log_progress(
        f"挖掘完成：candidate_count={summary.get('candidate_count', 0)}, "
        f"selected_count={summary.get('selected_count', 0)}, "
        f"factor_table={summary.get('factor_table_path', '')}",
        module="run_factor_mining",
    )

    # Step 10) 输出摘要（含候选数、入选数、落盘路径）。
    print("=== Factor Mining Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
