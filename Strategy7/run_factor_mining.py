"""CLI entry for Strategy7 factor mining framework."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from strategy7.config import (
    DEFAULT_INDEX_ROOT,
    DEFAULT_UNIVERSE,
    UNIVERSE_CHOICES,
    DataConfig,
    DateConfig,
    resolve_market_data_scope,
)
from strategy7.core.constants import INTRADAY_FREQS, SUPPORTED_FREQS
from strategy7.core.utils import log_progress, set_log_level
from strategy7.data.loaders import MarketUniverseDataLoader, build_feature_bundle, load_index_benchmark_data
from strategy7.factors.base import FactorLibrary, compute_factor_panel
from strategy7.factors.defaults import (
    list_default_factor_packages,
    register_default_factors,
    resolve_default_factor_set,
)
from strategy7.factors.labeling import add_labels, validate_label_frequency_alignment
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
        help="默认因子库素材包过滤（逗号分隔；空=该频率全部默认包）",
    )
    g_date.add_argument(
        "--disable-default-factor-materials",
        action="store_true",
        help="关闭默认因子库素材注入（默认开启）",
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
            "自定义因子规格 JSON 文件路径。"
            "当 framework=custom 时必填，文件内容需为列表或 {items:[...]}。"
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
        "--catalog-path",
        type=str,
        default="auto",
        help="catalog 路径；auto 时使用 <factor-store-root>/factor_mining/factor_catalog.json",
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
    log_progress(
        f"参数校验通过：framework={args.framework}, train={train_start.date()}~{train_end.date()}, "
        f"valid={valid_start.date()}~{valid_end.date()}。",
        module="run_factor_mining",
    )

    # Step 2) 构建数据配置并加载市场数据（含训练/验证所需回看窗口）。
    data_root, universe, stock_list_path = resolve_market_data_scope(
        data_root=args.data_root,
        universe=args.universe,
        stock_list_path=getattr(args, "stock_list_path", "auto"),
    )
    data_cfg = DataConfig(
        data_root=data_root,
        universe=universe,
        stock_list_path=stock_list_path,
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

    # Step 3) 选择主频面板。
    factor_freq = str(args.factor_freq)
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
    # Step 4) 注入指数上下文素材（来自 --index-root）。
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

    # Step 5) 注入默认因子库素材（可按 --factor-packages 过滤）。
    default_material_cols: list[str] = []
    if not bool(args.disable_default_factor_materials):
        fac_lib = FactorLibrary()
        register_default_factors(fac_lib)
        avail_packages = list_default_factor_packages(factor_freq)
        if avail_packages:
            log_progress(
                f"默认因子素材包@{factor_freq}: {avail_packages}",
                module="run_factor_mining",
                level="debug",
            )
        default_material_cols = resolve_default_factor_set(
            freq=factor_freq,
            package_expr=str(args.factor_packages),
        )
        log_progress(
            f"开始注入默认因子库素材：freq={factor_freq}, selected={len(default_material_cols)}。",
            module="run_factor_mining",
        )
        before_cols = set(panel.columns)
        panel = compute_factor_panel(
            base_df=panel,
            library=fac_lib,
            freq=factor_freq,
            selected_factors=default_material_cols,
        )
        added_cols = [c for c in default_material_cols if c not in before_cols]
        log_progress(
            f"默认因子素材注入完成：added={len(added_cols)}, panel_cols={len(panel.columns)}。",
            module="run_factor_mining",
        )
    else:
        log_progress("已关闭默认因子库素材注入。", module="run_factor_mining")

    # Step 6) 生成标签并做频率一致性检查。
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

    # Step 7) 如使用 custom 框架，加载用户表达式规格。
    custom_specs = []
    if str(args.framework).lower() == "custom":
        if not str(args.custom_spec_json).strip():
            raise RuntimeError("framework=custom requires --custom-spec-json")
        log_progress(f"开始加载 custom 规格：{args.custom_spec_json}", module="run_factor_mining")
        custom_specs = load_custom_specs(args.custom_spec_json)
        if not custom_specs:
            raise RuntimeError(f"no valid custom specs loaded from: {args.custom_spec_json}")
        log_progress(f"custom 规格加载完成：{len(custom_specs)} 条。", module="run_factor_mining")

    # Step 8) 解析因子落盘路径与 catalog 路径。
    store_root = str(args.factor_store_root).strip()
    if store_root.lower() in {"", "auto"}:
        store_root = str(_infer_data_baostock_root(data_root))

    catalog_path = str(args.catalog_path).strip()
    if catalog_path.lower() in {"", "auto"}:
        catalog_path = str((Path(store_root) / "factor_mining" / "factor_catalog.json").resolve())

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
        material_factor_count=int(len(default_material_cols)),
        index_context_cols=index_ctx_cols,
    )
    log_progress("开始执行因子挖掘核心流程。", module="run_factor_mining")

    summary = run_factor_mining(
        cfg=mine_cfg,
        panel_with_label=panel,
        minute_df=market_bundle.minute5,
        custom_specs=custom_specs,
    )
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
