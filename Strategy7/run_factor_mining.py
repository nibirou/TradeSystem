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
    parser = argparse.ArgumentParser(
        description="Strategy7 因子挖掘入口（基本面多目标 / 分钟参数化 / 自定义表达式）"
    )
    parser.add_argument(
        "--framework",
        type=str,
        choices=["fundamental_multiobj", "minute_parametric", "custom"],
        default="fundamental_multiobj",
        help=(
            "挖掘框架类型："
            "fundamental_multiobj=基本面参数化+NSGA-II；"
            "minute_parametric=分钟参数化+NSGA-III；"
            "custom=用户自定义表达式。"
        ),
    )

    # =========================
    # 数据加载参数
    # =========================
    g_data = parser.add_argument_group("数据加载配置")
    g_data.add_argument(
        "--data-root",
        type=str,
        default=r"D:/PythonProject/Quant/data_baostock/stock_hist/hs300",
        help="行情数据根目录（用于构建日线/分钟特征与标签）",
    )
    g_data.add_argument(
        "--hs300-list-path",
        type=str,
        default=r"D:/PythonProject/Quant/data_baostock/metadata/stock_list_hs300.csv",
        help="HS300 成分股文件路径",
    )
    g_data.add_argument(
        "--index-root",
        type=str,
        default=r"D:/PythonProject/Quant/data_baostock/ak_index",
        help="指数目录（该脚本中主要保持与主框架配置一致）",
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
        default="D",
        help="因子频率标签（写入 catalog 时使用，默认 D）",
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
