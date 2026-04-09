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
    factor_catalog_path: str | None
    auto_load_catalog_factors: bool


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
    dafat_seq_len: int
    dafat_hidden_size: int
    dafat_num_layers: int
    dafat_num_heads: int
    dafat_ffn_mult: int
    dafat_dropout: float
    dafat_local_window: int
    dafat_topk_ratio: float
    dafat_vol_quantile: float
    dafat_meso_scale: int
    dafat_macro_scale: int
    dafat_epochs: int
    dafat_lr: float
    dafat_weight_decay: float
    dafat_early_stop: int
    dafat_per_epoch_batch: int
    dafat_batch_size: int
    dafat_label_transform: str
    dafat_mse_weight: float
    dafat_use_dpe: bool
    dafat_use_sparse_attn: bool
    dafat_use_multiscale: bool
    dafat_device: str


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
    ic_eval_mode: str


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


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _infer_data_baostock_root(data_root: str) -> Path:
    p = Path(data_root).expanduser()
    for cand in [p, *p.parents]:
        if cand.name.lower() == "data_baostock":
            return cand
    # common shape: .../data_baostock/stock_hist/hs300
    if p.name.lower() in {"hs300", "all", "zz500", "zz1000", "sz50"} and p.parent.name.lower() == "stock_hist":
        return p.parent.parent
    if p.name.lower() == "stock_hist":
        return p.parent
    return p


def _resolve_factor_catalog_path(raw_value: str | None, data_root: str) -> str | None:
    v = str(raw_value).strip() if raw_value is not None else "auto"
    if v.lower() in {"", "none", "null"}:
        return None
    if v.lower() != "auto":
        return _resolve_path(v)

    base = _infer_data_baostock_root(data_root)
    return str((base / "factor_mining" / "factor_catalog.json").resolve())


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
    """构建 Strategy7 命令行参数。

    说明：
    1. 本函数只负责“参数定义与帮助文案”，具体取值合法性在 `build_run_config` 校验。
    2. 参数字段与 `DataConfig/FactorConfig/...` 数据类基本一一对应，便于追踪配置流。
    3. 布尔参数使用 `_parse_bool`，避免 argparse 中 `bool("false")==True` 的常见陷阱。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Strategy7：模块化量化研究引擎（数据/因子/模型/回测可插拔）"
        )
    )
    # =========================
    # 数据与外部因子源参数
    # =========================
    g_data = parser.add_argument_group("数据与因子源配置")
    g_data.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="行情数据根目录（通常为 .../data_baostock/stock_hist/hs300）",
    )
    g_data.add_argument(
        "--hs300-list-path",
        type=str,
        default=DEFAULT_HS300_LIST_PATH,
        help="HS300 成分股文件路径，用于限制股票池",
    )
    g_data.add_argument(
        "--index-root",
        type=str,
        default=DEFAULT_INDEX_ROOT,
        help="指数行情目录（用于 HS300/ZZ500/ZZ1000 基准对比）",
    )
    g_data.add_argument(
        "--file-format",
        type=str,
        choices=["auto", "csv", "parquet"],
        default="auto",
        help="行情文件格式：auto 自动识别；csv/parquet 强制指定",
    )
    g_data.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="最多读取多少只股票文件（调试加速用，默认全量）",
    )
    g_data.add_argument(
        "--main-board-only",
        action="store_true",
        help="仅使用主板股票（过滤创业板/科创板等）",
    )
    g_data.add_argument(
        "--extra-factor-paths",
        type=str,
        default="",
        help="额外因子表路径，多个用逗号分隔；会按 [date,code] 左连接到主面板",
    )
    g_data.add_argument(
        "--extra-source-module",
        type=str,
        default=None,
        help="自定义外部数据源插件路径，模块需实现 register_sources(registry)",
    )
    g_data.add_argument(
        "--factor-catalog-path",
        type=str,
        default="auto",
        help=(
            "因子挖掘 catalog(JSON) 路径。"
            "auto=自动定位到 data_baostock/factor_mining/factor_catalog.json；"
            "none/null=禁用 catalog。"
        ),
    )
    g_data.add_argument(
        "--disable-catalog-factors",
        action="store_true",
        help="关闭 catalog 因子的自动加载与注册",
    )

    # =========================
    # 时间区间参数
    # =========================
    g_date = parser.add_argument_group("训练/测试区间配置")
    g_date.add_argument("--train-start", type=str, default="2024-01-01", help="训练开始日期（含）")
    g_date.add_argument("--train-end", type=str, default="2024-12-31", help="训练结束日期（含）")
    g_date.add_argument("--test-start", type=str, default="2025-01-01", help="测试开始日期（含）")
    g_date.add_argument("--test-end", type=str, default="2025-12-31", help="测试结束日期（含）")

    # =========================
    # 因子与标签参数
    # =========================
    g_factor = parser.add_argument_group("因子与标签配置")
    g_factor.add_argument(
        "--factor-freq",
        type=str,
        choices=SUPPORTED_FREQS,
        default="D",
        help="因子频率：5min/15min/30min/60min/120min/D/W/M",
    )
    g_factor.add_argument(
        "--factor-list",
        type=str,
        default="",
        help="显式指定因子名列表（逗号分隔）；为空时使用该频率默认因子集",
    )
    g_factor.add_argument(
        "--custom-factor-py",
        type=str,
        default=None,
        help="自定义因子插件路径，模块需实现 register_factors(library)",
    )
    g_factor.add_argument(
        "--list-factors",
        action="store_true",
        help="仅列出当前频率可用因子并退出（默认不训练不回测）",
    )
    g_factor.add_argument(
        "--lookback-days",
        type=int,
        default=160,
        help="加载数据时向前额外回溯天数（用于滚动指标稳定计算）",
    )
    g_factor.add_argument(
        "--min-ic-cross-section",
        type=int,
        default=30,
        help="计算 IC/RankIC 时最小横截面样本数阈值",
    )
    g_factor.add_argument(
        "--label-task",
        type=str,
        choices=["direction", "return", "volatility", "multi_task"],
        default="direction",
        help=(
            "训练标签类型：direction=涨跌分类；return=连续收益；"
            "volatility=连续波动率；multi_task=多任务占位（当前以 direction 为主）"
        ),
    )

    # =========================
    # 选股模型参数
    # =========================
    g_stock = parser.add_argument_group("选股模型配置")
    g_stock.add_argument(
        "--stock-model-type",
        type=str,
        default="decision_tree",
        help="选股模型类型：decision_tree / factor_gcl / dafat / 自定义插件",
    )
    g_stock.add_argument(
        "--custom-stock-model-py",
        type=str,
        default=None,
        help="自定义选股模型插件路径，需实现 build_model(cfg)",
    )
    g_stock.add_argument("--max-depth", type=int, default=6, help="决策树最大深度（仅 decision_tree 生效）")
    g_stock.add_argument("--min-samples-leaf", type=int, default=200, help="决策树叶节点最小样本数")
    g_stock.add_argument("--random-state", type=int, default=42, help="随机种子（影响模型训练与采样）")
    g_stock.add_argument("--fgcl-seq-len", type=int, default=30, help="FactorGCL 历史序列长度")
    g_stock.add_argument("--fgcl-future-look", type=int, default=20, help="FactorGCL 未来对比分支长度")
    g_stock.add_argument("--fgcl-hidden-size", type=int, default=128, help="FactorGCL 隐层维度")
    g_stock.add_argument("--fgcl-num-layers", type=int, default=2, help="FactorGCL GRU 层数")
    g_stock.add_argument("--fgcl-num-factor", type=int, default=48, help="FactorGCL 隐式超边因子数量")
    g_stock.add_argument("--fgcl-gamma", type=float, default=1.0, help="FactorGCL 对比损失权重系数")
    g_stock.add_argument("--fgcl-tau", type=float, default=0.25, help="FactorGCL InfoNCE 温度系数")
    g_stock.add_argument("--fgcl-epochs", type=int, default=200, help="FactorGCL 最大训练轮数")
    g_stock.add_argument("--fgcl-lr", type=float, default=9e-5, help="FactorGCL 学习率")
    g_stock.add_argument("--fgcl-early-stop", type=int, default=20, help="FactorGCL 早停耐心轮数")
    g_stock.add_argument("--fgcl-smooth-steps", type=int, default=5, help="FactorGCL 最优权重平滑窗口")
    g_stock.add_argument("--fgcl-per-epoch-batch", type=int, default=100, help="每轮抽取多少交易日切片训练")
    g_stock.add_argument(
        "--fgcl-batch-size",
        type=int,
        default=-1,
        help="-1=单日横截面全量；正整数=单日随机抽样股票数",
    )
    g_stock.add_argument(
        "--fgcl-label-transform",
        type=str,
        choices=["raw", "csrank", "cszscore", "csranknorm"],
        default="csranknorm",
        help="FactorGCL 训练标签变换方式（截面排序/标准化）",
    )
    g_stock.add_argument("--fgcl-weight-decay", type=float, default=1e-4, help="FactorGCL L2 正则强度")
    g_stock.add_argument("--fgcl-dropout", type=float, default=0.0, help="FactorGCL GRU dropout 概率")
    g_stock.add_argument("--fgcl-device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="FactorGCL 训练设备")
    g_stock.add_argument("--dafat-seq-len", type=int, default=40, help="DAFAT 时序长度")
    g_stock.add_argument("--dafat-hidden-size", type=int, default=128, help="DAFAT 隐层维度")
    g_stock.add_argument("--dafat-num-layers", type=int, default=2, help="DAFAT Transformer 层数")
    g_stock.add_argument("--dafat-num-heads", type=int, default=4, help="DAFAT 多头注意力头数")
    g_stock.add_argument("--dafat-ffn-mult", type=int, default=4, help="DAFAT 前馈层扩展倍数")
    g_stock.add_argument("--dafat-dropout", type=float, default=0.10, help="DAFAT dropout 概率")
    g_stock.add_argument("--dafat-local-window", type=int, default=20, help="DAFAT 稀疏注意力局部窗口")
    g_stock.add_argument("--dafat-topk-ratio", type=float, default=0.30, help="DAFAT TopK 稀疏保留比例")
    g_stock.add_argument("--dafat-vol-quantile", type=float, default=0.40, help="DAFAT 波动率门控分位点")
    g_stock.add_argument("--dafat-meso-scale", type=int, default=5, help="DAFAT 多尺度中观池化尺度")
    g_stock.add_argument("--dafat-macro-scale", type=int, default=20, help="DAFAT 多尺度宏观池化尺度")
    g_stock.add_argument("--dafat-epochs", type=int, default=200, help="DAFAT 最大训练轮数")
    g_stock.add_argument("--dafat-lr", type=float, default=1e-4, help="DAFAT 学习率")
    g_stock.add_argument("--dafat-weight-decay", type=float, default=1e-4, help="DAFAT L2 正则强度")
    g_stock.add_argument("--dafat-early-stop", type=int, default=20, help="DAFAT 早停耐心轮数")
    g_stock.add_argument("--dafat-per-epoch-batch", type=int, default=120, help="DAFAT 每轮抽取交易日切片数")
    g_stock.add_argument("--dafat-batch-size", type=int, default=-1, help="DAFAT 单日截面采样数，-1=全量")
    g_stock.add_argument(
        "--dafat-label-transform",
        type=str,
        choices=["raw", "csrank", "cszscore", "csranknorm"],
        default="csranknorm",
        help="DAFAT 训练标签变换方式（截面排序/标准化）",
    )
    g_stock.add_argument("--dafat-mse-weight", type=float, default=0.05, help="DAFAT 损失函数中 MSE 权重")
    g_stock.add_argument("--dafat-use-dpe", type=_parse_bool, nargs="?", const=True, default=True, help="是否启用 DPE 模块")
    g_stock.add_argument(
        "--dafat-use-sparse-attn",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="是否启用稀疏注意力模块",
    )
    g_stock.add_argument(
        "--dafat-use-multiscale",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="是否启用多尺度融合模块",
    )
    g_stock.add_argument("--dafat-device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="DAFAT 训练设备")

    # =========================
    # 择时模型参数
    # =========================
    g_timing = parser.add_argument_group("择时模型配置")
    g_timing.add_argument(
        "--timing-model-type",
        type=str,
        choices=["none", "volatility_regime"],
        default="none",
        help="择时模型类型：none=不择时；volatility_regime=波动/拥挤/动量状态择时",
    )
    g_timing.add_argument(
        "--custom-timing-model-py",
        type=str,
        default=None,
        help="自定义择时模型插件路径，需实现 build_model(cfg)",
    )
    g_timing.add_argument("--timing-vol-threshold", type=float, default=0.0, help="波动阈值（0 表示训练期自动估计）")
    g_timing.add_argument("--timing-momentum-threshold", type=float, default=0.0, help="动量阈值（0 表示训练期自动估计）")

    # =========================
    # 执行模型参数
    # =========================
    g_exec = parser.add_argument_group("执行模型配置")
    g_exec.add_argument(
        "--execution-model-type",
        type=str,
        choices=["ideal_fill", "realistic_fill"],
        default="ideal_fill",
        help="执行模型类型：ideal_fill=理想成交；realistic_fill=含流动性/延迟惩罚",
    )
    g_exec.add_argument(
        "--custom-execution-model-py",
        type=str,
        default=None,
        help="自定义执行模型插件路径，需实现 build_model(cfg)",
    )
    g_exec.add_argument("--max-participation-rate", type=float, default=0.15, help="单标的最大成交参与率上限")
    g_exec.add_argument("--base-fill-rate", type=float, default=0.95, help="基础成交比例（realistic_fill）")
    g_exec.add_argument("--latency-bars", type=int, default=0, help="执行延迟 bars 数（realistic_fill）")

    # =========================
    # 回测交易参数
    # =========================
    g_bt = parser.add_argument_group("回测交易配置")
    g_bt.add_argument("--horizon", type=int, default=5, help="持有期长度（单位：当前频率 bars）")
    g_bt.add_argument("--top-k", type=int, default=10, help="每次调仓最多持仓股票数")
    g_bt.add_argument("--long-threshold", type=float, default=0.50, help="做多阈值：pred_score >= 该值才入选")
    g_bt.add_argument(
        "--execution-scheme",
        type=str,
        choices=sorted(EXECUTION_SCHEMES.keys()),
        default="vwap30_vwap30",
        help="买卖价格规则（如 open5_open5 / vwap30_vwap30 / open5_twap_last30）",
    )
    g_bt.add_argument("--fee-bps", type=float, default=1.5, help="单边交易费率（bps）")
    g_bt.add_argument("--slippage-bps", type=float, default=1.5, help="单边滑点（bps）")
    g_bt.add_argument("--rebalance-stride", type=int, default=0, help="调仓间隔（0 表示使用 horizon）")
    g_bt.add_argument(
        "--ic-eval-mode",
        type=str,
        choices=["strict_horizon", "per_bar"],
        default="strict_horizon",
        help=(
            "IC/分层评估模式：strict_horizon=按调仓步长(默认=H)抽样评估；"
            "per_bar=每个主频bar评估（等价H=1）。"
        ),
    )

    # =========================
    # 组合优化参数
    # =========================
    g_port = parser.add_argument_group("组合模型配置")
    g_port.add_argument(
        "--portfolio-model-type",
        type=str,
        choices=["equal_weight", "dynamic_opt"],
        default="equal_weight",
        help="组合模型：equal_weight=等权；dynamic_opt=动态优化",
    )
    g_port.add_argument(
        "--portfolio-weighting",
        type=str,
        choices=["equal_weight", "dynamic_opt"],
        default=None,
        help="兼容旧参数名（等价于 --portfolio-model-type）",
    )
    g_port.add_argument("--opt-max-weight", type=float, default=0.25, help="单标的权重上限")
    g_port.add_argument("--custom-portfolio-model-py", type=str, default=None, help="自定义组合模型插件路径")
    g_port.add_argument("--opt-max-turnover", type=float, default=1.20, help="单次调仓最大换手约束")
    g_port.add_argument("--opt-liquidity-scale", type=float, default=3.0, help="流动性容量缩放系数")
    g_port.add_argument("--opt-expected-return-weight", type=float, default=1.0, help="目标函数中收益项权重")
    g_port.add_argument("--opt-risk-aversion", type=float, default=1.2, help="目标函数中风险厌恶系数")
    g_port.add_argument("--opt-style-penalty", type=float, default=0.8, help="风格暴露偏离惩罚系数")
    g_port.add_argument("--opt-industry-penalty", type=float, default=0.7, help="行业偏离惩罚系数")
    g_port.add_argument("--opt-crowding-penalty", type=float, default=0.5, help="拥挤度暴露惩罚系数")
    g_port.add_argument("--opt-transaction-cost-penalty", type=float, default=0.6, help="交易成本惩罚系数")
    g_port.add_argument("--opt-max-iter", type=int, default=220, help="优化最大迭代次数")
    g_port.add_argument("--opt-step-size", type=float, default=0.08, help="梯度步长")
    g_port.add_argument("--opt-tolerance", type=float, default=1e-6, help="优化收敛阈值")

    # =========================
    # 输出与落盘参数
    # =========================
    g_out = parser.add_argument_group("输出配置")
    g_out.add_argument(
        "--output-dir",
        type=str,
        default="auto",
        help="输出目录；auto 会按时间戳与关键参数自动生成目录",
    )
    g_out.add_argument(
        "--save-models",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="是否保存模型文件（true/false）",
    )

    g_out.add_argument(
        "--log-level",
        type=str,
        choices=["quiet", "normal", "verbose"],
        default="normal",
        help="runtime progress log level: quiet / normal / verbose",
    )
    g_out.add_argument(
        "--quiet",
        action="store_true",
        help="equivalent to --log-level quiet (highest priority)",
    )
    g_out.add_argument(
        "--verbose",
        action="store_true",
        help="equivalent to --log-level verbose when --quiet is not set",
    )

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
    """Validate CLI args and convert them into typed dataclass configs."""
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
    if int(args.dafat_seq_len) <= 1:
        raise ValueError("dafat_seq_len must be greater than 1.")
    if int(args.dafat_hidden_size) <= 0:
        raise ValueError("dafat_hidden_size must be positive.")
    if int(args.dafat_num_layers) <= 0:
        raise ValueError("dafat_num_layers must be positive.")
    if int(args.dafat_num_heads) <= 0:
        raise ValueError("dafat_num_heads must be positive.")
    if int(args.dafat_hidden_size) % int(args.dafat_num_heads) != 0:
        raise ValueError("dafat_hidden_size must be divisible by dafat_num_heads.")
    if int(args.dafat_ffn_mult) <= 0:
        raise ValueError("dafat_ffn_mult must be positive.")
    if not (0.0 <= float(args.dafat_dropout) < 1.0):
        raise ValueError("dafat_dropout must be in [0, 1).")
    if int(args.dafat_local_window) <= 0:
        raise ValueError("dafat_local_window must be positive.")
    if not (0.0 < float(args.dafat_topk_ratio) <= 1.0):
        raise ValueError("dafat_topk_ratio must be in (0, 1].")
    if not (0.0 <= float(args.dafat_vol_quantile) <= 1.0):
        raise ValueError("dafat_vol_quantile must be in [0, 1].")
    if int(args.dafat_meso_scale) <= 0:
        raise ValueError("dafat_meso_scale must be positive.")
    if int(args.dafat_macro_scale) <= 0:
        raise ValueError("dafat_macro_scale must be positive.")
    if int(args.dafat_epochs) <= 0:
        raise ValueError("dafat_epochs must be positive.")
    if float(args.dafat_lr) <= 0.0:
        raise ValueError("dafat_lr must be positive.")
    if float(args.dafat_weight_decay) < 0.0:
        raise ValueError("dafat_weight_decay must be non-negative.")
    if int(args.dafat_early_stop) <= 0:
        raise ValueError("dafat_early_stop must be positive.")
    if int(args.dafat_per_epoch_batch) <= 0:
        raise ValueError("dafat_per_epoch_batch must be positive.")
    if int(args.dafat_batch_size) == 0 or int(args.dafat_batch_size) < -1:
        raise ValueError("dafat_batch_size must be -1 or positive.")
    if float(args.dafat_mse_weight) < 0.0:
        raise ValueError("dafat_mse_weight must be non-negative.")
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
    data_root_resolved = _resolve_path(args.data_root)
    factor_catalog_path = _resolve_factor_catalog_path(args.factor_catalog_path, data_root=data_root_resolved)

    data = DataConfig(
        data_root=data_root_resolved,
        hs300_list_path=_resolve_path(args.hs300_list_path),
        index_root=_resolve_path(args.index_root),
        file_format=args.file_format,
        max_files=args.max_files,
        main_board_only=bool(args.main_board_only),
        extra_factor_paths=extra_paths,
        extra_source_module=_resolve_path(args.extra_source_module) if args.extra_source_module else None,
        factor_catalog_path=factor_catalog_path,
        auto_load_catalog_factors=not bool(args.disable_catalog_factors),
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
        dafat_seq_len=int(args.dafat_seq_len),
        dafat_hidden_size=int(args.dafat_hidden_size),
        dafat_num_layers=int(args.dafat_num_layers),
        dafat_num_heads=int(args.dafat_num_heads),
        dafat_ffn_mult=int(args.dafat_ffn_mult),
        dafat_dropout=float(args.dafat_dropout),
        dafat_local_window=int(args.dafat_local_window),
        dafat_topk_ratio=float(args.dafat_topk_ratio),
        dafat_vol_quantile=float(args.dafat_vol_quantile),
        dafat_meso_scale=int(args.dafat_meso_scale),
        dafat_macro_scale=int(args.dafat_macro_scale),
        dafat_epochs=int(args.dafat_epochs),
        dafat_lr=float(args.dafat_lr),
        dafat_weight_decay=float(args.dafat_weight_decay),
        dafat_early_stop=int(args.dafat_early_stop),
        dafat_per_epoch_batch=int(args.dafat_per_epoch_batch),
        dafat_batch_size=int(args.dafat_batch_size),
        dafat_label_transform=str(args.dafat_label_transform),
        dafat_mse_weight=float(args.dafat_mse_weight),
        dafat_use_dpe=bool(args.dafat_use_dpe),
        dafat_use_sparse_attn=bool(args.dafat_use_sparse_attn),
        dafat_use_multiscale=bool(args.dafat_use_multiscale),
        dafat_device=str(args.dafat_device),
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
        ic_eval_mode=str(args.ic_eval_mode),
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
