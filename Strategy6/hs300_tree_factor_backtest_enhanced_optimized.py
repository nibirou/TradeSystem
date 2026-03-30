"""HS300 决策树量价因子回测脚本。

整体流程：
1. 读取 HS300 日频与 5 分钟数据；
2. 将 5 分钟数据聚合成日内微观结构特征与可执行价格；
3. 构建可插拔因子库并生成训练/测试因子面板；
4. 训练决策树并生成横截面打分；
5. 依据可选执行价方案进行回测；
6. 输出收益曲线、净值曲线、IC/RankIC/ICIR 等统计结果。

无未来函数约束（核心）：
- 因子仅使用信号日及历史数据；
- 信号在 T 日收盘后产生，交易从 T+1 开始；
- 标签收益使用 T+1 入场到 T+1+horizon 出场；
- 买卖价格均来自对应交易日内可观测 5 分钟数据。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# 数值稳定项，避免除零与极小分母放大。
EPS = 1e-12
# A 股年化换算通常使用 252 个交易日。
TRADING_DAYS_PER_YEAR = 252

# 三种可切换的执行价格方案（均不引入未来信息）。
# buy_col/sell_col 对应 build_minute_daily_features 生成的可执行价格列。
EXECUTION_SCHEMES: Dict[str, Dict[str, str]] = {
    "open5_open5": {
        "buy_col": "px_open5",
        "sell_col": "px_open5",
        "description": "信号日后首个交易日首根5分钟开盘价买入，持有n日后在退出日首根5分钟开盘价卖出。",
    },
    "vwap30_vwap30": {
        "buy_col": "px_vwap30",
        "sell_col": "px_vwap30",
        "description": "信号日后首个交易日开盘后前30分钟VWAP买入，持有n日后在退出日开盘后前30分钟VWAP卖出。",
    },
    "open5_twap_last30": {
        "buy_col": "px_open5",
        "sell_col": "px_twap_last30",
        "description": "信号日后首个交易日首根5分钟开盘价买入，持有n日后在退出日最后30分钟TWAP卖出。",
    },
    "daily_close_daily_close": {
        "buy_col": "px_daily_close",
        "sell_col": "px_daily_close",
        "description": "信号日后首个交易日收盘价买入，持有n日后在退出日收盘价卖出。",
    },
}


@dataclass
class FactorDef:
    """单个因子的元信息定义。"""
    name: str
    category: str
    description: str
    func: Callable[[pd.DataFrame], pd.Series]


class FactorLibrary:
    """可插拔因子注册表。

    - register: 注册因子
    - names/get: 查询因子
    - metadata: 导出因子说明表
    """
    def __init__(self) -> None:
        self._factors: Dict[str, FactorDef] = {}

    def register(self, name: str, category: str, description: str, func: Callable[[pd.DataFrame], pd.Series]) -> None:
        self._factors[name] = FactorDef(name=name, category=category, description=description, func=func)

    def names(self) -> List[str]:
        return sorted(self._factors.keys())

    def get(self, name: str) -> FactorDef:
        if name not in self._factors:
            raise KeyError(f"未找到因子: {name}")
        return self._factors[name]

    def metadata(self) -> pd.DataFrame:
        rows = [
            {"factor": f.name, "category": f.category, "description": f.description}
            for f in self._factors.values()
        ]
        return pd.DataFrame(rows).sort_values("factor")


# 默认推荐因子组合：偏向趋势 + 流动性 + 波动率 + 日内微结构。
DEFAULT_FACTOR_SET: List[str] = [
    "mom_5",
    "mom_10",
    "mom_20",
    "rev_1",
    "rev_3",
    "ma_gap_5",
    "ma_gap_20",
    "breakout_20",
    "vol_ratio_20",
    "amount_ratio_20",
    "turn_ratio_5",
    "atr_norm_14",
    "realized_vol_20",
    "downside_vol_ratio_20",
    "amihud_20",
    "ret_vol_corr_20",
    "close_to_vwap_day",
    "open_to_close_intraday",
    "morning_momentum_30m",
    "last30_momentum",
    "minute_up_ratio_5m",
    "minute_ret_skew_5m",
    "signed_vol_imbalance_5m",
    "jump_ratio_5m",
]


@dataclass
class PortfolioOptimizationConfig:
    """组合优化参数。"""
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


def _split_exchange_code(code_or_key: str) -> Tuple[str, str]:
    """解析代码并返回(交易所, 6位证券代码)。"""
    s = str(code_or_key).strip().lower()
    if "_" in s:
        ex, code = s.split("_", 1)
    elif "." in s:
        code, ex = s.split(".", 1)
    else:
        code, ex = s, ""

    code = re.sub(r"[^0-9]", "", code)
    code = code[-6:].zfill(6) if code else ""

    ex = ex.strip().lower()
    if ex not in {"sh", "sz"}:
        if code.startswith(("6", "9")):
            ex = "sh"
        elif code.startswith(("0", "1", "2", "3")):
            ex = "sz"
    return ex, code


def is_main_board_symbol_key(symbol_key: str) -> bool:
    """判断是否属于沪深主板。"""
    ex, code = _split_exchange_code(symbol_key)
    if ex == "sh":
        return code.startswith(("600", "601", "603", "605"))
    if ex == "sz":
        return code.startswith(("000", "001", "002", "003"))
    return False


def infer_board_type(code_or_key: str) -> str:
    """根据证券代码推断板块类型。"""
    ex, code = _split_exchange_code(code_or_key)
    if ex == "sh":
        if code.startswith(("600", "601", "603", "605")):
            return "main_board"
        if code.startswith("688"):
            return "star_board"
    if ex == "sz":
        if code.startswith(("000", "001", "002", "003")):
            return "main_board"
        if code.startswith("300"):
            return "chi_next"
    return "other_board"


def infer_industry_bucket(code_or_key: str) -> str:
    """构建行业代理分组（优先保证稳定分组，不依赖外部行业表）。"""
    ex, code = _split_exchange_code(code_or_key)
    if not code:
        return "unknown"
    return f"{ex}_{code[:2]}"


def cross_section_zscore(s: pd.Series) -> pd.Series:
    """横截面标准化，自动处理常数列与缺失。"""
    x = pd.to_numeric(s, errors="coerce")
    mean_v = float(x.mean()) if x.notna().any() else 0.0
    std_v = float(x.std(ddof=0)) if x.notna().sum() > 1 else 0.0
    if std_v <= EPS:
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    return (x - mean_v) / (std_v + EPS)


def robust_history_zscore(value: float, history: List[float]) -> float:
    """基于历史样本计算稳健z-score。"""
    hist = pd.Series(history, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(hist) < 5 or not np.isfinite(value):
        return 0.0

    med = float(hist.median())
    mad = float((hist - med).abs().median())
    scale = 1.4826 * mad if mad > EPS else float(hist.std(ddof=0))
    if not np.isfinite(scale) or scale <= EPS:
        return 0.0
    return float(np.clip((value - med) / (scale + EPS), -4.0, 4.0))


def ensure_feasible_caps(caps: np.ndarray, target_sum: float = 1.0, hard_cap: float = 1.0) -> np.ndarray:
    """保证上限向量可行（上限和至少覆盖target_sum）。"""
    u = np.clip(np.asarray(caps, dtype=float), 0.0, hard_cap)
    if u.sum() >= target_sum - 1e-10:
        return u

    slack = np.clip(hard_cap - u, 0.0, None)
    deficit = target_sum - float(u.sum())
    if slack.sum() > EPS:
        u = u + slack * min(1.0, deficit / (float(slack.sum()) + EPS))

    if u.sum() < target_sum - 1e-10:
        n = len(u)
        uniform_cap = min(hard_cap, max(target_sum / max(n, 1), np.max(u) if n > 0 else 1.0))
        u = np.full(n, uniform_cap, dtype=float)
    return u


def project_to_capped_simplex(v: np.ndarray, caps: np.ndarray, target_sum: float = 1.0) -> np.ndarray:
    """投影到带上限的概率单纯形：w>=0, w<=caps, sum(w)=target_sum。"""
    if len(v) == 0:
        return np.array([], dtype=float)

    u = ensure_feasible_caps(caps, target_sum=target_sum, hard_cap=1.0)
    x = np.asarray(v, dtype=float)

    lo = float(np.min(x - u))
    hi = float(np.max(x))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        w_mid = np.clip(x - mid, 0.0, u)
        if w_mid.sum() > target_sum:
            lo = mid
        else:
            hi = mid

    w = np.clip(x - hi, 0.0, u)
    residual = target_sum - float(w.sum())
    if abs(residual) > 1e-9:
        room = np.where(residual > 0.0, u - w, w)
        total_room = float(room[room > 0].sum())
        if total_room > EPS:
            adjust = room / (total_room + EPS) * residual
            w = np.clip(w + adjust, 0.0, u)
    total = float(w.sum())
    return w / (total + EPS) if total > EPS else np.full(len(w), 1.0 / len(w))


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    所有实验开关集中于此，便于复现、网格调参和批量回测。
    """
    parser = argparse.ArgumentParser(
        description=(
            "读取沪深300日频+5分钟数据，构建可插拔因子库，训练决策树并执行无未来函数回测。"
        )
    )
    parser.add_argument(
        "--data-root",
        type=str,
        # default=r"E:\pythonProject\data_baostock\stock_hist\hs300",
        # default=r"/workspace/Quant/data_baostock/stock_hist/hs300",
        default=r"D:/PythonProject/Quant/data_baostock/stock_hist/hs300",
        help="数据根目录，需包含 d(日频) 和 5(5分钟) 子目录。",
    )
    parser.add_argument("--train-start", type=str, default="2024-01-01", help="训练开始日期。")
    parser.add_argument("--train-end", type=str, default="2024-12-31", help="训练结束日期。")
    parser.add_argument("--test-start", type=str, default="2025-01-01", help="回测开始日期（信号日）。")
    parser.add_argument("--test-end", type=str, default="2025-12-31", help="回测结束日期（信号日）。")
    parser.add_argument("--horizon", type=int, default=5, help="持有n个交易日后卖出。")
    parser.add_argument("--top-k", type=int, default=10, help="每次调仓选取概率最高的股票数。")
    parser.add_argument("--long-threshold", type=float, default=0.50, help="做多阈值。")
    parser.add_argument("--max-depth", type=int, default=6, help="决策树最大深度。")
    parser.add_argument("--min-samples-leaf", type=int, default=200, help="叶子最小样本数。")
    parser.add_argument("--lookback-days", type=int, default=160, help="因子计算额外回看天数。")
    parser.add_argument("--max-files", type=int, default=None, help="仅加载前N只股票（调试用）。")
    parser.add_argument("--main-board-only", action="store_true", help="仅使用沪深主板股票（训练与回测保持一致）。")
    parser.add_argument("--file-format", type=str, choices=["auto", "parquet", "csv"], default="auto", help="Data file format preference.")
    parser.add_argument(
        "--hs300-list-path",
        type=str,
        # default=r"E:/pythonProject/data_baostock/metadata/stock_list_hs300.csv",
        default=r"/workspace/Quant/data_baostock/metadata/stock_list_hs300.csv",
        help="Path to HS300 constituent list CSV.",
    )
    parser.add_argument(
        "--index-root",
        type=str,
        # default=r"E:/pythonProject/data_baostock/ak_index",
        default=r"/workspace/Quant/data_baostock/ak_index",
        help="Directory containing hs300/zz500/zz1000 index price files.",
    )

    parser.add_argument(
        "--execution-scheme",
        type=str,
        choices=sorted(EXECUTION_SCHEMES.keys()),
        default="vwap30_vwap30",
        help="买卖价执行方案。",
    )
    parser.add_argument("--fee-bps", type=float, default=1.5, help="单边手续费（bps）。")
    parser.add_argument("--slippage-bps", type=float, default=1.5, help="单边滑点（bps）。")

    parser.add_argument("--factor-list", type=str, default="", help="使用逗号分隔的因子列表；为空则使用内置推荐因子集。")
    parser.add_argument("--custom-factor-py", type=str, default=None, help="自定义因子模块路径，需暴露 register_factors(library) 函数。")
    parser.add_argument("--list-factors", action="store_true", help="仅打印可用因子并退出。")
    parser.add_argument("--min-ic-cross-section", type=int, default=30, help="IC计算单日最小截面样本数。")
    parser.add_argument(
        "--portfolio-weighting",
        type=str,
        choices=["equal_weight", "dynamic_opt"],
        default="equal_weight",
        help="回测持仓权重构建方式：等权或状态感知动态优化。",
    )
    parser.add_argument("--opt-max-weight", type=float, default=0.25, help="动态优化时单票最大权重上限。")
    parser.add_argument("--opt-max-turnover", type=float, default=1.20, help="动态优化时单期最大换手约束（L1）。")
    parser.add_argument("--opt-liquidity-scale", type=float, default=3.0, help="基于成交额份额的流动性上限放大倍数。")
    parser.add_argument("--opt-expected-return-weight", type=float, default=1.0, help="优化目标中预期收益项权重。")
    parser.add_argument("--opt-risk-aversion", type=float, default=1.2, help="优化目标中波动风险惩罚系数。")
    parser.add_argument("--opt-style-penalty", type=float, default=0.8, help="Barra风格暴露偏离惩罚系数。")
    parser.add_argument("--opt-industry-penalty", type=float, default=0.7, help="行业偏离惩罚系数。")
    parser.add_argument("--opt-crowding-penalty", type=float, default=0.5, help="拥挤度惩罚系数。")
    parser.add_argument("--opt-transaction-cost-penalty", type=float, default=0.6, help="交易成本惩罚系数。")
    parser.add_argument("--opt-max-iter", type=int, default=220, help="动态优化最大迭代次数。")
    parser.add_argument("--opt-step-size", type=float, default=0.08, help="动态优化梯度步长。")
    parser.add_argument("--opt-tolerance", type=float, default=1e-6, help="动态优化收敛阈值。")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="auto",
        help="结果输出目录；auto 表示自动生成包含时间戳和关键信息的目录名。",
    )
    return parser.parse_args()


def _parse_date(date_str: str) -> pd.Timestamp:
    """统一日期解析并归一化到自然日。"""
    return pd.to_datetime(date_str).normalize()


def resolve_output_dir(
    args: argparse.Namespace,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> Path:
    """生成输出目录。output-dir=auto 时自动拼接时间戳与核心参数。"""
    if str(args.output_dir).strip().lower() not in {"", "auto"}:
        return Path(args.output_dir)

    ts = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d_%H%M%S")
    board_tag = "mainboard" if bool(args.main_board_only) else "allboards"
    mode_tag = "opt" if args.portfolio_weighting == "dynamic_opt" else "equal"
    run_name = (
        f"outputs_{ts}"
        f"_tr{train_start.strftime('%Y%m%d')}-{train_end.strftime('%Y%m%d')}"
        f"_te{test_start.strftime('%Y%m%d')}-{test_end.strftime('%Y%m%d')}"
        f"_h{int(args.horizon)}_k{int(args.top_k)}_{args.execution_scheme}_{board_tag}_{mode_tag}"
    )
    return Path(__file__).resolve().parent / run_name


def build_portfolio_opt_config(args: argparse.Namespace) -> PortfolioOptimizationConfig:
    """从命令行参数构建组合优化配置。"""
    return PortfolioOptimizationConfig(
        mode=args.portfolio_weighting,
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
    )


def _symbol_key_from_filename(filename: str) -> Optional[str]:
    """从文件名提取股票键（如 sh_600000）。"""
    match = re.match(r"^([a-z]{2}_\d{6})_(d|5)\.(csv|parquet)$", filename.lower())
    if not match:
        return None
    return match.group(1)


def list_symbol_keys(daily_dir: Path, file_format: str = "auto") -> List[str]:
    """扫描日频目录，返回可用股票键列表。"""
    keys: set[str] = set()
    if file_format in ("auto", "parquet"):
        for p in daily_dir.glob("*_d.parquet"):
            k = _symbol_key_from_filename(p.name)
            if k:
                keys.add(k)
    if file_format in ("auto", "csv"):
        for p in daily_dir.glob("*_d.csv"):
            k = _symbol_key_from_filename(p.name)
            if k:
                keys.add(k)
    return sorted(keys)


def pick_existing_file(folder: Path, symbol_key: str, freq_tag: str, file_format: str = "auto") -> Optional[Path]:
    """按格式偏好寻找存在的数据文件。"""
    if file_format == "parquet":
        candidates = [folder / f"{symbol_key}_{freq_tag}.parquet"]
    elif file_format == "csv":
        candidates = [folder / f"{symbol_key}_{freq_tag}.csv"]
    else:
        candidates = [folder / f"{symbol_key}_{freq_tag}.parquet", folder / f"{symbol_key}_{freq_tag}.csv"]
    for p in candidates:
        if p.exists():
            return p
    return None


def _read_data_file(path: Path, usecols: List[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """读取单文件并按日期截取。

    parquet 不可用时自动回退到同名 csv。
    """
    date_start = start_date.strftime("%Y-%m-%d")
    date_end = end_date.strftime("%Y-%m-%d")

    if path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(
                path,
                columns=usecols,
                filters=[("date", ">=", date_start), ("date", "<=", date_end)],
            )
        except Exception:
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                df = pd.read_csv(csv_path, usecols=lambda c: c in usecols)
            else:
                raise
    else:
        df = pd.read_csv(path, usecols=lambda c: c in usecols)

    if "date" not in df.columns:
        return pd.DataFrame(columns=usecols)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    return df

def load_hs300_constituent_keys(hs300_list_path: Path) -> List[str]:
    """从成分股清单读取股票键（格式如 sh_600000）。"""
    if not hs300_list_path.exists():
        raise FileNotFoundError(f"未找到沪深300成分股清单: {hs300_list_path}")

    df = pd.read_csv(hs300_list_path)
    if "code" in df.columns:
        codes = df["code"].astype(str)
    else:
        codes = df.iloc[:, 0].astype(str)

    keys = (
        codes.str.strip().str.lower().str.replace(".", "_", regex=False)
    )
    keys = keys[keys.str.match(r"^[a-z]{2}_\d{6}$", na=False)]
    return sorted(keys.drop_duplicates().tolist())


def load_hs300_data(
    data_root: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    file_format: str = "auto",
    max_files: Optional[int] = None,
    hs300_list_path: Optional[Path] = None,
    main_board_only: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """批量加载 HS300 日频与 5 分钟行情。

    股票池优先按 stock_list_hs300.csv 过滤，确保与指数成分一致。
    可选进一步筛选至沪深主板股票。
    """
    daily_dir = data_root / "d"
    minute_dir = data_root / "5"
    if not daily_dir.exists() or not minute_dir.exists():
        raise FileNotFoundError(f"未找到数据目录: {daily_dir} 或 {minute_dir}")

    keys_all = list_symbol_keys(daily_dir, file_format=file_format)
    keys = keys_all
    if hs300_list_path is not None:
        hs300_keys = set(load_hs300_constituent_keys(hs300_list_path))
        keys = [k for k in keys_all if k in hs300_keys]
    if main_board_only:
        keys = [k for k in keys if is_main_board_symbol_key(k)]

    if max_files is not None:
        keys = keys[:max_files]

    daily_cols = [
        "date",
        "code",
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "turn",
        "tradestatus",
    ]
    minute_cols = ["date", "time", "code", "open", "high", "low", "close", "volume", "amount"]

    daily_frames: List[pd.DataFrame] = []
    minute_frames: List[pd.DataFrame] = []

    for key in keys:
        daily_path = pick_existing_file(daily_dir, key, "d", file_format=file_format)
        minute_path = pick_existing_file(minute_dir, key, "5", file_format=file_format)
        if daily_path is None or minute_path is None:
            continue

        ddf = _read_data_file(daily_path, daily_cols, start_date, end_date)
        mdf = _read_data_file(minute_path, minute_cols, start_date, end_date)
        if ddf.empty or mdf.empty:
            continue

        for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn"]:
            if col in ddf.columns:
                ddf[col] = pd.to_numeric(ddf[col], errors="coerce")
        if "tradestatus" in ddf.columns:
            ddf["tradestatus"] = pd.to_numeric(ddf["tradestatus"], errors="coerce")

        for col in ["time", "open", "high", "low", "close", "volume", "amount"]:
            if col in mdf.columns:
                mdf[col] = pd.to_numeric(mdf[col], errors="coerce")

        if "tradestatus" in ddf.columns:
            ddf = ddf[ddf["tradestatus"].fillna(1) == 1].copy()

        daily_frames.append(ddf)
        minute_frames.append(mdf)

    if not daily_frames:
        raise RuntimeError("未加载到日频数据，请检查路径、日期范围或成分股清单。")
    if not minute_frames:
        raise RuntimeError("未加载到5分钟数据，请检查路径、日期范围或成分股清单。")

    daily_df = pd.concat(daily_frames, ignore_index=True)
    minute_df = pd.concat(minute_frames, ignore_index=True)
    return daily_df, minute_df

def build_minute_daily_features(minute_df: pd.DataFrame) -> pd.DataFrame:
    """将 5 分钟行情聚合为“日级微观结构特征 + 可执行价格”。

    主要产出：
    - 日内统计因子（波动、偏度、成交失衡、早盘/尾盘动量等）
    - 三类执行价格：open5 / vwap30 / twap_last30
    """
    m = minute_df.copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.normalize()
    m["time"] = pd.to_numeric(m["time"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors="coerce")

    m = m.dropna(subset=["date", "code", "time", "open", "close", "volume"]).copy()
    m = m.sort_values(["code", "date", "time"]).copy()

    grp_cols = ["code", "date"]
    m["bar_idx"] = m.groupby(grp_cols).cumcount()
    m["bar_rev_idx"] = m.groupby(grp_cols).cumcount(ascending=False)
    m["ret_5m"] = m.groupby(grp_cols)["close"].pct_change()
    m["pv"] = m["close"] * m["volume"]
    m["signed_vol"] = np.sign(m["ret_5m"].fillna(0.0)) * m["volume"]
    m["abs_ret_5m"] = m["ret_5m"].abs()

    base = m.groupby(grp_cols, as_index=False).agg(
        first_open_5m=("open", "first"),
        first_close_5m=("close", "first"),
        last_close_5m=("close", "last"),
        day_vol=("volume", "sum"),
        day_pv=("pv", "sum"),
        minute_realized_vol_5m=("ret_5m", "std"),
        minute_up_ratio_5m=("ret_5m", lambda x: float((x > 0).mean()) if x.notna().any() else np.nan),
        minute_ret_skew_5m=("ret_5m", lambda x: float(x.skew()) if x.notna().sum() > 2 else np.nan),
        minute_ret_kurt_5m=("ret_5m", lambda x: float(x.kurt()) if x.notna().sum() > 3 else np.nan),
        signed_vol_sum=("signed_vol", "sum"),
        abs_ret_max=("abs_ret_5m", "max"),
    )

    first30 = m[m["bar_idx"] < 6].groupby(grp_cols, as_index=False).agg(
        pv_first30=("pv", "sum"),
        vol_first30=("volume", "sum"),
        close_30m=("close", "last"),
    )

    first60 = m[m["bar_idx"] < 12].groupby(grp_cols, as_index=False).agg(
        pv_first60=("pv", "sum"),
        vol_first60=("volume", "sum"),
    )

    last30 = m[m["bar_rev_idx"] < 6].groupby(grp_cols, as_index=False).agg(
        twap_last30=("close", "mean"),
        close_last30_start=("close", "first"),
        close_last30_end=("close", "last"),
    )

    feat = base.merge(first30, on=grp_cols, how="left").merge(first60, on=grp_cols, how="left").merge(last30, on=grp_cols, how="left")

    feat["vwap_day"] = feat["day_pv"] / (feat["day_vol"] + EPS)
    feat["vwap_first30"] = feat["pv_first30"] / (feat["vol_first30"] + EPS)
    feat["vwap_first60"] = feat["pv_first60"] / (feat["vol_first60"] + EPS)
    feat["signed_vol_imbalance_5m"] = feat["signed_vol_sum"] / (feat["day_vol"] + EPS)
    feat["jump_ratio_5m"] = feat["abs_ret_max"] / (feat["minute_realized_vol_5m"] + EPS)

    feat["morning_momentum_30m"] = feat["close_30m"] / (feat["first_open_5m"] + EPS) - 1.0
    feat["last30_momentum"] = feat["close_last30_end"] / (feat["close_last30_start"] + EPS) - 1.0
    feat["open_to_close_intraday"] = feat["last_close_5m"] / (feat["first_open_5m"] + EPS) - 1.0
    feat["vwap30_vs_day"] = feat["vwap_first30"] / (feat["vwap_day"] + EPS) - 1.0

    feat["px_open5"] = feat["first_open_5m"]
    feat["px_vwap30"] = feat["vwap_first30"]
    feat["px_twap_last30"] = feat["twap_last30"]

    keep_cols = [
        "code",
        "date",
        "vwap_day",
        "vwap_first30",
        "vwap_first60",
        "minute_realized_vol_5m",
        "minute_up_ratio_5m",
        "minute_ret_skew_5m",
        "minute_ret_kurt_5m",
        "signed_vol_imbalance_5m",
        "jump_ratio_5m",
        "morning_momentum_30m",
        "last30_momentum",
        "open_to_close_intraday",
        "vwap30_vs_day",
        "px_open5",
        "px_vwap30",
        "px_twap_last30",
    ]
    return feat[keep_cols]

def _rolling_mean(g: pd.core.groupby.generic.SeriesGroupBy, window: int) -> pd.Series:
    """分组滚动均值封装，统一窗口与最小样本口径。"""
    return g.transform(lambda s: s.rolling(window, min_periods=window).mean())


def _rolling_std(g: pd.core.groupby.generic.SeriesGroupBy, window: int) -> pd.Series:
    """分组滚动标准差封装。"""
    return g.transform(lambda s: s.rolling(window, min_periods=window).std())


def build_daily_feature_base(daily_df: pd.DataFrame, minute_daily_feat: pd.DataFrame) -> pd.DataFrame:
    """构建日频基础特征底表。

    此函数只做“可复用底层特征”计算，具体因子由 FactorLibrary 再次组合。
    """
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "code", "open", "high", "low", "close", "volume"]).copy()
    df = df.sort_values(["code", "date"]).copy()

    if "preclose" not in df.columns:
        df["preclose"] = np.nan

    df["preclose"] = df["preclose"].where(df["preclose"].notna(), df.groupby("code")["close"].shift(1))
    if "turn" not in df.columns:
        df["turn"] = np.nan

    df = df.merge(minute_daily_feat, on=["code", "date"], how="left")

    g = df.groupby("code")

    df["ret_1d"] = g["close"].pct_change(1)
    df["ret_3d"] = g["close"].pct_change(3)
    df["ret_5d"] = g["close"].pct_change(5)
    df["ret_10d"] = g["close"].pct_change(10)
    df["ret_20d"] = g["close"].pct_change(20)
    df["vol_chg_1d"] = g["volume"].pct_change(1)

    df["ma5"] = _rolling_mean(g["close"], 5)
    df["ma10"] = _rolling_mean(g["close"], 10)
    df["ma20"] = _rolling_mean(g["close"], 20)
    df["roll_high_20"] = g["high"].transform(lambda s: s.rolling(20, min_periods=20).max())
    df["roll_low_20"] = g["low"].transform(lambda s: s.rolling(20, min_periods=20).min())

    df["vol_ma5"] = _rolling_mean(g["volume"], 5)
    df["vol_ma20"] = _rolling_mean(g["volume"], 20)
    df["amount_ma20"] = _rolling_mean(g["amount"], 20)
    df["turn_ma5"] = _rolling_mean(g["turn"], 5)

    df["ma_gap_5"] = df["close"] / (df["ma5"] + EPS) - 1.0
    df["ma_gap_10"] = df["close"] / (df["ma10"] + EPS) - 1.0
    df["ma_gap_20"] = df["close"] / (df["ma20"] + EPS) - 1.0
    df["ma_cross_5_20"] = df["ma5"] / (df["ma20"] + EPS) - 1.0
    df["breakout_20"] = df["close"] / (df["roll_high_20"] + EPS) - 1.0

    df["vol_ratio_5"] = df["volume"] / (df["vol_ma5"] + EPS)
    df["vol_ratio_20"] = df["volume"] / (df["vol_ma20"] + EPS)
    df["amount_ratio_20"] = df["amount"] / (df["amount_ma20"] + EPS)
    df["turn_ratio_5"] = df["turn"] / (df["turn_ma5"] + EPS)

    df["intraday_range"] = (df["high"] - df["low"]) / (df["preclose"] + EPS)
    hl = df["high"] - df["low"]
    df["body_ratio"] = (df["close"] - df["open"]) / (hl + EPS)
    df["close_pos"] = (df["close"] - df["low"]) / (hl + EPS)

    prev_close = g["close"].shift(1)
    tr = np.maximum.reduce(
        [
            (df["high"] - df["low"]).to_numpy(),
            (df["high"] - prev_close).abs().to_numpy(),
            (df["low"] - prev_close).abs().to_numpy(),
        ]
    )
    df["tr"] = tr
    df["atr14"] = g["tr"].transform(lambda s: s.rolling(14, min_periods=14).mean())
    df["atr_norm_14"] = df["atr14"] / (df["close"] + EPS)

    df["realized_vol_20"] = _rolling_std(g["ret_1d"], 20)
    df["ret_neg_1d"] = df["ret_1d"].where(df["ret_1d"] < 0.0, 0.0)
    df["downside_vol_20"] = df.groupby("code")["ret_neg_1d"].transform(lambda s: s.rolling(20, min_periods=20).std())
    df["downside_vol_ratio_20"] = df["downside_vol_20"] / (df["realized_vol_20"] + EPS)

    delta = g["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.groupby(df["code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    avg_loss = loss.groupby(df["code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    rs = avg_gain / (avg_loss + EPS)
    df["rsi14"] = 100.0 - (100.0 / (1.0 + rs))

    df["amihud_1d"] = np.abs(df["ret_1d"]) / (df["amount"] + EPS)
    df["amihud_20"] = df.groupby("code")["amihud_1d"].transform(lambda s: s.rolling(20, min_periods=20).mean())

    ret_vol_corr = (
        df.groupby("code", group_keys=False)
        .apply(lambda x: x["ret_1d"].rolling(20, min_periods=20).corr(x["vol_chg_1d"]))
        .reset_index(level=0, drop=True)
    )
    df["ret_vol_corr_20"] = ret_vol_corr

    df["close_to_vwap_day"] = df["close"] / (df["vwap_day"] + EPS) - 1.0
    df["overnight_gap"] = df["open"] / (df["preclose"] + EPS) - 1.0
    df["px_daily_close"] = df["close"]

    # 组合优化用的 Barra 风格代理与状态变量（均为当日可观测量）。
    df["barra_size_proxy"] = np.log(df["amount_ma20"].clip(lower=0.0) + 1.0)
    df["barra_momentum_proxy"] = df["ret_20d"]
    df["barra_volatility_proxy"] = df["realized_vol_20"]
    df["barra_liquidity_proxy"] = -df["amihud_20"]
    df["barra_beta_proxy"] = df["ret_vol_corr_20"]
    df["crowding_proxy_raw"] = (
        0.45 * df["vol_ratio_20"].abs()
        + 0.35 * df["turn_ratio_5"].abs()
        + 0.20 * df["ret_vol_corr_20"].abs()
    )
    df["board_type"] = df["code"].astype(str).map(infer_board_type)
    df["industry_bucket"] = df["code"].astype(str).map(infer_industry_bucket)

    return df

def register_default_factors(library: FactorLibrary) -> None:
    """注册内置因子库。

    分类包含：趋势、反转、流动性、波动率、日内微结构等。
    """
    library.register("mom_5", "trend", "5日动量", lambda d: d["ret_5d"])
    library.register("mom_10", "trend", "10日动量", lambda d: d["ret_10d"])
    library.register("mom_20", "trend", "20日动量", lambda d: d["ret_20d"])
    library.register("rev_1", "reversal", "1日反转（-ret_1d）", lambda d: -d["ret_1d"])
    library.register("rev_3", "reversal", "3日反转（-ret_3d）", lambda d: -d["ret_3d"])

    library.register("ma_gap_5", "trend", "close/MA5-1", lambda d: d["ma_gap_5"])
    library.register("ma_gap_10", "trend", "close/MA10-1", lambda d: d["ma_gap_10"])
    library.register("ma_gap_20", "trend", "close/MA20-1", lambda d: d["ma_gap_20"])
    library.register("ma_cross_5_20", "trend", "MA5/MA20-1", lambda d: d["ma_cross_5_20"])
    library.register("breakout_20", "trend", "close/20日最高价-1", lambda d: d["breakout_20"])

    library.register("vol_ratio_5", "liquidity", "volume/5日均量", lambda d: d["vol_ratio_5"])
    library.register("vol_ratio_20", "liquidity", "volume/20日均量", lambda d: d["vol_ratio_20"])
    library.register("amount_ratio_20", "liquidity", "amount/20日均成交额", lambda d: d["amount_ratio_20"])
    library.register("turn_ratio_5", "liquidity", "换手率/5日均换手", lambda d: d["turn_ratio_5"])
    library.register("amihud_20", "liquidity", "Amihud流动性（取负）", lambda d: -d["amihud_20"])

    library.register("ret_vol_corr_20", "flow", "收益与量变20日相关", lambda d: d["ret_vol_corr_20"])

    library.register("intraday_range", "volatility", "日内振幅（取负）", lambda d: -d["intraday_range"])
    library.register("body_ratio", "price_action", "实体占振幅比", lambda d: d["body_ratio"])
    library.register("close_pos", "price_action", "收盘在高低区间位置", lambda d: d["close_pos"])
    library.register("atr_norm_14", "volatility", "ATR14/close（取负）", lambda d: -d["atr_norm_14"])
    library.register("realized_vol_20", "volatility", "20日收益波动率（取负）", lambda d: -d["realized_vol_20"])
    library.register("downside_vol_ratio_20", "volatility", "下行波动占比（取负）", lambda d: -d["downside_vol_ratio_20"])
    library.register("rsi14", "oscillator", "RSI14", lambda d: d["rsi14"] / 100.0)

    library.register("close_to_vwap_day", "intraday_micro", "收盘相对日内VWAP偏离", lambda d: d["close_to_vwap_day"])
    library.register("morning_momentum_30m", "intraday_micro", "开盘前30分钟动量", lambda d: d["morning_momentum_30m"])
    library.register("last30_momentum", "intraday_micro", "尾盘30分钟动量", lambda d: d["last30_momentum"])
    library.register("vwap30_vs_day", "intraday_micro", "前30分钟VWAP相对日VWAP", lambda d: d["vwap30_vs_day"])
    library.register("minute_realized_vol_5m", "intraday_micro", "5分钟收益实现波动（取负）", lambda d: -d["minute_realized_vol_5m"])
    library.register("minute_up_ratio_5m", "intraday_micro", "5分钟K线上涨占比", lambda d: d["minute_up_ratio_5m"])
    library.register("minute_ret_skew_5m", "intraday_micro", "5分钟收益偏度", lambda d: d["minute_ret_skew_5m"])
    library.register("minute_ret_kurt_5m", "intraday_micro", "5分钟收益峰度（取负）", lambda d: -d["minute_ret_kurt_5m"])
    library.register("signed_vol_imbalance_5m", "intraday_micro", "符号成交量失衡", lambda d: d["signed_vol_imbalance_5m"])
    library.register("jump_ratio_5m", "intraday_micro", "跳跃强度（取负）", lambda d: -d["jump_ratio_5m"])
    library.register("open_to_close_intraday", "intraday_micro", "日内开收动量", lambda d: d["open_to_close_intraday"])
    library.register("overnight_gap", "overnight", "隔夜跳空", lambda d: d["overnight_gap"])


def load_custom_factor_module(library: FactorLibrary, module_path: str) -> None:
    """动态加载用户自定义因子模块。

    模块需暴露 register_factors(library) 接口。
    """
    path = Path(module_path)
    if not path.exists():
        raise FileNotFoundError(f"自定义因子模块不存在: {path}")

    spec = importlib.util.spec_from_file_location("custom_factor_module", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "register_factors"):
        raise RuntimeError("自定义模块需提供 register_factors(library) 函数。")
    module.register_factors(library)


def resolve_selected_factors(library: FactorLibrary, factor_list_arg: str) -> List[str]:
    """解析最终启用因子集合，并校验是否全部已注册。"""
    if factor_list_arg.strip():
        selected = [x.strip() for x in factor_list_arg.split(",") if x.strip()]
    else:
        selected = DEFAULT_FACTOR_SET.copy()

    available = set(library.names())
    missing = [f for f in selected if f not in available]
    if missing:
        raise ValueError(f"因子不存在: {missing}")
    return selected


def compute_factor_panel(base_df: pd.DataFrame, library: FactorLibrary, selected_factors: List[str]) -> pd.DataFrame:
    """按所选因子逐一计算并拼接到因子面板。"""
    panel = base_df.copy()
    for fac in selected_factors:
        values = library.get(fac).func(panel)
        if isinstance(values, pd.Series):
            panel[fac] = pd.to_numeric(values, errors="coerce")
        else:
            panel[fac] = pd.to_numeric(pd.Series(values, index=panel.index), errors="coerce")
    return panel


def add_label(
    panel: pd.DataFrame,
    horizon: int,
    price_table: pd.DataFrame,
    execution_scheme: str,
) -> pd.DataFrame:
    """构建监督学习标签，并与回测使用同一执行价方案。

    时间对齐规则：
    - 信号日: T
    - 入场日: T+1
    - 出场日: T+1+horizon
    - 标签收益: exit_label_price / entry_label_price - 1
    """
    out = panel.copy()
    g = out.groupby("code")

    out["entry_date"] = g["date"].shift(-1)
    out["exit_date"] = g["date"].shift(-(horizon + 1))

    conf = EXECUTION_SCHEMES[execution_scheme]
    buy_col = conf["buy_col"]
    sell_col = conf["sell_col"]

    buy = price_table[["code", "date", buy_col]].rename(
        columns={"date": "entry_date", buy_col: "entry_label_price"}
    )
    sell = price_table[["code", "date", sell_col]].rename(
        columns={"date": "exit_date", sell_col: "exit_label_price"}
    )

    out = out.merge(buy, on=["code", "entry_date"], how="left")
    out = out.merge(sell, on=["code", "exit_date"], how="left")

    out["future_ret_n"] = out["exit_label_price"] / (out["entry_label_price"] + EPS) - 1.0
    out["target_up"] = (out["future_ret_n"] > 0).astype(int)
    out["target_date"] = out["exit_date"]
    return out


def split_train_test(
    panel: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按信号日期切分训练/测试，并限制 target_date 不越界。

    这是防止时间穿越的关键步骤之一。
    """
    valid = panel.dropna(subset=["entry_date", "exit_date", "target_date", "future_ret_n"]).copy()
    valid = valid.replace([np.inf, -np.inf], np.nan)

    # 训练集要求：信号日在训练窗口内，且标签终点也在训练窗口内。
    train_mask = (
        (valid["date"] >= train_start)
        & (valid["date"] <= train_end)
        & (valid["target_date"] <= train_end)
    )
    # 测试集同理，避免测试样本标签穿越到窗口外。
    test_mask = (
        (valid["date"] >= test_start)
        & (valid["date"] <= test_end)
        & (valid["target_date"] <= test_end)
    )

    return valid[train_mask].copy(), valid[test_mask].copy()


def train_decision_tree(train_df: pd.DataFrame, factors: List[str], max_depth: int, min_samples_leaf: int) -> Tuple[DecisionTreeClassifier, pd.Series]:
    """训练决策树分类器，并返回训练阶段缺失值填充中位数。"""
    X_train = train_df[factors].replace([np.inf, -np.inf], np.nan)
    fill_values = X_train.median(numeric_only=True)
    X_train = X_train.fillna(fill_values)
    y_train = train_df["target_up"].astype(int)

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model, fill_values


def predict_up_prob(model: DecisionTreeClassifier, X: pd.DataFrame) -> np.ndarray:
    """稳健地提取“上涨类别(1)”概率，兼容单类别训练边界情形。"""
    prob = model.predict_proba(X)
    if prob.shape[1] == 1:
        cls = int(model.classes_[0])
        return np.ones(len(X), dtype=float) if cls == 1 else np.zeros(len(X), dtype=float)

    class_to_idx = {int(c): i for i, c in enumerate(model.classes_)}
    if 1 not in class_to_idx:
        return np.zeros(len(X), dtype=float)
    return prob[:, class_to_idx[1]]


def evaluate_classifier(model: DecisionTreeClassifier, test_df: pd.DataFrame, factors: List[str], fill_values: pd.Series) -> Dict[str, float]:
    """计算分类指标（accuracy/precision/recall/auc）。"""
    X_test = test_df[factors].replace([np.inf, -np.inf], np.nan).fillna(fill_values)
    y_true = test_df["target_up"].astype(int)

    pred = model.predict(X_test)
    pred_prob = predict_up_prob(model, X_test)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }
    if y_true.nunique() == 2:
        metrics["auc"] = float(roc_auc_score(y_true, pred_prob))
    else:
        metrics["auc"] = float("nan")
    return metrics

def attach_execution_prices(df: pd.DataFrame, price_table: pd.DataFrame, buy_col: str, sell_col: str) -> pd.DataFrame:
    """将入场/出场日期映射到对应执行价格。"""
    buy = price_table[["code", "date", buy_col]].rename(columns={"date": "entry_date", buy_col: "entry_price"})
    sell = price_table[["code", "date", sell_col]].rename(columns={"date": "exit_date", sell_col: "exit_price"})

    out = df.merge(buy, on=["code", "entry_date"], how="left")
    out = out.merge(sell, on=["code", "exit_date"], how="left")
    return out


def calc_trade_return(entry_price: pd.Series, exit_price: pd.Series, fee_bps: float, slippage_bps: float) -> pd.Series:
    """按双边交易成本计算单笔净收益率。"""
    cost = (fee_bps + slippage_bps) / 10000.0
    net_ret = (exit_price * (1.0 - cost)) / (entry_price * (1.0 + cost) + EPS) - 1.0
    return net_ret


def max_drawdown(net_values: pd.Series) -> float:
    """计算净值序列最大回撤。"""
    if net_values.empty:
        return float("nan")
    running_max = net_values.cummax()
    dd = net_values / (running_max + EPS) - 1.0
    return float(dd.min())


def compute_return_stats(returns: pd.Series, horizon: int) -> Dict[str, float]:
    """计算收益统计摘要。

    包含：胜率、期望收益、年化收益/波动、Sharpe、Sortino、回撤、Calmar 等。
    """
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return {
            "periods": 0.0,
            "win_rate": float("nan"),
            "avg_return": float("nan"),
            "cum_return": float("nan"),
            "annualized_return": float("nan"),
            "annualized_vol": float("nan"),
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "max_drawdown": float("nan"),
            "calmar": float("nan"),
            "profit_factor": float("nan"),
            "avg_win": float("nan"),
            "avg_loss": float("nan"),
        }

    periods = len(r)
    periods_per_year = TRADING_DAYS_PER_YEAR / max(horizon, 1)

    cum_return = float((1.0 + r).prod() - 1.0)
    annualized_return = float((1.0 + cum_return) ** (periods_per_year / periods) - 1.0)

    std = float(r.std(ddof=1)) if periods > 1 else float("nan")
    annualized_vol = float(std * np.sqrt(periods_per_year)) if periods > 1 else float("nan")

    if periods > 1 and std > 0:
        sharpe = float((r.mean() / std) * np.sqrt(periods_per_year))
    else:
        sharpe = float("nan")

    downside = r[r < 0]
    down_std = float(downside.std(ddof=1)) if len(downside) > 1 else float("nan")
    if periods > 1 and down_std and down_std > 0:
        sortino = float((r.mean() / down_std) * np.sqrt(periods_per_year))
    else:
        sortino = float("nan")

    net = (1.0 + r).cumprod()
    mdd = max_drawdown(net)
    calmar = float(annualized_return / abs(mdd)) if pd.notna(mdd) and mdd < 0 else float("nan")

    pos = r[r > 0]
    neg = r[r < 0]
    profit_factor = float(pos.sum() / abs(neg.sum())) if abs(neg.sum()) > EPS else float("nan")

    return {
        "periods": float(periods),
        "win_rate": float((r > 0).mean()),
        "avg_return": float(r.mean()),
        "cum_return": cum_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "profit_factor": profit_factor,
        "avg_win": float(pos.mean()) if not pos.empty else float("nan"),
        "avg_loss": float(neg.mean()) if not neg.empty else float("nan"),
    }


def calc_ic_for_column(df: pd.DataFrame, score_col: str, ret_col: str, min_cross_section: int) -> pd.DataFrame:
    """按交易日横截面计算 IC 与 RankIC 时间序列。"""
    records: List[Dict[str, object]] = []
    for dt, g in df.groupby("date"):
        sub = g[[score_col, ret_col]].replace([np.inf, -np.inf], np.nan).dropna()
        n = len(sub)
        if n < min_cross_section:
            continue
        if sub[score_col].nunique() < 2 or sub[ret_col].nunique() < 2:
            continue

        ic = sub[score_col].corr(sub[ret_col], method="pearson")
        rank_ic = sub[score_col].corr(sub[ret_col], method="spearman")
        records.append({"date": pd.Timestamp(dt), "ic": float(ic), "rank_ic": float(rank_ic), "n": int(n)})

    return pd.DataFrame(records).sort_values("date") if records else pd.DataFrame(columns=["date", "ic", "rank_ic", "n"])


def summarize_ic(ic_ts: pd.DataFrame) -> Dict[str, float]:
    """汇总 IC 序列统计：均值、标准差、IR、正值占比、t 统计。"""
    if ic_ts.empty:
        return {
            "obs": 0.0,
            "ic_mean": float("nan"),
            "ic_std": float("nan"),
            "ic_ir": float("nan"),
            "ic_positive_ratio": float("nan"),
            "ic_tstat": float("nan"),
            "rank_ic_mean": float("nan"),
            "rank_ic_std": float("nan"),
            "rank_ic_ir": float("nan"),
            "rank_ic_positive_ratio": float("nan"),
            "rank_ic_tstat": float("nan"),
        }

    n = len(ic_ts)
    ic_mean = float(ic_ts["ic"].mean())
    ic_std = float(ic_ts["ic"].std(ddof=1)) if n > 1 else float("nan")
    rank_mean = float(ic_ts["rank_ic"].mean())
    rank_std = float(ic_ts["rank_ic"].std(ddof=1)) if n > 1 else float("nan")

    ic_ir = float(ic_mean / ic_std * np.sqrt(TRADING_DAYS_PER_YEAR)) if pd.notna(ic_std) and ic_std > 0 else float("nan")
    rank_ir = float(rank_mean / rank_std * np.sqrt(TRADING_DAYS_PER_YEAR)) if pd.notna(rank_std) and rank_std > 0 else float("nan")

    ic_tstat = float(ic_mean / (ic_std / np.sqrt(n))) if pd.notna(ic_std) and ic_std > 0 else float("nan")
    rank_tstat = float(rank_mean / (rank_std / np.sqrt(n))) if pd.notna(rank_std) and rank_std > 0 else float("nan")

    return {
        "obs": float(n),
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "ic_positive_ratio": float((ic_ts["ic"] > 0).mean()),
        "ic_tstat": ic_tstat,
        "rank_ic_mean": rank_mean,
        "rank_ic_std": rank_std,
        "rank_ic_ir": rank_ir,
        "rank_ic_positive_ratio": float((ic_ts["rank_ic"] > 0).mean()),
        "rank_ic_tstat": rank_tstat,
    }

def compute_factor_ic_statistics(
    df: pd.DataFrame,
    factor_cols: List[str],
    ret_col: str,
    min_cross_section: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """批量计算每个因子的 IC/RankIC 统计，并输出汇总表与时序表。"""
    summary_rows: List[Dict[str, object]] = []
    ts_rows: List[pd.DataFrame] = []

    for fac in factor_cols:
        ic_ts = calc_ic_for_column(df, fac, ret_col, min_cross_section=min_cross_section)
        summary = summarize_ic(ic_ts)
        summary["factor"] = fac
        summary_rows.append(summary)

        if not ic_ts.empty:
            ic_tmp = ic_ts.copy()
            ic_tmp["factor"] = fac
            ts_rows.append(ic_tmp)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df[
            [
                "factor",
                "obs",
                "ic_mean",
                "ic_std",
                "ic_ir",
                "ic_positive_ratio",
                "ic_tstat",
                "rank_ic_mean",
                "rank_ic_std",
                "rank_ic_ir",
                "rank_ic_positive_ratio",
                "rank_ic_tstat",
            ]
        ].sort_values("rank_ic_ir", ascending=False)

    ts_df = pd.concat(ts_rows, ignore_index=True) if ts_rows else pd.DataFrame(columns=["date", "ic", "rank_ic", "n", "factor"])
    return summary_df, ts_df


def compute_score_spread(df: pd.DataFrame, score_col: str, ret_col: str, quantiles: int = 5) -> Dict[str, float]:
    """按模型分数分层，计算 Q5-Q1 收益差。"""
    spreads: List[float] = []

    for _, g in df.groupby("date"):
        sub = g[[score_col, ret_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < quantiles * 8:
            continue
        try:
            bucket = pd.qcut(sub[score_col], q=quantiles, labels=False, duplicates="drop")
        except Exception:
            continue
        if bucket.nunique() < 2:
            continue

        top = sub.loc[bucket == bucket.max(), ret_col].mean()
        bot = sub.loc[bucket == bucket.min(), ret_col].mean()
        spreads.append(float(top - bot))

    if not spreads:
        return {"obs": 0.0, "spread_mean": float("nan"), "spread_std": float("nan"), "spread_ir": float("nan")}

    s = pd.Series(spreads)
    std = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
    ir = float(s.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR)) if pd.notna(std) and std > 0 else float("nan")
    return {
        "obs": float(len(s)),
        "spread_mean": float(s.mean()),
        "spread_std": std,
        "spread_ir": ir,
    }



def pick_named_file(folder: Path, stem: str, file_format: str = "auto") -> Optional[Path]:
    """按文件名主干读取指数行情文件。"""
    if file_format == "parquet":
        candidates = [folder / f"{stem}.parquet"]
    elif file_format == "csv":
        candidates = [folder / f"{stem}.csv"]
    else:
        candidates = [folder / f"{stem}.parquet", folder / f"{stem}.csv"]

    for p in candidates:
        if p.exists():
            return p
    return None


def load_index_benchmark_data(
    index_root: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    file_format: str = "auto",
) -> Dict[str, pd.DataFrame]:
    """读取沪深300/中证500/中证1000指数收盘价。"""
    mapping = {
        "hs300": "hs300_price",
        "zz500": "zz500_price",
        "zz1000": "zz1000_price",
    }
    idx_data: Dict[str, pd.DataFrame] = {}

    for name, stem in mapping.items():
        fp = pick_named_file(index_root, stem, file_format=file_format)
        if fp is None:
            idx_data[name] = pd.DataFrame(columns=["date", "close"])
            continue

        df = _read_data_file(fp, usecols=["date", "close"], start_date=start_date, end_date=end_date)
        if df.empty:
            idx_data[name] = pd.DataFrame(columns=["date", "close"])
            continue

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date")
        idx_data[name] = df[["date", "close"]].copy()

    return idx_data


def lookup_index_period_return(index_df: pd.DataFrame, entry_date: pd.Timestamp, exit_date: pd.Timestamp) -> float:
    """计算指数在 [entry_date, exit_date] 区间的收益率。"""
    if index_df.empty or pd.isna(entry_date) or pd.isna(exit_date):
        return float("nan")

    s = index_df.sort_values("date")
    s_entry = s[s["date"] >= pd.Timestamp(entry_date)]
    s_exit = s[s["date"] >= pd.Timestamp(exit_date)]
    if s_entry.empty or s_exit.empty:
        return float("nan")

    entry_px = float(s_entry.iloc[0]["close"])
    exit_px = float(s_exit.iloc[0]["close"])
    return exit_px / (entry_px + EPS) - 1.0


def _to_numeric_series(df: pd.DataFrame, col: str, fill_value: float = 0.0) -> pd.Series:
    """安全读取数值列。"""
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            return s.fillna(float(s.median()))
    return pd.Series(np.full(len(df), fill_value, dtype=float), index=df.index)


def _build_style_matrix(selected_df: pd.DataFrame, universe_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """构建 Barra 风格代理暴露矩阵与目标暴露向量。"""
    style_cols = [
        "barra_size_proxy",
        "barra_momentum_proxy",
        "barra_volatility_proxy",
        "barra_liquidity_proxy",
        "barra_beta_proxy",
    ]

    if selected_df.empty:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float)

    u = universe_df.copy()
    u["code"] = u["code"].astype(str)
    sel_codes = selected_df["code"].astype(str)

    exposures: List[np.ndarray] = []
    targets: List[float] = []
    for col in style_cols:
        uni_series = _to_numeric_series(u, col, fill_value=0.0)
        uni_z = cross_section_zscore(uni_series)
        code_to_exp = dict(zip(u["code"], uni_z))
        exp = sel_codes.map(code_to_exp).fillna(0.0).astype(float).to_numpy()
        exposures.append(exp)
        targets.append(float(uni_z.mean()))

    style_matrix = np.column_stack(exposures) if exposures else np.zeros((len(selected_df), 0), dtype=float)
    target_vec = np.asarray(targets, dtype=float)
    return style_matrix, target_vec


def _build_industry_matrix(selected_df: pd.DataFrame, universe_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """构建行业暴露矩阵与行业基准权重。"""
    if selected_df.empty:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float)

    if "industry_bucket" in selected_df.columns:
        sel_ind = selected_df["industry_bucket"].astype(str).fillna("unknown")
    else:
        sel_ind = selected_df["code"].astype(str).map(infer_industry_bucket)

    if "industry_bucket" in universe_df.columns:
        uni_ind = universe_df["industry_bucket"].astype(str).fillna("unknown")
    else:
        uni_ind = universe_df["code"].astype(str).map(infer_industry_bucket)

    industries = sorted(set(sel_ind.tolist()) | set(uni_ind.tolist()))
    if not industries:
        return np.zeros((len(selected_df), 0), dtype=float), np.zeros(0, dtype=float)

    mat = np.column_stack([(sel_ind == ind).astype(float).to_numpy() for ind in industries]).astype(float)
    bench = uni_ind.value_counts(normalize=True)
    target = np.asarray([float(bench.get(ind, 0.0)) for ind in industries], dtype=float)
    return mat, target


def _compute_market_state(universe_df: pd.DataFrame, state_tracker: Dict[str, List[float]]) -> Dict[str, float]:
    """计算当前组合状态并基于历史做标准化。"""
    market_vol = float(_to_numeric_series(universe_df, "realized_vol_20", fill_value=0.0).median())
    crowd_raw = (
        0.45 * _to_numeric_series(universe_df, "vol_ratio_20", fill_value=0.0).abs()
        + 0.35 * _to_numeric_series(universe_df, "turn_ratio_5", fill_value=0.0).abs()
        + 0.20 * _to_numeric_series(universe_df, "ret_vol_corr_20", fill_value=0.0).abs()
    )
    crowd_level = float(crowd_raw.median())

    style_dispersion = float(
        np.nanmean(
            [
                cross_section_zscore(_to_numeric_series(universe_df, "barra_size_proxy", fill_value=0.0)).std(ddof=0),
                cross_section_zscore(_to_numeric_series(universe_df, "barra_momentum_proxy", fill_value=0.0)).std(ddof=0),
                cross_section_zscore(_to_numeric_series(universe_df, "barra_volatility_proxy", fill_value=0.0)).std(ddof=0),
            ]
        )
    )

    vol_z = robust_history_zscore(market_vol, state_tracker.get("market_vol", []))
    crowd_z = robust_history_zscore(crowd_level, state_tracker.get("crowding", []))
    style_z = robust_history_zscore(style_dispersion, state_tracker.get("style_disp", []))

    return {
        "market_vol": market_vol,
        "crowding": crowd_level,
        "style_dispersion": style_dispersion,
        "vol_z": vol_z,
        "crowding_z": crowd_z,
        "style_z": style_z,
    }


def optimize_portfolio_weights(
    day_pick: pd.DataFrame,
    day_universe: pd.DataFrame,
    prev_weights: Dict[str, float],
    opt_cfg: PortfolioOptimizationConfig,
    state_tracker: Dict[str, List[float]],
    fee_bps: float,
    slippage_bps: float,
) -> Tuple[pd.Series, Dict[str, float]]:
    """状态感知组合优化：收益、风格、行业、流动性、交易成本联合优化。"""
    n = len(day_pick)
    if n == 0:
        return pd.Series(dtype=float), {}
    if n == 1:
        code = str(day_pick.iloc[0]["code"])
        return pd.Series({code: 1.0}), {"opt_iterations": 0.0, "optimizer_fallback": 0.0}

    pick = day_pick.copy().reset_index(drop=True)
    uni = day_universe.copy()
    pick_codes = pick["code"].astype(str).tolist()
    pick_code_set = set(pick_codes)

    market_state = _compute_market_state(uni, state_tracker)
    vol_z = market_state["vol_z"]
    crowd_z = market_state["crowding_z"]
    style_z = market_state["style_z"]

    expected_scale = float(np.clip(1.0 + 0.18 * style_z - 0.25 * max(vol_z, 0.0) - 0.18 * max(crowd_z, 0.0), 0.4, 1.8))
    risk_scale = float(np.clip(1.0 + 0.40 * max(vol_z, 0.0) + 0.25 * max(crowd_z, 0.0), 0.7, 2.8))
    style_scale = float(np.clip(1.0 + 0.20 * max(crowd_z, 0.0) + 0.15 * abs(style_z), 0.8, 2.4))
    industry_scale = float(np.clip(1.0 + 0.20 * abs(style_z), 0.8, 2.2))
    tc_scale = float(np.clip(1.0 + 0.30 * max(crowd_z, 0.0), 0.8, 2.5))

    alpha_prob = cross_section_zscore(_to_numeric_series(pick, "pred_prob_up", fill_value=0.5))
    alpha_mom = cross_section_zscore(_to_numeric_series(pick, "ret_20d", fill_value=0.0))
    alpha_micro = cross_section_zscore(_to_numeric_series(pick, "morning_momentum_30m", fill_value=0.0))
    mu = (0.65 * alpha_prob + 0.25 * alpha_mom + 0.10 * alpha_micro).to_numpy(dtype=float)

    vol_series = _to_numeric_series(pick, "realized_vol_20", fill_value=0.0).clip(lower=0.0)
    vol_zs = cross_section_zscore(vol_series).to_numpy(dtype=float)
    risk_diag = np.clip(1.0 + 0.5 * vol_zs, 0.2, 3.0)

    style_mat, style_target = _build_style_matrix(pick, uni)
    ind_mat, ind_target = _build_industry_matrix(pick, uni)

    crowding = cross_section_zscore(_to_numeric_series(pick, "crowding_proxy_raw", fill_value=0.0)).to_numpy(dtype=float)

    liq_amt = _to_numeric_series(pick, "amount_ma20", fill_value=0.0).clip(lower=0.0)
    if liq_amt.sum() <= EPS:
        liq_amt = _to_numeric_series(pick, "amount", fill_value=0.0).clip(lower=0.0)
    if liq_amt.sum() <= EPS:
        liq_share = np.full(n, 1.0 / n, dtype=float)
    else:
        liq_share = (liq_amt / (liq_amt.sum() + EPS)).to_numpy(dtype=float)

    effective_max_weight = float(np.clip(max(opt_cfg.max_weight, 1.0 / n + 1e-6), 0.01, 1.0))
    liq_caps = np.clip(liq_share * max(opt_cfg.liquidity_scale, 0.1), 0.01, effective_max_weight)
    caps = ensure_feasible_caps(liq_caps, target_sum=1.0, hard_cap=effective_max_weight)

    prev_vec = np.asarray([float(prev_weights.get(c, 0.0)) for c in pick_codes], dtype=float)
    prev_dropped = float(sum(v for c, v in prev_weights.items() if c not in pick_code_set and v > 0))
    prev_total = float(prev_vec.sum())
    if prev_total <= EPS:
        w = np.full(n, 1.0 / n, dtype=float)
    else:
        carry = max(0.0, 1.0 - prev_dropped)
        if carry > EPS:
            w = prev_vec / (prev_total + EPS) * carry
            if carry < 1.0:
                w += (1.0 - carry) / n
        else:
            w = np.full(n, 1.0 / n, dtype=float)
    w = project_to_capped_simplex(w, caps)

    ret_coef = max(opt_cfg.expected_return_weight * expected_scale, 0.0)
    risk_coef = max(opt_cfg.risk_aversion * risk_scale, 0.0)
    style_coef = max(opt_cfg.style_penalty * style_scale, 0.0)
    industry_coef = max(opt_cfg.industry_penalty * industry_scale, 0.0)
    crowd_coef = max(opt_cfg.crowding_penalty * (1.0 + 0.25 * max(crowd_z, 0.0)), 0.0)
    tc_bps = (fee_bps + slippage_bps) / 10000.0
    tc_coef = max(opt_cfg.transaction_cost_penalty * tc_scale * tc_bps * 100.0, 0.0)
    step = max(opt_cfg.step_size, 1e-4)

    converged = 0.0
    iterations = 0
    for i in range(max(opt_cfg.max_iter, 1)):
        iterations = i + 1
        grad = ret_coef * mu - 2.0 * risk_coef * risk_diag * w

        if style_mat.size > 0:
            style_gap = style_mat.T @ w - style_target
            grad = grad - 2.0 * style_coef * (style_mat @ style_gap)

        if ind_mat.size > 0:
            ind_gap = ind_mat.T @ w - ind_target
            grad = grad - 2.0 * industry_coef * (ind_mat @ ind_gap)

        crowd_exp = float(np.dot(crowding, w))
        grad = grad - 2.0 * crowd_coef * crowding * crowd_exp

        if prev_vec.size == w.size:
            grad = grad - tc_coef * np.sign(w - prev_vec)

        w_next = project_to_capped_simplex(w + step * grad, caps)

        total_turnover = prev_dropped + float(np.abs(w_next - prev_vec).sum())
        if total_turnover > opt_cfg.max_turnover + 1e-10:
            allowed = max(opt_cfg.max_turnover - prev_dropped, 0.0)
            cur_turnover = float(np.abs(w_next - prev_vec).sum())
            if cur_turnover > EPS and allowed < cur_turnover:
                blend = allowed / (cur_turnover + EPS)
                w_next = prev_vec + blend * (w_next - prev_vec)
                w_next = project_to_capped_simplex(w_next, caps)

        if float(np.abs(w_next - w).sum()) < opt_cfg.tolerance:
            w = w_next
            converged = 1.0
            break
        w = w_next

    style_exposure_dev = float(np.linalg.norm((style_mat.T @ w - style_target), ord=2)) if style_mat.size > 0 else 0.0
    industry_dev = float(np.linalg.norm((ind_mat.T @ w - ind_target), ord=2)) if ind_mat.size > 0 else 0.0
    crowd_exposure = float(np.dot(crowding, w))
    liquidity_utilization = float(np.max(w / (caps + EPS)))
    turnover = prev_dropped + float(np.abs(w - prev_vec).sum())
    exp_ret_score = float(np.dot(mu, w))

    state_tracker.setdefault("market_vol", []).append(market_state["market_vol"])
    state_tracker.setdefault("crowding", []).append(market_state["crowding"])
    state_tracker.setdefault("style_disp", []).append(market_state["style_dispersion"])
    for k in ("market_vol", "crowding", "style_disp"):
        if len(state_tracker[k]) > 240:
            state_tracker[k] = state_tracker[k][-240:]

    diagnostics = {
        "opt_iterations": float(iterations),
        "opt_converged": float(converged),
        "state_market_vol": market_state["market_vol"],
        "state_crowding": market_state["crowding"],
        "state_style_dispersion": market_state["style_dispersion"],
        "state_vol_z": vol_z,
        "state_crowding_z": crowd_z,
        "state_style_z": style_z,
        "dynamic_expected_scale": expected_scale,
        "dynamic_risk_scale": risk_scale,
        "dynamic_style_scale": style_scale,
        "dynamic_industry_scale": industry_scale,
        "dynamic_tc_scale": tc_scale,
        "opt_turnover": turnover,
        "opt_expected_ret_score": exp_ret_score,
        "opt_style_exposure_dev": style_exposure_dev,
        "opt_industry_dev": industry_dev,
        "opt_crowding_exposure": crowd_exposure,
        "opt_liquidity_utilization": liquidity_utilization,
        "opt_effective_max_weight": effective_max_weight,
    }
    return pd.Series(w, index=pick_codes, dtype=float), diagnostics


def run_backtest(
    model: DecisionTreeClassifier,
    test_df: pd.DataFrame,
    factors: List[str],
    fill_values: pd.Series,
    price_table: pd.DataFrame,
    index_benchmarks: Dict[str, pd.DataFrame],
    horizon: int,
    top_k: int,
    long_threshold: float,
    execution_scheme: str,
    fee_bps: float,
    slippage_bps: float,
    portfolio_opt_cfg: PortfolioOptimizationConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object], pd.DataFrame]:
    """执行组合回测并同时计算多基准对比与超额曲线。"""
    conf = EXECUTION_SCHEMES[execution_scheme]
    buy_col = conf["buy_col"]
    sell_col = conf["sell_col"]

    pred_df = test_df.copy()
    X_test = pred_df[factors].replace([np.inf, -np.inf], np.nan).fillna(fill_values)
    pred_df["pred_prob_up"] = predict_up_prob(model, X_test)
    pred_df["pred_up"] = (pred_df["pred_prob_up"] >= long_threshold).astype(int)

    pred_df = attach_execution_prices(pred_df, price_table, buy_col=buy_col, sell_col=sell_col)
    pred_df["gross_trade_ret"] = pred_df["exit_price"] / (pred_df["entry_price"] + EPS) - 1.0
    pred_df["net_trade_ret"] = calc_trade_return(pred_df["entry_price"], pred_df["exit_price"], fee_bps=fee_bps, slippage_bps=slippage_bps)

    rebalance_dates = sorted(pred_df["date"].dropna().unique())
    trade_records: List[Dict[str, object]] = []
    position_records: List[Dict[str, object]] = []
    prev_weights: Dict[str, float] = {}
    state_tracker: Dict[str, List[float]] = {"market_vol": [], "crowding": [], "style_disp": []}

    def _portfolio_turnover(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
        all_codes = set(prev_w.keys()) | set(new_w.keys())
        if not all_codes:
            return 0.0
        return float(sum(abs(float(new_w.get(c, 0.0)) - float(prev_w.get(c, 0.0))) for c in all_codes))

    for idx in range(0, len(rebalance_dates), horizon):
        dt = rebalance_dates[idx]
        day_raw = pred_df[(pred_df["date"] == dt)].copy()
        day_all = day_raw.dropna(subset=["entry_price", "exit_price", "net_trade_ret", "future_ret_n"]).copy()

        if not day_all.empty:
            entry_dt = pd.Timestamp(day_all["entry_date"].mode().iloc[0])
            exit_dt = pd.Timestamp(day_all["exit_date"].mode().iloc[0])
        elif not day_raw.empty:
            entry_dt = pd.Timestamp(day_raw["entry_date"].dropna().iloc[0]) if day_raw["entry_date"].notna().any() else pd.NaT
            exit_dt = pd.Timestamp(day_raw["exit_date"].dropna().iloc[0]) if day_raw["exit_date"].notna().any() else pd.NaT
        else:
            entry_dt = pd.NaT
            exit_dt = pd.NaT

        benchmark_pool_ret = float(day_all["net_trade_ret"].mean()) if not day_all.empty else 0.0
        bench_size = int(len(day_all))

        benchmark_hs300_ret = lookup_index_period_return(index_benchmarks.get("hs300", pd.DataFrame()), entry_dt, exit_dt)
        benchmark_zz500_ret = lookup_index_period_return(index_benchmarks.get("zz500", pd.DataFrame()), entry_dt, exit_dt)
        benchmark_zz1000_ret = lookup_index_period_return(index_benchmarks.get("zz1000", pd.DataFrame()), entry_dt, exit_dt)

        day_pick = day_all[day_all["pred_prob_up"] >= long_threshold].copy()
        day_pick = day_pick.sort_values("pred_prob_up", ascending=False).head(top_k)
        opt_diag: Dict[str, float] = {}
        portfolio_turnover = float("nan")
        portfolio_concentration = float("nan")
        portfolio_entropy = float("nan")

        if day_pick.empty:
            strategy_ret = 0.0
            active_count = 0
            avg_prob = float("nan")
            new_weights: Dict[str, float] = {}
            portfolio_turnover = _portfolio_turnover(prev_weights, new_weights)
            prev_weights = new_weights
            if not day_all.empty:
                state_info = _compute_market_state(day_all, state_tracker)
                state_tracker.setdefault("market_vol", []).append(state_info["market_vol"])
                state_tracker.setdefault("crowding", []).append(state_info["crowding"])
                state_tracker.setdefault("style_disp", []).append(state_info["style_dispersion"])
                for k in ("market_vol", "crowding", "style_disp"):
                    if len(state_tracker[k]) > 240:
                        state_tracker[k] = state_tracker[k][-240:]
                opt_diag.update(
                    {
                        "state_market_vol": state_info["market_vol"],
                        "state_crowding": state_info["crowding"],
                        "state_style_dispersion": state_info["style_dispersion"],
                        "state_vol_z": state_info["vol_z"],
                        "state_crowding_z": state_info["crowding_z"],
                        "state_style_z": state_info["style_z"],
                    }
                )
        else:
            if portfolio_opt_cfg.mode == "dynamic_opt":
                weight_series, opt_diag = optimize_portfolio_weights(
                    day_pick=day_pick,
                    day_universe=day_all,
                    prev_weights=prev_weights,
                    opt_cfg=portfolio_opt_cfg,
                    state_tracker=state_tracker,
                    fee_bps=fee_bps,
                    slippage_bps=slippage_bps,
                )
                if weight_series.empty or weight_series.sum() <= EPS:
                    weight_series = pd.Series(np.full(len(day_pick), 1.0 / len(day_pick)), index=day_pick["code"].astype(str))
                    opt_diag["optimizer_fallback"] = 1.0
            else:
                if not day_all.empty:
                    state_info = _compute_market_state(day_all, state_tracker)
                    state_tracker.setdefault("market_vol", []).append(state_info["market_vol"])
                    state_tracker.setdefault("crowding", []).append(state_info["crowding"])
                    state_tracker.setdefault("style_disp", []).append(state_info["style_dispersion"])
                    for k in ("market_vol", "crowding", "style_disp"):
                        if len(state_tracker[k]) > 240:
                            state_tracker[k] = state_tracker[k][-240:]
                    opt_diag.update(
                        {
                            "state_market_vol": state_info["market_vol"],
                            "state_crowding": state_info["crowding"],
                            "state_style_dispersion": state_info["style_dispersion"],
                            "state_vol_z": state_info["vol_z"],
                            "state_crowding_z": state_info["crowding_z"],
                            "state_style_z": state_info["style_z"],
                        }
                    )
                weight_series = pd.Series(np.full(len(day_pick), 1.0 / len(day_pick)), index=day_pick["code"].astype(str))

            day_pick = day_pick.copy()
            day_pick["weight"] = day_pick["code"].astype(str).map(weight_series).fillna(0.0)
            sum_w = float(day_pick["weight"].sum())
            if sum_w <= EPS:
                day_pick["weight"] = 1.0 / len(day_pick)
            else:
                day_pick["weight"] = day_pick["weight"] / (sum_w + EPS)

            strategy_ret = float((day_pick["net_trade_ret"] * day_pick["weight"]).sum())
            active_count = int((day_pick["weight"] > 1e-6).sum())
            avg_prob = float((day_pick["pred_prob_up"] * day_pick["weight"]).sum())
            portfolio_concentration = float(np.square(day_pick["weight"]).sum())
            portfolio_entropy = float(-(day_pick["weight"] * np.log(day_pick["weight"] + EPS)).sum())

            new_weights = {
                str(code): float(w)
                for code, w in day_pick[["code", "weight"]].itertuples(index=False, name=None)
                if w > 1e-8
            }
            portfolio_turnover = float(opt_diag.get("opt_turnover", _portfolio_turnover(prev_weights, new_weights)))
            prev_weights = new_weights

            for _, r in day_pick.iterrows():
                position_records.append(
                    {
                        "signal_date": pd.Timestamp(r["date"]),
                        "code": r["code"],
                        "entry_date": pd.Timestamp(r["entry_date"]),
                        "exit_date": pd.Timestamp(r["exit_date"]),
                        "entry_price": float(r["entry_price"]),
                        "exit_price": float(r["exit_price"]),
                        "pred_prob_up": float(r["pred_prob_up"]),
                        "future_ret_ref": float(r["future_ret_n"]),
                        "gross_trade_ret": float(r["gross_trade_ret"]),
                        "net_trade_ret": float(r["net_trade_ret"]),
                        "weight": float(r.get("weight", np.nan)),
                        "weighting_mode": portfolio_opt_cfg.mode,
                        "industry_bucket": str(r.get("industry_bucket", "")),
                        "board_type": str(r.get("board_type", "")),
                    }
                )

        trade_records.append(
            {
                "trade_date": pd.Timestamp(dt),
                "entry_date": entry_dt,
                "exit_date": exit_dt,
                "hold_n_days": int(horizon),
                "active_stocks": active_count,
                "pool_size": bench_size,
                "avg_pred_prob": avg_prob,
                "strategy_ret": strategy_ret,
                "benchmark_pool_ret": benchmark_pool_ret,
                "benchmark_hs300_ret": benchmark_hs300_ret,
                "benchmark_zz500_ret": benchmark_zz500_ret,
                "benchmark_zz1000_ret": benchmark_zz1000_ret,
                "weighting_mode": portfolio_opt_cfg.mode,
                "portfolio_turnover": portfolio_turnover,
                "portfolio_concentration": portfolio_concentration,
                "portfolio_weight_entropy": portfolio_entropy,
                "state_market_vol": float(opt_diag.get("state_market_vol", np.nan)),
                "state_crowding": float(opt_diag.get("state_crowding", np.nan)),
                "state_style_dispersion": float(opt_diag.get("state_style_dispersion", np.nan)),
                "state_vol_z": float(opt_diag.get("state_vol_z", np.nan)),
                "state_crowding_z": float(opt_diag.get("state_crowding_z", np.nan)),
                "state_style_z": float(opt_diag.get("state_style_z", np.nan)),
                "dynamic_expected_scale": float(opt_diag.get("dynamic_expected_scale", np.nan)),
                "dynamic_risk_scale": float(opt_diag.get("dynamic_risk_scale", np.nan)),
                "dynamic_style_scale": float(opt_diag.get("dynamic_style_scale", np.nan)),
                "dynamic_industry_scale": float(opt_diag.get("dynamic_industry_scale", np.nan)),
                "dynamic_tc_scale": float(opt_diag.get("dynamic_tc_scale", np.nan)),
                "opt_iterations": float(opt_diag.get("opt_iterations", np.nan)),
                "opt_converged": float(opt_diag.get("opt_converged", np.nan)),
                "opt_expected_ret_score": float(opt_diag.get("opt_expected_ret_score", np.nan)),
                "opt_style_exposure_dev": float(opt_diag.get("opt_style_exposure_dev", np.nan)),
                "opt_industry_dev": float(opt_diag.get("opt_industry_dev", np.nan)),
                "opt_crowding_exposure": float(opt_diag.get("opt_crowding_exposure", np.nan)),
                "opt_liquidity_utilization": float(opt_diag.get("opt_liquidity_utilization", np.nan)),
                "opt_effective_max_weight": float(opt_diag.get("opt_effective_max_weight", np.nan)),
                "opt_fallback": float(opt_diag.get("optimizer_fallback", 0.0)),
            }
        )

    trades_df = pd.DataFrame(trade_records)
    positions_df = pd.DataFrame(position_records)

    if trades_df.empty:
        empty_curve = pd.DataFrame()
        summary = {
            "execution_scheme": execution_scheme,
            "execution_description": conf["description"],
            "portfolio_weighting_mode": portfolio_opt_cfg.mode,
            "strategy": compute_return_stats(pd.Series(dtype=float), horizon=horizon),
            "benchmark": compute_return_stats(pd.Series(dtype=float), horizon=horizon),
            "excess": compute_return_stats(pd.Series(dtype=float), horizon=horizon),
            "benchmark_pool": compute_return_stats(pd.Series(dtype=float), horizon=horizon),
            "benchmark_hs300": compute_return_stats(pd.Series(dtype=float), horizon=horizon),
            "benchmark_zz500": compute_return_stats(pd.Series(dtype=float), horizon=horizon),
            "benchmark_zz1000": compute_return_stats(pd.Series(dtype=float), horizon=horizon),
        }
        return trades_df, positions_df, empty_curve, summary, pred_df

    trades_df = trades_df.sort_values("trade_date").reset_index(drop=True)

    benchmark_tags = ["pool", "hs300", "zz500", "zz1000"]
    for tag in benchmark_tags:
        ret_col = f"benchmark_{tag}_ret"
        trades_df[ret_col] = pd.to_numeric(trades_df[ret_col], errors="coerce").fillna(0.0)

    trades_df["strategy_ret"] = pd.to_numeric(trades_df["strategy_ret"], errors="coerce").fillna(0.0)

    trades_df["strategy_net"] = (1.0 + trades_df["strategy_ret"]).cumprod()
    trades_df["strategy_cum_return"] = trades_df["strategy_net"] - 1.0
    trades_df["strategy_cum_ret"] = trades_df["strategy_cum_return"]

    for tag in benchmark_tags:
        ret_col = f"benchmark_{tag}_ret"
        net_col = f"benchmark_{tag}_net"
        cum_col = f"benchmark_{tag}_cum_return"
        cum_ret_col = f"benchmark_{tag}_cum_ret"
        trades_df[net_col] = (1.0 + trades_df[ret_col]).cumprod()
        trades_df[cum_col] = trades_df[net_col] - 1.0
        trades_df[cum_ret_col] = trades_df[cum_col]

    for tag in benchmark_tags:
        ex_ret = f"excess_vs_{tag}_ret"
        ex_net = f"excess_vs_{tag}_net"
        ex_cum = f"excess_vs_{tag}_cum_return"
        ex_cum_ret = f"excess_vs_{tag}_cum_ret"
        trades_df[ex_ret] = trades_df["strategy_ret"] - trades_df[f"benchmark_{tag}_ret"]
        trades_df[ex_net] = (1.0 + trades_df[ex_ret]).cumprod()
        trades_df[ex_cum] = trades_df[ex_net] - 1.0
        trades_df[ex_cum_ret] = trades_df[ex_cum]

    trades_df["benchmark_ret"] = trades_df["benchmark_pool_ret"]
    trades_df["benchmark_net"] = trades_df["benchmark_pool_net"]
    trades_df["benchmark_cum_return"] = trades_df["benchmark_pool_cum_return"]
    trades_df["benchmark_cum_ret"] = trades_df["benchmark_cum_return"]
    trades_df["excess_ret"] = trades_df["excess_vs_pool_ret"]
    trades_df["excess_net"] = trades_df["excess_vs_pool_net"]
    trades_df["excess_cum_return"] = trades_df["excess_vs_pool_cum_return"]
    trades_df["excess_cum_ret"] = trades_df["excess_cum_return"]

    strategy_stats = compute_return_stats(trades_df["strategy_ret"], horizon=horizon)
    benchmark_pool_stats = compute_return_stats(trades_df["benchmark_pool_ret"], horizon=horizon)
    benchmark_hs300_stats = compute_return_stats(trades_df["benchmark_hs300_ret"], horizon=horizon)
    benchmark_zz500_stats = compute_return_stats(trades_df["benchmark_zz500_ret"], horizon=horizon)
    benchmark_zz1000_stats = compute_return_stats(trades_df["benchmark_zz1000_ret"], horizon=horizon)

    excess_pool_stats = compute_return_stats(trades_df["excess_vs_pool_ret"], horizon=horizon)
    excess_hs300_stats = compute_return_stats(trades_df["excess_vs_hs300_ret"], horizon=horizon)
    excess_zz500_stats = compute_return_stats(trades_df["excess_vs_zz500_ret"], horizon=horizon)
    excess_zz1000_stats = compute_return_stats(trades_df["excess_vs_zz1000_ret"], horizon=horizon)

    summary = {
        "execution_scheme": execution_scheme,
        "execution_description": conf["description"],
        "portfolio_weighting_mode": portfolio_opt_cfg.mode,
        "strategy": strategy_stats,
        "benchmark": benchmark_pool_stats,
        "excess": excess_pool_stats,
        "benchmark_pool": benchmark_pool_stats,
        "benchmark_hs300": benchmark_hs300_stats,
        "benchmark_zz500": benchmark_zz500_stats,
        "benchmark_zz1000": benchmark_zz1000_stats,
        "excess_vs_pool": excess_pool_stats,
        "excess_vs_hs300": excess_hs300_stats,
        "excess_vs_zz500": excess_zz500_stats,
        "excess_vs_zz1000": excess_zz1000_stats,
        "active_trade_ratio": float((trades_df["active_stocks"] > 0).mean()),
        "portfolio_avg_turnover": float(pd.to_numeric(trades_df["portfolio_turnover"], errors="coerce").dropna().mean()),
        "portfolio_avg_concentration": float(pd.to_numeric(trades_df["portfolio_concentration"], errors="coerce").dropna().mean()),
        "portfolio_avg_entropy": float(pd.to_numeric(trades_df["portfolio_weight_entropy"], errors="coerce").dropna().mean()),
        "optimizer_converged_ratio": float(pd.to_numeric(trades_df["opt_converged"], errors="coerce").fillna(0.0).mean()),
        "optimizer_fallback_ratio": float(pd.to_numeric(trades_df["opt_fallback"], errors="coerce").fillna(0.0).mean()),
        "state_avg_vol_z": float(pd.to_numeric(trades_df["state_vol_z"], errors="coerce").dropna().mean()),
        "state_avg_crowding_z": float(pd.to_numeric(trades_df["state_crowding_z"], errors="coerce").dropna().mean()),
        "state_avg_style_z": float(pd.to_numeric(trades_df["state_style_z"], errors="coerce").dropna().mean()),
    }

    curve_df = trades_df.copy()
    return trades_df, positions_df, curve_df, summary, pred_df


def plot_backtest_curves(
    curve_df: pd.DataFrame,
    output_main_png: Path,
    output_excess_png: Path,
    title_prefix: str,
) -> Dict[str, bool]:
    """Plot strategy and excess curves: period return, net value, and cumulative return."""
    plot_status = {"main": False, "excess": False}
    if curve_df.empty or plt is None:
        return plot_status

    dt = pd.to_datetime(curve_df["trade_date"])

    # Main panel: strategy against 4 benchmarks (pool mean + HS300 + ZZ500 + ZZ1000).
    fig1, axes1 = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    main_lines = [
        ("strategy_ret", "Strategy"),
        ("benchmark_pool_ret", "Benchmark-PoolMean"),
        ("benchmark_hs300_ret", "HS300"),
        ("benchmark_zz500_ret", "ZZ500"),
        ("benchmark_zz1000_ret", "ZZ1000"),
    ]
    for col, label in main_lines:
        if col in curve_df.columns:
            axes1[0].plot(dt, curve_df[col], label=label, linewidth=1.2)
    axes1[0].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    axes1[0].set_ylabel("Period Return")
    axes1[0].legend(loc="best")
    axes1[0].grid(alpha=0.2)

    main_net_lines = [
        ("strategy_net", "Strategy"),
        ("benchmark_pool_net", "Benchmark-PoolMean"),
        ("benchmark_hs300_net", "HS300"),
        ("benchmark_zz500_net", "ZZ500"),
        ("benchmark_zz1000_net", "ZZ1000"),
    ]
    for col, label in main_net_lines:
        if col in curve_df.columns:
            axes1[1].plot(dt, curve_df[col], label=label, linewidth=1.3)
    axes1[1].set_ylabel("Net Value")
    axes1[1].legend(loc="best")
    axes1[1].grid(alpha=0.2)

    main_cum_lines = [
        ("strategy_cum_ret", "Strategy"),
        ("benchmark_pool_cum_ret", "Benchmark-PoolMean"),
        ("benchmark_hs300_cum_ret", "HS300"),
        ("benchmark_zz500_cum_ret", "ZZ500"),
        ("benchmark_zz1000_cum_ret", "ZZ1000"),
    ]
    for col, label in main_cum_lines:
        if col in curve_df.columns:
            axes1[2].plot(dt, curve_df[col], label=label, linewidth=1.3)
    axes1[2].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    axes1[2].set_ylabel("Cum Return")
    axes1[2].set_xlabel("Date")
    axes1[2].legend(loc="best")
    axes1[2].grid(alpha=0.2)

    fig1.suptitle(f"{title_prefix} - Main", fontsize=14)
    fig1.tight_layout()
    fig1.savefig(output_main_png, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    plot_status["main"] = True

    # Excess panel: excess period/net/cumulative return vs 4 benchmarks.
    fig2, axes2 = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    excess_ret_lines = [
        ("excess_vs_pool_ret", "Excess vs PoolMean"),
        ("excess_vs_hs300_ret", "Excess vs HS300"),
        ("excess_vs_zz500_ret", "Excess vs ZZ500"),
        ("excess_vs_zz1000_ret", "Excess vs ZZ1000"),
    ]
    for col, label in excess_ret_lines:
        if col in curve_df.columns:
            axes2[0].plot(dt, curve_df[col], label=label, linewidth=1.2)
    axes2[0].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    axes2[0].set_ylabel("Excess Period Return")
    axes2[0].legend(loc="best")
    axes2[0].grid(alpha=0.2)

    excess_net_lines = [
        ("excess_vs_pool_net", "Excess vs PoolMean"),
        ("excess_vs_hs300_net", "Excess vs HS300"),
        ("excess_vs_zz500_net", "Excess vs ZZ500"),
        ("excess_vs_zz1000_net", "Excess vs ZZ1000"),
    ]
    for col, label in excess_net_lines:
        if col in curve_df.columns:
            axes2[1].plot(dt, curve_df[col], label=label, linewidth=1.3)
    axes2[1].set_ylabel("Excess Net Value")
    axes2[1].legend(loc="best")
    axes2[1].grid(alpha=0.2)

    excess_cum_lines = [
        ("excess_vs_pool_cum_ret", "Excess vs PoolMean"),
        ("excess_vs_hs300_cum_ret", "Excess vs HS300"),
        ("excess_vs_zz500_cum_ret", "Excess vs ZZ500"),
        ("excess_vs_zz1000_cum_ret", "Excess vs ZZ1000"),
    ]
    for col, label in excess_cum_lines:
        if col in curve_df.columns:
            axes2[2].plot(dt, curve_df[col], label=label, linewidth=1.3)
    axes2[2].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    axes2[2].set_ylabel("Excess Cum Return")
    axes2[2].set_xlabel("Date")
    axes2[2].legend(loc="best")
    axes2[2].grid(alpha=0.2)

    fig2.suptitle(f"{title_prefix} - Excess", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(output_excess_png, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    plot_status["excess"] = True

    return plot_status

def main() -> None:
    """Main entry point.

    Run loading, factor building, training, backtesting, and output persistence end-to-end.
    """
    args = parse_args()

    train_start = _parse_date(args.train_start)
    train_end = _parse_date(args.train_end)
    test_start = _parse_date(args.test_start)
    test_end = _parse_date(args.test_end)
    if args.horizon <= 0:
        raise ValueError("horizon must be a positive integer.")
    if args.top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if not (0.0 <= args.long_threshold <= 1.0):
        raise ValueError("long_threshold must be between 0 and 1.")
    if not (train_start <= train_end < test_start <= test_end):
        raise ValueError("Date constraint: train_start <= train_end < test_start <= test_end")

    factor_lib = FactorLibrary()
    register_default_factors(factor_lib)
    if args.custom_factor_py:
        load_custom_factor_module(factor_lib, args.custom_factor_py)

    if args.list_factors:
        print(factor_lib.metadata().to_string(index=False))
        return

    selected_factors = resolve_selected_factors(factor_lib, args.factor_list)

    # Load a longer window than train/test range for rolling factors and trade horizon alignment.
    load_start = train_start - pd.Timedelta(days=int(args.lookback_days))
    # Because exit_date can be later than signal date, extend load_end beyond test_end by horizon.
    load_end = test_end + pd.Timedelta(days=int(args.horizon) + 5)
    output_dir = resolve_output_dir(args, train_start, train_end, test_start, test_end)
    output_dir.mkdir(parents=True, exist_ok=True)
    portfolio_opt_cfg = build_portfolio_opt_config(args)

    print("=== Parameters ===")
    print(f"data_root         : {args.data_root}")
    print(f"hs300_list_path   : {args.hs300_list_path}")
    print(f"index_root        : {args.index_root}")
    print(f"train period      : {train_start.date()} ~ {train_end.date()}")
    print(f"test period       : {test_start.date()} ~ {test_end.date()}")
    print(f"horizon           : {args.horizon}")
    print(f"top_k             : {args.top_k}")
    print(f"long_threshold    : {args.long_threshold}")
    print(f"main_board_only   : {args.main_board_only}")
    print(f"execution_scheme  : {args.execution_scheme}")
    print(f"portfolio_mode    : {args.portfolio_weighting}")
    print(f"fee_bps/slip_bps  : {args.fee_bps}/{args.slippage_bps}")
    print(f"load window       : {load_start.date()} ~ {load_end.date()}")
    print(f"factor_count      : {len(selected_factors)}")
    print(f"output_dir        : {output_dir}")
    if args.portfolio_weighting == "dynamic_opt":
        print(
            "opt cfg           : "
            f"max_w={portfolio_opt_cfg.max_weight}, "
            f"max_to={portfolio_opt_cfg.max_turnover}, "
            f"liq_scale={portfolio_opt_cfg.liquidity_scale}, "
            f"risk={portfolio_opt_cfg.risk_aversion}, "
            f"style_pen={portfolio_opt_cfg.style_penalty}, "
            f"ind_pen={portfolio_opt_cfg.industry_penalty}, "
            f"crowd_pen={portfolio_opt_cfg.crowding_penalty}, "
            f"tc_pen={portfolio_opt_cfg.transaction_cost_penalty}"
        )
    print()

    daily_df, minute_df = load_hs300_data(
        data_root=Path(args.data_root),
        start_date=load_start,
        end_date=load_end,
        file_format=args.file_format,
        max_files=args.max_files,
        hs300_list_path=Path(args.hs300_list_path),
        main_board_only=args.main_board_only,
    )
    print(f"Loaded daily rows : {len(daily_df):,}")
    print(f"Loaded minute rows: {len(minute_df):,}")

    index_benchmarks = load_index_benchmark_data(
        index_root=Path(args.index_root),
        start_date=load_start,
        end_date=load_end,
        file_format=args.file_format,
    )
    print(f"Loaded index rows : hs300={len(index_benchmarks.get('hs300', pd.DataFrame())):,}, "
          f"zz500={len(index_benchmarks.get('zz500', pd.DataFrame())):,}, "
          f"zz1000={len(index_benchmarks.get('zz1000', pd.DataFrame())):,}")

    minute_daily_feat = build_minute_daily_features(minute_df)
    base_df = build_daily_feature_base(daily_df, minute_daily_feat)

    price_cols = ["code", "date", "px_open5", "px_vwap30", "px_twap_last30", "px_daily_close"]
    missing_price_cols = [c for c in price_cols if c not in base_df.columns]
    if missing_price_cols:
        raise RuntimeError(f"Missing execution price columns: {missing_price_cols}")
    price_table = base_df[price_cols].copy()

    panel = compute_factor_panel(base_df, factor_lib, selected_factors)
    panel = add_label(
        panel,
        horizon=args.horizon,
        price_table=price_table,
        execution_scheme=args.execution_scheme,
    )

    train_df, test_df = split_train_test(
        panel,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    if train_df.empty:
        raise RuntimeError("Training set is empty. Expand train range or check data.")
    if test_df.empty:
        raise RuntimeError("Test set is empty. Expand test range or check data.")

    print(f"Train samples      : {len(train_df):,}")
    print(f"Test samples       : {len(test_df):,}")

    model, fill_values = train_decision_tree(
        train_df=train_df,
        factors=selected_factors,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )
    clf_metrics = evaluate_classifier(model, test_df, selected_factors, fill_values)

    trades_df, positions_df, curve_df, bt_summary, pred_df = run_backtest(
        model=model,
        test_df=test_df,
        factors=selected_factors,
        fill_values=fill_values,
        price_table=price_table,
        index_benchmarks=index_benchmarks,
        horizon=args.horizon,
        top_k=args.top_k,
        long_threshold=args.long_threshold,
        execution_scheme=args.execution_scheme,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        portfolio_opt_cfg=portfolio_opt_cfg,
    )

    factor_ic_summary_df, factor_ic_series_df = compute_factor_ic_statistics(
        pred_df,
        factor_cols=selected_factors,
        ret_col="future_ret_n",
        min_cross_section=args.min_ic_cross_section,
    )

    model_ic_series_df = calc_ic_for_column(
        pred_df,
        score_col="pred_prob_up",
        ret_col="future_ret_n",
        min_cross_section=args.min_ic_cross_section,
    )
    model_ic_summary = summarize_ic(model_ic_series_df)

    score_spread = compute_score_spread(pred_df, score_col="pred_prob_up", ret_col="future_ret_n", quantiles=5)

    run_tag = (
        f"tr_{train_start.strftime('%Y%m%d')}_{train_end.strftime('%Y%m%d')}"
        f"_te_{test_start.strftime('%Y%m%d')}_{test_end.strftime('%Y%m%d')}"
        f"_h{args.horizon}_{args.execution_scheme}"
        f"_{'mainboard' if args.main_board_only else 'allboards'}"
        f"_{args.portfolio_weighting}"
    )

    pred_path = output_dir / f"predictions_{run_tag}.csv"
    trades_path = output_dir / f"trades_{run_tag}.csv"
    positions_path = output_dir / f"positions_{run_tag}.csv"
    curve_path = output_dir / f"curve_{run_tag}.csv"
    factor_ic_summary_path = output_dir / f"factor_ic_summary_{run_tag}.csv"
    factor_ic_series_path = output_dir / f"factor_ic_series_{run_tag}.csv"
    model_ic_series_path = output_dir / f"model_ic_series_{run_tag}.csv"
    factor_meta_path = output_dir / f"factor_library_{run_tag}.csv"
    plot_main_path = output_dir / f"backtest_curve_main_{run_tag}.png"
    plot_excess_path = output_dir / f"backtest_curve_excess_{run_tag}.png"
    summary_path = output_dir / f"summary_{run_tag}.json"

    pred_cols = [
        "date",
        "code",
        "entry_date",
        "exit_date",
        "pred_prob_up",
        "pred_up",
        "target_up",
        "future_ret_n",
        "entry_price",
        "exit_price",
        "gross_trade_ret",
        "net_trade_ret",
    ] + selected_factors

    pred_df[pred_cols].to_csv(pred_path, index=False, encoding="utf-8-sig")
    trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
    positions_df.to_csv(positions_path, index=False, encoding="utf-8-sig")
    curve_df.to_csv(curve_path, index=False, encoding="utf-8-sig")
    factor_ic_summary_df.to_csv(factor_ic_summary_path, index=False, encoding="utf-8-sig")
    factor_ic_series_df.to_csv(factor_ic_series_path, index=False, encoding="utf-8-sig")
    model_ic_series_df.to_csv(model_ic_series_path, index=False, encoding="utf-8-sig")
    factor_lib.metadata().to_csv(factor_meta_path, index=False, encoding="utf-8-sig")

    plot_status = plot_backtest_curves(
        curve_df,
        output_main_png=plot_main_path,
        output_excess_png=plot_excess_path,
        title_prefix=f"Strategy vs Benchmarks ({args.execution_scheme})",
    )

    top_factors = factor_ic_summary_df.head(10).copy() if not factor_ic_summary_df.empty else pd.DataFrame()

    summary = {
        "config": {
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "lookback_days": int(args.lookback_days),
            "horizon": int(args.horizon),
            "top_k": int(args.top_k),
            "long_threshold": float(args.long_threshold),
            "max_depth": int(args.max_depth),
            "min_samples_leaf": int(args.min_samples_leaf),
            "max_files": None if args.max_files is None else int(args.max_files),
            "file_format": args.file_format,
            "main_board_only": bool(args.main_board_only),
            "hs300_list_path": args.hs300_list_path,
            "index_root": args.index_root,
            "execution_scheme": args.execution_scheme,
            "execution_scheme_desc": EXECUTION_SCHEMES[args.execution_scheme]["description"],
            "fee_bps": float(args.fee_bps),
            "slippage_bps": float(args.slippage_bps),
            "portfolio_weighting": args.portfolio_weighting,
            "portfolio_opt_config": {
                "max_weight": float(portfolio_opt_cfg.max_weight),
                "max_turnover": float(portfolio_opt_cfg.max_turnover),
                "liquidity_scale": float(portfolio_opt_cfg.liquidity_scale),
                "expected_return_weight": float(portfolio_opt_cfg.expected_return_weight),
                "risk_aversion": float(portfolio_opt_cfg.risk_aversion),
                "style_penalty": float(portfolio_opt_cfg.style_penalty),
                "industry_penalty": float(portfolio_opt_cfg.industry_penalty),
                "crowding_penalty": float(portfolio_opt_cfg.crowding_penalty),
                "transaction_cost_penalty": float(portfolio_opt_cfg.transaction_cost_penalty),
                "max_iter": int(portfolio_opt_cfg.max_iter),
                "step_size": float(portfolio_opt_cfg.step_size),
                "tolerance": float(portfolio_opt_cfg.tolerance),
            },
            "selected_factors": selected_factors,
            "note_no_future_leakage": (
                "Factors use only signal-day and historical info; labels/backtest share the same execution scheme;"
                "target_date is bounded in train/test windows; prices use only observable execution-day data (5m or daily close)."
            ),
        },
        "sample_count": {
            "daily_rows": int(len(daily_df)),
            "minute_rows": int(len(minute_df)),
            "index_hs300_rows": int(len(index_benchmarks.get("hs300", pd.DataFrame()))),
            "index_zz500_rows": int(len(index_benchmarks.get("zz500", pd.DataFrame()))),
            "index_zz1000_rows": int(len(index_benchmarks.get("zz1000", pd.DataFrame()))),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "trade_points": int(len(trades_df)),
            "position_count": int(len(positions_df)),
        },
        "classifier_metrics": clf_metrics,
        "backtest_metrics": bt_summary,
        "model_ic_summary": model_ic_summary,
        "model_score_spread": score_spread,
        "factor_ic_top10": top_factors.to_dict(orient="records") if not top_factors.empty else [],
        "outputs": {
            "output_dir": str(output_dir),
            "predictions_csv": str(pred_path),
            "trades_csv": str(trades_path),
            "positions_csv": str(positions_path),
            "curve_csv": str(curve_path),
            "factor_ic_summary_csv": str(factor_ic_summary_path),
            "factor_ic_series_csv": str(factor_ic_series_path),
            "model_ic_series_csv": str(model_ic_series_path),
            "factor_library_csv": str(factor_meta_path),
            "backtest_main_plot_png": str(plot_main_path) if plot_status.get("main", False) else "Not generated (matplotlib unavailable or curve empty)",
            "backtest_excess_plot_png": str(plot_excess_path) if plot_status.get("excess", False) else "Not generated (matplotlib unavailable or curve empty)",
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=== Classifier Metrics ===")
    print(json.dumps(clf_metrics, ensure_ascii=False, indent=2))
    print("=== Backtest Metrics (Strategy) ===")
    print(json.dumps(bt_summary.get("strategy", {}), ensure_ascii=False, indent=2))
    print("=== Model IC Summary ===")
    print(json.dumps(model_ic_summary, ensure_ascii=False, indent=2))
    print("=== Score Spread (Q5-Q1) ===")
    print(json.dumps(score_spread, ensure_ascii=False, indent=2))

    print()
    print(f"Predictions saved       : {pred_path}")
    print(f"Trades saved            : {trades_path}")
    print(f"Positions saved         : {positions_path}")
    print(f"Curve saved             : {curve_path}")
    print(f"Factor IC summary saved : {factor_ic_summary_path}")
    print(f"Model IC series saved   : {model_ic_series_path}")
    print(f"Summary saved           : {summary_path}")
    if plot_status.get("main", False):
        print(f"Backtest main plot saved: {plot_main_path}")
    else:
        print("Backtest main plot skip : matplotlib unavailable or empty curve")
    if plot_status.get("excess", False):
        print(f"Backtest excess plot saved: {plot_excess_path}")
    else:
        print("Backtest excess plot skip : matplotlib unavailable or empty curve")


if __name__ == "__main__":
    main()
















