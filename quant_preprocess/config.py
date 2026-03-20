# quant_preprocess/config.py
# 模块 0：通用类型与配置 quant_preprocess/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Literal

Freq = Literal["1min", "5min", "15min", "30min", "60min", "120min", "D", "W", "M", "CUSTOM"]

@dataclass
class PathsConfig:
    base_dir: str = "/workspace/Quant/data_baostock"

    # 量价
    hist_dir: str = "stock_hist"        # {base_dir}/stock_hist/{pool}/{freq}/...
    meta_dir: str = "metadata"          # {base_dir}/metadata/...

    # 股票池
    hs300_list: str = "stock_list_hs300.csv"
    zz500_list: str = "stock_list_zz500.csv"

    # 频率目录名（与你现有一致）
    daily_freq_dir: str = "d"
    minute5_freq_dir: str = "5"

    # 交易日历、全市场列表（你现有）
    trade_calendar_csv: str = "metadata/trade_datas.csv"
    all_stock_list_csv: str = "metadata/stock_list_all.csv"

    # 基本面落盘目录（与你现有 GNN 模块一致）
    baostock_fundamental_q_dir: str = "baostock_fundamental_q"   # {base_dir}/baostock_fundamental_q/{pool}/{cat}/...
    fundamental_akshare_dir: str = "fundamental_akshare"         # {base_dir}/fundamental_akshare/...

    def abs_path(self, rel: str) -> Path:
        return Path(self.base_dir) / rel


@dataclass
class SchemaConfig:
    # 统一字段
    code_col: str = "code"
    date_col: str = "date"
    time_col: str = "time"
    datetime_col: str = "datetime"

    # 日频常用（你现有 daily 里）
    px_close: str = "close"
    px_open: str = "open"
    px_high: str = "high"
    px_low: str = "low"
    vol: str = "volume"
    turn: str = "turn"
    tradestatus: str = "tradestatus"


@dataclass
class PeriodConfig:
    factor_start: str = "2019-01-01"
    factor_end: str = "2024-12-31"

    train_start: str = "2019-01-01"
    train_end: str = "2024-12-31"

    backtest_start: str = "2020-01-01"
    backtest_end: str = "2024-12-31"

    inference_day: str = "2024-12-31"

    # 用于 rolling 因子前置 buffer（你已有）
    factor_buffer_n: int = 30

    # 用于 计算 收益率y 的后置buffer
    label_buffer_n: int = 30

    def __post_init__(self):
        # 这里不强制校验关系，保留你原来的灵活性
        pass


@dataclass
class LoadRequest:
    pools: Sequence[str] = ("hs300", "zz500")
    load_minute5: bool = True
    load_daily: bool = True

    # 统一加载窗口（如果为空，则默认 factor_start~factor_end 及 buffer 自动算）
    start: Optional[str] = None
    end: Optional[str] = None

    # 历史行情采样频率
    trade_freq: Freq = "D"

    # 因子计算/标签用的“采样频率”
    # D/W/M 或 CUSTOM（CUSTOM 你可以传自定义窗口生成器）
    factor_freq: Freq = "D"

    # 标签行情采样频率
    label_freq: Freq = "D"

    # 5min 标签需要的“未来窗口内最优价”规则
    # minute5_best_price: Literal["best_close", "best_vwap", "best_high"] = "best_close"

    # 是否加载基本面（可插拔）
    use_akshare_fundamental: bool = False
    use_baostock_q_fundamental: bool = True

    # 额外数据源：你后续的“算法挖掘新因子/文本因子”等
    extra_sources: list[str] = field(default_factory=list)


