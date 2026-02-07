# moudle1_gp_config.py
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Config:
    # data
    base_dir: str
    pool: str                 # hs300 / zz500 / sz50 / all
    price_freq: str = "d"
    start_date: str = "2019-01-01"
    end_date: Optional[str] = None
    forward_ret_days: int = 10

    # runtime
    seed: int = 42
    device: str = "cpu"
    max_stocks_for_debug: Optional[int] = None

    # fundamental source
    use_ak_fundamental: bool = False

    # ---------- GP params ----------
    pop_size: int = 200
    n_generations: int = 30
    tournament_k: int = 5
    elitism: int = 10

    p_crossover: float = 0.7
    p_mutation: float = 0.25
    p_reproduce: float = 0.05

    max_tree_depth: int = 6
    max_tree_nodes: int = 60

    # fitness
    fitness_metric: str = "icir"   # "ic_mean" / "icir"
    ic_min_obs: int = 80           # 每期横截面最少样本数
    turnover_penalty: float = 0.05 # 换手惩罚权重
    complexity_penalty: float = 0.002 # 表达式复杂度惩罚

    # backtest
    topk: int = 50
    rebalance_freq: str = "W-FRI"  # 周五调仓
    long_short: bool = True

    # feature columns (你可以按需扩展)
    price_features: List[str] = None
    fundamental_features: List[str] = None

    def __post_init__(self):
        if self.price_features is None:
            # 来自 baostock 日线字段，你的 hist csv里一般有这些
            self.price_features = [
                "open", "high", "low", "close", "volume", "amount", "turn",
                "pctChg", "peTTM", "psTTM", "pcfNcfTTM", "pbMRQ"
            ]
        if self.fundamental_features is None:
            # 先选一批“季频关键指标”，来自你 baostock_fundamental_q 下各表的字段
            # 实际字段名以你的 csv 为准；你可先print列名再微调
            self.fundamental_features = [
                # profit表常见
                "roeAvg", "npMargin", "gpMargin",
                # operation表常见
                "turnoverRatio", "inventoryTurnover",
                # growth表常见
                "YOYEquity", "YOYNetProfit",
                # dupont表常见
                "dupontROE", "dupontAssetStoEquity",
                # balance/cashflow（如有）
                "currentRatio", "quickRatio",
                "cashFlowPerShare"
            ]
