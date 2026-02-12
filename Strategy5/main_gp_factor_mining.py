# main_gp_factor_mining.py
import os
import random
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from moudle1_gp_config import Config
from moudle2_data_hub import DataHub
from moudle3_gp_operators import OperatorSet
from moudle4_gp_tree import GPTree
from moudle5_evaluator import FactorEvaluator
from moudle6_gp_engine import GPEngine


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    cfg = Config(
        base_dir="/workspace/Quant/data_baostock",  # 改成你的 BASE_DIR
        pool="zz500",
        price_freq="d",
        start_date="2019-01-01",
        end_date=None,          # None=用历史行情最大日期
        forward_ret_days=10,    # 标签：未来10日收益
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_ak_fundamental=False,  # 先False跑通；你需要时再True并完善字段映射
        max_stocks_for_debug=200,  # debug时可设小点
    )
    seed_everything(cfg.seed)

    # 1) 数据
    hub = DataHub(cfg)
    panel = hub.build_panel()  # MultiIndex (date, code) 的 DataFrame，含 features + label
    print("[panel]", panel.shape, panel.columns[:10])

    # 2) 算子集
    op_set = OperatorSet(cfg)

    # 3) 评估器
    evaluator = FactorEvaluator(cfg, panel)

    # 4) GP 引擎
    engine = GPEngine(cfg, op_set, evaluator)

    # 5) 进化
    best_exprs = engine.evolve()

    print("\n===== TOP EXPRESSIONS =====")
    for i, item in enumerate(best_exprs[:10], 1):
        print(f"[{i}] score={item['score']:.6f}  size={item['size']}  expr={item['expr']}")

    # 6) 对最优因子做完整回测报告
    best = best_exprs[0]["tree"]
    report = evaluator.full_report(best)
    print("\n===== BEST FACTOR REPORT =====")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
