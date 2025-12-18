# 使用最新的T日数据，计算每只股票的策略因子综合得分，排名，给出T+1日的选股

import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from moudle1_dataloader import DataConfig, PeriodConfig, load_data_bundle
from moudle2_factorengine import FactorEngine
from moudle4_backtest import Analyzer

