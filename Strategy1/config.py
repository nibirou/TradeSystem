# config.py
import os

# ========== 基础路径 ==========
BASE_DIR = "data_baostock"
HIST_DIR = os.path.join(BASE_DIR, "stock_hist")

# ========== 策略与资金参数 ==========
INIT_CAPITAL = 60000.0          # 初始资金
REB_FREQ_DAYS = 5               # 每隔几个交易日调仓
MAX_STOCKS = 15                 # 最大持仓股票数
MAX_WEIGHT = 0.10               # 单票权重上限

# 成本与滑点
SLIPPAGE = 0.0005               # 0.05% 滑点
COMMISSION = 0.0003             # 佣金率
STAMP_DUTY = 0.001              # 印花税（卖出）

# 流动性约束
MIN_DAILY_AMOUNT = 3e6          # 当日成交额下限（选股过滤）
MAX_VOL_RATIO = 0.05            # 单日交易量不能超过当日成交量的5%

# 因子相关
IC_LOOKBACK = 120               # 因子IC滚动窗口长度（截面个数）
IC_MIN_DATES = 60               # 至少有多少天才开始计算IC
FACTOR_WINSOR_LOWER = 0.05      # 因子去极值下分位
FACTOR_WINSOR_UPPER = 0.95      # 因子去极值上分位

# 使用的股票池和频率
POOLS = ("hs300", "zz500")
DAILY_FREQ = "d"
INTRADAY_FREQ = "5"