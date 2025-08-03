# 进一步支持【自动重试 + 多线程并发下载】
import os
import pandas as pd
import akshare as ak
from tqdm import tqdm
from datetime import datetime
import time
import random
from tenacity import retry, stop_after_attempt, wait_random
from concurrent.futures import ThreadPoolExecutor, as_completed
from xtquant import xtdata

# ========== 配置路径 ==========
BASE_DIR = "data"
HIST_DIR = os.path.join(BASE_DIR, "stock_hist")
FIN_DIR = os.path.join(BASE_DIR, "stock_finance")
META_DIR = os.path.join(BASE_DIR, "metadata")
for d in [BASE_DIR, HIST_DIR, FIN_DIR, META_DIR]:
    os.makedirs(d, exist_ok=True)

failed_hist = []
failed_fin = []

# ========== 保存函数：CSV + Parquet + MySQL ==========
def save_data(df, path_prefix, table_name):
    # 保存 CSV
    df.to_csv(f"{path_prefix}.csv", index=False)
    # 保存 Parquet
    df.to_parquet(f"{path_prefix}.parquet", index=False)
    # 保存到 MySQL
    # df.to_sql(table_name, engine, if_exists='replace', index=False)

# 判断下载的CSV是否存在且有效（非空）
def is_valid_csv(path):
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
        return not df.empty
    except:
        return False

# ========== 获取最新的交易日（不晚于今天） ==========
def get_latest_trade_date():
    """
    获取最新的交易日（不晚于今天）
    """
    today = datetime.today().date()  # 👈 转成 datetime.date 类型
    trade_dates = ak.tool_trade_date_hist_sina()
    trade_dates["trade_date"] = pd.to_datetime(trade_dates["trade_date"]).dt.date  # 👈 确保列为 date 类型
    trade_dates = trade_dates[trade_dates["trade_date"] < today]
    latest_date = trade_dates["trade_date"].max()
    return latest_date.strftime("%Y%m%d")  # 👈 最终返回字符串格式如 '20250729'


# ========== 批量下载 ==========
def init_all_data():
    stock_df = get_stock_list()
    codes = stock_df["代码"].tolist()
    end_date = get_latest_trade_date()
    start_date = "2010-01-01"
    
def save_failed_logs():
    if failed_hist:
        pd.DataFrame({"代码": failed_hist}).to_csv("failed_hist.csv", index=False)
        print(f"[警告] {len(failed_hist)} 只股票历史行情下载失败，已记录 failed_hist.csv")
    if failed_fin:
        pd.DataFrame({"代码": failed_fin}).to_csv("failed_fin.csv", index=False)
        print(f"[警告] {len(failed_fin)} 只股票财务数据下载失败，已记录 failed_fin.csv")

# ========== 启动 ==========
if __name__ == "__main__":
    init_all_data()
    save_failed_logs()