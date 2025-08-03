import os
import pandas as pd
import akshare as ak
import time
import random
from datetime import datetime
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random
from concurrent.futures import ThreadPoolExecutor, as_completed

from xtquant import xtdata
# from xtquant import fundamental

# ========== 路径配置 ==========
BASE_DIR = "data_xt"
HIST_DIR = os.path.join(BASE_DIR, "stock_hist")
FIN_DIR = os.path.join(BASE_DIR, "stock_finance")
META_DIR = os.path.join(BASE_DIR, "metadata")

failed_hist = []
failed_fin = []

for d in [BASE_DIR, HIST_DIR, META_DIR]:
    os.makedirs(d, exist_ok=True)

# ========== 辅助函数 ==========
def is_valid_csv(path):
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
        return not df.empty
    except:
        return False

def save_data(df, path_prefix):
    df.to_csv(f"{path_prefix}.csv", index=False)
    df.to_parquet(f"{path_prefix}.parquet", index=False)

# ========== 获取股票列表 ==========
def get_stock_list():
    path_prefix = os.path.join(META_DIR, "stock_list")
    if is_valid_csv(f"{path_prefix}.csv"):
        return pd.read_csv(f"{path_prefix}.csv")

    codes = xtdata.get_stock_list_in_sector("沪深A股")
    df = pd.DataFrame({"代码": codes})
    df = df[~df["代码"].str.startswith(("300", "688"))]  # 剔除创业/科创板
    save_data(df, path_prefix)
    return df

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

# ========== 获取行情数据（xtquant） ==========
@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_xtquant_hist(code, start_date, end_date, freq="1d"):
    time.sleep(random.uniform(0.5, 1.5))
    fields = ["open", "high", "low", "close", "volume", "amount"]
    df = xtdata.get_market_data(code, freq, start_date, end_date, fields, dividend_type=1)
    if df is None or df.empty:
        return pd.DataFrame()
    df.reset_index(inplace=True)
    df.rename(columns={"datetime": "日期"}, inplace=True)
    return df

# ========== 下载单只股票历史行情 ==========
def get_stock_hist(code, start_date, end_date, freq="1d"):
    freq_map = {"1d": "D", "1w": "W", "1m": "M"}
    path_prefix = os.path.join(HIST_DIR, f"{code}_{freq_map[freq]}")
    if is_valid_csv(f"{path_prefix}.csv"):
        return
    try:
        df = fetch_xtquant_hist(code, start_date, end_date, freq)
        if df.empty:
            failed_hist.append(code)
            return
        save_data(df, path_prefix)
    except Exception as e:
        print(f"[失败] {code} {freq} 行情下载失败: {e}")
        failed_hist.append(code)

@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_financial_with_retry(code):
    time.sleep(random.uniform(0.5, 1.5))
    df = xtdata.get_financial_data(stock_code=code, report_type="全部", start_year=2010)
    if df is None or df.empty:
        return pd.DataFrame()
    return df

def get_financial_data(code):
    path_prefix = os.path.join(FIN_DIR, f"{code}_finance")
    if is_valid_csv(f"{path_prefix}.csv"):
        return
    try:
        df = fetch_financial_with_retry(code)
        if df.empty:
            failed_fin.append(code)
            return
        save_data(df, path_prefix)
    except Exception as e:
        print(f"[失败] 财务数据获取失败 {code}: {e}")
        failed_fin.append(code)


# ========== 批量下载 ==========
def init_all_data():
    stock_df = get_stock_list()
    codes = stock_df["代码"].tolist()
    end_date = get_latest_trade_date()
    start_date = "2010-01-01"

    max_workers = 10

    print("[并发] 下载日线行情...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(get_stock_hist, code, start_date, end_date, "1d") for code in codes]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    # print("[并发] 下载周线行情...")
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     futures = [executor.submit(get_stock_hist, code, start_date, end_date, "1w") for code in codes]
    #     for _ in tqdm(as_completed(futures), total=len(futures)):
    #         pass

    # print("[并发] 下载月线行情...")
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     futures = [executor.submit(get_stock_hist, code, start_date, end_date, "1m") for code in codes]
    #     for _ in tqdm(as_completed(futures), total=len(futures)):
    #         pass

    # print("[并发] 下载财务指标...")
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     futures = [executor.submit(get_financial_data, code) for code in codes]
    #     for _ in tqdm(as_completed(futures), total=len(futures)):
    #         pass


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
