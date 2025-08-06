# 个股量价面 + 个股财务基本面数据获取【自动重试 + 多线程并发下载 + 增量更新覆盖】
# 要注意两点：
# 1、akshare要时刻保持在最新版本 pip install --upgrade akshare
# 2、东方财富访问频繁后会封ip，手动登录东方财富网可以解决

import os
import pandas as pd
import akshare as ak   # akshare要时刻保持在最新版本 pip install --upgrade akshare
from tqdm import tqdm
from datetime import datetime
import time
import random
from tenacity import retry, stop_after_attempt, wait_random
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== 配置路径 ==========
BASE_DIR = "data"
HIST_DIR = os.path.join(BASE_DIR, "stock_hist")
FIN_DIR = os.path.join(BASE_DIR, "stock_finance")
META_DIR = os.path.join(BASE_DIR, "metadata")
SNAPSHOT_DIR = os.path.join(META_DIR, "stock_snapshots")

for d in [BASE_DIR, HIST_DIR, FIN_DIR, META_DIR, SNAPSHOT_DIR]:
    os.makedirs(d, exist_ok=True)

empty_finance_codes = []
empty_hist_codes = []

def save_data(df, path_prefix, table_name):
    df.to_csv(f"{path_prefix}.csv", index=False)
    df.to_parquet(f"{path_prefix}.parquet", index=False)

def is_valid_csv(path):
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path) # 数据存在且不为空，返回True
        return not df.empty
    except:
        return False
# 获取最新交易日（不包含当日）
def get_latest_trade_date():
    today = datetime.today().date()
    trade_dates = ak.tool_trade_date_hist_sina()
    trade_dates["trade_date"] = pd.to_datetime(trade_dates["trade_date"]).dt.date
    trade_dates = trade_dates[trade_dates["trade_date"] < today]
    latest_date = trade_dates["trade_date"].max()
    return latest_date.strftime("%Y%m%d")

def get_stock_list(refresh=False):
    path_prefix = os.path.join(META_DIR, "stock_list")
    snapshot_path = os.path.join(SNAPSHOT_DIR, f"stock_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    if os.path.exists(f"{path_prefix}.csv") and not refresh:
        return pd.read_csv(f"{path_prefix}.csv", dtype=str)

    df = ak.stock_zh_a_spot_em()
    df["总市值"] = pd.to_numeric(df["总市值"], errors="coerce")
    df = df[~df["名称"].str.contains("ST", na=False)]
    df = df[df["总市值"] > 200e8]
    df["代码"] = df["代码"].apply(lambda x: x[:6])
    df = df[~df["代码"].str.startswith(("300", "688"))]

    end_date = get_latest_trade_date()
    print(f"[使用交易日] {end_date}")

    filtered = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        code = row["代码"]
        try:
            hist = ak.stock_zh_a_hist(symbol=code, start_date=end_date, end_date=end_date, adjust="qfq")
            if hist.empty:
                continue
            volume = pd.to_numeric(hist.at[0, "成交量"], errors="coerce")
            if volume > 0:
                filtered.append({
                    "代码": code,
                    "名称": row["名称"],
                    "总市值": row["总市值"],
                    "成交量": volume
                })
        except:
            continue

    df_final = pd.DataFrame(filtered)

    # 保存快照
    if os.path.exists(f"{path_prefix}.csv"):
        old_df = pd.read_csv(f"{path_prefix}.csv", dtype=str)
        old_codes = set(old_df["代码"])
        new_codes = set(df_final["代码"])
        removed_codes = old_codes - new_codes
        print(f"[更新] 股票池变化：新增 {len(new_codes - old_codes)}，剔除 {len(removed_codes)}")
        # 对于新增的，下载新的数据；对于剔除的，不下载新数据不管他；把历史快照保留好，以最新历史快照中的股票池为准
        # for code in removed_codes:
        #     hist_path = os.path.join(HIST_DIR, f"{code}_D.csv")
        #     fin_path = os.path.join(FIN_DIR, f"{code}.csv")
        #     for path in [hist_path, fin_path]:
        #         if os.path.exists(path):
        #             os.remove(path)
        #             print(f"[删除] {path}")
    else:
        print("[首次运行] 无旧股票池对比")

    df_final.to_csv(path_prefix + ".csv", index=False)
    df_final.to_csv(snapshot_path, index=False)  # stock_list.csv 永远和 最新的快照.csv 保持一致
    print(f"[快照] 股票池快照已保存至 {snapshot_path}")
    return df_final

@retry(stop=stop_after_attempt(3), wait=wait_random(min=1, max=3))
def fetch_hist_with_retry(symbol, start_date, end_date, adjust):
    time.sleep(random.uniform(0.5, 1.5))
    return ak.stock_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust=adjust)

def get_stock_hist(code, start_date="20100101", end_date=None, adjust="qfq", freq="D"):
    symbol = code
    path_prefix = os.path.join(HIST_DIR, f"{symbol}_{freq}")
    csv_path = f"{path_prefix}.csv"

    # 当前最新交易日
    latest_date = get_latest_trade_date() if end_date is None else end_date
    # 如果文件存在，尝试增量下载
    old_df = None
    if is_valid_csv(csv_path):
        old_df = pd.read_csv(csv_path, parse_dates=["日期"])
        max_date = old_df["日期"].max().strftime("%Y%m%d")

        if max_date >= latest_date:
            return  # 数据已是最新

        start_date = (pd.to_datetime(max_date) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        print(f"[增量] {code} 从 {start_date} 更新至 {latest_date}")

    try:
        raw = fetch_hist_with_retry(symbol, start_date, latest_date, adjust)
        print(raw)
        print(f"下载到历史行情数据了: {symbol}")
    except Exception as e:
        print(f"[失败] 历史行情获取失败：{symbol} → {e}")
        empty_hist_codes.append(code)
        return

    if raw.empty:
        print(f"历史行情数据为空: {symbol}")
        empty_hist_codes.append(code)
        return

    raw["日期"] = pd.to_datetime(raw["日期"])
    raw.set_index("日期", inplace=True)
    raw = raw.rename(columns={
        "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume",
        "成交额": "amount", "振幅": "amplitude", "涨跌幅": "change_percent",
        "涨跌额": "change", "换手率": "turnover_rate"
    })

    if freq == "W":
        df = raw.resample("W").agg({...})
    elif freq == "M":
        df = raw.resample("M").agg({...})
    else:
        df = raw[[
            "open", "high", "low", "close", "volume",
            "amount", "amplitude", "change_percent", "change", "turnover_rate"
        ]]

    df = df.dropna().reset_index()
    if old_df is not None:
        df = pd.concat([old_df, df]).drop_duplicates(subset="日期").sort_values("日期")
        print(df)
    save_data(df, path_prefix, f"stock_hist_{freq}_{symbol}")

@retry(stop=stop_after_attempt(3), wait=wait_random(min=1, max=3))
def fetch_finance_with_retry(symbol):
    time.sleep(random.uniform(0.5, 1.5))
    return ak.stock_financial_analysis_indicator(symbol=symbol, start_year="2010")

def get_finance_data(code):
    path_prefix = os.path.join(FIN_DIR, code)
    csv_path = f"{path_prefix}.csv"

    old_df = None
    if is_valid_csv(csv_path):
        try:
            old_df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[警告] 读取旧财务数据失败：{code} → {e}")
            old_df = None

    try:
        df = fetch_finance_with_retry(symbol=code)
        print(df)
        print(f"下载到财务数据了: {code}")
        
        if df.empty:
            print(f"财务数据为空: {code}")
            empty_finance_codes.append(code)
            return

        # 如果旧数据存在，进行合并去重 + 排序
        if old_df is not None:
            combined = pd.concat([old_df, df]).drop_duplicates().sort_values("报告期")

            # ✅ 判断内容是否一致（避免重复保存）
            if combined.equals(old_df):
                return  # 数据未更新，跳过保存
            else:
                df = combined  # 使用合并后的 df

        save_data(df, path_prefix, f"stock_finance_{code}")

    except Exception as e:
        print(f"[失败] 财务数据获取失败：{code} → {e}")
        empty_finance_codes.append(code)

def get_stock_concept():
    path_prefix = os.path.join(META_DIR, "stock_concept")
    if is_valid_csv(f"{path_prefix}.csv"):
        return pd.read_csv(f"{path_prefix}.csv", dtype=str)
    
    old_df = None
    if is_valid_csv(f"{path_prefix}.csv"):
        old_df = pd.read_csv(f"{path_prefix}.csv", dtype=str)

    df = ak.stock_board_concept_name_em()
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        concept = row["板块名称"]
        try:
            members = ak.stock_board_concept_cons_em(concept)
            for _, m in members.iterrows():
                records.append({"代码": m["代码"], "名称": m["名称"], "概念": concept})
        except:
            continue

    concept_df = pd.DataFrame(records)
    if old_df is not None:
        concept_df = pd.concat([old_df, concept_df]).drop_duplicates()
    save_data(concept_df, path_prefix, "stock_concept")
    return concept_df

def init_all_data():
    stocks = get_stock_list(refresh=True) # 是否开启增量更新
    codes = stocks["代码"].tolist()

    max_workers = 10
    # print("[并发] 下载历史行情中...")
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     futures = {executor.submit(get_stock_hist, code): code for code in codes}
    #     for _ in tqdm(as_completed(futures), total=len(futures)):
    #         pass

    print("[并发] 下载财务指标中...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_finance_data, code): code for code in codes}
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    # print("[执行] 下载概念板块中...")
    # try:
    #     get_stock_concept()
    # except Exception as e:
    #     print(f"[跳过] 概念板块失败：{e}")

def save_failed_logs():
    if empty_hist_codes:
        pd.DataFrame({"代码": empty_hist_codes}).to_csv("empty_hist.csv", index=False)
    if empty_finance_codes:
        pd.DataFrame({"代码": empty_finance_codes}).to_csv("empty_finance.csv", index=False)

if __name__ == '__main__':
    init_all_data()
    save_failed_logs()
