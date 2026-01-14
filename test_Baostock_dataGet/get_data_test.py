import os
import time
import random
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import baostock as bs
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random
from tqdm import tqdm


# ===========================
#        路径配置
# ===========================
BASE_DIR = "data_baostock"
HIST_DIR = os.path.join(BASE_DIR, "stock_hist")
META_DIR = os.path.join(BASE_DIR, "metadata")
SNAPSHOT_DIR = os.path.join(META_DIR, "stock_snapshots")

for d in [BASE_DIR, HIST_DIR, META_DIR, SNAPSHOT_DIR]:
    os.makedirs(d, exist_ok=True)

# ===========================
#        Baostock登录
# ===========================
def bs_login():
    lg = bs.login()
    if lg.error_code != "0":
        raise Exception(f"Baostock login failed: {lg.error_msg}")
    print("[登录] Baostock 登录成功")

def bs_logout():
    bs.logout()
    print("[退出] Baostock 已退出登录")

# ===========================
#         获取最新交易日
#   （Baostock 无 trade_date 接口 → 我们用 yesterday）
# ===========================
def get_latest_end_date():
    today = datetime.now().date()
    end_date = today - timedelta(days=1)
    return end_date.strftime("%Y-%m-%d")

def _bs_query_to_df(rs):
    data_list = []
    while (rs.error_code == "0") & rs.next():
        data_list.append(rs.get_row_data())
    return pd.DataFrame(data_list, columns=rs.fields)

def get_stock_list_bs(mode="hs300", day=None):
    """
    mode = sz50 | hs300 | zz500 | all
    """
    if mode == "sz50":
        rs = bs.query_sz50_stocks()
    elif mode == "hs300":
        rs = bs.query_hs300_stocks()
    elif mode == "zz500":
        rs = bs.query_zz500_stocks()
    elif mode == "all":
        if day is None:
            raise ValueError("mode='all' 时需传入 day='YYYY-MM-DD'")
        rs = bs.query_all_stock(day=day)
    else:
        raise ValueError(f"未知股票池模式：{mode}")

    df = _bs_query_to_df(rs)

    # 全市场没有 name 字段，补充一个
    if "code_name" in df.columns:
        df = df.rename(columns={"code_name": "name"})
    else:
        df["name"] = df["code"]

    return df[["code", "name"]]

    
# ===========================
#        并发下载入口
# ===========================
def run_history_download(pool="hs300", freq="d", workers=8):
    bs_login()

    # 获取股票池
    if pool == "all":
        stocks = get_stock_list_bs("all", day=get_latest_end_date())
    else:
        stocks = get_stock_list_bs(pool)

    codes = stocks["code"].tolist()

    print(f"[股票池] {pool} 共 {len(codes)} 只股票")
    print(f"[开始下载] 频率: {freq}")
    
    for code in codes:
        rs = bs.query_history_k_data_plus(code,
            # "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
            "date,time,code,open,high,low,close,volume,amount,adjustflag",
            start_date='1990-12-19', end_date=get_latest_end_date(),
            frequency=freq, adjustflag="2")
        print('query_history_k_data_plus respond error_code:'+rs.error_code)
        print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)
        
        data_list = []
        while (rs.error_code == "0") & rs.next():
            data_list.append(rs.get_row_data())
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 文件名保持 code_freq 格式
        save_dir = os.path.join(HIST_DIR, pool, freq)
        os.makedirs(save_dir, exist_ok=True)
        code_clean = code.replace(".", "_")
        filename = f"{code_clean}_{freq}"
        path_prefix = os.path.join(save_dir, filename)

        csv_path = f"{path_prefix}.csv"
        
        df.to_csv(csv_path, index=False)
        df.to_parquet(f"{path_prefix}.parquet", index=False)

        print(f"[完成] {code} → {csv_path}")


if __name__ == "__main__":
    # freq=D/W/M/5/15/30/60
    # run_history_download(pool="hs300", freq="5", workers=10)
    run_history_download(pool="hs300", freq="15", workers=10)