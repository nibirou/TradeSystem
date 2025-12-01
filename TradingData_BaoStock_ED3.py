import os
import time
import random
from datetime import datetime, timedelta

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

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

failed_list = []


# ===========================
#      Baostock登录
# ===========================
def bs_login():
    lg = bs.login()
    if lg.error_code != "0":
        raise Exception(f"Baostock login failed: {lg.error_msg}")


def bs_logout():
    bs.logout()


# ===========================
#      股票池读取
# ===========================
def _bs_query_to_df(rs):
    data_list = []
    while (rs.error_code == "0") & rs.next():
        data_list.append(rs.get_row_data())
    return pd.DataFrame(data_list, columns=rs.fields)


def get_stock_list_bs(mode="hs300", day=None):
    if mode == "sz50":
        rs = bs.query_sz50_stocks()
    elif mode == "hs300":
        rs = bs.query_hs300_stocks()
    elif mode == "zz500":
        rs = bs.query_zz500_stocks()
    elif mode == "all":
        if day is None:
            raise ValueError("mode='all' 需要 day='YYYY-MM-DD'")
        rs = bs.query_all_stock(day=day)
    else:
        raise ValueError(f"未知股票池模式：{mode}")

    df = _bs_query_to_df(rs)

    if "code_name" in df.columns:
        df = df.rename(columns={"code_name": "name"})
    else:
        df["name"] = df["code"]

    return df[["code", "name"]]


# ===========================
#       获取最新交易日
# ===========================
def get_latest_end_date():
    today = datetime.now().date()
    return (today - timedelta(days=1)).strftime("%Y-%m-%d")


# ===========================
#       字段映射
# ===========================
def get_fields(freq):
    if freq == "d":
        return ("date,code,open,high,low,close,preclose,volume,amount,"
                "adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST")
    if freq in ["w", "m"]:
        return "date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg"
    if freq in ["5", "15", "30", "60"]:
        return "date,time,code,open,high,low,close,volume,amount,adjustflag"
    raise ValueError(f"未知 freq: {freq}")


# ===========================
#       自动重试调用
# ===========================
@retry(stop=stop_after_attempt(3), wait=wait_random(1, 2))
def safe_query(code, fields, start_date, end_date, freq, adjustflag):
    rs = bs.query_history_k_data_plus(
        code,
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency=freq,
        adjustflag=adjustflag
    )
    if rs.error_code != "0":
        raise Exception(rs.error_msg)
    return rs


# ===========================
#     分钟线 → 自然月拆分
# ===========================
def split_months(start_date, end_date):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    periods = []
    cur = start.replace(day=1)

    while cur <= end:
        month_end = (cur + pd.offsets.MonthEnd()).normalize()
        if month_end > end:
            month_end = end
        periods.append((cur.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")))
        cur = month_end + timedelta(days=1)

    return periods


# ===========================
#         下载单只股票
# ===========================
def get_stock_hist_bs(code, pool, start_date, end_date, freq, adjustflag):

    # ========== 最早起始日期 ==========
    is_index = (
        code.startswith("sh.000")
        or code.startswith("sz.399")
        or code.startswith("sh.880")
    )

    if freq in ["d", "w", "m"]:
        min_date = "2006-01-01" if is_index else "1990-12-19"
    else:
        if is_index:
            print(f"[跳过] 指数不支持分钟线：{code}")
            return
        min_date = "2019-01-01"

    if start_date is None:
        start_date = min_date
    else:
        start_date = max(pd.to_datetime(start_date).date(),
                         pd.to_datetime(min_date).date()).strftime("%Y-%m-%d")

    end_date = end_date or get_latest_end_date()

    # ========== 文件路径 ==========
    save_dir = os.path.join(HIST_DIR, pool, freq)
    os.makedirs(save_dir, exist_ok=True)

    code_clean = code.replace(".", "_")
    filename = f"{code_clean}_{freq}"
    csv_path = os.path.join(save_dir, filename + ".csv")
    parquet_path = os.path.join(save_dir, filename + ".parquet")

    fields = get_fields(freq)

    # ========== 断点续传 ==========
    old_df = None
    if os.path.exists(csv_path):
        try:
            old_df = pd.read_csv(csv_path)
            if not old_df.empty:
                max_date = old_df["date"].max()
                if max_date >= end_date:
                    return
                start_date = (pd.to_datetime(max_date) + timedelta(days=1)).strftime("%Y-%m-%d")
        except:
            old_df = None

    # ========== 日 / 周 / 月 ==========
    if freq in ["d", "w", "m"]:
        rs = safe_query(code, fields, start_date, end_date, freq, adjustflag)

        data_list = []
        while (rs.error_code == "0") & rs.next():
            data_list.append(rs.get_row_data())

    else:
        # ========== 分钟线，自然月拆分 ==========
        # periods = split_months(start_date, end_date)
        # data_list = []

        # for s, e in periods:
        #     rs = safe_query(code, fields, s, e, freq, adjustflag)
        #     while (rs.error_code == "0") & rs.next():
        #         data_list.append(rs.get_row_data())
        
        # 不按照自然月拆分
        rs = safe_query(code, fields, start_date, end_date, freq, adjustflag)
        data_list = []
        while (rs.error_code == "0") & rs.next():
            data_list.append(rs.get_row_data())

    if not data_list:
        failed_list.append(code)
        print(f"[空数据] {code}")
        return

    df = pd.DataFrame(data_list, columns=rs.fields)

    if old_df is not None:
        df = pd.concat([old_df, df]).drop_duplicates(subset="date").sort_values("date")

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)


# ===========================
#           主流程
# ===========================
def run_history_download(pool="hs300", freq="d"):
    # 登录一次，用于股票池查询
    bs_login()
    if pool == "all":
        stocks = get_stock_list_bs("all", day=get_latest_end_date())
    else:
        stocks = get_stock_list_bs(pool)
    bs_logout()

    # 保存快照
    snapshot_path = os.path.join(
        SNAPSHOT_DIR, f"{pool}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    stocks.to_csv(snapshot_path, index=False, encoding="utf-8")
    stocks.to_csv(os.path.join(META_DIR, f"stock_list_{pool}.csv"),
                  index=False, encoding="utf-8")

    codes = stocks["code"].tolist()
    print(f"[股票池] {pool} 共 {len(codes)} 只股票")
    print(f"[开始顺序下载] 频率: {freq}")

    # 顺序逐只下载
    for code in tqdm(codes):
        try:
            bs_login()
            get_stock_hist_bs(code, pool, None, None, freq, adjustflag="2")
        except Exception as e:
            failed_list.append(code)
            print(f"[失败] {code}: {e}")
        finally:
            bs_logout()

    # 写入失败日志
    if failed_list:
        pd.DataFrame({"code": failed_list}).to_csv("failed_hist.csv", index=False)
        print(f"[失败数量] {len(failed_list)}，已写入 failed_hist.csv")
    else:
        print("[成功] 全部数据下载成功！")


# ===========================
#             MAIN
# ===========================
if __name__ == "__main__":
    run_history_download(pool="all", freq="5")
