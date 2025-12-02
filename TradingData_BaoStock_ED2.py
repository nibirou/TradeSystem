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

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

failed_hist = []
failed_retry = []     # 第二轮失败记录


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
#      股票池统一封装
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
            raise ValueError("mode='all' 时需传入 day='YYYY-MM-DD'")
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
#         获取最新交易日
# ===========================
def get_latest_end_date():
    today = datetime.now().date()
    return (today - timedelta(days=1)).strftime("%Y-%m-%d")


# ===========================
#   BaoStock 官方字段配置
# ===========================
def get_fields(freq):
    if freq == "d":
        return "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST"
    if freq in ["w", "m"]:
        return "date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg"
    if freq in ["5", "15", "30", "60"]:
        return "date,time,code,open,high,low,close,volume,amount,adjustflag"
    raise ValueError(f"未知 freq: {freq}")


# ===========================
#         自动重试封装
# ===========================
@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
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
        raise Exception(f"Baostock 返回错误: {rs.error_msg}")
    return rs


# ===========================
#     分钟线 → 按自然月拆分
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
#         历史行情爬取
# ===========================
def get_stock_hist_bs(code, pool="hs300", start_date=None, end_date=None,
                      freq="d", adjustflag="2"):

    # bs_login()
    try:
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
                print(f"[跳过] 指数 {code} 不支持分钟线")
                return
            min_date = "2019-01-01"   # 实际以 end_date - 5y 为准

        if start_date is None:
            start_date = min_date
        else:
            start_date = max(pd.to_datetime(start_date).date(),
                             pd.to_datetime(min_date).date()).strftime("%Y-%m-%d")

        end_date = end_date or get_latest_end_date()

        # ========== 文件结构 ==========
        save_dir = os.path.join(HIST_DIR, pool, freq)
        os.makedirs(save_dir, exist_ok=True)

        code_clean = code.replace(".", "_")
        filename = f"{code_clean}_{freq}"
        path_prefix = os.path.join(save_dir, filename)
        csv_path = f"{path_prefix}.csv"

        fields = get_fields(freq)

        # ========== 断点续传：从 CSV 最大日期继续 ==========
        old_df = None
        if os.path.exists(csv_path):
            try:
                old_df = pd.read_csv(csv_path, encoding="gbk")
                if old_df.empty:
                    old_df = None
                else:
                    max_date = old_df["date"].max()
                    if max_date >= end_date:
                        return
                    start_date = (pd.to_datetime(max_date) + timedelta(days=1)).strftime("%Y-%m-%d")
            except:
                old_df = None

        # ========== 日线/周线/月线（一段抓取） ==========
        if freq in ["d", "w", "m"]:
            rs = safe_query(code, fields, start_date, end_date, freq, adjustflag)

            data_list = []
            while (rs.error_code == "0") & rs.next():
                data_list.append(rs.get_row_data())

        # ========== 分钟线（自然月拆分） ==========
        else:
            periods = split_months(start_date, end_date)
            data_list = []

            for s, e in periods:
                rs = safe_query(code, fields, s, e, freq, adjustflag)
                while (rs.error_code == "0") & rs.next():
                    data_list.append(rs.get_row_data())

        if not data_list:
            failed_hist.append(code)
            return

        df = pd.DataFrame(data_list, columns=rs.fields)

        if old_df is not None:
            df = pd.concat([old_df, df]).drop_duplicates(subset="date").sort_values("date")

        df.to_csv(csv_path, index=False, encoding="gbk")
        df.to_parquet(f"{path_prefix}.parquet", index=False)

    except Exception as e:
        failed_hist.append(code)
        print(f"[失败] {code}: {e}")

    finally:
        bs_logout()


# ===========================
#        并发下载入口
# ===========================
def run_history_download(pool="hs300", freq="d", workers=8):

    # ========== 主线程必须登录一次 ==========
    bs_login()

    # 获取股票池
    if pool == "all":
        stocks = get_stock_list_bs("all", day=get_latest_end_date())
    else:
        stocks = get_stock_list_bs(pool)

    # 保存股票池快照
    snapshot_path = os.path.join(SNAPSHOT_DIR, f"{pool}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    stocks.to_csv(snapshot_path, index=False, encoding="utf-8")
    stocks.to_csv(os.path.join(META_DIR, f"stock_list_{pool}.csv"), index=False, encoding="utf-8")

    # 主线程不参与 K 线下载，可直接 logout
    bs_logout()

    codes = stocks["code"].tolist()
    print(f"[股票池] {pool} 共 {len(codes)} 只股票")
    print(f"[开始下载] 频率: {freq}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(get_stock_hist_bs, code, pool, None, None, freq)
            for code in codes
        ]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    # ========== 重试 + 失败处理 ==========
    ...


# ===========================
#             主程序
# ===========================
if __name__ == "__main__":
    run_history_download(pool="hs300", freq="d", workers=10)
