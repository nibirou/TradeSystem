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

failed_hist = []


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
#      股票池统一封装
# ===========================
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
#     频率转换（你习惯的格式→Baostock格式）
# ===========================
# def map_freq(freq):
#     mapper = {
#         "D": "d",
#         "W": "w",
#         "M": "m",
#         "5": "5",
#         "15": "15",
#         "30": "30",
#         "60": "60",
#     }
#     return mapper[freq]


# ===========================
#         获取最新交易日
#   （Baostock 无 trade_date 接口 → 我们用 yesterday）
# ===========================
def get_latest_end_date():
    today = datetime.now().date()
    end_date = today - timedelta(days=1)
    return end_date.strftime("%Y-%m-%d")

def get_fields(freq):
    """
    根据频率自动选择 BaoStock 官方规范字段
    """
    if freq == "d":
        return "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST"
    if freq in ["w", "m"]:
        return "date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg"
    elif freq in ["5", "15", "30", "60"]:
        return "date,time,code,open,high,low,close,volume,amount,adjustflag"
        
    else:
        raise ValueError(f"未知 freq: {freq}")


# ===========================
#         历史行情爬取
# ===========================
@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def fetch_hist_bs(code, fields, start_date, end_date, freq, adjustflag):
    time.sleep(random.uniform(0.5, 1.5))
    return bs.query_history_k_data_plus(
        code,
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency=freq,
        adjustflag=adjustflag
    )


def get_stock_hist_bs(code, pool="hs300", start_date=None, end_date=None,
                      freq="d", adjustflag="2"):
    """
    BaoStock 历史行情下载
    自动根据频率 & 股票/指数类型裁剪开始日期，确保不请求无效数据

    pool: 股票池（sz50 / hs300 / zz500 / all）
    freq: D/W/M/5/15/30/60
    adjustflag:
        1 后复权
        2 前复权
        3 不复权
    """

    # ================================
    #         自动计算最大可用起始日期
    # ================================
    # 判断是否指数：指数通常以 .000 开头 或 399xxx（深证指数）
    is_index = (
        code.startswith("sh.000") or
        code.startswith("sz.399") or
        code.startswith("sh.880")  # 行业指数
    )

    if freq in ["d", "w", "m"]:
        if is_index:
            min_date = "2006-01-01"          # 指数有效起始
        else:
            min_date = "1990-12-19"          # 股票有效起始
    else:
        # 分钟线
        if is_index:
            print(f"[跳过] 指数 {code} 不支持分钟线")
            return
        min_date = "2019-01-02"              # 分钟线有效起始（最近 5 年）

    # 最终开始日期（必须 >= min_date）
    if start_date is None:
        start_date = min_date
    else:
        # 若用户传入 start_date 过小 → 自动提升到 min_date
        start_date = max(pd.to_datetime(start_date).date(),
                         pd.to_datetime(min_date).date()).strftime("%Y-%m-%d")

    # ================================
    #         结束日期计算
    # ================================
    end_date = end_date or get_latest_end_date()

    freq_bs = freq

    # ================================
    #        目录结构 (pool/freq)
    # ================================
    save_dir = os.path.join(HIST_DIR, pool, freq)
    os.makedirs(save_dir, exist_ok=True)

    # 文件名保持 code_freq 格式
    code_clean = code.replace(".", "_")
    filename = f"{code_clean}_{freq}"
    path_prefix = os.path.join(save_dir, filename)

    csv_path = f"{path_prefix}.csv"

    fields = get_fields(freq)

    # ================================
    #         增量更新逻辑
    # ================================
    old_df = None
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        max_date = old_df["date"].max()

        # 增量结束条件
        if max_date >= end_date:
            return

        start_date = (pd.to_datetime(max_date) + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"[增量] {code}: {start_date} → {end_date}")

    # ================================
    #         执行抓取
    # ================================
    print(f"[执行抓取] {code}")
    print(f"[code] {code}")
    print(f"[fields] {fields}")
    print(f"[start_date] {start_date}")
    print(f"[end_date] {end_date}")
    print(f"[freq_bs] {freq_bs}")
    print(f"[adjustflag] {adjustflag}")
    try:
        rs = fetch_hist_bs(code, fields, start_date, end_date, freq_bs, adjustflag)
        print(f"[抓取成功] {code}: {rs}")
    except Exception as e:
        print(f"[失败] {code}: {e}")
        failed_hist.append(code)
        return

    data_list = []
    while (rs.error_code == "0") & rs.next():
        data_list.append(rs.get_row_data())

    if not data_list:
        print(f"[空数据] {code}")
        failed_hist.append(code)
        return

    df = pd.DataFrame(data_list, columns=rs.fields)

    # ================================
    #       合并旧数据（增量）
    # ================================
    if old_df is not None:
        df = pd.concat([old_df, df]).drop_duplicates(subset="date").sort_values("date")

    # ================================
    #             保存
    # ================================
    df.to_csv(csv_path, index=False, encoding="gbk")
    df.to_parquet(f"{path_prefix}.parquet", index=False)

    print(f"[完成] {code} → {csv_path}")


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

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(get_stock_hist_bs, code, freq=freq)
            for code in codes
        ]

        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    bs_logout()

    # 失败日志
    if failed_hist:
        pd.DataFrame({"code": failed_hist}).to_csv("failed_hist.csv", index=False)
        print(f"[失败数量] {len(failed_hist)}，已写入 failed_hist.csv")
    else:
        print("[成功] 无失败代码")


# ===========================
#             主程序
# ===========================
if __name__ == "__main__":
    # freq=D/W/M/5/15/30/60
    # run_history_download(pool="hs300", freq="5", workers=10)
    run_history_download(pool="hs300", freq="d", workers=10)
