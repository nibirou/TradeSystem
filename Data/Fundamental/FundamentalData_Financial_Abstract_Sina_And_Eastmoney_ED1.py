# 获取新浪财经 东方财富个股基本面财务数据（主要关键指标）
# by akshare

import os
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import baostock as bs
import pandas as pd
import akshare as ak
from tenacity import retry, stop_after_attempt, wait_random
from tqdm import tqdm


# ===========================
#        路径配置
# ===========================
BASE_DIR = "/workspace/Quant/data_baostock"

META_DIR = os.path.join(BASE_DIR, "metadata")
SNAPSHOT_DIR = os.path.join(META_DIR, "stock_snapshots")

AK_FUND_DIR = os.path.join(BASE_DIR, "ak_fundamental")

# 三个接口各自一个目录
DIR_ABSTRACT_SINA = os.path.join(AK_FUND_DIR, "financial_abstract_sina")
DIR_INDICATOR_EM = os.path.join(AK_FUND_DIR, "financial_indicator_em")
DIR_INDICATOR_SINA = os.path.join(AK_FUND_DIR, "financial_indicator_sina")

for d in [BASE_DIR, META_DIR, SNAPSHOT_DIR, AK_FUND_DIR,
          DIR_ABSTRACT_SINA, DIR_INDICATOR_EM, DIR_INDICATOR_SINA]:
    os.makedirs(d, exist_ok=True)

failed_list: List[Dict] = []


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
#      股票池读取（与你上面一致）
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


def get_latest_end_date():
    today = datetime.now().strftime("%Y-%m-%d")

    year_start = f"{datetime.now().year}-01-01"
    year_second = f"{datetime.now().year}-01-02"
    year_third = f"{datetime.now().year}-01-03"
    year_fourth = f"{datetime.now().year}-01-04"

    if today in {year_start, year_second, year_third, year_fourth}:
        year_start = f"{datetime.now().year - 1}-01-01"

    rs = bs.query_trade_dates(start_date=year_start, end_date=today)

    data_list = []
    while (rs.error_code == "0") & rs.next():
        data_list.append(rs.get_row_data())

    if not data_list:
        raise Exception("baostock query_trade_dates 返回空数据")

    df = pd.DataFrame(data_list, columns=rs.fields)
    df["calendar_date"] = pd.to_datetime(df["calendar_date"])
    df["is_trading_day"] = df["is_trading_day"].astype(int)

    trade_days = df[df["is_trading_day"] == 1]["calendar_date"].tolist()
    if not trade_days:
        raise Exception("没有找到任何交易日")

    return trade_days[-1].strftime("%Y-%m-%d")


# ===========================
#  股票池动态更新：读取旧快照
# ===========================
def load_last_snapshot(pool: str) -> Set[str]:
    path = os.path.join(META_DIR, f"stock_list_{pool}.csv")
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(df["code"].tolist())


# ===========================
#   AkShare code 转换
# ===========================
def bs_code_to_plain(code: str) -> str:
    """
    baostock: sh.600000 / sz.000001 -> AkShare 新浪接口需要 "600000"
    """
    if "." not in code:
        return code
    return code.split(".", 1)[1]


def bs_code_to_em_suffix(code: str) -> str:
    """
    baostock: sh.600000 -> 600000.SH
             sz.300750 -> 300750.SZ
    """
    if not code.startswith(("sh.", "sz.")) or "." not in code:
        raise ValueError(f"unexpected code: {code}")
    prefix, num = code.split(".", 1)
    suffix = "SH" if prefix == "sh" else "SZ"
    return f"{num}.{suffix}"


# ===========================
#   通用：清洗 + 存盘
# ===========================
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("", pd.NA)
    return df


def save_df(df: pd.DataFrame, csv_path: str, parquet_path: str):
    df = clean_df(df)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_parquet(parquet_path, index=False)


# ===========================
#   通用：目录反解历史 code（动态更新用）
# ===========================
def codes_from_dir(folder: str) -> Set[str]:
    """
    文件名统一用 baostock 格式：sh_600000.csv -> sh.600000
    """
    if not os.path.exists(folder):
        return set()
    s = set()
    for fn in os.listdir(folder):
        if not fn.endswith(".csv"):
            continue
        code = fn.replace(".csv", "").replace("_", ".")
        s.add(code)
    return s


# ===========================
#   AkShare 重试调用 + 限速
# ===========================
@retry(stop=stop_after_attempt(3), wait=wait_random(1, 2))
def safe_ak_call(fn, **kwargs):
    return fn(**kwargs)


def polite_sleep(base: float = 0.4, jitter: float = 0.8):
    time.sleep(base + random.random() * jitter)


# ===========================
#   1) 关键指标-新浪：stock_financial_abstract
#      单次返回全历史（接口定义）
# ===========================
def download_financial_abstract_sina(code_bs: str, pool: str):
    symbol = bs_code_to_plain(code_bs)

    save_dir = os.path.join(DIR_ABSTRACT_SINA, pool)
    os.makedirs(save_dir, exist_ok=True)

    code_clean = code_bs.replace(".", "_")
    csv_path = os.path.join(save_dir, f"{code_clean}.csv")
    parquet_path = os.path.join(save_dir, f"{code_clean}.parquet")

    # 拉全量
    polite_sleep()
    df = safe_ak_call(ak.stock_financial_abstract, symbol=symbol)
    if df is None or df.empty:
        return

    # “伪断点”：若本地已有且最大报告期未变化，则不重写
    if os.path.exists(csv_path):
        try:
            old = pd.read_csv(csv_path)
            # 该表是宽表：列名里有很多报告期（如 20220930）
            old_periods = [c for c in old.columns if c.isdigit() and len(c) == 8]
            new_periods = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 8]
            if old_periods and new_periods and max(old_periods) == max(map(str, new_periods)):
                return
        except Exception:
            pass

    save_df(df, csv_path, parquet_path)


# ===========================
#   2) 主要指标-东方财富：stock_financial_analysis_indicator_em
#      单次返回全历史（接口定义）
# ===========================
def download_financial_indicator_em(code_bs: str, pool: str, indicator: str = "按报告期"):
    symbol = bs_code_to_em_suffix(code_bs)

    save_dir = os.path.join(DIR_INDICATOR_EM, pool)
    os.makedirs(save_dir, exist_ok=True)

    code_clean = code_bs.replace(".", "_")
    # indicator 也写入文件名，避免“按报告期/按单季度”互相覆盖
    ind_tag = "report" if indicator == "按报告期" else "singleq"
    csv_path = os.path.join(save_dir, f"{code_clean}_{ind_tag}.csv")
    parquet_path = os.path.join(save_dir, f"{code_clean}_{ind_tag}.parquet")

    polite_sleep()
    df = safe_ak_call(ak.stock_financial_analysis_indicator_em, symbol=symbol, indicator=indicator)
    if df is None or df.empty:
        return

    # “伪断点”：REPORT_DATE 最大值不变则不重写
    if os.path.exists(csv_path) and "REPORT_DATE" in df.columns:
        try:
            old = pd.read_csv(csv_path)
            if "REPORT_DATE" in old.columns:
                old_max = pd.to_datetime(old["REPORT_DATE"], errors="coerce").max()
                new_max = pd.to_datetime(df["REPORT_DATE"], errors="coerce").max()
                if pd.notna(old_max) and pd.notna(new_max) and old_max == new_max:
                    return
        except Exception:
            pass

    # 去重/排序
    if "REPORT_DATE" in df.columns:
        df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"], errors="coerce")
        df = df.drop_duplicates(subset=["REPORT_DATE"], keep="last").sort_values("REPORT_DATE")

    save_df(df, csv_path, parquet_path)


# ===========================
#   3) 财务指标-新浪：stock_financial_analysis_indicator
#      支持 start_year：可做真正断点续传
# ===========================
def _infer_start_year_from_local(csv_path: str, min_year: int = 2000) -> str:
    if not os.path.exists(csv_path):
        return str(min_year)

    try:
        old = pd.read_csv(csv_path)
        if old.empty or "日期" not in old.columns:
            return str(min_year)
        last_date = pd.to_datetime(old["日期"], errors="coerce").max()
        if pd.isna(last_date):
            return str(min_year)
        # 回拉一整年，避免同年内季度缺口
        y = max(min_year, int(last_date.year) - 1)
        return str(y)
    except Exception:
        return str(min_year)


def download_financial_indicator_sina(code_bs: str, pool: str, min_year: int = 2000):
    symbol = bs_code_to_plain(code_bs)

    save_dir = os.path.join(DIR_INDICATOR_SINA, pool)
    os.makedirs(save_dir, exist_ok=True)

    code_clean = code_bs.replace(".", "_")
    csv_path = os.path.join(save_dir, f"{code_clean}.csv")
    parquet_path = os.path.join(save_dir, f"{code_clean}.parquet")

    start_year = _infer_start_year_from_local(csv_path, min_year=min_year)

    polite_sleep()
    df = safe_ak_call(ak.stock_financial_analysis_indicator, symbol=symbol, start_year=start_year)
    if df is None or df.empty:
        return

    # 合并旧数据实现增量
    if os.path.exists(csv_path):
        try:
            old = pd.read_csv(csv_path)
            df = pd.concat([old, df], ignore_index=True)
        except Exception:
            pass

    # 去重/排序（日期唯一）
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        df = df.drop_duplicates(subset=["日期"], keep="last").sort_values("日期")

    save_df(df, csv_path, parquet_path)


# ===========================
#           主流程
# ===========================
def run_ak_fundamental_download(
    pool: str = "hs300",
    indicator_em_mode: str = "按报告期",   # 也可传 "按单季度"
    min_year_sina_indicator: int = 2000,
):
    """
    三个接口分别落盘到：
      ak_fundamental/financial_abstract_sina/{pool}/
      ak_fundamental/financial_indicator_em/{pool}/
      ak_fundamental/financial_indicator_sina/{pool}/
    """

    bs_login()

    # ---- 股票池 ----
    if pool == "all":
        stocks = get_stock_list_bs("all", day=get_latest_end_date())
    else:
        stocks = get_stock_list_bs(pool)

    # ---- 快照（与你K线脚本一致）----
    snapshot_path = os.path.join(
        SNAPSHOT_DIR, f"{pool}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    stocks.to_csv(snapshot_path, index=False, encoding="utf-8")
    stocks.to_csv(os.path.join(META_DIR, f"stock_list_{pool}.csv"),
                  index=False, encoding="utf-8")

    current_codes = set(stocks["code"].tolist())
    last_snapshot_codes = load_last_snapshot(pool)

    # 历史目录 codes（3 个目录取并集）
    hist_codes = set()
    hist_codes |= codes_from_dir(os.path.join(DIR_ABSTRACT_SINA, pool))
    hist_codes |= codes_from_dir(os.path.join(DIR_INDICATOR_EM, pool))
    hist_codes |= codes_from_dir(os.path.join(DIR_INDICATOR_SINA, pool))

    final_codes = current_codes | hist_codes | last_snapshot_codes
    codes = sorted(final_codes)

    print(f"[股票池] 本次 {len(current_codes)} 只")
    print(f"[历史文件] {len(hist_codes)} 只")
    print(f"[最终更新] 共 {len(codes)} 只")
    print(f"[EM模式] {indicator_em_mode}")

    for code in tqdm(codes):
        # 逐接口抓取（每个接口失败不影响其他接口/其他股票）
        try:
            download_financial_abstract_sina(code, pool)
        except Exception as e:
            failed_list.append({
                "code": code, "api": "stock_financial_abstract",
                "error": str(e), "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        try:
            download_financial_indicator_em(code, pool, indicator=indicator_em_mode)
        except Exception as e:
            failed_list.append({
                "code": code, "api": "stock_financial_analysis_indicator_em",
                "error": str(e), "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        try:
            download_financial_indicator_sina(code, pool, min_year=min_year_sina_indicator)
        except Exception as e:
            failed_list.append({
                "code": code, "api": "stock_financial_analysis_indicator",
                "error": str(e), "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    # 失败日志
    if failed_list:
        fail_path = os.path.join(BASE_DIR, "failed_ak_fundamental.csv")
        pd.DataFrame(failed_list).to_csv(fail_path, index=False, encoding="utf-8")
        print(f"[失败数量] {len(failed_list)}，已写入 {fail_path}")
    else:
        print("[成功] AkShare 基本面数据下载完成，无失败！")


if __name__ == "__main__":
    # 示例：中证500
    run_ak_fundamental_download(pool="zz500", indicator_em_mode="按报告期", min_year_sina_indicator=2000)
    bs_logout()
