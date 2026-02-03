# 获取baostock 个股基本面财务数据

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import baostock as bs
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random
from tqdm import tqdm


# ===========================
#        路径配置
# ===========================
# BASE_DIR = "/workspace/Quant/data_baostock"
BASE_DIR = "D:/PythonProject/Quant/data_baostock"
META_DIR = os.path.join(BASE_DIR, "metadata")
SNAPSHOT_DIR = os.path.join(META_DIR, "stock_snapshots")

FUND_DIR = os.path.join(BASE_DIR, "baostock_fundamental_q")  # <- 新增：季频基本面目录

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(FUND_DIR, exist_ok=True)

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
    if len(trade_days) == 0:
        raise Exception("没有找到任何交易日")

    return trade_days[-1].strftime("%Y-%m-%d")


# ===========================
#  股票池动态更新（复用思路）
# ===========================
def get_codes_from_data_dir(pool: str, category: str) -> set:
    """
    从 fundamental_q/{pool}/{category} 目录下反解出 code
    文件名例：sh_600000.csv
    """
    d = os.path.join(FUND_DIR, pool, category)
    if not os.path.exists(d):
        return set()

    codes = set()
    for fn in os.listdir(d):
        if not fn.endswith(".csv"):
            continue
        code = fn.replace(".csv", "").replace("_", ".")
        codes.add(code)
    return codes


def load_last_snapshot(pool: str) -> set:
    path = os.path.join(META_DIR, f"stock_list_{pool}.csv")
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(df["code"].tolist())


# ===========================
#   季度工具：year/quarter 迭代
# ===========================
def to_year_quarter_from_statdate(statdate: str) -> Tuple[int, int]:
    """
    statDate 形如 2017-06-30
    """
    d = pd.to_datetime(statdate)
    y = int(d.year)
    m = int(d.month)
    q = (m - 1) // 3 + 1
    return y, q


def next_year_quarter(y: int, q: int) -> Tuple[int, int]:
    if q < 4:
        return y, q + 1
    return y + 1, 1


def current_year_quarter(today: Optional[datetime] = None) -> Tuple[int, int]:
    today = today or datetime.now()
    y = today.year
    q = (today.month - 1) // 3 + 1
    return y, q


def iter_year_quarters(start_y: int, start_q: int, end_y: int, end_q: int):
    y, q = start_y, start_q
    while (y < end_y) or (y == end_y and q <= end_q):
        yield y, q
        y, q = next_year_quarter(y, q)


# ===========================
#     通用：Baostock 重试调用
# ===========================
@retry(stop=stop_after_attempt(3), wait=wait_random(1, 2))
def safe_query_rs(fn: Callable, *args, **kwargs):
    rs = fn(*args, **kwargs)
    if getattr(rs, "error_code", "0") != "0":
        raise Exception(getattr(rs, "error_msg", "unknown error"))
    return rs


def rs_to_df(rs) -> pd.DataFrame:
    rows = []
    while (rs.error_code == "0") & rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def clean_df_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("", pd.NA)
    # 不强行指定全部 numeric：因为不同表字段差异很大
    # 只把明显的日期字段转 datetime，便于排序/断点
    for c in ["pubDate", "statDate",
              "performanceExpPubDate", "performanceExpStatDate", "performanceExpUpdateDate",
              "profitForcastExpPubDate", "profitForcastExpStatDate"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


# ===========================
#   A类：按 year/quarter 的表
# ===========================
@dataclass(frozen=True)
class YQEndpoint:
    name: str
    fn: Callable  # bs.query_xxx_data


YQ_ENDPOINTS: Dict[str, YQEndpoint] = {
    "profit":    YQEndpoint("profit", bs.query_profit_data),
    "operation": YQEndpoint("operation", bs.query_operation_data),
    "growth":    YQEndpoint("growth", bs.query_growth_data),
    "balance":   YQEndpoint("balance", bs.query_balance_data),
    "cash_flow": YQEndpoint("cash_flow", bs.query_cash_flow_data),
    "dupont":    YQEndpoint("dupont", bs.query_dupont_data),
}


def get_last_statdate(csv_path: str) -> Optional[str]:
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        if df.empty or "statDate" not in df.columns:
            return None
        # statDate 可能是字符串
        s = df["statDate"].dropna().astype(str)
        if s.empty:
            return None
        return s.max()
    except Exception:
        return None


def download_yq_table_for_stock(
    code: str,
    pool: str,
    category: str,
    start_year: int = 2007,
    start_quarter: int = 1,
):
    ep = YQ_ENDPOINTS[category]

    save_dir = os.path.join(FUND_DIR, pool, category)
    os.makedirs(save_dir, exist_ok=True)

    code_clean = code.replace(".", "_")
    csv_path = os.path.join(save_dir, f"{code_clean}.csv")
    parquet_path = os.path.join(save_dir, f"{code_clean}.parquet")

    # ---- 断点续传：基于 statDate ----
    last_stat = get_last_statdate(csv_path)
    if last_stat:
        y, q = to_year_quarter_from_statdate(last_stat)
        y, q = next_year_quarter(y, q)
        start_year, start_quarter = y, q

    end_year, end_quarter = current_year_quarter()

    all_new = []
    for y, q in iter_year_quarters(start_year, start_quarter, end_year, end_quarter):
        rs = safe_query_rs(ep.fn, code=code, year=y, quarter=q)
        dfq = rs_to_df(rs)
        if dfq.empty:
            continue
        all_new.append(dfq)

    if not all_new:
        return  # 没有新增就不动文件

    new_df = pd.concat(all_new, ignore_index=True)

    # 合并旧数据
    if os.path.exists(csv_path):
        try:
            old_df = pd.read_csv(csv_path)
            new_df = pd.concat([old_df, new_df], ignore_index=True)
        except Exception:
            pass

    # 去重：同一季度用 statDate 唯一（保留最后）
    if "statDate" in new_df.columns:
        new_df = new_df.drop_duplicates(subset=["statDate"], keep="last")
        new_df = new_df.sort_values("statDate")

    new_df = clean_df_for_parquet(new_df)
    new_df.to_csv(csv_path, index=False, encoding="utf-8")
    new_df.to_parquet(parquet_path, index=False)


# ===========================
#   B类：按 start/end 的表
# ===========================
DATE_ENDPOINTS = {
    "perf_express": {
        "fn": bs.query_performance_express_report,
        "date_col": "performanceExpUpdateDate",  # 用“最新披露日”做断点更稳
        "start_min": "2006-01-01",
    },
    "forecast": {
        "fn": bs.query_forecast_report,
        "date_col": "profitForcastExpPubDate",   # 以发布日断点
        "start_min": "2003-01-01",
    },
}


def get_last_date(csv_path: str, date_col: str) -> Optional[str]:
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        if df.empty or date_col not in df.columns:
            return None
        s = df[date_col].dropna().astype(str)
        if s.empty:
            return None
        return s.max()
    except Exception:
        return None


def download_date_table_for_stock(code: str, pool: str, category: str):
    meta = DATE_ENDPOINTS[category]
    fn = meta["fn"]
    date_col = meta["date_col"]
    start_min = meta["start_min"]

    save_dir = os.path.join(FUND_DIR, pool, category)
    os.makedirs(save_dir, exist_ok=True)

    code_clean = code.replace(".", "_")
    csv_path = os.path.join(save_dir, f"{code_clean}.csv")
    parquet_path = os.path.join(save_dir, f"{code_clean}.parquet")

    # ---- 断点续传：基于 date_col 最大值 + 1天 ----
    last_d = get_last_date(csv_path, date_col)
    if last_d:
        try:
            start_date = (pd.to_datetime(last_d) + timedelta(days=1)).strftime("%Y-%m-%d")
        except Exception:
            start_date = start_min
    else:
        start_date = start_min

    end_date = datetime.now().strftime("%Y-%m-%d")

    rs = safe_query_rs(fn, code, start_date=start_date, end_date=end_date)
    new_df = rs_to_df(rs)
    if new_df.empty:
        return

    # 合并旧数据
    if os.path.exists(csv_path):
        try:
            old_df = pd.read_csv(csv_path)
            new_df = pd.concat([old_df, new_df], ignore_index=True)
        except Exception:
            pass

    # 去重：同一条记录一般用 (statDate, pubDate) 或 (updateDate, statDate) 去重
    subset = []
    if "performanceExpStatDate" in new_df.columns:
        subset.append("performanceExpStatDate")
    if "performanceExpUpdateDate" in new_df.columns:
        subset.append("performanceExpUpdateDate")
    if "profitForcastExpStatDate" in new_df.columns:
        subset.append("profitForcastExpStatDate")
    if "profitForcastExpPubDate" in new_df.columns:
        subset.append("profitForcastExpPubDate")
    if subset:
        new_df = new_df.drop_duplicates(subset=subset, keep="last")

    # 排序：优先按断点列
    if date_col in new_df.columns:
        new_df = new_df.sort_values(date_col)

    new_df = clean_df_for_parquet(new_df)
    new_df.to_csv(csv_path, index=False, encoding="utf-8")
    new_df.to_parquet(parquet_path, index=False)


# ===========================
#           主流程
# ===========================
def run_fundamental_download(
    pool: str = "hs300",
    categories: Optional[List[str]] = None,
):
    """
    categories:
      A类（按年季）：profit/operation/growth/balance/cash_flow/dupont
      B类（按日期）：perf_express/forecast
    """
    categories = categories or [
        "profit", "operation", "growth", "balance", "cash_flow", "dupont",
        "perf_express", "forecast"
    ]

    bs_login()

    # --- 股票池 ---
    if pool == "all":
        stocks = get_stock_list_bs("all", day=get_latest_end_date())
    else:
        stocks = get_stock_list_bs(pool)

    # --- 快照（与你K线一致）---
    snapshot_path = os.path.join(
        SNAPSHOT_DIR, f"{pool}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    stocks.to_csv(snapshot_path, index=False, encoding="utf-8")
    stocks.to_csv(os.path.join(META_DIR, f"stock_list_{pool}.csv"),
                  index=False, encoding="utf-8")

    current_codes = set(stocks["code"].tolist())
    last_snapshot_codes = load_last_snapshot(pool)

    # fundamental 目录里历史出现过的代码：对每个 category 都汇总
    hist_codes = set()
    for c in categories:
        hist_codes |= get_codes_from_data_dir(pool, c)

    final_codes = current_codes | hist_codes | last_snapshot_codes
    codes = sorted(final_codes)

    print(f"[股票池] 本次 {len(current_codes)} 只")
    print(f"[历史文件] {len(hist_codes)} 只")
    print(f"[最终更新] 共 {len(codes)} 只")
    print(f"[类别] {categories}")

    for code in tqdm(codes):
        for cat in categories:
            try:
                if cat in YQ_ENDPOINTS:
                    download_yq_table_for_stock(code, pool, cat)
                elif cat in DATE_ENDPOINTS:
                    download_date_table_for_stock(code, pool, cat)
                else:
                    raise ValueError(f"unknown category: {cat}")
            except Exception as e:
                failed_list.append({
                    "code": code,
                    "category": cat,
                    "error": str(e),
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

    # 失败日志
    if failed_list:
        fail_path = os.path.join(BASE_DIR, "failed_fundamental.csv")
        pd.DataFrame(failed_list).to_csv(fail_path, index=False, encoding="utf-8")
        print(f"[失败数量] {len(failed_list)}，已写入 {fail_path}")
    else:
        print("[成功] 基本面数据下载完成，无失败！")


if __name__ == "__main__":
    # 示例：中证500全量更新
    run_fundamental_download(pool="zz500")
    bs_logout()
