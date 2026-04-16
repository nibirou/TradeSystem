# 使用akshare 申万行业查询接口 确定个股行业信息

# 先用你现有的股票池获取逻辑拿到 hs300 / zz500 / all 的股票列表。
# 再用 ak.sw_index_third_info() 拿到全部申万三级行业代码。
# 对每个三级行业调用 ak.sw_index_third_cons(symbol=...)，把所有行业成份股拉下来。
# 建立 股票代码 -> 行业成份明细行 的映射。
# 最后按股票池逐只保存成 csv/parquet，每只股票保存的表结构保持和 sw_index_third_cons 返回一致。

import os
import time
import random
from datetime import datetime
from typing import Dict, List, Optional

import akshare as ak
import baostock as bs
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random
from tqdm import tqdm

# ===========================
#        路径配置
# ===========================
# BASE_DIR = "E:/pythonProject/data_baostock"
# BASE_DIR = "/workspace/Quant/data_baostock"
BASE_DIR = "D:/PythonProject/Quant/data_baostock"
HIST_DIR = os.path.join(BASE_DIR, "stock_hist")
META_DIR = os.path.join(BASE_DIR, "metadata")
SNAPSHOT_DIR = os.path.join(META_DIR, "stock_snapshots")

# 新增：申万行业保存目录
SW_DIR = os.path.join(BASE_DIR, "sw_industry_third_cons")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(SW_DIR, exist_ok=True)

failed_industry_list = []
failed_stock_list = []


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


def get_latest_end_date():
    """
    根据 Baostock 交易日历，获取最近一个交易日
    """
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

    latest_trade_day = trade_days[-1].strftime("%Y-%m-%d")
    return latest_trade_day


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
        # rs = bs.query_all_stock(day=day)
        rs = bs.query_all_stock(day="2026-04-15")
    else:
        raise ValueError(f"未知股票池模式：{mode}")

    df = _bs_query_to_df(rs)

    if "code_name" in df.columns:
        df = df.rename(columns={"code_name": "name"})
    else:
        df["name"] = df["code"]

    return df[["code", "name"]]


# ===========================
#      代码格式转换
# ===========================
def bs_code_to_ak(code: str) -> str:
    """
    baostock: sh.600000 / sz.000001
    akshare申万成份: 600000.SH / 000001.SZ
    """
    if not isinstance(code, str) or "." not in code:
        return code

    market, symbol = code.split(".")
    market = market.upper()
    return f"{symbol}.{market}"


def ak_code_to_bs(code: str) -> str:
    """
    600000.SH / 000001.SZ -> sh.600000 / sz.000001
    """
    if not isinstance(code, str) or "." not in code:
        return code

    symbol, market = code.split(".")
    market = market.lower()
    return f"{market}.{symbol}"


def code_to_filename(code: str) -> str:
    return code.replace(".", "_")


# ===========================
#      通用清洗
# ===========================
def clean_df_for_save(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace("", pd.NA)
    return df


# ===========================
#      AKShare 安全调用
# ===========================
@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def safe_sw_index_third_info() -> pd.DataFrame:
    df = ak.sw_index_third_info()
    if df is None or df.empty:
        raise ValueError("ak.sw_index_third_info 返回空数据")
    return df


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def safe_sw_index_third_cons(symbol: str) -> pd.DataFrame:
    df = ak.sw_index_third_cons(symbol=symbol)
    if df is None:
        raise ValueError(f"ak.sw_index_third_cons({symbol}) 返回 None")
    return df


# ===========================
#      拉取全部申万三级成份
# ===========================
def build_stock_industry_mapping() -> Dict[str, pd.DataFrame]:
    """
    返回:
        {
            "600000.SH": DataFrame(该股票对应的申万三级行业行，列结构与 sw_index_third_cons 一致),
            ...
        }

    正常情况下，一只股票通常只有1行；
    为防止历史口径/异常情况，这里统一按 DataFrame 保存。
    """
    info_df = safe_sw_index_third_info()
    info_df = clean_df_for_save(info_df)

    if "行业代码" not in info_df.columns:
        raise KeyError("sw_index_third_info 返回结果缺少 '行业代码' 列")

    industry_codes = info_df["行业代码"].dropna().astype(str).unique().tolist()
    print(f"[申万三级行业] 共 {len(industry_codes)} 个行业代码")

    stock_map: Dict[str, List[pd.DataFrame]] = {}

    for industry_code in tqdm(industry_codes, desc="拉取申万三级行业成份"):
        try:
            cons_df = safe_sw_index_third_cons(symbol=industry_code)
            cons_df = clean_df_for_save(cons_df)

            if cons_df.empty:
                continue

            if "股票代码" not in cons_df.columns:
                print(f"[警告] {industry_code} 返回结果缺少 '股票代码' 列，跳过")
                failed_industry_list.append(industry_code)
                continue

            # 按股票代码聚合
            for stock_code, g in cons_df.groupby("股票代码", dropna=False):
                if pd.isna(stock_code):
                    continue
                stock_code = str(stock_code)
                stock_map.setdefault(stock_code, []).append(g.copy())

            time.sleep(random.uniform(0.3, 0.8))

        except Exception as e:
            failed_industry_list.append(industry_code)
            print(f"[失败][行业] {industry_code}: {e}")

    # 合并同股票的多块 DataFrame
    merged_stock_map: Dict[str, pd.DataFrame] = {}
    for stock_code, dfs in stock_map.items():
        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.drop_duplicates()
        merged_stock_map[stock_code] = merged

    print(f"[行业映射完成] 共覆盖股票 {len(merged_stock_map)} 只")
    return merged_stock_map


# ===========================
#      保存单只股票行业信息
# ===========================
def save_stock_industry_info(
    stock_code_ak: str,
    df: pd.DataFrame,
    pool: str,
):
    """
    保存格式:
      /workspace/Quant/data_baostock/sw_industry_third_cons/{pool}/{code}.csv
      /workspace/Quant/data_baostock/sw_industry_third_cons/{pool}/{code}.parquet
    """
    save_dir = os.path.join(SW_DIR, pool)
    os.makedirs(save_dir, exist_ok=True)

    filename = code_to_filename(stock_code_ak)
    csv_path = os.path.join(save_dir, f"{filename}.csv")
    parquet_path = os.path.join(save_dir, f"{filename}.parquet")

    out_df = clean_df_for_save(df)
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    out_df.to_parquet(parquet_path, index=False)

# ===========================
#  更新：解决股票列表动态更新问题
#  需要更新的股票 = 本次股票池 ∪ 历史下载目录中已存在的股票 ∪ 上一次快照中存在但本次消失的股票
# ===========================
def get_codes_from_hist_dir(pool, freq="d"):
    hist_dir = os.path.join(HIST_DIR, pool, freq)
    if not os.path.exists(hist_dir):
        return set()

    codes = set()
    for fn in os.listdir(hist_dir):
        if not fn.endswith(".csv"):
            continue
        # 例如：sh_600519_5.csv
        code = fn.replace(".csv", "").rsplit("_", 1)[0]
        code = code.replace("_", ".")
        codes.add(code)
    return codes

# 读取旧快照
def load_last_snapshot(pool):
    path = os.path.join(META_DIR, f"stock_list_{pool}.csv")
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(df["code"].tolist())

# ===========================
#      保存股票池快照
# ===========================
def save_pool_snapshot(stocks: pd.DataFrame, pool: str):
    snapshot_path = os.path.join(
        SNAPSHOT_DIR, f"{pool}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    stocks.to_csv(snapshot_path, index=False, encoding="utf-8")
    stocks.to_csv(os.path.join(META_DIR, f"stock_list_{pool}.csv"),
                  index=False, encoding="utf-8")


# ===========================
#      主流程
# ===========================
def run_sw_industry_download(pool: str = "hs300"):
    """
    pool 支持:
      - hs300
      - zz500
      - all
      - sz50（如果你也想支持，可以直接用）
    """
    # 1) 获取股票池
    bs_login()
    try:
        if pool == "all":
            stocks = get_stock_list_bs("all", day=get_latest_end_date())
        else:
            stocks = get_stock_list_bs(pool)
    finally:
        bs_logout()

    # save_pool_snapshot(stocks, pool)

    current_codes = set(stocks["code"].tolist()) # 本次股票池
    
    hist_codes = get_codes_from_hist_dir(pool, "d") # 历史股票池（有数据存储记录的股票）
    last_snapshot_codes = load_last_snapshot(pool) # 上一次快照股票池

    final_codes = current_codes | hist_codes | last_snapshot_codes

    print(f"[股票池] 本次 {len(current_codes)} 只")
    print(f"[历史文件] {len(hist_codes)} 只")
    print(f"[股票池] 上次 {len(last_snapshot_codes)} 只")

    save_pool_snapshot(stocks, pool) # 保存本次新快照

    # baostock -> akshare 代码格式
    stocks = final_codes.copy()
    print(len(stocks))

    stocks_df = pd.DataFrame({"code": list(stocks)})
    stocks_df["code_ak"] = stocks_df["code"].astype(str).map(bs_code_to_ak)

    pool_codes_ak = set(stocks_df["code_ak"].dropna().tolist())
    print(f"[最终更新] 共 {len(pool_codes_ak)} 只")

    # 2) 构建 股票 -> 申万三级行业明细 映射
    stock_industry_map = build_stock_industry_mapping()

    # 3) 保存
    hit = 0
    miss = 0

    for stock_code_ak in tqdm(sorted(pool_codes_ak), desc=f"保存 {pool} 个股申万行业信息"):
        try:
            df = stock_industry_map.get(stock_code_ak)

            if df is None or df.empty:
                # 没查到时，也可以选择保存一个空表；这里先记失败更直观
                failed_stock_list.append(stock_code_ak)
                miss += 1
                continue

            save_stock_industry_info(stock_code_ak=stock_code_ak, df=df, pool=pool)
            hit += 1

        except Exception as e:
            failed_stock_list.append(stock_code_ak)
            print(f"[失败][个股] {stock_code_ak}: {e}")

    print(f"[完成] pool={pool}, 命中 {hit} 只, 未命中/失败 {miss + (len(failed_stock_list) - miss)} 只")

    # 4) 写失败日志
    if failed_industry_list:
        pd.DataFrame({"行业代码": failed_industry_list}).drop_duplicates().to_csv(
            os.path.join(META_DIR, f"failed_sw_industry_codes_{pool}.csv"),
            index=False,
            encoding="utf-8-sig"
        )
        print(f"[行业失败] {len(set(failed_industry_list))} 个，已写入 failed_sw_industry_codes_{pool}.csv")

    if failed_stock_list:
        pd.DataFrame({"股票代码": failed_stock_list}).drop_duplicates().to_csv(
            os.path.join(META_DIR, f"failed_sw_stock_codes_{pool}.csv"),
            index=False,
            encoding="utf-8-sig"
        )
        print(f"[个股失败] {len(set(failed_stock_list))} 只，已写入 failed_sw_stock_codes_{pool}.csv")
    else:
        print("[成功] 个股行业信息全部保存完成！")


# ===========================
#             MAIN
# ===========================
if __name__ == "__main__":
    # run_sw_industry_download(pool="hs300")
    # run_sw_industry_download(pool="zz500")
    run_sw_industry_download(pool="all")