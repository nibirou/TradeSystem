# -*- coding: utf-8 -*-
"""
场内 ETF 历史行情下载器（AkShare 版）
------------------------------------------------
功能：
1. 通过 ETF 实时行情接口获取 ETF 代码列表
   - 东方财富: ak.fund_etf_spot_em()
   - 新浪财经: ak.fund_etf_category_sina(symbol="ETF基金")

2. 再基于 ETF 代码下载历史行情
   - 东方财富: ak.fund_etf_hist_em()
   - 新浪财经: ak.fund_etf_hist_sina()

3. 东财 / 新浪数据分开保存
4. 每只 ETF 单独保存 csv + parquet
5. 支持 ETF 列表快照
6. 支持失败记录
7. 支持增量更新
"""

import os
import time
import random
from datetime import datetime, timedelta
from typing import Optional, Set

import pandas as pd
import akshare as ak
from tenacity import retry, stop_after_attempt, wait_random
from tqdm import tqdm

# ===========================
#         路径配置
# ===========================
# BASE_DIR = "/workspace/Quant/data_baostock"
BASE_DIR = "D:/PythonProject/Quant/data_baostock"
ETF_BASE_DIR = os.path.join(BASE_DIR, "etf_hist")
META_DIR = os.path.join(BASE_DIR, "metadata")
ETF_META_DIR = os.path.join(META_DIR, "etf_metadata")
SNAPSHOT_DIR = os.path.join(ETF_META_DIR, "etf_snapshots")

# 东财
EM_DIR = os.path.join(ETF_BASE_DIR, "eastmoney")
EM_HIST_DIR = os.path.join(EM_DIR, "daily")
EM_LIST_PATH = os.path.join(ETF_META_DIR, "etf_list_eastmoney.csv")

# 新浪
SINA_DIR = os.path.join(ETF_BASE_DIR, "sina")
SINA_HIST_DIR = os.path.join(SINA_DIR, "daily")
SINA_LIST_PATH = os.path.join(ETF_META_DIR, "etf_list_sina.csv")

for path in [
    BASE_DIR, ETF_BASE_DIR, META_DIR, ETF_META_DIR, SNAPSHOT_DIR,
    EM_DIR, EM_HIST_DIR, SINA_DIR, SINA_HIST_DIR
]:
    os.makedirs(path, exist_ok=True)

failed_em = []
failed_sina = []


# ===========================
#         通用工具
# ===========================
def safe_sleep(a: float = 0.3, b: float = 1.0):
    time.sleep(random.uniform(a, b))


def normalize_date_str(dt) -> str:
    return pd.to_datetime(dt).strftime("%Y-%m-%d")


def get_today_ymd() -> str:
    return datetime.now().strftime("%Y%m%d")


def get_now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_dual(df: pd.DataFrame, csv_path: str, parquet_path: str):
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_parquet(parquet_path, index=False)


def load_last_snapshot_codes(list_path: str, code_col: str = "code") -> Set[str]:
    if not os.path.exists(list_path):
        return set()
    try:
        df = pd.read_csv(list_path, dtype={code_col: str})
        if code_col not in df.columns:
            return set()
        return set(df[code_col].dropna().astype(str).tolist())
    except Exception:
        return set()


def get_codes_from_hist_dir(hist_dir: str) -> Set[str]:
    if not os.path.exists(hist_dir):
        return set()

    codes = set()
    for fn in os.listdir(hist_dir):
        if not fn.endswith(".csv"):
            continue
        # 约定文件名: 510050.csv / sh510050.csv
        code = fn.replace(".csv", "")
        codes.add(code)
    return codes


def infer_incremental_start_from_csv(csv_path: str, date_col: str) -> Optional[str]:
    """
    根据已有 csv 中最大日期，推断下一次增量起始日 = max_date + 1天
    """
    if not os.path.exists(csv_path):
        return None

    try:
        old_df = pd.read_csv(csv_path)
        if old_df.empty or date_col not in old_df.columns:
            return None

        max_date = pd.to_datetime(old_df[date_col]).max()
        if pd.isna(max_date):
            return None

        next_date = max_date + timedelta(days=1)
        return next_date.strftime("%Y%m%d")
    except Exception:
        return None


def clean_em_hist_df(df: pd.DataFrame, symbol: str, name: Optional[str] = None, adjust: str = "") -> pd.DataFrame:
    """
    东方财富 ETF 历史行情标准化
    原字段示例：
    日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df = df.replace("", pd.NA)

    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_chg",
        "涨跌额": "chg",
        "换手率": "turnover",
    }
    df = df.rename(columns=rename_map)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    numeric_cols = [
        "open", "close", "high", "low",
        "volume", "amount", "amplitude",
        "pct_chg", "chg", "turnover"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["code"] = str(symbol)
    if name is not None:
        df["name"] = name
    df["source"] = "eastmoney"
    df["adjust"] = adjust if adjust else "raw"

    # 排序去重
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    return df


def clean_sina_hist_df(df: pd.DataFrame, symbol: str, name: Optional[str] = None) -> pd.DataFrame:
    """
    新浪 ETF 历史行情标准化
    原字段示例：
    date, open, high, low, close, volume
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df = df.replace("", pd.NA)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["code"] = str(symbol)
    if name is not None:
        df["name"] = name
    df["source"] = "sina"

    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    return df


# ===========================
#       ETF 列表获取
# ===========================
@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def get_etf_list_em() -> pd.DataFrame:
    df = ak.fund_etf_spot_em()
    if df is None or df.empty:
        raise ValueError("ak.fund_etf_spot_em() 返回空数据")

    df = df.copy()
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    df["名称"] = df["名称"].astype(str)

    out = pd.DataFrame({
        "code": df["代码"],
        "name": df["名称"],
        "source": "eastmoney",
    })

    # 如有数据日期/更新时间，一并保留
    if "数据日期" in df.columns:
        out["data_date"] = pd.to_datetime(df["数据日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "更新时间" in df.columns:
        out["update_time"] = pd.to_datetime(df["更新时间"], errors="coerce").astype("string")

    out = out.drop_duplicates(subset=["code"], keep="last").sort_values("code").reset_index(drop=True)
    return out


@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def get_etf_list_sina() -> pd.DataFrame:
    df = ak.fund_etf_category_sina(symbol="ETF基金")
    if df is None or df.empty:
        raise ValueError('ak.fund_etf_category_sina(symbol="ETF基金") 返回空数据')

    df = df.copy()
    df["代码"] = df["代码"].astype(str)
    df["名称"] = df["名称"].astype(str)

    out = pd.DataFrame({
        "code": df["代码"],   # 例如 sh510050
        "name": df["名称"],
        "source": "sina",
    })
    out = out.drop_duplicates(subset=["code"], keep="last").sort_values("code").reset_index(drop=True)
    return out


def save_etf_list_snapshot(df: pd.DataFrame, source: str):
    snapshot_path = os.path.join(SNAPSHOT_DIR, f"etf_list_{source}_{get_now_ts()}.csv")
    df.to_csv(snapshot_path, index=False, encoding="utf-8-sig")

    if source == "eastmoney":
        df.to_csv(EM_LIST_PATH, index=False, encoding="utf-8-sig")
    elif source == "sina":
        df.to_csv(SINA_LIST_PATH, index=False, encoding="utf-8-sig")
    else:
        raise ValueError(f"未知 source: {source}")


# ===========================
#     历史行情下载 - 东方财富
# ===========================
@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def query_em_hist(symbol: str, start_date: str, end_date: str, period: str = "daily", adjust: str = "") -> pd.DataFrame:
    df = ak.fund_etf_hist_em(
        symbol=symbol,
        period=period,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )
    if df is None:
        raise ValueError(f"东财历史行情返回 None: {symbol}")
    return df


def get_etf_hist_em(symbol: str,
                    name: Optional[str] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    period: str = "daily",
                    adjust: str = "hfq"):
    """
    symbol: 6位 ETF 代码，如 510050 / 159915
    adjust: "", "qfq", "hfq"
    """

    end_date = end_date or get_today_ymd()
    save_dir = os.path.join(EM_HIST_DIR, adjust if adjust else "raw")
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, f"{symbol}.csv")
    parquet_path = os.path.join(save_dir, f"{symbol}.parquet")

    old_df = None
    inc_start = infer_incremental_start_from_csv(csv_path, date_col="date")

    if inc_start is not None:
        start_date = inc_start
        try:
            old_df = pd.read_csv(csv_path)
        except Exception:
            old_df = None
    else:
        start_date = start_date or "20000101"

    # 若已是最新，则跳过
    if inc_start is not None and pd.to_datetime(start_date) > pd.to_datetime(end_date):
        return

    raw_df = query_em_hist(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        period=period,
        adjust=adjust,
    )

    if raw_df is None or raw_df.empty:
        print(f"[东财空数据] {symbol}")
        failed_em.append(symbol)
        return

    new_df = clean_em_hist_df(raw_df, symbol=symbol, name=name, adjust=adjust)

    if new_df.empty:
        print(f"[东财清洗后空数据] {symbol}")
        failed_em.append(symbol)
        return

    if old_df is not None and not old_df.empty:
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
        merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    else:
        merged = new_df

    save_dual(merged, csv_path, parquet_path)


# ===========================
#      历史行情下载 - 新浪
# ===========================
@retry(stop=stop_after_attempt(3), wait=wait_random(1, 3))
def query_sina_hist(symbol: str) -> pd.DataFrame:
    df = ak.fund_etf_hist_sina(symbol=symbol)
    if df is None:
        raise ValueError(f"新浪历史行情返回 None: {symbol}")
    return df


def get_etf_hist_sina(symbol: str, name: Optional[str] = None):
    """
    symbol: 新浪格式，如 sh510050 / sz159915
    注：
    - 新浪接口通常直接返回全历史
    - 这里为了稳妥，采取“全量拉取 + 本地去重覆盖”
    """
    csv_path = os.path.join(SINA_HIST_DIR, f"{symbol}.csv")
    parquet_path = os.path.join(SINA_HIST_DIR, f"{symbol}.parquet")

    raw_df = query_sina_hist(symbol)
    if raw_df is None or raw_df.empty:
        print(f"[新浪空数据] {symbol}")
        failed_sina.append(symbol)
        return

    df = clean_sina_hist_df(raw_df, symbol=symbol, name=name)
    if df.empty:
        print(f"[新浪清洗后空数据] {symbol}")
        failed_sina.append(symbol)
        return

    if os.path.exists(csv_path):
        try:
            old_df = pd.read_csv(csv_path)
            if not old_df.empty:
                merged = pd.concat([old_df, df], ignore_index=True)
                merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
                merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
            else:
                merged = df
        except Exception:
            merged = df
    else:
        merged = df

    save_dual(merged, csv_path, parquet_path)


# ===========================
#       主流程 - 东方财富
# ===========================
def run_em_history_download(adjust: str = "hfq",
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None):
    etf_df = get_etf_list_em()
    save_etf_list_snapshot(etf_df, source="eastmoney")

    current_codes = set(etf_df["code"].tolist())
    hist_codes = get_codes_from_hist_dir(os.path.join(EM_HIST_DIR, adjust if adjust else "raw"))
    last_codes = load_last_snapshot_codes(EM_LIST_PATH, code_col="code")

    final_codes = sorted(current_codes | hist_codes | last_codes)
    name_map = dict(zip(etf_df["code"], etf_df["name"]))

    print(f"[东财ETF] 本次列表 {len(current_codes)} 只")
    print(f"[东财ETF] 历史文件 {len(hist_codes)} 只")
    print(f"[东财ETF] 最终更新 {len(final_codes)} 只")

    for code in tqdm(final_codes, desc=f"EastMoney ETF hist ({adjust if adjust else 'raw'})"):
        try:
            get_etf_hist_em(
                symbol=code,
                name=name_map.get(code, code),
                start_date=start_date,
                end_date=end_date,
                period="daily",
                adjust=adjust,
            )
            safe_sleep(0.2, 0.8)
        except Exception as e:
            failed_em.append(code)
            print(f"[东财失败] {code}: {e}")

    if failed_em:
        failed_path = os.path.join(ETF_META_DIR, f"failed_etf_eastmoney_{adjust if adjust else 'raw'}.csv")
        pd.DataFrame({"code": sorted(set(failed_em))}).to_csv(failed_path, index=False, encoding="utf-8-sig")
        print(f"[东财失败数量] {len(set(failed_em))}，已写入: {failed_path}")
    else:
        print("[东财成功] 全部下载完成！")


# ===========================
#        主流程 - 新浪
# ===========================
def run_sina_history_download():
    etf_df = get_etf_list_sina()
    save_etf_list_snapshot(etf_df, source="sina")

    current_codes = set(etf_df["code"].tolist())
    hist_codes = get_codes_from_hist_dir(SINA_HIST_DIR)
    last_codes = load_last_snapshot_codes(SINA_LIST_PATH, code_col="code")

    final_codes = sorted(current_codes | hist_codes | last_codes)
    name_map = dict(zip(etf_df["code"], etf_df["name"]))

    print(f"[新浪ETF] 本次列表 {len(current_codes)} 只")
    print(f"[新浪ETF] 历史文件 {len(hist_codes)} 只")
    print(f"[新浪ETF] 最终更新 {len(final_codes)} 只")

    for code in tqdm(final_codes, desc="Sina ETF hist"):
        try:
            get_etf_hist_sina(
                symbol=code,
                name=name_map.get(code, code)
            )
            safe_sleep(0.2, 0.8)
        except Exception as e:
            failed_sina.append(code)
            print(f"[新浪失败] {code}: {e}")

    if failed_sina:
        failed_path = os.path.join(ETF_META_DIR, "failed_etf_sina.csv")
        pd.DataFrame({"code": sorted(set(failed_sina))}).to_csv(failed_path, index=False, encoding="utf-8-sig")
        print(f"[新浪失败数量] {len(set(failed_sina))}，已写入: {failed_path}")
    else:
        print("[新浪成功] 全部下载完成！")


# ===========================
#            MAIN
# ===========================
if __name__ == "__main__":
    # 东方财富：建议量化研究默认用 hfq
    # run_em_history_download(adjust="hfq")
    # run_em_history_download(adjust="qfq")
    # run_em_history_download(adjust="")

    # 新浪：仅一套日线历史
    run_sina_history_download()