# -*- coding: utf-8 -*-
"""
个股 消息面 + 资金面 数据提取（单文件加强版）
- 列名鲁棒性 + 日期排序
- 缓存过期刷新
- 重试 + 指数退避 + 失败日志
- 龙虎榜：flag & 次数
- 并发提速 + 轻量限速
"""
import os
import time
import random
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import akshare as ak
from snownlp import SnowNLP
from tqdm import tqdm

# =================== 可配置参数 ===================
META_DIR = "data/metadata"
SENTIMENT_DIR = os.path.join(META_DIR, "sentiment")
FUND_DIR = os.path.join(META_DIR, "fund")
OUT_PATH = os.path.join(META_DIR, "sentiment_fund_factors.csv")
FAIL_LOG = os.path.join(META_DIR, "failures.log")

# 缓存文件最大有效期（小时），超过则刷新抓取
MAX_AGE_HOURS_NEWS = 24
MAX_AGE_HOURS_FUND = 6

# 抓取重试
RETRIES = 3
BASE_WAIT = 0.8  # 秒，指数退避基数

# 并发线程数
MAX_WORKERS = 8

# 轻量限速：每只股票抓完后随机 sleep 区间（秒）
RANDOM_SLEEP_RANGE = (0.05, 0.2)

# 新闻情绪：取“最近 topk 条”用于情绪打分
NEWS_TOPK = 20

# 资金面：近 N 日主力净流入合计（默认 5 日）
FUND_DAYS = 5

# 龙虎榜：统计近 N 天是否上榜（flag）以及上榜次数（count）
LHB_DAYS = 30

# =================================================

os.makedirs(SENTIMENT_DIR, exist_ok=True)
os.makedirs(FUND_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)


# =================== 工具函数 ===================
def _need_refresh(path: str, max_age_hours: int) -> bool:
    if not os.path.exists(path):
        return True
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return (datetime.now() - mtime).total_seconds() > max_age_hours * 3600
    except Exception:
        return True


def _retry(fn, tries=RETRIES, base_wait=BASE_WAIT):
    last_err = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if i < tries - 1:
                time.sleep(base_wait * (2 ** i) + random.random() * 0.3)
    raise last_err


def _log_fail(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(FAIL_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def _random_pause():
    low, high = RANDOM_SLEEP_RANGE
    time.sleep(random.uniform(low, high))


def _get_first_existing_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_datetime_sorted(df: pd.DataFrame, date_candidates=("日期", "时间", "发布时间", "time", "pub_time")):
    """
    找到一个日期列，转换为 datetime 并按升序排序；若找到多列，取第一个
    """
    for c in date_candidates:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
                df = df.sort_values(c, ascending=True)  # 升序，便于 tail(N) 取最近 N 日
                return df
            except Exception:
                continue
    return df  # 找不到日期列就原样返回


# =================== 新闻情绪因子 ===================
def get_news_sentiment(code: str, topk: int = NEWS_TOPK) -> float:
    """
    获取个股最近新闻标题情感平均值（[-1, 1]），缓存 24h 刷新
    """
    path = os.path.join(SENTIMENT_DIR, f"{code}.csv")

    def fetch():
        return ak.stock_news_em(symbol=code)

    # 读取或刷新缓存
    df = None
    try:
        if _need_refresh(path, MAX_AGE_HOURS_NEWS):
            df = _retry(fetch)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(path, index=False, encoding="utf-8-sig")
        else:
            df = pd.read_csv(path)
    except Exception as e:
        _log_fail(f"NEWS_FETCH_FAIL {code} -> {e}")
        # 失败就尝试用旧缓存
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
            except Exception:
                df = None

    if df is None or df.empty:
        return 0.0

    # 找标题列
    title_col = _get_first_existing_col(df, ["新闻内容", "新闻标题", "标题", "title"])
    if not title_col:
        # 没有可用标题列
        return 0.0

    # 日期排序：优先找日期列，升序后用 tail(topk) 取“最近 topk 条”
    df = _to_datetime_sorted(df)
    if len(df) > topk:
        df_recent = df.tail(topk)
    else:
        df_recent = df

    titles = df_recent[title_col].dropna().astype(str).tolist()
    if not titles:
        return 0.0

    scores = []
    for t in titles:
        try:
            s = SnowNLP(t).sentiments  # [0,1]
            scores.append(s)
        except Exception as e:
            _log_fail(f"SNOWNLP_FAIL {code} -> {e}")
            continue

    if not scores:
        return 0.0

    # 映射到 [-1, 1] 后取均值
    mapped = [(s - 0.5) * 2 for s in scores]
    return round(float(pd.Series(mapped).mean()), 4)


# =================== 资金流因子 ===================
def get_fund_flow(code: str, days: int = FUND_DAYS) -> float:
    """
    获取近 days 日“主力净流入-净额”的合计，单位“万元”。
    缓存 6h 刷新；显式按日期排序后取最近 N 日。
    """
    path = os.path.join(FUND_DIR, f"{code}.csv")

    def fetch():
        # market="沪深"：A股
        return ak.stock_individual_fund_flow(stock=code, market="沪深")

    df = None
    try:
        if _need_refresh(path, MAX_AGE_HOURS_FUND):
            df = _retry(fetch)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(path, index=False, encoding="utf-8-sig")
        else:
            df = pd.read_csv(path)
    except Exception as e:
        _log_fail(f"FUND_FETCH_FAIL {code} -> {e}")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
            except Exception:
                df = None

    if df is None or df.empty:
        return 0.0

    # 日期排序
    df = _to_datetime_sorted(df)

    # 数值列名（不同接口版本可能有差异）
    flow_col = _get_first_existing_col(df, ["主力净流入-净额", "主力净流入净额", "净流入净额"])
    if not flow_col:
        return 0.0

    df[flow_col] = pd.to_numeric(df[flow_col], errors="coerce").fillna(0.0)

    # 取最近 N 日
    if len(df) >= days:
        recent = df.tail(days)
    else:
        recent = df

    total = float(recent[flow_col].sum())

    # 常见情况：接口给的是“元”。如果你确认是“万元”，把下面一行注释掉即可。
    total_wan = total / 1e4

    return round(total_wan, 2)


# =================== 龙虎榜热度 ===================
def get_lhb_features(code: str, days: int = LHB_DAYS):
    """
    返回 (flag, count)：
    - flag: 近 days 天是否上过龙虎榜（0/1）
    - count: 近 days 天上过几次（整数）
    """
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    def fetch():
        return ak.stock_lhb_stock_statistic_em(symbol=code, start_date=start_date, end_date=end_date)

    try:
        df = _retry(fetch)
    except Exception as e:
        _log_fail(f"LHB_FETCH_FAIL {code} -> {e}")
        return 0, 0

    if df is None or df.empty:
        return 0, 0

    # 简单计数：返回的每行即一次上榜记录（不同接口版本可能统计口径略不同）
    count = len(df)
    flag = 1 if count > 0 else 0
    return flag, int(count)


# =================== 批量提取（并发） ===================
def _one_code(code: str) -> dict:
    try:
        senti = get_news_sentiment(code, topk=NEWS_TOPK)
    except Exception as e:
        _log_fail(f"SENTIMENT_FAIL {code} -> {e}")
        senti = 0.0

    try:
        net_in = get_fund_flow(code, days=FUND_DAYS)
    except Exception as e:
        _log_fail(f"FUND_FAIL {code} -> {e}")
        net_in = 0.0

    try:
        lhb_flag, lhb_count = get_lhb_features(code, days=LHB_DAYS)
    except Exception as e:
        _log_fail(f"LHB_FAIL {code} -> {e}")
        lhb_flag, lhb_count = 0, 0

    _random_pause()

    return {
        "代码": code,
        "新闻情绪": senti,
        f"主力净流入_{FUND_DAYS}日(万元)": net_in,
        "龙虎榜": lhb_flag,
        f"龙虎榜_{LHB_DAYS}日次数": lhb_count,
    }


def extract_sentiment_fund_features(code_list, max_workers: int = MAX_WORKERS) -> pd.DataFrame:
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_one_code, code): code for code in code_list}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Extracting"):
            try:
                rows.append(fut.result())
            except Exception as e:
                code = "UNKNOWN"
                try:
                    code = futs[fut]
                except Exception:
                    pass
                _log_fail(f"TASK_FAIL {code} -> {e}")
    df = pd.DataFrame(rows)
    # 附加元数据列（可选）
    df.insert(0, "生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    df.insert(1, "新闻topk", NEWS_TOPK)
    df.insert(2, "资金天数", FUND_DAYS)
    df.insert(3, "龙虎榜天数", LHB_DAYS)
    return df


# =================== 主程序 ===================
if __name__ == '__main__':
    # 示例：跑一遍股票列表（需包含“代码”列；建议字符串类型保留前导0）
    stock_list_path = os.path.join(META_DIR, "stock_list.csv")
    if not os.path.exists(stock_list_path):
        raise FileNotFoundError(f"未找到股票列表：{stock_list_path}，请先准备包含‘代码’列的CSV。")

    df_list = pd.read_csv(stock_list_path, dtype=str)
    if "代码" not in df_list.columns:
        # 做一个兜底：常见字段名 candidates
        cand = _get_first_existing_col(df_list, ["代码", "symbol", "ts_code", "code"])
        if cand and cand != "代码":
            df_list.rename(columns={cand: "代码"}, inplace=True)
        if "代码" not in df_list.columns:
            raise ValueError("股票列表缺少‘代码’列（或常见等价列）。")

    codes = df_list["代码"].dropna().astype(str).unique().tolist()
    df_feat = extract_sentiment_fund_features(codes, max_workers=MAX_WORKERS)
    df_feat.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(df_feat.head())
    print(f"\n✅ 已保存：{OUT_PATH}")
    print(f"📄 失败日志（如有）：{FAIL_LOG}")
