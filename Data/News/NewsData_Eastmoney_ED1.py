# 通过东方财富新闻接口，获取个股新闻数据

# -*- coding: utf-8 -*-
"""
Baostock 股票池 + 东方财富个股新闻抓取（含正文）并保存
- 按股票保存 CSV
- 支持增量更新（按 新闻链接 去重）
- 超限裁剪（按 发布时间 保留最近 max_rows）
- 股票池并集：当前股票池 ∪ 历史新闻目录 ∪ 上次快照
- 失败/空结果记录
"""

import os
import json
import time
import random
from datetime import datetime

import baostock as bs
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests


# ===========================
#        路径配置
# ===========================
# BASE_DIR = "/workspace/Quant/data_baostock"
BASE_DIR = "D:/PythonProject/Quant/data_baostock"

NEWS_DIR = os.path.join(BASE_DIR, "data_em_news")       # 输出目录：按股票 CSV
META_DIR = os.path.join(BASE_DIR, "metadata")           # 股票池快照等
HIST_DIR = os.path.join(BASE_DIR, "stock_hist")         # 历史行情目录（用于补股票）

os.makedirs(NEWS_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)

FAILED_LIST = []
EMPTY_LIST = []


# ===========================
#      Baostock 登录
# ===========================
def bs_login():
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"Baostock login failed: {lg.error_msg}")

def bs_logout():
    bs.logout()

def _bs_query_to_df(rs):
    data_list = []
    while (rs.error_code == "0") & rs.next():
        data_list.append(rs.get_row_data())
    return pd.DataFrame(data_list, columns=rs.fields)

def get_latest_end_date():
    """
    根据 Baostock 交易日历，获取距离程序运行当天最近的一个交易日
    - 如果今天是交易日 → 返回今天
    - 如果今天不是交易日（周末/节假日）→ 返回最近交易日
    """
    today = datetime.now().strftime("%Y-%m-%d")
    year_start = f"{datetime.now().year}-01-01"

    # 避免 1 月 1~4 日查询不到交易日（跨年）
    if today in {f"{datetime.now().year}-01-01",
                 f"{datetime.now().year}-01-02",
                 f"{datetime.now().year}-01-03",
                 f"{datetime.now().year}-01-04"}:
        year_start = f"{datetime.now().year - 1}-01-01"

    rs = bs.query_trade_dates(start_date=year_start, end_date=today)

    data_list = []
    while (rs.error_code == "0") & rs.next():
        data_list.append(rs.get_row_data())

    if not data_list:
        raise RuntimeError("baostock query_trade_dates 返回空数据")

    df = pd.DataFrame(data_list, columns=rs.fields)
    df["calendar_date"] = pd.to_datetime(df["calendar_date"])
    df["is_trading_day"] = df["is_trading_day"].astype(int)

    trade_days = df[df["is_trading_day"] == 1]["calendar_date"].tolist()
    if not trade_days:
        raise RuntimeError("没有找到任何交易日")

    return trade_days[-1].strftime("%Y-%m-%d")

def get_stock_list_bs(mode="hs300", day=None):
    """
    返回 DataFrame: [code, name]
    code 形式类似：sh.600519 / sz.000001
    """
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
#   股票池并集：历史兜底
# ===========================
def get_codes_from_hist_dir(pool: str) -> set:
    """
    从 HIST_DIR/{pool}/*.csv 解析股票 code
    目录存在：/stock_hist/hs300/sh_600519.csv 这种
    """
    d = os.path.join(HIST_DIR, pool)
    if not os.path.exists(d):
        return set()

    codes = set()
    for fn in os.listdir(d):
        if fn.endswith(".csv"):
            codes.add(fn.replace(".csv", "").replace("_", "."))
    return codes

def load_last_snapshot(pool: str) -> set:
    """
    读取上次股票池快照：metadata/stock_list_{pool}.csv
    """
    path = os.path.join(META_DIR, f"stock_list_{pool}.csv")
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(df["code"].tolist())


# ===========================
#   东方财富：新闻列表接口
# ===========================
def stock_news_em(
    symbol: str = "600519",
    max_pages: int = 5,
    page_size: int = 10,
    sleep: float = 0.5
) -> pd.DataFrame:
    """
    东方财富 - 个股新闻（支持分页）
    - symbol: 纯数字证券代码，如 600519
    """
    url = "https://search-api-web.eastmoney.com/search/jsonp"
    all_items = []

    for page in range(1, max_pages + 1):
        ts = int(pd.Timestamp.now().timestamp() * 1000)
        cb = f"jQuery3510{ts}"
        _ts = str(ts)

        inner_param = {
            "uid": "",
            "keyword": symbol,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "default",
                    "sort": "time",
                    "pageIndex": page,
                    "pageSize": page_size,
                    "preTag": "<em>",
                    "postTag": "</em>",
                }
            },
        }

        params = {
            "cb": cb,
            "param": json.dumps(inner_param, ensure_ascii=False),
            "_": _ts,
        }

        headers = {
            "accept": "*/*",
            "accept-language": "zh-CN,zh;q=0.9",
            "referer": f"https://so.eastmoney.com/news/s?keyword={symbol}&sort=time",
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/143.0.0.0 Safari/537.36"
            ),
        }

        resp = cffi_requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()

        text = resp.text
        prefix = f"{cb}("
        if not text.startswith(prefix):
            break

        data_json = json.loads(text[len(prefix):-1])
        items = data_json.get("result", {}).get("cmsArticleWebOld", [])
        if not items:
            break

        all_items.extend(items)
        time.sleep(sleep)

    if not all_items:
        return pd.DataFrame()

    df = pd.DataFrame(all_items)
    df.rename(
        columns={
            "date": "发布时间",
            "mediaName": "文章来源",
            "title": "新闻标题",
            "content": "新闻内容",
            "url": "新闻链接",
        },
        inplace=True,
    )

    df["关键词"] = symbol
    df = df[["关键词", "新闻标题", "新闻内容", "发布时间", "文章来源", "新闻链接"]]

    for col in ["新闻标题", "新闻内容"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"<em>|</em>", "", regex=True)
            .str.replace(r"\u3000|\r\n", " ", regex=True)
        )

    df["发布时间"] = pd.to_datetime(df["发布时间"], errors="coerce")
    return df


# ===========================
#   东方财富：新闻正文抓取
# ===========================
def fetch_article_html(url: str, timeout: int = 15) -> str:
    headers = {
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/143.0.0.0 Safari/537.36"
        ),
        "accept-language": "zh-CN,zh;q=0.9",
        "referer": "https://finance.eastmoney.com/",
    }
    resp = cffi_requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text

def parse_article_content(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    content_div = soup.find("div", id="ContentBody")
    if not content_div:
        return ""

    paragraphs = []
    # 只取 ContentBody 下直接 p（避免广告/脚本区）
    for p in content_div.find_all("p", recursive=False):
        if "em_media" in p.get("class", []):
            continue
        txt = p.get_text(strip=True)
        if txt:
            paragraphs.append(txt)

    return "\n".join(paragraphs)

def enrich_news_content(
    df: pd.DataFrame,
    sleep: float = 0.5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    为每条新闻 URL 抓取完整正文
    """
    full_contents = []
    for _, row in df.iterrows():
        url = str(row["新闻链接"])
        try:
            html = fetch_article_html(url)
            content = parse_article_content(html)
        except Exception as e:
            if verbose:
                print(f"[正文失败] {url}: {e}")
            content = ""
        full_contents.append(content)
        time.sleep(sleep)

    df2 = df.copy()
    df2["新闻正文"] = full_contents
    return df2


# ===========================
#   单股票：新闻增量维护
# ===========================
def update_single_stock_news(
    code: str,
    pool: str,
    init_fetch: int = 50,     # 首次希望覆盖的新闻条数（粗略）
    max_rows: int = 2000,     # 单股票 CSV 最大行数
    list_sleep: float = 0.4,  # 列表分页 sleep
    detail_sleep: float = 0.4 # 正文抓取 sleep
):
    """
    单只股票新闻维护规则：
    - 第一次：抓最近 init_fetch 条
    - 后续：只增量追加（按 新闻链接 去重）
    - 当 CSV 行数 > max_rows：按 发布时间 删除最旧，仅保留最新 max_rows
    """

    save_dir = os.path.join(NEWS_DIR, pool)
    os.makedirs(save_dir, exist_ok=True)

    code_clean = code.replace(".", "_")  # sh.600519 -> sh_600519
    csv_path = os.path.join(save_dir, f"{code_clean}.csv")

    # 1) 读取旧数据
    old_df = None
    old_urls = set()
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        if not old_df.empty and "新闻链接" in old_df.columns:
            old_urls = set(old_df["新闻链接"].astype(str))

    # 2) 抓新闻列表（分页数按 init_fetch 估算）
    max_pages = max(1, init_fetch // 10 + 1)
    symbol = code.split(".")[-1]  # sh.600519 -> 600519

    df = stock_news_em(
        symbol=symbol,
        max_pages=max_pages,
        page_size=10,
        sleep=list_sleep
    )

    if df.empty:
        EMPTY_LIST.append(code)
        return

    # 3) 增量过滤
    df["新闻链接"] = df["新闻链接"].astype(str)
    df_new = df[~df["新闻链接"].isin(old_urls)].copy()

    if df_new.empty:
        return

    # 4) 抓全文正文
    df_new = enrich_news_content(df_new, sleep=detail_sleep, verbose=False)

    # 5) 合并
    if old_df is not None and not old_df.empty:
        df_all = pd.concat([old_df, df_new], ignore_index=True)
    else:
        df_all = df_new

    # 6) 去重
    df_all["新闻链接"] = df_all["新闻链接"].astype(str)
    df_all = df_all.drop_duplicates(subset=["新闻链接"], keep="first")

    # 7) 按发布时间排序 + 裁剪
    df_all["发布时间"] = pd.to_datetime(df_all["发布时间"], errors="coerce")
    df_all = df_all.sort_values("发布时间", ascending=True)  # 旧 -> 新

    if len(df_all) > max_rows:
        df_all = df_all.iloc[-max_rows:]

    # 8) 写回
    df_all.to_csv(csv_path, index=False, encoding="utf-8")
    return


# ===========================
#          主流程
# ===========================
def run_news_download(pool="hs300", save_snapshot: bool = True):
    """
    pool: sz50 / hs300 / zz500 / all
    """
    bs_login()

    # 1) 股票池
    if pool == "all":
        stocks = get_stock_list_bs("all", day=get_latest_end_date())
    else:
        stocks = get_stock_list_bs(pool)

    # 2) 快照（可选）
    if save_snapshot:
        stocks.to_csv(
            os.path.join(META_DIR, f"stock_list_{pool}.csv"),
            index=False,
            encoding="utf-8",
        )

    # 3) 并集兜底
    current_codes = set(stocks["code"])
    hist_codes = get_codes_from_hist_dir(pool)
    last_snapshot_codes = load_last_snapshot(pool)

    final_codes = current_codes | hist_codes | last_snapshot_codes
    codes = sorted(final_codes)

    print(f"[股票池] 本次 {len(current_codes)}")
    print(f"[历史目录] {len(hist_codes)}")
    print(f"[上次快照] {len(last_snapshot_codes)}")
    print(f"[最终更新] {len(codes)}")

    # 4) 遍历更新
    for code in tqdm(codes):
        try:
            update_single_stock_news(
                code=code,
                pool=pool,
                init_fetch=60,
                max_rows=2000,
                list_sleep=random.uniform(0.25, 0.55),
                detail_sleep=random.uniform(0.25, 0.55),
            )
            time.sleep(random.uniform(0.6, 1.3))
        except Exception as e:
            FAILED_LIST.append(code)
            print(f"[失败] {code}: {e}")

    # 5) 日志输出
    if FAILED_LIST:
        pd.DataFrame({"code": FAILED_LIST}).to_csv(
            os.path.join(NEWS_DIR, f"failed_news_{pool}.csv"),
            index=False,
            encoding="utf-8",
        )

    if EMPTY_LIST:
        pd.DataFrame({"code": EMPTY_LIST}).to_csv(
            os.path.join(NEWS_DIR, f"empty_news_{pool}.csv"),
            index=False,
            encoding="utf-8",
        )

    bs_logout()


if __name__ == "__main__":
    # run_news_download(pool="sz50")
    run_news_download(pool="hs300")
    # run_news_download(pool="zz500")
    # run_news_download(pool="all")
