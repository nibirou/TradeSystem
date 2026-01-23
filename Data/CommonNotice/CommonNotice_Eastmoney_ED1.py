# 通过东方财富接口获取个股公告

# -*- coding: utf-8 -*-
"""
Baostock 股票池 + 东方财富个股公告（noticeWeb）
- 按股票 CSV 保存
- 增量更新（按 公告ID 去重）
- 公告正文抓取
- PDF 文件下载
- 超限裁剪
"""

import os
import json
import time
import random
from datetime import datetime
from pathlib import Path

import baostock as bs
import pandas as pd
from tqdm import tqdm
from curl_cffi import requests


# ======================================================
#                  路径配置
# ======================================================
# BASE_DIR = "/workspace/Quant/data_baostock"
BASE_DIR = "D:/PythonProject/Quant/data_baostock"

NOTICE_DIR = os.path.join(BASE_DIR, "data_em_notices")   # CSV 输出
PDF_DIR = os.path.join(BASE_DIR, "data_em_notice_pdfs")  # PDF 存储
META_DIR = os.path.join(BASE_DIR, "metadata")
HIST_DIR = os.path.join(BASE_DIR, "stock_hist")

for d in [NOTICE_DIR, PDF_DIR, META_DIR, HIST_DIR]:
    os.makedirs(d, exist_ok=True)

FAILED_LIST = []
EMPTY_LIST = []


# ======================================================
#                Baostock 工具
# ======================================================
def bs_login():
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(lg.error_msg)

def bs_logout():
    bs.logout()

def _bs_query_to_df(rs):
    rows = []
    while (rs.error_code == "0") & rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)

def get_latest_end_date():
    today = datetime.now().strftime("%Y-%m-%d")
    year_start = f"{datetime.now().year}-01-01"
    rs = bs.query_trade_dates(start_date=year_start, end_date=today)

    data = []
    while (rs.error_code == "0") & rs.next():
        data.append(rs.get_row_data())

    df = pd.DataFrame(data, columns=rs.fields)
    df["calendar_date"] = pd.to_datetime(df["calendar_date"])
    df["is_trading_day"] = df["is_trading_day"].astype(int)

    trade_days = df[df["is_trading_day"] == 1]["calendar_date"]
    return trade_days.iloc[-1].strftime("%Y-%m-%d")

def get_stock_list_bs(mode="hs300", day=None):
    if mode == "hs300":
        rs = bs.query_hs300_stocks()
    elif mode == "zz500":
        rs = bs.query_zz500_stocks()
    elif mode == "sz50":
        rs = bs.query_sz50_stocks()
    elif mode == "all":
        rs = bs.query_all_stock(day=day)
    else:
        raise ValueError(mode)

    df = _bs_query_to_df(rs)
    if "code_name" in df.columns:
        df = df.rename(columns={"code_name": "name"})
    else:
        df["name"] = df["code"]

    return df[["code", "name"]]


# ======================================================
#         东方财富：公告列表（noticeWeb）
# ======================================================
def stock_notice_em(
    symbol: str,
    max_pages: int = 10,
    page_size: int = 10,
    sleep: float = 0.4
) -> pd.DataFrame:

    url = "https://search-api-web.eastmoney.com/search/jsonp"
    all_items = []

    for page in range(1, max_pages + 1):

        ts = int(pd.Timestamp.now().timestamp() * 1000)
        cb = f"jQuery3510{ts}"

        inner_param = {
            "uid": "",
            "keyword": symbol,
            "type": ["noticeWeb"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "noticeWeb": {
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
            "_": ts + 1,
        }

        headers = {
            "referer": f"https://so.eastmoney.com/ann/s?keyword={symbol}",
            "user-agent": "Mozilla/5.0",
        }

        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()

        text = r.text
        if not text.startswith(f"{cb}("):
            break

        data = json.loads(text[len(cb) + 1:-1])
        items = data.get("result", {}).get("noticeWeb", [])
        if not items:
            break

        all_items.extend(items)
        time.sleep(sleep)

    if not all_items:
        return pd.DataFrame()

    df = pd.DataFrame(all_items)
    df.rename(
        columns={
            "title": "公告标题",
            "content": "公告摘要",
            "date": "公告日期",
            "securityFullName": "公司名称",
            "url": "公告链接",
            "code": "公告ID",
            "columnCode": "公告栏目",
        },
        inplace=True,
    )

    df["股票代码"] = symbol
    df["公告日期"] = pd.to_datetime(df["公告日期"], errors="coerce")

    return df[
        [
            "股票代码",
            "公司名称",
            "公告标题",
            "公告摘要",
            "公告日期",
            "公告栏目",
            "公告ID",
            "公告链接",
        ]
    ]


# ======================================================
#           公告正文 + PDF
# ======================================================
def fetch_notice_detail(art_code: str) -> dict:
    url = "https://np-cnotice-stock.eastmoney.com/api/content/ann"
    ts = int(pd.Timestamp.now().timestamp() * 1000)
    cb = f"jQuery1123{ts}"

    params = {
        "cb": cb,
        "art_code": art_code,
        "client_source": "web",
        "page_index": 1,
        "_": ts + 1,
    }

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()

    text = r.text
    data = json.loads(text[len(cb) + 1:-1])

    if not data.get("success"):
        raise RuntimeError("公告正文接口失败")

    return data["data"]

def parse_notice_detail(data: dict) -> dict:
    return {
        "公告正文": data.get("notice_content", "").replace("\r", "").strip(),
        "PDF链接": data.get("attach_url_web", ""),
        "公告发布时间": pd.to_datetime(data.get("notice_date"), errors="coerce"),
    }

def download_pdf(pdf_url: str, save_dir: str, filename: str) -> str:
    if not pdf_url:
        return ""

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / filename

    r = requests.get(pdf_url, timeout=20)
    r.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(r.content)

    return str(save_path)


# ======================================================
#           单股票：公告维护
# ======================================================
def update_single_stock_notice(
    code: str,
    pool: str,
    init_fetch: int = 50,
    max_rows: int = 2000,
):
    save_dir = os.path.join(NOTICE_DIR, pool)
    os.makedirs(save_dir, exist_ok=True)

    pdf_stock_dir = os.path.join(PDF_DIR, pool, code.replace(".", "_"))
    os.makedirs(pdf_stock_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, f"{code.replace('.', '_')}.csv")

    old_df = None
    old_ids = set()

    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        if not old_df.empty:
            old_ids = set(old_df["公告ID"].astype(str))

    df = stock_notice_em(
        symbol=code.split(".")[-1],
        max_pages=max(1, init_fetch // 10 + 1),
    )

    if df.empty:
        EMPTY_LIST.append(code)
        return

    df["公告ID"] = df["公告ID"].astype(str)
    df_new = df[~df["公告ID"].isin(old_ids)]

    if df_new.empty:
        return

    texts, pdfs, times = [], [], []

    for _, row in df_new.iterrows():
        try:
            data = fetch_notice_detail(row["公告ID"])
            parsed = parse_notice_detail(data)

            pdf_path = download_pdf(
                parsed["PDF链接"],
                pdf_stock_dir,
                filename=f"{row['公告ID']}.pdf",
            )

            texts.append(parsed["公告正文"])
            pdfs.append(pdf_path)
            times.append(parsed["公告发布时间"])

        except Exception as e:
            texts.append("")
            pdfs.append("")
            times.append(pd.NaT)

        time.sleep(random.uniform(0.4, 0.7))

    df_new["公告正文"] = texts
    df_new["PDF本地路径"] = pdfs
    df_new["公告发布时间"] = times

    if old_df is not None:
        df_all = pd.concat([old_df, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all = df_all.drop_duplicates(subset=["公告ID"], keep="first")
    df_all = df_all.sort_values("公告日期")

    if len(df_all) > max_rows:
        df_all = df_all.iloc[-max_rows:]

    df_all.to_csv(csv_path, index=False, encoding="utf-8")


# ======================================================
#                  主流程
# ======================================================
def run_notice_download(pool="hs300"):
    bs_login()

    if pool == "all":
        stocks = get_stock_list_bs("all", day=get_latest_end_date())
    else:
        stocks = get_stock_list_bs(pool)

    stocks.to_csv(
        os.path.join(META_DIR, f"stock_list_{pool}.csv"),
        index=False,
        encoding="utf-8",
    )

    codes = sorted(stocks["code"].tolist())

    for code in tqdm(codes):
        try:
            update_single_stock_notice(code, pool)
            time.sleep(random.uniform(0.6, 1.2))
        except Exception as e:
            FAILED_LIST.append(code)
            print(f"[失败] {code}: {e}")

    if FAILED_LIST:
        pd.DataFrame({"code": FAILED_LIST}).to_csv(
            os.path.join(NOTICE_DIR, f"failed_notice_{pool}.csv"),
            index=False,
        )

    if EMPTY_LIST:
        pd.DataFrame({"code": EMPTY_LIST}).to_csv(
            os.path.join(NOTICE_DIR, f"empty_notice_{pool}.csv"),
            index=False,
        )

    bs_logout()


if __name__ == "__main__":
    run_notice_download(pool="hs300")
