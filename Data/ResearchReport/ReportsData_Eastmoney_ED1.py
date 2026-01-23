# 通过东方财富接口获取个股研报数据

# -*- coding: utf-8 -*-
"""
Baostock 股票池 + 东方财富个股研报（reportapi）
- 研报列表
- 研报 HTML 解析正文
- PDF 下载
- 按股票 CSV 保存
- 增量更新（按 研报ID）
"""

import os
import json
import time
import random
import re
from datetime import datetime
from pathlib import Path

import baostock as bs
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from curl_cffi import requests


# ======================================================
#                   路径配置
# ======================================================
# BASE_DIR = "/workspace/Quant/data_baostock"
BASE_DIR = "D:/PythonProject/Quant/data_baostock"

REPORT_DIR = os.path.join(BASE_DIR, "data_em_reports")     # CSV
PDF_DIR = os.path.join(BASE_DIR, "data_em_report_pdfs")    # PDF
META_DIR = os.path.join(BASE_DIR, "metadata")

for d in [REPORT_DIR, PDF_DIR, META_DIR]:
    os.makedirs(d, exist_ok=True)

FAILED_LIST = []
EMPTY_LIST = []


# ======================================================
#                  Baostock 工具
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
    rs = bs.query_trade_dates(
        start_date=f"{datetime.now().year}-01-01",
        end_date=today
    )
    data = []
    while (rs.error_code == "0") & rs.next():
        data.append(rs.get_row_data())

    df = pd.DataFrame(data, columns=rs.fields)
    df["calendar_date"] = pd.to_datetime(df["calendar_date"])
    df["is_trading_day"] = df["is_trading_day"].astype(int)
    return df[df["is_trading_day"] == 1]["calendar_date"].iloc[-1].strftime("%Y-%m-%d")

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
#        东方财富：研报列表（reportapi）
# ======================================================
def stock_report_em(
    symbol: str,
    begin: str,
    end: str,
    max_pages: int = 20,
    page_size: int = 50,
    sleep: float = 0.4,
) -> pd.DataFrame:

    url = "https://reportapi.eastmoney.com/report/list"
    all_items = []

    for page in range(1, max_pages + 1):
        params = {
            "cb": "datatable",
            "pageNo": page,
            "pageSize": page_size,
            "code": symbol,
            "industryCode": "*",
            "industry": "*",
            "rating": "*",
            "ratingchange": "*",
            "beginTime": begin,
            "endTime": end,
            "qType": 0,
            "_": int(time.time() * 1000),
        }

        headers = {
            "referer": f"https://data.eastmoney.com/report/{symbol}.html",
            "user-agent": "Mozilla/5.0",
        }

        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()

        text = r.text
        if not text.startswith("datatable("):
            break

        data = json.loads(text[len("datatable("):-1])
        items = data.get("data", [])
        if not items:
            break

        all_items.extend(items)
        time.sleep(sleep)

    return pd.DataFrame(all_items)


def normalize_report_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.copy()
    df.rename(columns={
        "infoCode": "研报ID",
        "title": "研报标题",
        "orgName": "机构名称",
        "researcher": "研究员",
        "publishDate": "发布日期",
        "emRatingName": "评级",
        "ratingChange": "评级变动",
    }, inplace=True)

    df["股票代码"] = symbol
    df["发布日期"] = pd.to_datetime(df["发布日期"], errors="coerce")

    return df[
        [
            "股票代码",
            "研报ID",
            "研报标题",
            "机构名称",
            "研究员",
            "评级",
            "评级变动",
            "发布日期",
        ]
    ]


# ======================================================
#        研报详情 HTML + PDF
# ======================================================
def fetch_report_html(info_code: str) -> str:
    url = f"https://data.eastmoney.com/report/info/{info_code}.html"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.text

def parse_report_detail(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    scripts = soup.find_all("script")

    zwinfo_text = None
    for s in scripts:
        if s.string and "var zwinfo" in s.string:
            zwinfo_text = s.string
            break

    if not zwinfo_text:
        raise RuntimeError("未找到 zwinfo")

    m = re.search(r"var\s+zwinfo\s*=\s*(\{.*?\});", zwinfo_text, re.S)
    if not m:
        raise RuntimeError("zwinfo 解析失败")

    data = json.loads(m.group(1))

    return {
        "研报正文": data.get("notice_content", "").strip(),
        "PDF链接": data.get("attach_url", ""),
        "研报发布时间": pd.to_datetime(data.get("notice_date"), errors="coerce"),
        "页数": data.get("attach_pages"),
        "文件大小KB": data.get("attach_size"),
    }

def download_report_pdf(pdf_url: str, save_dir: str, filename: str) -> str:
    if not pdf_url:
        return ""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = Path(save_dir) / filename

    r = requests.get(pdf_url, timeout=20)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return str(path)


# ======================================================
#        单股票：研报增量维护
# ======================================================
def update_single_stock_report(
    code: str,
    pool: str,
    begin: str,
    end: str,
    init_fetch: int = 200,
    max_rows: int = 2000,
):
    save_dir = os.path.join(REPORT_DIR, pool)
    os.makedirs(save_dir, exist_ok=True)

    pdf_stock_dir = os.path.join(PDF_DIR, pool, code.replace(".", "_"))
    os.makedirs(pdf_stock_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, f"{code.replace('.', '_')}.csv")

    old_df = None
    old_ids = set()

    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        if not old_df.empty:
            old_ids = set(old_df["研报ID"].astype(str))

    df = stock_report_em(
        symbol=code.split(".")[-1],
        begin=begin,
        end=end,
        max_pages=max(1, init_fetch // 50 + 1),
    )

    if df.empty:
        EMPTY_LIST.append(code)
        return

    df = normalize_report_df(df, code.split(".")[-1])
    df["研报ID"] = df["研报ID"].astype(str)

    df_new = df[~df["研报ID"].isin(old_ids)]
    if df_new.empty:
        return

    texts, pdfs, times, pages, sizes = [], [], [], [], []

    for _, row in df_new.iterrows():
        try:
            html = fetch_report_html(row["研报ID"])
            parsed = parse_report_detail(html)

            pdf_path = download_report_pdf(
                parsed["PDF链接"],
                pdf_stock_dir,
                f"{row['研报ID']}.pdf"
            )

            texts.append(parsed["研报正文"])
            pdfs.append(pdf_path)
            times.append(parsed["研报发布时间"])
            pages.append(parsed["页数"])
            sizes.append(parsed["文件大小KB"])

        except Exception as e:
            texts.append("")
            pdfs.append("")
            times.append(pd.NaT)
            pages.append(None)
            sizes.append(None)

        time.sleep(random.uniform(0.4, 0.7))

    df_new["研报正文"] = texts
    df_new["PDF本地路径"] = pdfs
    df_new["研报发布时间"] = times
    df_new["页数"] = pages
    df_new["文件大小KB"] = sizes

    if old_df is not None:
        df_all = pd.concat([old_df, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all = df_all.drop_duplicates(subset=["研报ID"], keep="first")
    df_all = df_all.sort_values("发布日期")

    if len(df_all) > max_rows:
        df_all = df_all.iloc[-max_rows:]

    df_all.to_csv(csv_path, index=False, encoding="utf-8")


# ======================================================
#                  主流程
# ======================================================
def run_report_download(pool="hs300"):
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

    begin = "2015-01-01"
    end = datetime.now().strftime("%Y-%m-%d")

    for code in tqdm(stocks["code"]):
        try:
            update_single_stock_report(code, pool, begin, end)
            time.sleep(random.uniform(0.8, 1.4))
        except Exception as e:
            FAILED_LIST.append(code)
            print(f"[失败] {code}: {e}")

    bs_logout()


if __name__ == "__main__":
    run_report_download(pool="hs300")
