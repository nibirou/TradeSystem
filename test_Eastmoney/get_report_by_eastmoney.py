import json
import time
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests

def stock_report_em(
    symbol: str,
    begin: str,
    end: str,
    max_pages: int = 20,
    page_size: int = 50,
    sleep: float = 0.5,
) -> pd.DataFrame:
    """
    东方财富 - 个股研报列表
    """
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
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/143.0.0.0 Safari/537.36"
            ),
        }

        resp = requests.get(url, params=params, headers=headers)
        resp.raise_for_status()

        text = resp.text
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

def fetch_report_detail(
    info_code: str,
    timeout: int = 10
) -> str:
    """
    获取研报 HTML 页面
    """
    url = f"https://data.eastmoney.com/report/info/{info_code}.html"

    headers = {
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/143.0.0.0 Safari/537.36"
        )
    }

    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text

def parse_report_detail(html: str) -> dict:
    """
    从 HTML 中解析 zwinfo
    """
    soup = BeautifulSoup(html, "lxml")

    script_texts = soup.find_all("script")
    zwinfo_text = None

    for s in script_texts:
        if s.string and "var zwinfo" in s.string:
            zwinfo_text = s.string
            break

    if zwinfo_text is None:
        raise ValueError("未找到 zwinfo 数据")

    match = re.search(r"var\s+zwinfo\s*=\s*(\{.*?\});", zwinfo_text, re.S)
    if not match:
        raise ValueError("zwinfo JSON 解析失败")

    data = json.loads(match.group(1))

    return {
        "研报正文": data.get("notice_content", "").strip(),
        "PDF链接": data.get("attach_url", ""),
        "研报发布时间": pd.to_datetime(data.get("notice_date"), errors="coerce"),
        "页数": data.get("attach_pages"),
        "文件大小KB": data.get("attach_size"),
    }

def download_report_pdf(
    pdf_url: str,
    save_dir: str,
    filename: str,
    timeout: int = 15
) -> str:
    if not pdf_url:
        return ""

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / filename

    headers = {
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/143.0.0.0 Safari/537.36"
        )
    }

    resp = requests.get(pdf_url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(resp.content)

    return str(save_path)

def enrich_report_content(
    df: pd.DataFrame,
    pdf_dir: str,
    sleep: float = 0.6,
    verbose: bool = True
) -> pd.DataFrame:

    texts, pdfs, times, pages, sizes = [], [], [], [], []

    for _, row in df.iterrows():
        info_code = row["研报ID"]

        try:
            html = fetch_report_detail(info_code)
            parsed = parse_report_detail(html)

            pdf_path = download_report_pdf(
                parsed["PDF链接"],
                save_dir=pdf_dir,
                filename=f"{info_code}.pdf"
            )

            texts.append(parsed["研报正文"])
            pdfs.append(pdf_path)
            times.append(parsed["研报发布时间"])
            pages.append(parsed["页数"])
            sizes.append(parsed["文件大小KB"])

        except Exception as e:
            if verbose:
                print(f"[失败] {info_code}: {e}")
            texts.append("")
            pdfs.append("")
            times.append(pd.NaT)
            pages.append(None)
            sizes.append(None)

        time.sleep(sleep)

    df = df.copy()
    df["研报正文"] = texts
    df["PDF本地路径"] = pdfs
    df["研报发布时间"] = times
    df["页数"] = pages
    df["文件大小KB"] = sizes

    return df

if __name__ == "__main__":

    stock_code = "000858"

    # 1. 研报列表
    df = stock_report_em(
        stock_code,
        begin="2025-12-20",
        end="2026-12-31",
        max_pages=1
    )

    df = normalize_report_df(df, stock_code)

    # 2. enrich 研报正文 + PDF
    df = enrich_report_content(
        df,
        pdf_dir=f"./report_pdfs/{stock_code}"
    )

    # 3. 保存
    out = f"report_full_{stock_code}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print("✅ 研报正文 + PDF 下载完成")
    print(df.head())
