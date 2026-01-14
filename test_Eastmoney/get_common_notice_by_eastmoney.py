import json
import time
import pandas as pd
from curl_cffi import requests


def stock_notice_em(
    symbol: str = "600519",
    max_pages: int = 10,
    page_size: int = 10,
    sleep: float = 0.5
) -> pd.DataFrame:
    """
    东方财富 - 个股公告（noticeWeb，支持分页）
    """

    url = "https://search-api-web.eastmoney.com/search/jsonp"
    all_items = []

    for page in range(1, max_pages + 1):

        ts = int(pd.Timestamp.now().timestamp() * 1000)
        cb = f"jQuery3510{ts}"
        _ts = str(ts + 1)

        inner_param = {
            "uid": "",
            "keyword": symbol,
            "type": ["noticeWeb"],
            "client": "web",
            "clientVersion": "curr",
            "clientType": "web",
            "param": {
                "noticeWeb": {
                    "preTag": "<em class=\"red\">",
                    "postTag": "</em>",
                    "pageSize": page_size,
                    "pageIndex": page,
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
            "referer": f"https://so.eastmoney.com/ann/s?keyword={symbol}",
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/143.0.0.0 Safari/537.36"
            ),
        }

        resp = requests.get(url, params=params, headers=headers)
        resp.raise_for_status()

        text = resp.text
        prefix = f"{cb}("

        if not text.startswith(prefix):
            break

        data_json = json.loads(text[len(prefix):-1])

        items = data_json.get("result", {}).get("noticeWeb", [])
        if not items:
            break

        all_items.extend(items)
        time.sleep(sleep)

    if not all_items:
        return pd.DataFrame()

    df = pd.DataFrame(all_items)

    # 字段重命名
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

    # 字段顺序
    df = df[
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

    # 清洗 em 标签与空白
    for col in ["公告标题", "公告摘要"]:
        df[col] = (
            df[col]
            .str.replace(r"<em.*?>|</em>", "", regex=True)
            .str.replace(r"\u3000|\r\n", " ", regex=True)
            .str.strip()
        )

    df["公告日期"] = pd.to_datetime(df["公告日期"], errors="coerce")

    return df

def fetch_notice_detail(
    art_code: str,
    timeout: int = 10
) -> dict:
    """
    东方财富 - 公告正文接口（JSONP）
    """
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

    headers = {
        "accept": "*/*",
        "accept-language": "zh-CN,zh;q=0.9",
        "referer": "https://data.eastmoney.com/notices/",
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/143.0.0.0 Safari/537.36"
        ),
    }

    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()

    text = resp.text
    prefix = f"{cb}("

    if not text.startswith(prefix):
        raise ValueError("JSONP callback 不匹配")

    data = json.loads(text[len(prefix):-1])

    if not data.get("success"):
        raise ValueError("公告正文接口返回失败")

    return data["data"]

def parse_notice_detail(data: dict) -> dict:
    """
    解析公告正文内容 & PDF 信息
    """
    notice_text = (
        data.get("notice_content", "")
        .replace("\r", "")
        .strip()
    )

    pdf_url = data.get("attach_url_web", "")

    return {
        "公告正文": notice_text,
        "PDF链接": pdf_url,
        "公告发布时间": pd.to_datetime(
            data.get("notice_date"),
            errors="coerce"
        ),
    }

from pathlib import Path


def download_pdf(
    pdf_url: str,
    save_dir: str,
    filename: str | None = None,
    timeout: int = 15
) -> str:
    """
    下载公告 PDF 文件
    """
    if not pdf_url:
        return ""

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = pdf_url.split("/")[-1].split("?")[0]

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

def enrich_notice_content(
    df: pd.DataFrame,
    pdf_dir: str = "./notice_pdfs",
    sleep: float = 0.5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    为公告 DataFrame 补充：公告正文 + PDF 本地路径
    """

    full_texts = []
    pdf_paths = []
    publish_times = []

    for _, row in df.iterrows():

        art_code = row["公告ID"]

        try:
            data = fetch_notice_detail(art_code)
            parsed = parse_notice_detail(data)

            pdf_path = download_pdf(
                parsed["PDF链接"],
                save_dir=pdf_dir,
                filename=f"{art_code}.pdf"
            )

            full_texts.append(parsed["公告正文"])
            pdf_paths.append(pdf_path)
            publish_times.append(parsed["公告发布时间"])

        except Exception as e:
            if verbose:
                print(f"[失败] {art_code}: {e}")

            full_texts.append("")
            pdf_paths.append("")
            publish_times.append(pd.NaT)

        time.sleep(sleep)

    df = df.copy()
    df["公告正文"] = full_texts
    df["PDF本地路径"] = pdf_paths
    df["公告发布时间"] = publish_times

    return df

if __name__ == "__main__":

    stock_code = "600519"

    # 1. 先抓公告列表
    df_notice = stock_notice_em(
        stock_code,
        max_pages=5,
        page_size=1
    )

    # 2. enrich 公告正文 + PDF
    df_notice = enrich_notice_content(
        df_notice,
        pdf_dir=f"./pdfs/{stock_code}",
        sleep=0.6
    )

    # 3. 落盘
    df_notice.to_csv(
        f"notice_full_{stock_code}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("✅ 公告正文 + PDF 已完成")
    print(df_notice.head())
