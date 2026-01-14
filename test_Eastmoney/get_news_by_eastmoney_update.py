import json
import re
import pandas as pd
from curl_cffi import requests
import time
from bs4 import BeautifulSoup

def stock_news_em(
    symbol: str = "600519",
    max_pages: int = 5,
    page_size: int = 10,
    sleep: float = 0.5
) -> pd.DataFrame:
    """
    东方财富 - 个股新闻（支持分页）
    """

    url = "https://search-api-web.eastmoney.com/search/jsonp"
    all_items = []

    for page in range(1, max_pages + 1):

        cb = f"jQuery3510{int(pd.Timestamp.now().timestamp() * 1000)}"
        _ts = str(int(pd.Timestamp.now().timestamp() * 1000))

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

        resp = requests.get(url, params=params, headers=headers)
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

    df = df[
        ["关键词", "新闻标题", "新闻内容", "发布时间", "文章来源", "新闻链接"]
    ]

    for col in ["新闻标题", "新闻内容"]:
        df[col] = (
            df[col]
            .str.replace(r"<em>|</em>", "", regex=True)
            .str.replace(r"\u3000|\r\n", " ", regex=True)
        )

    df["发布时间"] = pd.to_datetime(df["发布时间"], errors="coerce")

    return df

def news_statistics(df: pd.DataFrame, recent_days: int = 7) -> dict:
    """
    新闻统计分析
    """
    stats = {}

    if df.empty:
        return stats

    # 总条数
    stats["total_news"] = len(df)

    # 按日期统计
    stats["by_date"] = (
        df.groupby(df["发布时间"].dt.date)
        .size()
        .sort_index()
        .to_dict()
    )

    # 按媒体统计
    stats["by_media"] = (
        df["文章来源"]
        .value_counts()
        .head(10)
        .to_dict()
    )

    # 最近 N 天新闻数量
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=recent_days)
    stats[f"last_{recent_days}_days"] = df[df["发布时间"] >= cutoff].shape[0]

    return stats

def news_keywords(df: pd.DataFrame, topk: int = 20) -> pd.Series:
    """
    简单高频关键词统计（不分词版，适合事件爆发度）
    """
    text = (
        df["新闻标题"].fillna("") + " " +
        df["新闻内容"].fillna("")
    )

    words = (
        text
        .str.replace(r"[^\u4e00-\u9fa5]", " ", regex=True)
        .str.split()
        .explode()
    )

    return words.value_counts().head(topk)

def fetch_article_html(url: str, timeout: int = 10) -> str:
    headers = {
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/143.0.0.0 Safari/537.36"
        ),
        "accept-language": "zh-CN,zh;q=0.9",
        "referer": "https://finance.eastmoney.com/",
    }

    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text

def parse_article_content(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    content_div = soup.find("div", id="ContentBody")
    if not content_div:
        return ""

    paragraphs = []

    for p in content_div.find_all("p", recursive=False):
        # 跳过“文章来源”那一行
        if "em_media" in p.get("class", []):
            continue

        text = p.get_text(strip=True)
        if not text:
            continue

        paragraphs.append(text)

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

    for i, row in df.iterrows():
        url = row["新闻链接"]

        try:
            html = fetch_article_html(url)
            content = parse_article_content(html)
        except Exception as e:
            if verbose:
                print(f"[失败] {url}: {e}")
            content = ""

        full_contents.append(content)
        time.sleep(sleep)

    df = df.copy()
    df["新闻正文"] = full_contents

    return df



if __name__ == "__main__":
    stock_code = "600519"
    df = stock_news_em(stock_code, max_pages=10)
    df = enrich_news_content(df, sleep=0.5)

    df.to_csv(f"news_{stock_code}.csv", index=False, encoding="utf-8-sig")
    print(f"✅ CSV 已输出: ", f"news_{stock_code}.csv")

    print(df.head())

    stats = news_statistics(df, recent_days=5)
    print("统计信息:", stats)

    keywords = news_keywords(df)
    print("高频词:\n", keywords)


