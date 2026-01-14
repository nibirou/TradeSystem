#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
东方财富 - 个股新闻（严格仿浏览器抓包）
接口：
https://search-api-web.eastmoney.com/search/jsonp
"""

import json
import re
import pandas as pd
from curl_cffi import requests


def stock_news_em(symbol: str = "600519") -> pd.DataFrame:
    """
    东方财富 - 个股新闻（严格按照浏览器抓包）
    """

    # ------------------------------------------------------------------
    # 1. 请求 URL
    # ------------------------------------------------------------------
    url = "https://search-api-web.eastmoney.com/search/jsonp"

    # ------------------------------------------------------------------
    # 2. 请求参数（params）——与你给的抓包完全一致
    # ------------------------------------------------------------------
    cb = "jQuery35105297082506338707_1768374302811"

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
                "pageIndex": 1,
                "pageSize": 10,
                "preTag": "<em>",
                "postTag": "</em>",
            }
        },
    }

    params = {
        "cb": cb,
        "param": json.dumps(inner_param, ensure_ascii=False),
        "_": "1768374302813",
    }

    # ------------------------------------------------------------------
    # 3. Headers —— 按你提供的抓包逐项写
    # ------------------------------------------------------------------
    headers = {
        "accept": "*/*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "zh-CN,zh;q=0.9",
        "cache-control": "no-cache",
        "connection": "keep-alive",
        "cookie": (
            "st_nvi=XlqqBTSaAakLWldpAKWNude3d; "
            "qgqp_b_id=434e3c6ebe4e7392921147e2a4730249; "
            "st_si=30675574105663; "
            "nid18=0d415f38386bc9e8d96209ac2ca689a2; "
            "nid18_create_time=1768358951054; "
            "gviem=BNpOSWj8MmUMbiaWags3Pc091; "
            "gviem_create_time=1768358951054; "
            "p_origin=https%3A%2F%2Fpassport2.eastmoney.com; "
            "fullscreengg=1; "
            "fullscreengg2=1; "
            "emshistory=%5B%22600519%22%2C%22%E8%B4%B5%E5%B7%9E%E8%8C%85%E5%8F%B0%22%5D; "
            "st_pvi=99776927593724; "
            "st_sp=2023-11-23%2008%3A39%3A57; "
            "st_inirUrl=https%3A%2F%2Fwww.google.com%2F; "
            "st_sn=73; "
            "st_psi=20260114150503105-118000300905-3995203244; "
            "st_asi=20260114150503105-118000300905-3995203244-dfcfwss.ssh.mrpx.xxdj-1"
        ),
        "host": "search-api-web.eastmoney.com",
        "pragma": "no-cache",
        "referer": f"https://so.eastmoney.com/news/s?keyword={symbol}&sort=time",
        "sec-ch-ua": "\"Google Chrome\";v=\"143\", \"Chromium\";v=\"143\", \"Not A(Brand\";v=\"24\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "script",
        "sec-fetch-mode": "no-cors",
        "sec-fetch-site": "same-site",
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/143.0.0.0 Safari/537.36"
        ),
    }

    # ------------------------------------------------------------------
    # 4. 发请求（curl_cffi，完全模拟浏览器）
    # ------------------------------------------------------------------
    resp = requests.get(
        url,
        params=params,
        headers=headers
    )
    resp.raise_for_status()

    text = resp.text

    # ------------------------------------------------------------------
    # 5. JSONP 解析（严格按 cb）
    # ------------------------------------------------------------------
    prefix = f"{cb}("
    if not text.startswith(prefix):
        raise RuntimeError("JSONP callback 不匹配")

    data_json = json.loads(text[len(prefix):-1])

    if data_json.get("code") != 0:
        raise RuntimeError(f"接口返回异常: {data_json}")

    items = data_json.get("result", {}).get("cmsArticleWebOld", [])
    if not items:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 6. DataFrame 整理
    # ------------------------------------------------------------------
    df = pd.DataFrame(items)

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

    # 清理 <em> 标签
    for col in ["新闻标题", "新闻内容"]:
        df[col] = (
            df[col]
            .str.replace(r"<em>", "", regex=True)
            .str.replace(r"</em>", "", regex=True)
            .str.replace(r"\u3000", "", regex=True)
            .str.replace(r"\r\n", " ", regex=True)
        )

    return df


# ======================================================================
# 示例
# ======================================================================
if __name__ == "__main__":
    df = stock_news_em("600519")
    print(df)
