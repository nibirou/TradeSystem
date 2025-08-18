# -*- coding: utf-8 -*-
"""
个股 消息面 + 资金面 数据提取（单文件 · 正文抓取升级版）
- 新闻标题列名鲁棒性 + 日期排序
- 正文抓取：优先用“新闻链接”获取完整文章内容（多策略 + 站点特化）
- 正文/新闻缓存（避免重复抓）
- 缓存过期刷新（新闻列表/资金流/正文）
- 重试 + 指数退避 + 失败日志
- 龙虎榜：flag & 次数
- 并发提速 + 轻量限速
"""

import os
import re
import time
import json
import hashlib
import random
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import akshare as ak
from snownlp import SnowNLP
from tqdm import tqdm

# —— 可选依赖：自动优先使用，缺少则自动降级 ——
try:
    from newspaper import Article  # pip install newspaper3k
    _HAS_NEWSPAPER = True
except Exception:
    _HAS_NEWSPAPER = False

try:
    from readability import Document  # pip install readability-lxml
    _HAS_READABILITY = True
except Exception:
    _HAS_READABILITY = False

try:
    import requests
    from bs4 import BeautifulSoup  # pip install beautifulsoup4 lxml
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False


# =================== 可配置参数 ===================
R00T_DIR = "data"
META_DIR = "data/metadata"
STOCK_NEWS_DIR = os.path.join(R00T_DIR, "stock_news")
RESULT_DIR = os.path.join(STOCK_NEWS_DIR, "news_nlp_result_by_day")
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_PATH = os.path.join(RESULT_DIR, f"stock_news_factors_{RUN_TS}.csv") # 新闻评分结果输出(带日期)
FAIL_LOG = os.path.join(RESULT_DIR, f"stock_news_failures_{RUN_TS}.log") # 新闻获取失败结果（带日期）

# Akshare 东财个股新闻接口
AKSHARE_STOCK_NEWS_EM_DIR = os.path.join(STOCK_NEWS_DIR, "akshare_stock_news_em")
AKSHARE_STOCK_NEWS_EM_ARTICLES_DIR = os.path.join(AKSHARE_STOCK_NEWS_EM_DIR, "articles")     # 正文缓存目录
os.makedirs(AKSHARE_STOCK_NEWS_EM_DIR, exist_ok=True)
os.makedirs(AKSHARE_STOCK_NEWS_EM_ARTICLES_DIR, exist_ok=True)



# 其他方法/接口目录


# 缓存文件最大有效期（小时），超过则刷新抓取
MAX_AGE_HOURS_NEWS = 24        # 新闻列表缓存
MAX_AGE_HOURS_ARTICLE = 24*7   # 正文缓存：一周

# 抓取重试
RETRIES = 3
BASE_WAIT = 0.8  # 秒，指数退避基数

# 并发线程数
MAX_WORKERS = 8

# 轻量限速：每只股票抓完后随机 sleep 区间（秒）
RANDOM_SLEEP_RANGE = (0.05, 0.2)

# 新闻情绪：取“最近 topk 条”用于情绪打分
NEWS_TOPK = 20

# 正文最小长度阈值（太短说明抽取失败，回退标题/摘要）
MIN_ARTICLE_CHARS = 60
# 对超长正文可截断做情感（避免超慢），按字符截断
MAX_ARTICLE_CHARS = 6000

# Requests 抓取参数
REQ_TIMEOUT = 12
REQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}


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
                df = df.sort_values(c, ascending=True)  # 升序，便于 tail(N) 取最近 N 条
                return df
            except Exception:
                continue
    return df  # 找不到日期列就原样返回


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _sanitize_text(txt: str) -> str:
    if not txt:
        return ""
    # 去脚注/重复空白
    txt = re.sub(r"\s+", " ", txt)
    # 常见版权尾巴粗略剔除（可按需扩展）
    txt = re.sub(r"(责任编辑[:：].*?$)", "", txt)
    return txt.strip()


# =================== 正文抓取与缓存 ===================
def _article_cache_path(url: str) -> str:
    return os.path.join(AKSHARE_STOCK_NEWS_EM_ARTICLES_DIR, f"{_md5(url)}.json")


def _read_article_cache(url: str):
    path = _article_cache_path(url)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 过期判断
        ts = datetime.fromisoformat(data.get("fetched_at"))
        if (datetime.now() - ts).total_seconds() > MAX_AGE_HOURS_ARTICLE * 3600:
            return None
        return data.get("text", "")
    except Exception:
        return None


def _write_article_cache(url: str, text: str):
    path = _article_cache_path(url)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"url": url, "text": text, "fetched_at": datetime.now().isoformat()}, f, ensure_ascii=False)
    except Exception as e:
        _log_fail(f"WRITE_ARTICLE_CACHE_FAIL {url} -> {e}")


def _requests_get(url: str):
    if not _HAS_REQUESTS:
        raise RuntimeError("Missing 'requests' / 'beautifulsoup4' dependency")
    return requests.get(url, headers=REQ_HEADERS, timeout=REQ_TIMEOUT)


def _requests_html(url: str) -> str:
    """
    请求网页并返回解码后的 HTML 文本（处理 gbk/gb2312 等）
    """
    resp = _requests_get(url)
    # 优先用 apparent_encoding（依赖 chardet），再退回 response.encoding
    enc = resp.apparent_encoding or resp.encoding or "utf-8"
    resp.encoding = enc
    return resp.text


def _extract_with_bs4(url: str, html: str) -> str:
    """
    通用 + 站点特化的正文抽取（BeautifulSoup）
    """
    soup = BeautifulSoup(html, "lxml")

    # —— 站点特化：东方财富（eastmoney）常见容器选择器 —— 
    # 不同频道/年代 class/id 可能不同，准备多个候选
    if "eastmoney.com" in url:
        candidates = [
            "#ContentBody",               # 常见内容容器
            ".newsContent",               # 旧版
            ".content",                   # 通用
            ".articleBody", ".article", ".media_article", "#newsContent"
        ]
        for sel in candidates:
            node = soup.select_one(sel)
            if node:
                text = node.get_text(separator=" ", strip=True)
                if len(text) >= MIN_ARTICLE_CHARS:
                    return text

    # —— 每日经济新闻（nbd.com.cn） / 同源转载到 eastmoney —— 
    if "nbd.com.cn" in url:
        candidates = [".g-article", ".article", ".content", ".m-article"]
        for sel in candidates:
            node = soup.select_one(sel)
            if node:
                text = node.get_text(separator=" ", strip=True)
                if len(text) >= MIN_ARTICLE_CHARS:
                    return text

    # —— 通用：尝试寻找常见正文容器 —— 
    common_candidates = [
        "article", ".article", ".article-content", ".content", ".post", "#article",
        ".main-content", ".detail", ".news_content", "#content", ".rich_media_content"
    ]
    for sel in common_candidates:
        node = soup.select_one(sel)
        if node:
            text = node.get_text(separator=" ", strip=True)
            if len(text) >= MIN_ARTICLE_CHARS:
                return text

    # —— 退化：取最大文本块（段落综合） —— 
    blocks = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "div", "section"]) if p]
    blocks = [b for b in blocks if b and len(b) > 10]
    blocks.sort(key=lambda x: len(x), reverse=True)
    text = " ".join(blocks[:15])  # 取若干大块拼接
    return text


def _extract_with_readability(url: str, html: str) -> str:
    if not _HAS_READABILITY:
        return ""
    try:
        doc = Document(html)
        summary_html = doc.summary()  # 提取后的 HTML
        soup = BeautifulSoup(summary_html, "lxml")
        text = soup.get_text(separator=" ", strip=True)
        return text
    except Exception as e:
        _log_fail(f"READABILITY_FAIL {url} -> {e}")
        return ""


def _extract_with_newspaper(url: str) -> str:
    if not _HAS_NEWSPAPER:
        return ""
    try:
        art = Article(url, language="zh")
        art.download()
        art.parse()
        return (art.text or "").strip()
    except Exception as e:
        _log_fail(f"NEWSPAPER_FAIL {url} -> {e}")
        return ""


def fetch_article_text(url: str) -> str:
    """
    获取文章完整正文（带缓存），步骤：
    1) 命中缓存返回
    2) newspaper3k
    3) requests + readability
    4) requests + BeautifulSoup（站点特化 + 通用）
    """
    if not url or not isinstance(url, str) or not url.startswith("http"):
        return ""

    # 命中缓存
    cached = _read_article_cache(url)
    if cached:
        return cached

    text = ""

    # 方案 1：newspaper3k
    if _HAS_NEWSPAPER:
        try:
            text = _retry(lambda: _extract_with_newspaper(url))
        except Exception as e:
            _log_fail(f"NEWSPAPER_FETCH_FAIL {url} -> {e}")

    # 方案 2/3：requests + readability / bs4
    if not text or len(text) < MIN_ARTICLE_CHARS:
        if not _HAS_REQUESTS:
            _log_fail(f"REQUESTS_MISSING {url}")
        else:
            try:
                html = _retry(lambda: _requests_html(url))
                # 2) readability
                if _HAS_READABILITY:
                    try:
                        text = _extract_with_readability(url, html)
                    except Exception:
                        pass
                # 3) BeautifulSoup（站点特化 + 通用）
                if not text or len(text) < MIN_ARTICLE_CHARS:
                    text = _extract_with_bs4(url, html)
            except Exception as e:
                _log_fail(f"REQUESTS_FETCH_FAIL {url} -> {e}")

    # 清洗 + 截断
    text = _sanitize_text(text)
    if len(text) > MAX_ARTICLE_CHARS:
        text = text[:MAX_ARTICLE_CHARS]

    # 写缓存（即使较短也写入，避免频繁请求）
    if text:
        _write_article_cache(url, text)

    return text


# =================== 新闻情绪因子 ===================
def get_news_AKSHARE_STOCK_NEWS_EM(code: str, topk: int = NEWS_TOPK) -> float:
    """
    获取个股最近新闻“完整正文”的情感平均值（[-1, 1]），抓不到正文则回退标题/摘要。
    """
    path = os.path.join(AKSHARE_STOCK_NEWS_EM_DIR, f"{code}.csv")

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

    # 标题/内容/链接列
    title_col = _get_first_existing_col(df, ["新闻标题", "标题", "news_title", "title"])
    content_col = _get_first_existing_col(df, ["新闻内容", "摘要", "content", "desc"])
    link_col = _get_first_existing_col(df, ["新闻链接", "链接", "url", "link"])
    # 日期排序
    df = _to_datetime_sorted(df)

    # 取最近 topk 条
    df_recent = df.tail(topk) if len(df) > topk else df

    texts = []
    for _, row in df_recent.iterrows():
        full_text = ""
        # 1) 优先抓取新闻链接的完整正文
        if link_col and pd.notna(row.get(link_col, None)):
            url = str(row[link_col]).strip()
            if url.startswith("http"):
                try:
                    full_text = fetch_article_text(url)
                except Exception as e:
                    _log_fail(f"FETCH_ARTICLE_FAIL {code} {url} -> {e}")

        # 2) 抓不到正文则回退到“新闻内容/摘要/标题”
        if not full_text or len(full_text) < MIN_ARTICLE_CHARS:
            fallback = None
            if content_col and pd.notna(row.get(content_col, None)):
                fallback = str(row[content_col])
            elif title_col and pd.notna(row.get(title_col, None)):
                fallback = str(row[title_col])
            if fallback:
                full_text = fallback

        if full_text:
            texts.append(full_text)

    if not texts:
        return 0.0

    # SnowNLP 情感：对每篇取分，再平均
    scores = []
    for t in texts:
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


# =================== 批量提取（并发 + 可扩展） ===================
def _one_code(code: str) -> dict:
    # 获取Akshare个股新闻（东财接口）
    try:
        senti = get_news_AKSHARE_STOCK_NEWS_EM(code, topk=NEWS_TOPK)
    except Exception as e:
        _log_fail(f"AKSHARE_STOCK_NEWS_EM_FAIL {code} -> {e}")
        senti = 0.0
        
    # 其他新闻模块
    
    
    # 其他消息模块 

    _random_pause()

    return {
        "代码": code,
        "新闻情绪": senti,
        # 其他结果返回
    }


def extract_AKSHARE_STOCK_NEWS_EM_fund_features(code_list, max_workers: int = MAX_WORKERS) -> pd.DataFrame:
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
    # df.insert(2, "资金天数", FUND_DAYS)
    # df.insert(3, "龙虎榜天数", LHB_DAYS)
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
    df_feat = extract_AKSHARE_STOCK_NEWS_EM_fund_features(codes, max_workers=MAX_WORKERS)
    df_feat.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(df_feat.head())
    print(f"\n✅ 已保存：{OUT_PATH}")
    print(f"📄 失败日志（如有）：{FAIL_LOG}")
