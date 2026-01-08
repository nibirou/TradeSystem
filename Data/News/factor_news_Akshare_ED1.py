# -*- coding: utf-8 -*-
"""
个股 消息面 + 资金面 数据提取（单文件 · 正文抓取升级版）
- 新闻标题列名鲁棒性 + 日期排序
- 正文抓取：优先用“新闻链接”获取完整文章内容（多策略 + 站点特化）
- 正文/新闻缓存（避免重复抓）
- 缓存过期刷新（新闻列表/资金流/正文）
- 重试 + 指数退避 + 失败日志
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

try:
    import trafilatura  # pip install trafilatura
    _HAS_TRAFILATURA = True
except Exception:
    _HAS_TRAFILATURA = False

try:
    from boilerpy3 import extractors as _bp_extractors
    _HAS_BOILERPY3 = True
except Exception:
    _HAS_BOILERPY3 = False

try:
    from goose3 import Goose
    _HAS_GOOSE3 = True
except Exception:
    _HAS_GOOSE3 = False   


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
def _format_pub_time(val) -> str:
    """
    将发布时间值统一格式化为 YYYYMMDD_HHMMSS；失败则返回 'unknown'
    """
    try:
        ts = pd.to_datetime(val, errors="coerce")
        if pd.isna(ts):
            return "unknown"
        return ts.strftime("%Y%m%d_%H%M%S")
    except Exception:
        return "unknown"
    
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

# —— 常见“免责声明/声明/版权尾巴”关键词 —— 
_DISCLAIMER_PREFIXES = [
    "郑重声明", "特别声明", "免责声明", "风险提示"
]
_DISCLAIMER_KEYWORDS = [
    "不构成投资建议", "仅供参考", "东方财富网不对", "以中国证监会指定",
    "版权", "转载", "责任编辑", "来源：东方财富"
]

def _looks_like_disclaimer(p: str) -> bool:
    if not p: 
        return False
    p = p.strip()
    if any(p.startswith(pre) for pre in _DISCLAIMER_PREFIXES):
        return True
    if len(p) < 120 and any(k in p for k in _DISCLAIMER_KEYWORDS):
        # 段落很短而且包含声明关键词，更像声明/版权尾巴或提示条
        return True
    return False

def _strip_disclaimer_tail(txt: str) -> str:
    if not txt:
        return txt
    # 从出现声明关键词的位置截断（保守做法：仅在靠后时截）
    for kw in _DISCLAIMER_PREFIXES + _DISCLAIMER_KEYWORDS:
        idx = txt.rfind(kw)
        if idx != -1 and idx > len(txt) * 0.6:
            return txt[:idx].strip()
    return txt

def _clean_join_paragraphs(paras):
    """合并段落并去掉声明/工具条/很短的UI文案"""
    out = []
    for p in paras:
        p = re.sub(r"\s+", " ", p or "").strip()
        if not p: 
            continue
        if _looks_like_disclaimer(p):
            continue
        # 过滤分享、编辑、收藏、打印等工具条
        if len(p) < 12 and any(w in p for w in ["微信", "微博", "QQ", "收藏", "打印", "客户端", "APP", "分享"]):
            continue
        out.append(p)
    text = " ".join(out).strip()
    return _strip_disclaimer_tail(text)

def _score_candidate_text(t: str) -> float:
    """给多个抽取结果打分，选最像“正文”的那个"""
    if not t:
        return -1e9
    n = len(t)
    zh = sum(0x4e00 <= ord(c) <= 0x9fff for c in t)
    digits = sum(c.isdigit() for c in t)
    punct = sum(c in "，。；：？！,.:%；、" for c in t)
    has_disclaimer = any(k in t for k in _DISCLAIMER_PREFIXES + _DISCLAIMER_KEYWORDS)

    # 长度（上限 5000）、中文占比、是否含数字&标点（财经稿常见）、声明惩罚
    score = min(n, 5000) / 5000.0
    if n > 0:
        score += 0.5 * (zh / n)
    if digits > 0:
        score += 0.15
    if punct > 5:
        score += 0.1
    if has_disclaimer and n < 400:
        score -= 1.0
    return score

def _sanitize_text(txt: str) -> str:
    if not txt:
        return ""
    # 去脚注/重复空白
    txt = re.sub(r"\s+", " ", txt)
    # 常见版权尾巴粗略剔除（可按需扩展）
    txt = re.sub(r"(责任编辑[:：].*?$)", "", txt)
    return txt.strip()


# =================== 正文抓取与缓存 ===================
def _article_cache_path(url: str, cache_key: str = None) -> str:
    """
    若提供 cache_key（形如 '000001_20250815_180505'），则用它命名；
    否则退回用 url 的 md5 命名（向后兼容）。
    """
    if cache_key:
        fname = f"{cache_key}.json"
    else:
        fname = f"{_md5(url)}.json"
    return os.path.join(AKSHARE_STOCK_NEWS_EM_ARTICLES_DIR, fname)

def _read_article_cache(url: str, cache_key: str = None):
    path = _article_cache_path(url, cache_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ts = datetime.fromisoformat(data.get("fetched_at"))
        if (datetime.now() - ts).total_seconds() > MAX_AGE_HOURS_ARTICLE * 3600:
            return None
        return data.get("text", "")
    except Exception:
        return None


def _write_article_cache(url: str, text: str, cache_key: str = None):
    path = _article_cache_path(url, cache_key)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"url": url, "text": text, "fetched_at": datetime.now().isoformat()},
                f,
                ensure_ascii=False
            )
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

def _extract_eastmoney_main(html: str) -> str:
    """
    东方财富站点（finance.eastmoney.com 等）的定制抽取：
    - 优先锁定正文容器
    - 仅收集正文段落，并过滤“免责声明/版权尾巴/工具条”
    """
    soup = BeautifulSoup(html, "lxml")
    # 常见容器候选（东财不同频道 class/id 会变，所以多准备几个）
    containers = [
        "#ContentBody", "#newsContent", ".newsContent", ".articleBody", ".article", ".content", ".media_article",
        ".main-content", ".m-article", ".g-article", ".detail"
    ]
    candidates = []
    for sel in containers:
        node = soup.select_one(sel)
        if not node:
            continue
        paras = [tag.get_text(" ", strip=True) for tag in node.find_all(["p","div","section","span"], recursive=True)]
        t = _clean_join_paragraphs(paras)
        if len(t) >= 60:
            candidates.append(t)

    # 如果上述容器都不靠谱，退化为“从页面所有段落中选密度最高的若干段”
    if not candidates:
        blocks = []
        for tag in soup.find_all(["p","div","section"]):
            txt = tag.get_text(" ", strip=True)
            if not txt or len(txt) < 15:
                continue
            blocks.append(txt)
        blocks.sort(key=lambda x: len(x), reverse=True)
        t = _clean_join_paragraphs(blocks[:20])
        if len(t) >= 60:
            candidates.append(t)

    if not candidates:
        return ""
    # 选分数最高的候选
    candidates.sort(key=_score_candidate_text, reverse=True)
    return candidates[0]

def _extract_with_boilerpy3(html: str) -> str:
    """
    boilerpy3 抽取正文（对中文也友好）
    """
    if not _HAS_BOILERPY3 or not html:
        return ""
    try:
        extractor = _bp_extractors.ArticleExtractor()
        text = extractor.get_content(html) or ""
        return text.strip()
    except Exception:
        return ""


def _extract_with_goose3(url: str, html: str) -> str:
    """
    goose3 抽取正文：优先用 raw_html，避免再次发起网络请求
    """
    if not _HAS_GOOSE3 or not html:
        return ""
    try:
        g = Goose({
            "enable_image_fetching": False,
            "use_meta_language": False,
            "target_language": "zh",
            "browser_user_agent": REQ_HEADERS.get("User-Agent", ""),
        })
        art = g.extract(raw_html=html)
        text = (art.cleaned_text or art.meta_description or "") if art else ""
        return text.strip()
    except Exception:
        return ""
def _extract_with_bs4(url: str, html: str) -> str:
    """
    通用 + 站点特化（优先东财），返回尽可能接近正文的文本。
    """
    try:
        host = ""
        m = re.search(r"https?://([^/]+)/", url)
        if m:
            host = m.group(1).lower()
    except Exception:
        host = ""

    # —— 东方财富强定制 —— 
    if "eastmoney.com" in host:
        t = _extract_eastmoney_main(html)
        if len(t) >= MIN_ARTICLE_CHARS:
            return t

    # —— 通用容器 —— 
    soup = BeautifulSoup(html, "lxml")
    common_candidates = [
        "article", ".article", ".article-content", ".content", ".post", "#article",
        ".main-content", ".detail", ".news_content", "#content", ".rich_media_content"
    ]
    texts = []
    for sel in common_candidates:
        node = soup.select_one(sel)
        if not node:
            continue
        paras = [tag.get_text(" ", strip=True) for tag in node.find_all(["p","div","section","span"], recursive=True)]
        t = _clean_join_paragraphs(paras)
        if len(t) >= 60:
            texts.append(t)

    # 退化：大块拼接
    if not texts:
        blocks = []
        for tag in soup.find_all(["p","div","section"]):
            txt = tag.get_text(" ", strip=True)
            if not txt or len(txt) < 15:
                continue
            blocks.append(txt)
        blocks.sort(key=lambda x: len(x), reverse=True)
        t = _clean_join_paragraphs(blocks[:20])
        if len(t) >= 60:
            texts.append(t)

    if not texts:
        return ""
    texts.sort(key=_score_candidate_text, reverse=True)
    return texts[0]



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

def _maybe_fetch_amp_html(url: str, html: str) -> str:
    """
    如页面提供 <link rel="amphtml" href="...">，抓取 AMP 版本再尝试抽取。
    """
    try:
        soup = BeautifulSoup(html, "lxml")
        amp = soup.find("link", rel=lambda v: v and "amphtml" in v.lower())
        if amp and amp.get("href"):
            amp_url = amp["href"]
            if amp_url.startswith("//"):
                amp_url = "https:" + amp_url
            if amp_url.startswith("/"):
                # 相对路径情况
                base = re.match(r"(https?://[^/]+)", url)
                if base:
                    amp_url = base.group(1) + amp_url
            amp_html = _requests_html(amp_url)
            return amp_html
    except Exception as e:
        _log_fail(f"AMP_FETCH_FAIL {url} -> {e}")
    return ""
def fetch_article_text(url: str, cache_key: str = None) -> str:
    """
    获取文章完整正文（带缓存）：
    - 命中缓存（若给了 cache_key 则用 {code}_{pubtime}.json 命名）
    - 抽取逻辑：newspaper3k → readability → bs4（站点特化+通用）→ AMP 回退 → trafilatura
    - 多候选打分择优 + 免责声明去除
    """
    if not url or not isinstance(url, str) or not url.startswith("http"):
        return ""

    # 1) 缓存
    cached = _read_article_cache(url, cache_key)
    if cached:
        return cached

    candidates = []

    # 2) newspaper3k
    if _HAS_NEWSPAPER:
        try:
            t = _retry(lambda: _extract_with_newspaper(url))
            t = _strip_disclaimer_tail(_sanitize_text(t))
            if len(t) >= MIN_ARTICLE_CHARS:
                candidates.append(("newspaper", t))
        except Exception as e:
            _log_fail(f"NEWSPAPER_FETCH_FAIL {url} -> {e}")

    # 3) requests + readability / bs4
    html = ""
    try:
        html = _retry(lambda: _requests_html(url))
    except Exception as e:
        _log_fail(f"REQUESTS_FETCH_FAIL {url} -> {e}")

    if html:
        # 3a) readability
        if _HAS_READABILITY:
            try:
                t = _extract_with_readability(url, html)
                t = _strip_disclaimer_tail(_sanitize_text(t))
                if len(t) >= MIN_ARTICLE_CHARS:
                    candidates.append(("readability", t))
            except Exception:
                pass

        # 3b) bs4（含 eastmoney 特化）
        try:
            t = _extract_with_bs4(url, html)
            t = _strip_disclaimer_tail(_sanitize_text(t))
            if len(t) >= MIN_ARTICLE_CHARS:
                candidates.append(("bs4", t))
        except Exception as e:
            _log_fail(f"BS4_EXTRACT_FAIL {url} -> {e}")
        
        # —— boilerpy3 候选 —— 
        try:
            t = _extract_with_boilerpy3(html)
            t = _strip_disclaimer_tail(_sanitize_text(t))
            if len(t) >= MIN_ARTICLE_CHARS:
                candidates.append(("boilerpy3", t))
        except Exception as e:
            _log_fail(f"BOILERPY3_FAIL {url} -> {e}")

        # —— goose3 候选 —— 
        try:
            t = _extract_with_goose3(url, html)
            t = _strip_disclaimer_tail(_sanitize_text(t))
            if len(t) >= MIN_ARTICLE_CHARS:
                candidates.append(("goose3", t))
        except Exception as e:
            _log_fail(f"GOOSE3_FAIL {url} -> {e}")

        # 3c) AMP 回退再试
        try:
            amp_html = _maybe_fetch_amp_html(url, html)
            if amp_html:
                # 先 readability
                if _HAS_READABILITY:
                    t = _extract_with_readability(url, amp_html)
                    t = _strip_disclaimer_tail(_sanitize_text(t))
                    if len(t) >= MIN_ARTICLE_CHARS:
                        candidates.append(("amp_readability", t))
                # 再 bs4
                t = _extract_with_bs4(url, amp_html)
                t = _strip_disclaimer_tail(_sanitize_text(t))
                if len(t) >= MIN_ARTICLE_CHARS:
                    candidates.append(("amp_bs4", t))
                # AMP + boilerpy3
                try:
                    t = _extract_with_boilerpy3(amp_html)
                    t = _strip_disclaimer_tail(_sanitize_text(t))
                    if len(t) >= MIN_ARTICLE_CHARS:
                        candidates.append(("amp_boilerpy3", t))
                except Exception as e:
                    _log_fail(f"AMP_BOILERPY3_FAIL {url} -> {e}")

                # AMP + goose3
                try:
                    t = _extract_with_goose3(url, amp_html)
                    t = _strip_disclaimer_tail(_sanitize_text(t))
                    if len(t) >= MIN_ARTICLE_CHARS:
                        candidates.append(("amp_goose3", t))
                except Exception as e:
                    _log_fail(f"AMP_GOOSE3_FAIL {url} -> {e}")
        except Exception as e:
            _log_fail(f"AMP_EXTRACT_FAIL {url} -> {e}")

    # 4) trafilatura 兜底
    if (not candidates) and _HAS_TRAFILATURA and html:
        try:
            t = trafilatura.extract(html, include_comments=False, favor_recall=True, output_format="txt")
            t = _strip_disclaimer_tail(_sanitize_text(t or ""))
            if len(t) >= MIN_ARTICLE_CHARS:
                candidates.append(("trafilatura", t))
        except Exception as e:
            _log_fail(f"TRAFILATURA_FAIL {url} -> {e}")

    # 5) 选择最佳候选
    best = ""
    if candidates:
        # 根据 _score_candidate_text 打分择优
        best = max(candidates, key=lambda kv: _score_candidate_text(kv[1]))[1]

    # 6) 截断 + 写缓存
    if best and len(best) > MAX_ARTICLE_CHARS:
        best = best[:MAX_ARTICLE_CHARS]
    if best:
        _write_article_cache(url, best, cache_key)

    return best or ""



# =================== 新闻情绪因子 ===================
def get_news_AKSHARE_STOCK_NEWS_EM(code: str, name: str = None, topk: int = NEWS_TOPK) -> float:
    """
    获取个股最近新闻“完整正文”的情感平均值（[-1, 1]）。
    - symbol 优先用股票名称（若传入 name），否则用 code
    - 正文缓存命名：{code}_{pub_time}
    - SnowNLP sentiments 修正 + 简单去重
    """
    path = os.path.join(AKSHARE_STOCK_NEWS_EM_DIR, f"{code}.csv")

    def fetch():
        symbol_str = name if (name and isinstance(name, str) and name.strip()) else code
        return ak.stock_news_em(symbol=symbol_str)

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
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
            except Exception:
                df = None

    if df is None or df.empty:
        return 0.0

    # 列名鲁棒：标题/内容/链接/时间
    title_col = _get_first_existing_col(df, ["新闻标题", "标题", "news_title", "title"])
    content_col = _get_first_existing_col(df, ["新闻内容", "摘要", "content", "desc"])
    link_col = _get_first_existing_col(df, ["新闻链接", "链接", "url", "link"])
    time_col = _get_first_existing_col(df, ["日期", "时间", "发布时间", "time", "pub_time"])

    # 时间列转 datetime 并升序
    df = _to_datetime_sorted(df)
    if time_col in df.columns:
        try:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        except Exception:
            pass

    # 取最近 topk
    df_recent = df.tail(topk) if len(df) > topk else df

    # 简单去重：优先按链接去重；没有链接则按 [标题, 时间] 去重
    if link_col and link_col in df_recent.columns:
        df_recent = df_recent.drop_duplicates(subset=[link_col])
    else:
        keys = [c for c in [title_col, time_col] if c and c in df_recent.columns]
        if keys:
            df_recent = df_recent.drop_duplicates(subset=keys)

    texts = []
    for _, row in df_recent.iterrows():
        full_text = ""
        pub_key = None

        # 生成缓存 key：{code}_{发布时间}
        if time_col and pd.notna(row.get(time_col, None)):
            pub_key = f"{code}_{_format_pub_time(row[time_col])}"
        else:
            pub_key = f"{code}_unknown"

        # 1) 优先抓链接正文（带 cache_key 命名）
        if link_col and pd.notna(row.get(link_col, None)):
            url = str(row[link_col]).strip()
            if url.startswith("http"):
                try:
                    full_text = fetch_article_text(url, cache_key=pub_key)
                except Exception as e:
                    _log_fail(f"FETCH_ARTICLE_FAIL {code} {url} -> {e}")

        # 2) 回退摘要/标题
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

    # SnowNLP 情感：修正为 .sentiments ∈ [0,1]
    scores = []
    for t in texts:
        try:
            s = SnowNLP(t).sentiments
            scores.append(s)
        except Exception as e:
            _log_fail(f"SNOWNLP_FAIL {code} -> {e}")

    if not scores:
        return 0.0

    mapped = [(s - 0.5) * 2 for s in scores]
    return round(float(pd.Series(mapped).mean()), 4)


# =================== 批量提取（并发 + 可扩展） ===================
def _one_code(code: str, name: str = None) -> dict:
    # 获取Akshare个股新闻（东财接口）
    try:
        senti_akshre_em = get_news_AKSHARE_STOCK_NEWS_EM(code, name=name, topk=NEWS_TOPK)
    except Exception as e:
        _log_fail(f"AKSHARE_STOCK_NEWS_EM_FAIL {code} -> {e}")
        senti_akshre_em = 0.0
        
    # 其他新闻模块
    
    
    # 其他消息模块 

    _random_pause()

    return {
        "代码": code,
        "AKSHARE东财接口的新闻情绪": senti_akshre_em,
        # 其他结果返回
    }


def extract_AKSHARE_STOCK_NEWS_EM_fund_features(code_list, max_workers: int = MAX_WORKERS) -> pd.DataFrame:
    """
    code_list 支持两种元素形态：
    - "000001"（只有代码）
    - ("000001", "平安银行")（代码 + 名称）
    """
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {}
        for item in code_list:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                code, name = item[0], item[1]
            else:
                code, name = item, None
            fut = ex.submit(_one_code, code, name)
            futs[fut] = code

        for fut in tqdm(as_completed(futs), total=len(futs), desc="Extracting"):
            try:
                rows.append(fut.result())
            except Exception as e:
                code = futs.get(fut, "UNKNOWN")
                _log_fail(f"TASK_FAIL {code} -> {e}")

    df = pd.DataFrame(rows)
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
    # 获取股票代码列
    if "代码" not in df_list.columns:
        # 做一个兜底：常见字段名 candidates
        cand = _get_first_existing_col(df_list, ["代码", "symbol", "ts_code", "code"])
        if cand and cand != "代码":
            df_list.rename(columns={cand: "代码"}, inplace=True)
        if "代码" not in df_list.columns:
            raise ValueError("股票列表缺少‘代码’列（或常见等价列）。")

    codes = df_list["代码"].dropna().astype(str).unique().tolist()

    # 获取股票名称列（可选）：['名称','股票名称','股票简称','name']
    name_col = _get_first_existing_col(df_list, ["名称", "股票名称", "股票简称", "name"])
    if name_col and name_col in df_list.columns:
        # 组装 (code, name) 列表
        merged = (
            df_list.dropna(subset=["代码"])
                .astype({"代码": str})
        )
        # 每个代码只取第一条名称
        merged = merged.drop_duplicates(subset=["代码"])
        code_name_list = list(zip(merged["代码"].tolist(), merged[name_col].astype(str).tolist()))
    else:
        code_name_list = codes  # 没有名称列就只用代码

    df_feat = extract_AKSHARE_STOCK_NEWS_EM_fund_features(code_name_list, max_workers=MAX_WORKERS)
    df_feat.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(df_feat.head())
    print(f"\n✅ 已保存：{OUT_PATH}")
    print(f"📄 失败日志（如有）：{FAIL_LOG}")
