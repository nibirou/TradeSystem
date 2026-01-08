# 调用akshare的个股研报接口并进行nlp分析
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

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from curl_cffi import requests as cf_requests
    _HAS_CURL_CFFI = True
except Exception:
    _HAS_CURL_CFFI = False

try:
    import tls_client
    _HAS_TLS_CLIENT = True
except Exception:
    _HAS_TLS_CLIENT = False

try:
    import requests
    from bs4 import BeautifulSoup  # pip install beautifulsoup4 lxml
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False

# —— 可选依赖：PDF 正文抽取（多路兜底） ——
try:
    from pdfminer.high_level import extract_text as _pdfminer_extract_text  # pip install pdfminer.six -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple 
    _HAS_PDFMINER = True
except Exception:
    _HAS_PDFMINER = False

try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    _HAS_PYMUPDF = False

R00T_DIR = "data"
META_DIR = "data/metadata"
STOCK_NEWS_DIR = os.path.join(R00T_DIR, "stock_news")
RESULT_DIR = os.path.join(STOCK_NEWS_DIR, "reports_nlp_result_by_day")
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# —— 研报（个股研报）缓存目录 ——
AKSHARE_STOCK_REPORT_EM_DIR = os.path.join(STOCK_NEWS_DIR, "akshare_stock_report_em")
AKSHARE_STOCK_REPORT_EM_TEXT_DIR = os.path.join(AKSHARE_STOCK_REPORT_EM_DIR, "reports")
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_PATH = os.path.join(RESULT_DIR, f"stock_reports_factors_{RUN_TS}.csv") # 新闻评分结果输出(带日期)
FAIL_LOG = os.path.join(RESULT_DIR, f"stock_reports_failures_{RUN_TS}.log") # 新闻获取失败结果（带日期）

os.makedirs(AKSHARE_STOCK_REPORT_EM_DIR, exist_ok=True)
os.makedirs(AKSHARE_STOCK_REPORT_EM_TEXT_DIR, exist_ok=True)

# 仅分析近 N 年研报
REPORT_WINDOW_YEARS = 2

# 并发线程数
MAX_WORKERS = 8

# 轻量限速：每只股票抓完后随机 sleep 区间（秒）
RANDOM_SLEEP_RANGE = (0.05, 0.2)

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

# 缓存文件最大有效期（小时），超过则刷新抓取
MAX_AGE_HOURS_NEWS = 24        # 新闻列表缓存
MAX_AGE_HOURS_REPORT = 24 * 7  # 研报文本缓存有效期（一周）
REPORT_TOPK = 10               # 最近 topk 篇研报用于打分
REPORT_TEXT_MAX_CHARS = 12000  # 解析 PDF 后用于情感的最大字符数

# 抓取重试
RETRIES = 3
BASE_WAIT = 0.8  # 秒，指数退避基数

# 研报条数：取“最近 topk 条”用于情绪打分
REPORTS_TOPK = 20
# 正文最小长度阈值（太短说明抽取失败，回退标题/摘要）
MIN_ARTICLE_CHARS = 150

# “东财评级”映射到 [-1,1] 的得分（可按需扩展）
_ECF_RATING2SCORE = {
    "强烈推荐": 1.0, "买入": 0.9, "增持": 0.6, "推荐": 0.7, "审慎增持": 0.5,
    "跑赢行业": 0.5, "持有": 0.0, "中性": 0.0, "谨慎推荐": 0.2,
    "减持": -0.6, "卖出": -0.9, "跑输行业": -0.5
}
# 研报最终得分 = 评级分 * α + 文本情感分 * (1-α)
REPORT_ALPHA_RATING = 0.6

# —— 下载/调试相关（可选） ——
VERIFY_SSL = True          # 如你在内网被拦，可暂时置 False（不推荐）
DEBUG_SAVE_RAW = True      # 把拿到的 PDF/HTML 原始内容落盘便于排查
DEBUG_SAVE_OK_PDF = False  # True=连成功的 PDF 也落盘

# 可能启动OCR
ENABLE_OCR = True
OCR_MAX_PAGES = 3  # 只 OCR 前几页以控时

VERIFY_SSL = True              # 如确实被内网拦，可临时 False（不推荐）
DEBUG_SAVE_RAW = True          # 失败时把 HTML/PDF 原始字节落盘以便排查
DEBUG_SAVE_OK_PDF = False      # True=连成功的 PDF 也保存（调试用）
ENABLE_OCR = True              # 启用 OCR 兜底
OCR_MAX_PAGES = 3              # 仅 OCR 前几页控制耗时

# —— 研报（个股研报）缓存工具 ——
def _log_fail(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(FAIL_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")

def _sanitize_text(txt: str) -> str:
    if not txt:
        return ""
    # 去脚注/重复空白
    txt = re.sub(r"\s+", " ", txt)
    # 常见版权尾巴粗略剔除（可按需扩展）
    txt = re.sub(r"(责任编辑[:：].*?$)", "", txt)
    return txt.strip()

def _report_cache_path(key: str) -> str:
    """研报文本缓存：以 {code}_report_{YYYYMMDD_HHMMSS}_{seq}.json 命名"""
    fname = f"{key}.json"
    return os.path.join(AKSHARE_STOCK_REPORT_EM_TEXT_DIR, fname)

def _read_report_cache(key: str):
    path = _report_cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ts = datetime.fromisoformat(data.get("fetched_at"))
        if (datetime.now() - ts).total_seconds() > MAX_AGE_HOURS_REPORT * 3600:
            return None
        return data.get("text", "")
    except Exception:
        return None

def _write_report_cache(key: str, text: str):
    path = _report_cache_path(key)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"key": key, "text": text, "fetched_at": datetime.now().isoformat()}, f, ensure_ascii=False)
    except Exception as e:
        _log_fail(f"WRITE_REPORT_CACHE_FAIL {key} -> {e}")

# —— PDF 下载与文本抽取工具 ——
def _get_session():
    s = requests.Session()
    s.headers.update(REQ_HEADERS)
    retry = Retry(
        total=2, backoff_factor=0.4,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

def _debug_dump_bytes(content: bytes, url: str, suffix: str):
    if not DEBUG_SAVE_RAW or not content:
        return
    md = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    path = os.path.join(AKSHARE_STOCK_REPORT_EM_TEXT_DIR, f"_raw_{md}{suffix}")
    try:
        with open(path, "wb") as f:
            f.write(content)
    except Exception:
        pass

def _requests_get(url: str):
    if not _HAS_REQUESTS:
        raise RuntimeError("Missing 'requests' / 'beautifulsoup4' dependency")
    return requests.get(url, headers=REQ_HEADERS, timeout=REQ_TIMEOUT)

def _is_pdf_bytes(b: bytes) -> bool:
    return isinstance(b, (bytes, bytearray)) and len(b) > 5 and b[:5] == b"%PDF-"

def _download_pdf_bytes(url: str) -> bytes:
    """
    智能下载器：
    1) curl_cffi（Chrome 指纹）+ 预热 Cookie
    2) tls_client（Chrome 指纹）+ 预热 Cookie
    3) 回退 requests（你原本的逻辑）
    全程带 Referer/Origin/Accept，失败落盘 HTML 便于排查
    """
    # 通用头
    h2 = {
        "Referer": "https://data.eastmoney.com/report/stock.jshtml",
        "Origin": "https://data.eastmoney.com",
        "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    }

    # ---------- 1) curl_cffi ----------
    if _HAS_CURL_CFFI:
        try:
            s = cf_requests.Session(impersonate="chrome")
            # 预热：获取 Cookie
            s.get("https://data.eastmoney.com/report/stock.jshtml",
                  headers=REQ_HEADERS, timeout=REQ_TIMEOUT, verify=VERIFY_SSL)
            r = s.get(url, headers={**REQ_HEADERS, **h2},
                      timeout=REQ_TIMEOUT, verify=VERIFY_SSL, allow_redirects=True)
            c = r.content or b""
            ct = (r.headers.get("Content-Type") or "").lower()
            if _is_pdf_bytes(c):
                if DEBUG_SAVE_OK_PDF: _debug_dump_bytes(c, url, ".pdf")
                return c
            else:
                if DEBUG_SAVE_RAW and c and ("pdf" not in ct):
                    _debug_dump_bytes(c, url, ".html")
        except Exception as e:
            _log_fail(f"CURL_CFFI_DOWNLOAD_FAIL {url} -> {e}")

    # ---------- 2) tls_client ----------
    if _HAS_TLS_CLIENT:
        try:
            s = tls_client.Session(client_identifier="chrome_120")
            s.headers.update(REQ_HEADERS)
            s.get("https://data.eastmoney.com/report/stock.jshtml",
                  timeout=REQ_TIMEOUT, allow_redirects=True, verify=VERIFY_SSL)
            r = s.get(url, headers=h2, timeout=REQ_TIMEOUT,
                      allow_redirects=True, verify=VERIFY_SSL)
            c = r.content or b""
            ct = (r.headers.get("Content-Type") or "").lower()
            if _is_pdf_bytes(c):
                if DEBUG_SAVE_OK_PDF: _debug_dump_bytes(c, url, ".pdf")
                return c
            else:
                if DEBUG_SAVE_RAW and c and ("pdf" not in ct):
                    _debug_dump_bytes(c, url, ".html")
        except Exception as e:
            _log_fail(f"TLS_CLIENT_DOWNLOAD_FAIL {url} -> {e}")

    # ---------- 3) 回退：requests（保留你原逻辑） ----------
    if not _HAS_REQUESTS:
        raise RuntimeError("Missing 'requests' for PDF download")

    # 简易会话 + 重试
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    sess = requests.Session()
    sess.headers.update(REQ_HEADERS)
    retry = Retry(total=2, backoff_factor=0.4,
                  status_forcelist=[429,500,502,503,504], allowed_methods=["GET"])
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    sess.mount("http://", HTTPAdapter(max_retries=retry))

    # 第一次
    r1 = sess.get(url, timeout=REQ_TIMEOUT, allow_redirects=True, verify=VERIFY_SSL)
    c1 = r1.content or b""
    ct1 = (r1.headers.get("Content-Type") or "").lower()
    if _is_pdf_bytes(c1):
        if DEBUG_SAVE_OK_PDF: _debug_dump_bytes(c1, url, ".pdf")
        return c1
    else:
        if DEBUG_SAVE_RAW and c1 and ("pdf" not in ct1):
            _debug_dump_bytes(c1, url, ".html")

    # 第二次：加 Referer/Origin
    r2 = sess.get(url, headers=h2, timeout=REQ_TIMEOUT, allow_redirects=True, verify=VERIFY_SSL)
    c2 = r2.content or b""
    ct2 = (r2.headers.get("Content-Type") or "").lower()
    if _is_pdf_bytes(c2):
        if DEBUG_SAVE_OK_PDF: _debug_dump_bytes(c2, url, ".pdf")
        return c2
    else:
        if DEBUG_SAVE_RAW and c2 and ("pdf" not in ct2):
            _debug_dump_bytes(c2, url, ".html")

    # 若 HTML，尝试从中提取真实 .pdf 再拉
    try:
        enc = r2.apparent_encoding or r2.encoding or "utf-8"; r2.encoding = enc
        html = r2.text or ""
    except Exception:
        html = ""
    if html:
        m = re.search(r"https?://[^\"'>]+\.pdf", html, flags=re.I)
        if m:
            real_pdf = m.group(0)
            r3 = sess.get(real_pdf, headers=h2, timeout=REQ_TIMEOUT, allow_redirects=True, verify=VERIFY_SSL)
            c3 = r3.content or b""
            ct3 = (r3.headers.get("Content-Type") or "").lower()
            if _is_pdf_bytes(c3):
                if DEBUG_SAVE_OK_PDF: _debug_dump_bytes(c3, real_pdf, ".pdf")
                return c3
            else:
                _log_fail(f"REPORT_PDF_NOT_PDF_AFTER_REDIRECT {real_pdf} -> {ct3}")
                if DEBUG_SAVE_RAW and c3: _debug_dump_bytes(c3, real_pdf, ".html")
        else:
            _log_fail(f"REPORT_PDF_HTML_RETURNED_NO_PDF_LINK {url}")
    else:
        _log_fail(f"REPORT_PDF_NOT_PDF {url} -> {ct2}")

    _log_fail(f"REPORT_PDF_DOWNLOAD_FAIL_NOT_PDF {url}")
    return b""


def _ocr_pdf_bytes_with_tesseract(pdf_bytes: bytes, max_pages: int = 3) -> str:
    if not (_HAS_PYMUPDF):
        return ""
    try:
        import pytesseract
        from PIL import Image
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        texts = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(dpi=200)  # 提高一点清晰度
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            t = pytesseract.image_to_string(img, lang="chi_sim")
            texts.append(t or "")
        return _sanitize_text("\n".join(texts))
    except Exception as e:
        _log_fail(f"OCR_FAIL -> {e}")
        return ""
def _extract_text_from_pdf_bytes(pdf_bytes: bytes, max_chars: int = REPORT_TEXT_MAX_CHARS) -> str:
    """优先 pdfminer；失败或过短再用 PyMuPDF 多模式；最后清洗 + 截断"""
    if not pdf_bytes:
        return ""

    text_all = ""

    # A) pdfminer（版面模式对中文研报通常较稳）
    if _HAS_PDFMINER:
        try:
            import io
            from pdfminer.high_level import extract_text_to_fp
            from pdfminer.layout import LAParams

            output = io.StringIO()
            laparams = LAParams(
                all_texts=True,
                line_margin=0.2,
                char_margin=1.0,
                word_margin=0.1,
                boxes_flow=None
            )
            extract_text_to_fp(io.BytesIO(pdf_bytes), output, laparams=laparams, output_type='text', codec='utf-8')
            txt = output.getvalue()
            text_all = (txt or "").replace("\x00", "")
        except Exception as e:
            _log_fail(f"PDFMINER_EXTRACT_FAIL -> {e}")
            text_all = ""

    # B) 若 pdfminer 为空或很短，尝试 PyMuPDF 多模式
    if (not text_all or len(text_all) < 50) and _HAS_PYMUPDF:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            # 解密（空密码试一下）
            if doc.is_encrypted:
                try:
                    doc.authenticate("")
                except Exception:
                    pass

            parts = []
            # 先用 'text'（布局重排）
            for i, page in enumerate(doc):
                t = page.get_text("text") or ""
                parts.append(t)
                if sum(len(p) for p in parts) >= max_chars:
                    break
            t1 = "\n".join(parts).replace("\x00", "").strip()

            # 再用 'blocks'（按块），两个拼谁长用谁
            parts2 = []
            for i, page in enumerate(doc):
                t = page.get_text("blocks") or ""
                # blocks 返回的是列表时也兼容
                if isinstance(t, list):
                    t = "\n".join(block[4] for block in t if isinstance(block, (list, tuple)) and len(block) >= 5)
                parts2.append(t or "")
                if sum(len(p) for p in parts2) >= max_chars:
                    break
            t2 = "\n".join(parts2).replace("\x00", "").strip()

            text_all = t1 if len(t1) >= len(t2) else t2
        except Exception as e:
            _log_fail(f"PYMUPDF_EXTRACT_FAIL -> {e}")

    # 清洗 + 截断（先做一次）
    text_all = _sanitize_text(text_all)

    # —— 如果还是很短/为空，尝试 OCR 兜底 ——
    if ENABLE_OCR and (not text_all or len(text_all) < 50) and _HAS_PYMUPDF:
        try:
            ocr_text = _ocr_pdf_bytes_with_tesseract(pdf_bytes, max_pages=OCR_MAX_PAGES)
            ocr_text = _sanitize_text(ocr_text)
            if len(ocr_text) > len(text_all):
                text_all = ocr_text
        except Exception as e:
            _log_fail(f"OCR_PIPELINE_FAIL -> {e}")

    # 最终截断
    if len(text_all) > max_chars:
        text_all = text_all[:max_chars]
    return text_all


# =================== 工具函数 ===================
def _random_pause():
    low, high = RANDOM_SLEEP_RANGE
    time.sleep(random.uniform(low, high))

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
def _get_first_existing_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _need_refresh(path: str, max_age_hours: int) -> bool:
    if not os.path.exists(path):
        return True
    try:
        # mtime = datetime.fromtimestamp(os.path.getmtime(path))
        # return (datetime.now() - mtime).total_seconds() > max_age_hours * 3600
        return True
    except Exception:
        return True
def get_research_report_em_score(code: str, topk: int = REPORT_TOPK) -> float:
    """
    东方财富-个股研报：综合“东财评级”与 PDF 文本情感，得到 [-1,1] 的研报得分。
    - 列表缓存：{AKSHARE_STOCK_REPORT_EM_DIR}/{code}.csv（24h 与 NEWS 同逻辑可复用）
    - 文本缓存：{AKSHARE_STOCK_REPORT_EM_TEXT_DIR}/{code}_report_{YYYYMMDD_HHMMSS}_{seq}.json
    """
    path = os.path.join(AKSHARE_STOCK_REPORT_EM_DIR, f"{code}.csv")

    def fetch():
        # ak 接口 symbol 需要代码；若需按名称检索，可自行在外层做映射
        return ak.stock_research_report_em(symbol=code)

    # 读取或刷新列表缓存（与新闻相同策略）
    df = None
    try:
        if _need_refresh(path, MAX_AGE_HOURS_NEWS):
            df = _retry(fetch)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(path, index=False, encoding="utf-8-sig")
        else:
            df = pd.read_csv(path)
    except Exception as e:
        _log_fail(f"REPORT_LIST_FETCH_FAIL {code} -> {e}")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
            except Exception:
                df = None

    if df is None or df.empty:
        return 0.0

    # 列名鲁棒
    title_col  = _get_first_existing_col(df, ["报告名称", "标题", "name", "报告标题"])
    rating_col = _get_first_existing_col(df, ["东财评级", "评级", "rating"])
    date_col   = _get_first_existing_col(df, ["日期", "time", "发布时间", "pub_time"])
    pdf_col    = _get_first_existing_col(df, ["报告PDF链接", "pdf链接", "pdf", "url"])

    # 仅保留“近两年”的研报，然后排序取最近 topk
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        except Exception:
            pass

        # 计算时间窗：当前时刻往回 REPORT_WINDOW_YEARS 年，只获取近年来的研报数据
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=REPORT_WINDOW_YEARS)
        # 丢掉没有日期的、以及早于窗口的
        df = df[df[date_col].notna() & (df[date_col] >= cutoff)]

        # 若过滤后为空，直接返回 0（也可记录一条日志便于排查）
        if df.empty:
            # _log_fail(f"REPORT_DATE_FILTER_EMPTY {code} cutoff={cutoff.date()}")
            return 0.0

        # 升序排序，便于 tail(topk) 取最近
        df = df.sort_values(date_col, ascending=True)

    # 取最近 topk
    df_recent = df.tail(topk) if len(df) > topk else df

    # 简单去重（优先按 PDF 链接）
    if pdf_col and pdf_col in df_recent.columns:
        df_recent = df_recent.drop_duplicates(subset=[pdf_col])
    else:
        keys = [c for c in [title_col, date_col] if c and c in df_recent.columns]
        if keys:
            df_recent = df_recent.drop_duplicates(subset=keys)

    item_scores = []

    for i, (_, row) in enumerate(df_recent.iterrows(), start=1):
        # 评级 → 分值（[-1,1]）
        rating_score = None
        if rating_col and pd.notna(row.get(rating_col, None)):
            raw = str(row[rating_col]).strip()
            # 兼容“买入（首次）/维持买入/上调至增持”等
            key_hit = None
            for k in _ECF_RATING2SCORE.keys():
                if k in raw:
                    key_hit = k
                    break
            if key_hit is not None:
                rating_score = _ECF_RATING2SCORE[key_hit]

        # 文本情感
        text_sent = None
        # 先 PDF 正文
        pdf_url = str(row.get(pdf_col, "")).strip() if pdf_col else ""
        date_val = row.get(date_col, None)
        pub_key = f"{code}_report_{_format_pub_time(date_val)}_{i:03d}"

        if pdf_url.startswith("http"):
            try:
                cached = _read_report_cache(pub_key)
                if cached:
                    text = cached
                else:
                    pdf_bytes = _retry(lambda: _download_pdf_bytes(pdf_url))
                    if not _is_pdf_bytes(pdf_bytes):
                        _log_fail(f"REPORT_PDF_BYTES_NOT_PDF {code} {pdf_url}")
                        text = ""
                    else:
                        text = _extract_text_from_pdf_bytes(pdf_bytes, max_chars=REPORT_TEXT_MAX_CHARS)
                        print(text)
                        if text:
                            _write_report_cache(pub_key, text)
                if text and len(text) >= MIN_ARTICLE_CHARS:
                    s = SnowNLP(text).sentiments
                    text_sent = (s - 0.5) * 2
            except Exception as e:
                _log_fail(f"REPORT_PDF_PARSE_FAIL {code} {pdf_url} -> {e}")

        # 若 PDF 失败，回退标题情感
        if text_sent is None and title_col and pd.notna(row.get(title_col, None)):
            try:
                t = _sanitize_text(str(row[title_col]))
                if t:
                    s = SnowNLP(t).sentiments
                    text_sent = (s - 0.5) * 2
            except Exception as e:
                _log_fail(f"REPORT_TITLE_SENT_FAIL {code} -> {e}")

        # 融合：评级/文本均无则跳过
        if rating_score is None and text_sent is None:
            continue
        if rating_score is None:
            final = text_sent
        elif text_sent is None:
            final = rating_score
        else:
            final = REPORT_ALPHA_RATING * rating_score + (1 - REPORT_ALPHA_RATING) * text_sent

        item_scores.append(final)

    if not item_scores:
        return 0.0
    return round(float(pd.Series(item_scores).mean()), 4)

def _one_code(code: str, name: str = None) -> dict:
    # 研报情绪（东财）
    try:
        senti_report_em = get_research_report_em_score(code, topk=REPORT_TOPK)
    except Exception as e:
        _log_fail(f"AKSHARE_STOCK_REPORT_EM_FAIL {code} -> {e}")
        senti_report_em = 0.0

    _random_pause()

    return {
        "代码": code,
        "AKSHARE东财接口的研报情绪": senti_report_em,
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
    df.insert(1, "研报topk", REPORTS_TOPK)
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

