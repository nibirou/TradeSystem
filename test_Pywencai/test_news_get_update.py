import time
import random
from typing import List, Dict, Optional
import requests

from bs4 import BeautifulSoup
import re
import pandas as pd

class IwencaiNewsClientSafe:
    """
    风控友好型：低频、限页、退避、遇403熔断。
    """

    URL = "https://www.iwencai.com/unifiedwap/unified-wap/v1/information/news"

    def __init__(
        self,
        cookie_str: str,
        min_interval_sec: float = 6.0,
        max_pages_per_stock: int = 5,
        page_size: int = 15,
        cooldown_403_sec: int = 15 * 60,
        timeout_sec: int = 10,
        user_agent: Optional[str] = None,
    ):
        # ✅ 一定要先创建 session
        self.session = requests.Session()

        self.cookie_str = cookie_str.strip()
        self.min_interval_sec = float(min_interval_sec)
        self.max_pages_per_stock = int(max_pages_per_stock)
        self.page_size = int(page_size)
        self.cooldown_403_sec = int(cooldown_403_sec)
        self.timeout_sec = int(timeout_sec)

        ua = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/143.0.0.0 Safari/537.36"
        )

        # ✅ headers 固定下来，模拟真实页面请求
        self.session.headers.update({
            # "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": "https://www.iwencai.com",
            "Referer": "https://www.iwencai.com/unifiedwap/inforesult",
            "User-Agent": ua,
            "Cookie": self.cookie_str,
            # "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9",
            # # ✅ 关键：禁止 br / zstd，避免 requests 解不出来导致“乱码”
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })

        self._last_request_ts = 0.0

        # ✅ 初始化自检：确保 session 存在
        if not hasattr(self, "session") or self.session is None:
            raise RuntimeError("Session 初始化失败：self.session is None")

    def _rate_limit_and_human_pause(self):
        """
        1) 保证两次请求之间 >= min_interval
        2) 增加轻微随机停顿，模拟“阅读/翻页”
        """
        now = time.time()
        dt = now - self._last_request_ts
        if dt < self.min_interval_sec:
            time.sleep(self.min_interval_sec - dt)

        # 轻微随机：不要抖太大，抖太大也不像人
        time.sleep(random.uniform(1.0, 3.0))
        self._last_request_ts = time.time()

    def _post(self, data: Dict) -> requests.Response:
        """
        带退避的 POST：
        - 403：熔断冷却
        - 429/5xx：指数退避
        """
        backoff = 2.0
        for attempt in range(1, 6):  # 最多尝试 5 次
            self._rate_limit_and_human_pause()
            resp = self.session.post(self.URL, data=data, timeout=self.timeout_sec)

            # 403： confirmed 风控
            if resp.status_code == 403:
                print(f"⚠️ 403 Forbidden（attempt={attempt}），冷却 {self.cooldown_403_sec}s 后退出")
                time.sleep(self.cooldown_403_sec)
                raise RuntimeError("触发 403 风控，已冷却并终止本次任务")

            # 429 或 5xx：退避再试（不硬撞）
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                sleep_s = backoff + random.uniform(0.0, 1.5)
                print(f"⚠️ HTTP {resp.status_code}（attempt={attempt}），退避 {sleep_s:.1f}s 后重试")
                time.sleep(sleep_s)
                backoff *= 1.8
                continue

            resp.raise_for_status()
            return resp

        raise RuntimeError("多次重试仍失败（429/5xx）")

    # >>> ADD
    def _gen_random_userid_like(self, old_userid: str) -> str:
        """
        生成一个“位数相同”的随机 userid（仅用于一次性验证）
        """
        n = len(old_userid)
        # 避免前导 0
        first = str(random.randint(1, 9))
        rest = "".join(str(random.randint(0, 9)) for _ in range(n - 1))
        return first + rest

    def _replace_userid_in_cookie(self, new_userid: str):
        """
        只替换 Cookie 中的 userid 字段
        """
        parts = self.cookie_str.split(";")
        new_parts = []
        replaced = False

        for p in parts:
            p_strip = p.strip()
            if p_strip.startswith("userid="):
                new_parts.append(f" userid={new_userid}")
                replaced = True
            else:
                new_parts.append(p)

        if not replaced:
            # 理论上不会发生，但兜底
            new_parts.append(f" userid={new_userid}")

        self.cookie_str = ";".join(new_parts).strip()
        self.session.headers["Cookie"] = self.cookie_str

    def fetch_news_page(
        self,
        code: str,
        offset: int = 0,
        dl: int = 120,
        tl: int = 41,
        date_range: int = 3,
    ) -> Dict:
        payload = {
            "query": code,
            "size": str(self.page_size),
            "offset": str(offset),
            "dl": str(dl),
            "tl": str(tl),
            "date_range": str(date_range),
            "mobile": "3",
        }

        resp = self._post(payload)
        data = resp.json()

        # >>> MODIFY：捕获“查询结果为空”
        if data.get("status_code") != 0:
            msg = data.get("status_msg", "")
            if "查询结果为空" in msg:
                print("⚠️ 返回空结果，尝试更换 userid 进行一次验证")

                # 解析当前 userid
                old_userid = None
                for p in self.cookie_str.split(";"):
                    p = p.strip()
                    if p.startswith("userid="):
                        old_userid = p.split("=", 1)[1]
                        break

                if old_userid:
                    new_userid = self._gen_random_userid_like(old_userid)
                    print(f"   userid: {old_userid} -> {new_userid}")
                    self._replace_userid_in_cookie(new_userid)

                    # 🔁 仅 retry 一次
                    resp2 = self._post(payload)
                    data2 = resp2.json()

                    if data2.get("status_code") == 0:
                        return data2["data"]

                # retry 仍失败，抛异常
                raise RuntimeError(f"接口返回异常（retry 后）：{msg}")

            raise RuntimeError(f"接口返回异常：{msg}")

        return data["data"]


    def crawl_stock_news(self, code: str) -> List[Dict]:
        """
        单股票抓取：严格限制页数，避免触发风控。
        """
        results_all: List[Dict] = []
        offset = 0

        for page in range(self.max_pages_per_stock):
            data = self.fetch_news_page(code=code, offset=offset)
            results = data.get("results", [])
            if not results:
                break

            results_all.extend(results)
            offset += len(results)

        return results_all

    def _decode_html_bytes(self, raw: bytes, header_ct: str | None = None) -> str:
        """
        对 bytes 做稳健解码：优先 header charset，其次常见中文编码，再用 utf-8 兜底。
        """
        # 1) header 里有 charset
        if header_ct:
            m = re.search(r"charset=([-\w]+)", header_ct, re.I)
            if m:
                enc = m.group(1).strip().lower()
                try:
                    return raw.decode(enc, errors="replace")
                except Exception:
                    pass

        # 2) 常见中文站点编码优先尝试
        for enc in ("utf-8", "gb18030", "gbk", "gb2312"):
            try:
                txt = raw.decode(enc, errors="replace")
                # 简单判定：解出来含大量中文才更可信（可选）
                return txt
            except Exception:
                continue

        # 3) 最后兜底
        return raw.decode("utf-8", errors="replace")

    def fetch_full_article(self, url: str) -> str:
        if not url:
            return ""
        if url.startswith("//"):
            url = "https:" + url

        # ✅ 更像真实用户：正文页比接口慢很多
        time.sleep(random.uniform(3.0, 6.0))

        resp = self.session.get(url, timeout=15, allow_redirects=True)
        # 403 / 429 也别硬撞（你可以沿用你已有的熔断逻辑）
        resp.raise_for_status()

        # ✅ 关键：不要用 resp.text（容易乱码）
        html = self._decode_html_bytes(resp.content, resp.headers.get("Content-Type"))

        soup = BeautifulSoup(html, "lxml")

        # 1) article
        article = soup.find("article")
        if article:
            return self._clean_text(article.get_text("\n", strip=True))

        # 2) 常见正文容器（覆盖微信/财经站常见结构）
        candidates = [
            {"id": "js_content"},               # 微信正文
            {"class_": re.compile(r"rich_media_content")},
            {"class_": re.compile(r"article-content|articleContent|content|main-content|post-content|article_content")},
            {"class_": re.compile(r"TRS_Editor|article|detail|text")},  # 一些门户
        ]
        for kw in candidates:
            node = soup.find(**kw)
            if node:
                txt = node.get_text("\n", strip=True)
                if len(txt) > 80:
                    return self._clean_text(txt)

        # 3) 兜底：聚合 p
        ps = soup.find_all("p")
        if ps:
            texts = []
            for p in ps:
                t = p.get_text(strip=True)
                if len(t) >= 20:
                    texts.append(t)
            if texts:
                return self._clean_text("\n".join(texts))

        return ""

    def _clean_text(self, text: str) -> str:
        """
        简单清洗正文
        """
        if not text:
            return ""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()
    
    def export_news_to_csv(self, news_list: List[Dict], csv_path: str):
        """
        对每条新闻抓取全文并导出 CSV
        """
        rows = []

        for i, n in enumerate(news_list, 1):
            print(f"📄 抓取正文 {i}/{len(news_list)}")

            full_content = self.fetch_full_article(n.get("url", ""))

            rows.append({
                "publish_time": n.get("publish_time", ""),
                "publish_source": n.get("publish_source", ""),
                "title": n.get("title", ""),
                "summary": n.get("summary", ""),
                "full_content": full_content,
                "source_url": n.get("url", ""),
            })

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"✅ CSV 已输出: {csv_path}")



if __name__ == "__main__":
    # ✅ 把你那条 cookie 拼成一行字符串粘贴在这里（原样即可）
    COOKIE_STR = (
                "cuc=xx8ezzz2zvmw; "
                "escapename=mx_566796434; "
                "u_name=mx_566796434; "
                "ta_random_userid=ktsynuvtgz; "
                "sess_tk=eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiIsImtpZCI6InNlc3NfdGtfMSIsImJ0eSI6InNlc3NfdGsifQ."
                "eyJqdGkiOiI2ZTAwYTQ4MzE4MTdjNjlhZGUwZjEzZWY2MjRmMTFkNTEiLCJpYXQiOjE3NjgyMTAwNzYsImV4cCI6"
                "MTc2ODgxNDg3Niwic3ViIjoiNTY2Nzk2NDM0IiwiaXNzIjoidXBhc3MuaXdlbmNhaS5jb20iLCJhdWQiOiIyMDIw"
                "MTExODUyODg5MDcyIiwiYWN0Ijoib2ZjIiwiY3VocyI6IjNjZGIzNWNiOTdmZmQ2Mzk0M2U3OTdiMGJmNzg2NjY4"
                "ZDEzOGNhZGU0Mzg0N2IyMjI1NjkyZTVlYWMzMzA0NmMifQ."
                "4fJThGiPP8-Vsm_HAbRoS9v81mX6jTxa5riLlJDuoLrBPzafG8YzJl2OtrSFtcZeITMRzJvxCZF03Cy1IIw5qw; "
                "ticket=a7f38de4cf7f3dee19d63006e77965c3; "
                "ttype=WEB; "
                "u_ttype=WEB; "
                "other_uid=Ths_iwencai_Xuangu_9t4ghwjskfabmu9iy7daondbz4hnjia0; "
                "user=MDpteF81NjY3OTY0MzQ6Ok5vbmU6NTAwOjU3Njc5NjQzNDo3LDExMTExMTExMTExLDQwOzQ0LDExLDQwOzYs"
                "MSw0MDs1LDEsNDA6MTY6Ojo1NjY3OTY0MzQ6MTc2ODIxMDA3Njo6OjE2MTI5MjUzNDA6NjA0ODAwOjA6MWQ1M"
                "TE0ZjYyZWYxMzBmZGU5YWM2MTcxODgzYTQwMDZlOmRlZmF1bHRfNTow; "
                "u_ukey=A10702B8689642C6BE607730E11E6E4A; "
                "v=A090c19BeiKTqX5e7o9euutl3uhcdKKgPcmnmWFc6oDGeGGWaUQz5k2YN81y; "
                "PHPSESSID=96090d860f932f660d55e0b685d0442d; "
                "userid=362796439; "    # 最好是每换一只股票随机生成一个userid
                "utk=3262e41d288421bc9c6340644220039b; "
                "cid=2c9c0c9b495a5e33e6226c22e7cc3ed91768187607; "
                "ComputerID=2c9c0c9b495a5e33e6226c22e7cc3ed91768187607; "
                "u_dpass=2CcKkH00sroyYH%2FsI12MMJvbN4IRtVjK3sUwYW%2F1mHvt%2B8LjA3QxTinPHALynlBfHi80LrSsTFH9a%2B6rtRvqGg%3D%3D; "
                "u_did=233EB705EF0C4596BD1A255BE0753091; "
                "u_uver=1.0.0; "
                "WafStatus=1; "
                "user_status=0"
            )

    client = IwencaiNewsClientSafe(
        cookie_str=COOKIE_STR,
        min_interval_sec=8.0,        # 更保守一点
        max_pages_per_stock=1,       # 单次最多 3 页
        cooldown_403_sec=15 * 60,    # 403 冷却 15 分钟
    )

    stock_code = "600223"  # ✅ 在这里指定
    try:
        news_list = client.crawl_stock_news(stock_code)
        print(f"抓取完成：{stock_code} 共 {len(news_list)} 条\n")

        for n in news_list[:5]:
            print(n.get("publish_time"), n.get("publish_source"))
            print(n.get("title"))
            print("-" * 80)
        
        client.export_news_to_csv(
            news_list,
            csv_path=f"news_{stock_code}.csv"
        )

    except Exception as e:
        print("终止原因：", e)
