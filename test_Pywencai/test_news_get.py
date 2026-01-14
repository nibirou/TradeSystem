import requests
import time
from typing import List, Dict
import random

class IwencaiNewsClient:
    """
    同花顺问财 - 个股新闻接口客户端
    """

    URL = "https://www.iwencai.com/unifiedwap/unified-wap/v1/information/news"

    def __init__(self):
        self.headers = {
            "accept": "application/json, text/plain, */*",
            "content-type": "application/x-www-form-urlencoded",
            "origin": "https://www.iwencai.com",
            "referer": "https://www.iwencai.com/unifiedwap/inforesult",
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/143.0.0.0 Safari/537.36"
            ),

            # ✅ 使用你提供的完整 Cookie（已整理）
            "cookie": (
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
                "userid=566756439; "    # 最好是每换一只股票随机生成一个userid 566796434
                "utk=3262e41d288421bc9c6340644220039b; "
                "cid=2c9c0c9b495a5e33e6226c22e7cc3ed91768187607; "
                "ComputerID=2c9c0c9b495a5e33e6226c22e7cc3ed91768187607; "
                "u_dpass=2CcKkH00sroyYH%2FsI12MMJvbN4IRtVjK3sUwYW%2F1mHvt%2B8LjA3QxTinPHALynlBfHi80LrSsTFH9a%2B6rtRvqGg%3D%3D; "
                "u_did=233EB705EF0C4596BD1A255BE0753091; "
                "u_uver=1.0.0; "
                "WafStatus=1; "
                "user_status=0"
            ),
        }
        self.cookie_str = self.headers["cookie"]
        self.session = requests.Session()

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

    def fetch_news(
        self,
        code: str,
        offset: int = 0,
        size: int = 15,
        dl: int = 120,
        tl: int = 41,
        date_range: int = 3,
    ) -> Dict:
        payload = {
            "query": code,
            "size": str(size),
            "offset": str(offset),
            "dl": str(dl),
            "tl": str(tl),
            "date_range": str(date_range),
            "mobile": "3",
        }

        resp = requests.post(
            self.URL,
            headers=self.headers,
            data=payload,
            timeout=10,
        )

        resp.raise_for_status()
        data = resp.json()

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

    def crawl_all_news(
        self,
        code: str,
        max_pages: int = 10,
        sleep_sec: float = 1.0,
    ) -> List[Dict]:
        all_results = []
        offset = 0

        for _ in range(max_pages):
            data = self.fetch_news(code=code, offset=offset)
            results = data.get("results", [])

            if not results:
                break

            all_results.extend(results)
            offset += len(results)
            time.sleep(sleep_sec)

        return all_results


if __name__ == "__main__":
    client = IwencaiNewsClient()

    # ======== 在这里指定股票代码 =========
    stock_code = "600519"

    news_list = client.crawl_all_news(
        code=stock_code,
        max_pages=10,  # 连着爬10页以上有时候爬不到
    )

    print(f"共抓取到 {len(news_list)} 条新闻\n")

    for n in news_list[:5]:
        print(n["publish_time"], n["publish_source"])
        print(n["title"])
        print(n["summary"])
        print("-" * 80)
