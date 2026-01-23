# 通过wencai接口获取个股研报内容

import requests
import time
import random
from typing import List, Dict
import pandas as pd

class IwencaiReportClient:
    """
    同花顺问财 - 个股研报接口客户端
    """

    URL = "https://www.iwencai.com/unifiedwap/unified-wap/v1/information/report"
    DETAIL_URL = "https://www.iwencai.com/unifiedwap/unified-wap/v1/information/notice-detail"

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
            # ⚠️ 使用你已有的完整 Cookie（示例）
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
                "userid=326736439; "    # 最好是每换一只股票随机生成一个userid 566796434
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
        self.session.headers.update(self.headers)

    # ==========================================================
    # 反风控：userid 软切换
    # ==========================================================
    def _gen_random_userid_like(self, old_userid: str) -> str:
        """
        生成一个与旧 userid 位数一致的随机 userid
        """
        n = len(old_userid)
        first = str(random.randint(1, 9))
        rest = "".join(str(random.randint(0, 9)) for _ in range(n - 1))
        return first + rest

    def _replace_userid_in_cookie(self, new_userid: str):
        """
        仅替换 Cookie 中的 userid 字段
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
            new_parts.append(f" userid={new_userid}")

        self.cookie_str = ";".join(new_parts).strip()
        self.session.headers["cookie"] = self.cookie_str

    # ==========================================================
    # 单页研报抓取
    # ==========================================================
    def fetch_reports(
        self,
        code: str,
        offset: int = 0,
        size: int = 15,
        dl: int = 120,
        tl: int = 41,
        date_range: str = "",
        sort: str = "",
        cat_id: str = "",
    ) -> Dict:
        payload = {
            "query": code,
            "query_source": "user",
            "size": str(size),
            "offset": str(offset),
            "dl": str(dl),
            "tl": str(tl),
            "range": "",
            "sort": sort,
            "cat_id": cat_id,
            "date_range": date_range,
        }

        resp = self.session.post(
            self.URL,
            data=payload,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status_code") != 0:
            msg = data.get("status_msg", "")

            # ⚠️ iwencai 常见风控：伪“空结果”
            if "查询结果为空" in msg:
                print("⚠️ 研报接口返回空结果，尝试更换 userid 进行一次验证")

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

                    # 🔁 retry 一次
                    resp2 = self.session.post(
                        self.URL,
                        data=payload,
                        timeout=10,
                    )
                    resp2.raise_for_status()
                    data2 = resp2.json()

                    if data2.get("status_code") == 0:
                        return data2["data"]

                raise RuntimeError(f"研报接口异常（retry 后仍失败）：{msg}")

            raise RuntimeError(f"研报接口异常：{msg}")

        return data["data"]

    # ==========================================================
    # 多页研报爬取
    # ==========================================================
    def crawl_all_reports(
        self,
        code: str,
        max_pages: int = 10,
        sleep_sec: float = 1.0,
    ) -> List[Dict]:
        all_results = []
        offset = 0

        for _ in range(max_pages):
            data = self.fetch_reports(code=code, offset=offset)
            results = data.get("results", [])

            if not results:
                break

            all_results.extend(results)
            offset += len(results)

            time.sleep(random.uniform(sleep_sec * 0.8, sleep_sec * 1.6))

        return all_results

    def fetch_report_detail(
        self,
        uid: str,
        code: str,
    ) -> Dict:
        """
        根据研报 uid 获取研报全文（content）
        """
        payload = {
            "type": "report",
            "duid": uid,
            "query_source": "guide",
            "query": code,
        }

        resp = self.session.post(
            self.DETAIL_URL,
            data=payload,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status_code") != 0:
            msg = data.get("status_msg", "")

            # iwencai 典型风控：返回空 / 校验失败
            if "查询结果为空" in msg or "权限" in msg:
                print("⚠️ 研报详情接口触发校验，尝试更换 userid")

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

                    # 🔁 retry 一次
                    resp2 = self.session.post(
                        self.DETAIL_URL,
                        data=payload,
                        timeout=10,
                    )
                    resp2.raise_for_status()
                    data2 = resp2.json()

                    if data2.get("status_code") == 0:
                        return data2["data"]

                raise RuntimeError(f"研报详情接口异常（retry 后仍失败）：{msg}")

        return data["data"]

if __name__ == "__main__":
    client = IwencaiReportClient()

    stock_code = "600519"

    reports = client.crawl_all_reports(
        code=stock_code,
        max_pages=2,
    )

    print(f"研报条数：{len(reports)}")

    # 只示例抓前 3 篇全文（避免频率过高）
    for r in reports[:3]:
        uid = r["uid"]
        print(f"\n抓取研报详情: {uid}")

        detail = client.fetch_report_detail(uid, stock_code)
        word = detail["wordData"]
        print(r)

        print("标题:", r["title"])
        print("机构:", r["organization"])
        print("作者:", r["author"])
        print("发布日期:", r["publish_time"])
        print("正文:\n", word["content"])
        # print("正文前200字:\n", word["content"][:200])
    
    df = pd.DataFrame(r)
    df.to_csv(f"reports_{stock_code}.csv", index=False, encoding="utf-8-sig")
    print(f"✅ CSV 已输出: reports_{stock_code}.csv")
