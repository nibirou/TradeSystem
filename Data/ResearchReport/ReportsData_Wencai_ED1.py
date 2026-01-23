# 通过同花顺wencai问财获取个股研报信息并保存、支持增量更新
import os
import time
import random
from datetime import datetime, timedelta

import baostock as bs
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random
from tqdm import tqdm

import requests
import time
import random
from typing import List, Dict
import pandas as pd


# ===========================
#        路径配置
# ===========================
BASE_DIR = "/workspace/Quant/data_baostock"
REPORT_DIR = os.path.join(BASE_DIR, "data_iwencai_reports")
META_DIR = os.path.join(BASE_DIR, "metadata")
HIST_DIR = os.path.join(BASE_DIR, "stock_hist")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)

FAILED_LIST = []
EMPTY_LIST = []

def bs_login():
    lg = bs.login()
    if lg.error_code != "0":
        raise Exception(f"Baostock login failed: {lg.error_msg}")

def bs_logout():
    bs.logout()

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
        max_request_count: int = 10
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

        # 最多重复访问10次
        if data.get("status_code") != 0:
            msg = data.get("status_msg", "")
            request_count = 1
            # ⚠️ iwencai 常见风控：伪“空结果”
            while "查询结果为空" in msg and request_count <= max_request_count:
                print(f"⚠️ 研报接口返回空结果，尝试更换 userid 进行一次验证，当前第{request_count}次")

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
                
                request_count += 1
                time.sleep(1)
            
            raise RuntimeError(f"研报接口异常（retry {max_request_count}次后仍失败）：{msg}")

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
        max_request_count: int = 10
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
        print("data:", data)

        request_count = 1
        # 如果研报详情为空，反复修改userid请求（最多10次）
        while data["data"].get("wordData") is None and request_count <= max_request_count:
            print(f"⚠️ 获取到的研报详情为空，尝试更换 userid，当前第{request_count}次")
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
            data = resp2.json()
            request_count += 1
            time.sleep(1)

        # 如果有其它异常状态，仅重复请求一次
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


def get_codes_from_report_hist_dir(pool):
    d = os.path.join(HIST_DIR, pool)
    if not os.path.exists(d):
        return set()

    codes = set()
    for fn in os.listdir(d):
        if fn.endswith(".csv"):
            codes.add(fn.replace(".csv", "").replace("_", "."))
    return codes


def load_last_snapshot(pool):
    path = os.path.join(META_DIR, f"stock_list_{pool}.csv")
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(df["code"].tolist())

def update_single_stock_reports(
    client: IwencaiReportClient,
    code: str,
    pool: str,
    init_fetch: int = 20,     # 首次下载条数
    max_rows: int = 500,      # CSV 最大行数阈值
):
    """
    单只股票研报维护规则：
    - 第一次：抓最近 init_fetch 条
    - 后续：只增量追加，不删旧数据
    - 当 CSV 行数 > max_rows 时：
        按 publish_time 删除最旧的，只保留 max_rows
    """

    save_dir = os.path.join(REPORT_DIR, pool)
    os.makedirs(save_dir, exist_ok=True)

    code_clean = code.replace(".", "_")
    csv_path = os.path.join(save_dir, f"{code_clean}.csv")

    # ======================================================
    # 1. 读取已有 CSV（如果存在）
    # ======================================================
    old_df = None
    old_uids = set()

    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        if not old_df.empty:
            old_df["uid"] = old_df["uid"].astype(str)
            old_uids = set(old_df["uid"])

    # ======================================================
    # 2. 抓取研报列表
    #    - 首次：只需要覆盖 init_fetch
    #    - 后续：抓多一点，靠 uid 去重
    # ======================================================
    max_pages = max(1, init_fetch // 15 + 1)

    reports = client.crawl_all_reports(
        code=code.split(".")[-1],
        max_pages=max_pages,
    )

    if not reports:
        EMPTY_LIST.append(code)
        return

    # ======================================================
    # 3. 增量抓取研报全文
    # ======================================================
    new_rows = []

    for r in reports:
        uid = str(r.get("uid"))
        if uid in old_uids:
            continue
        print(r)
        try:
            detail = client.fetch_report_detail(uid, code.split(".")[-1])
            print(detail)
            word = detail.get("wordData", {})

            # row = {
            #     **r,
            #     "uid": uid,
            #     "content": word.get("content", ""),
            # }
            row = {
                "uid": uid,
                "title": word.get("title", ""),
                "organization":  word.get("organize", ""),
                "author":  word.get("researcher", ""),
                "publish_time":  word.get("pubtime", ""),
                "content": word.get("content", ""),
            }
            new_rows.append(row)

        except Exception as e:
            FAILED_LIST.append(code)
            print(f"[研报详情失败] {code} {uid}: {e}")

    if not new_rows:
        print(f"⚠️⚠️⚠️⚠️⚠️⚠️[当前股票爬取到的研报数据条数为0] {code}")
        return

    new_df = pd.DataFrame(new_rows)

    # ======================================================
    # 4. 合并新旧数据（只追加）
    # ======================================================
    if old_df is not None:
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df

    # ======================================================
    # 5. 去重（最终兜底）
    # ======================================================
    df["uid"] = df["uid"].astype(str)
    df = df.drop_duplicates(subset=["uid"], keep="first")

    # ======================================================
    # 6. 超过阈值 → 删除最旧的
    # ======================================================
    if "publish_time" in df.columns:
        df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
        df = df.sort_values("publish_time", ascending=True)  # 旧 → 新

    if len(df) > max_rows:
        df = df.iloc[-max_rows:]   # 只保留最新 max_rows 条

    # ======================================================
    # 7. 写回 CSV
    # ======================================================
    df.to_csv(csv_path, index=False, encoding="utf-8")

def _bs_query_to_df(rs):
    data_list = []
    while (rs.error_code == "0") & rs.next():
        data_list.append(rs.get_row_data())
    return pd.DataFrame(data_list, columns=rs.fields)

def get_latest_end_date():
    """
    根据 Baostock 交易日历，获取距离程序运行当天最近的一个交易日
    - 如果今天是交易日 → 返回今天
    - 如果今天不是交易日（周末/节假日）→ 返回最近交易日
    """

    # ---- 登录 ----
    # lg = bs.login()
    # if lg.error_code != "0":
    #     raise Exception(f"Baostock login failed: {lg.error_msg}")

    today = datetime.now().strftime("%Y-%m-%d")

    # 仅查询今年的交易日即可（快）
    year_start = f"{datetime.now().year}-01-01"
    year_second= f"{datetime.now().year}-01-02"
    year_third = f"{datetime.now().year}-01-03"
    year_fourth = f"{datetime.now().year}-01-04"
    
    # 如果当日是某年的1月1日，应该查询前一年的最后一个交易日为end_date，year_start需要相应调整
    if today == year_start or today == year_second or today == year_third or today ==year_fourth:
        year_start = f"{datetime.now().year - 1}-01-01"
        
    rs = bs.query_trade_dates(start_date=year_start, end_date=today)

    # ---- 解析交易日数据 ----
    data_list = []
    while (rs.error_code == "0") & rs.next():
        data_list.append(rs.get_row_data())

    # 退出
    # bs.logout()

    if not data_list:
        raise Exception("baostock query_trade_dates 返回空数据")

    df = pd.DataFrame(data_list, columns=rs.fields)

    # DataFrame 字段：calendar_date, is_trading_day
    df["calendar_date"] = pd.to_datetime(df["calendar_date"])
    df["is_trading_day"] = df["is_trading_day"].astype(int)

    # 提取所有交易日
    trade_days = df[df["is_trading_day"] == 1]["calendar_date"].tolist()

    if len(trade_days) == 0:
        raise Exception("没有找到任何交易日")

    # 最近一个交易日
    latest_trade_day = trade_days[-1].strftime("%Y-%m-%d")
    return latest_trade_day

def get_stock_list_bs(mode="hs300", day=None):
    if mode == "sz50":
        rs = bs.query_sz50_stocks()
    elif mode == "hs300":
        rs = bs.query_hs300_stocks()
    elif mode == "zz500":
        rs = bs.query_zz500_stocks()
    elif mode == "all":
        if day is None:
            raise ValueError("mode='all' 需要 day='YYYY-MM-DD'")
        print(day)
        # rs = bs.query_all_stock(day="2025-12-22")
        rs = bs.query_all_stock(day=day)
    else:
        raise ValueError(f"未知股票池模式：{mode}")

    df = _bs_query_to_df(rs)
    print(df)

    if "code_name" in df.columns:
        df = df.rename(columns={"code_name": "name"})
    else:
        df["name"] = df["code"]

    return df[["code", "name"]]

def run_report_download(pool="hs300"):
    bs_login()
    client = IwencaiReportClient()

    # === 股票池 ===
    if pool == "all":
        stocks = get_stock_list_bs("all", day=get_latest_end_date())
    else:
        stocks = get_stock_list_bs(pool)

    # stocks.to_csv(
    #     os.path.join(META_DIR, f"stock_list_{pool}.csv"),
    #     index=False,
    #     encoding="utf-8",
    # )

    # snapshot_path = os.path.join(
    #     SNAPSHOT_DIR,
    #     f"{pool}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    # )
    # stocks.to_csv(snapshot_path, index=False, encoding="utf-8")

    current_codes = set(stocks["code"])
    hist_codes = get_codes_from_report_hist_dir(pool)
    last_snapshot_codes = load_last_snapshot(pool)
    # print(hist_codes)

    final_codes = current_codes | hist_codes | last_snapshot_codes
    codes = sorted(final_codes)

    print(f"[股票池] 本次 {len(current_codes)}")
    print(f"[历史研报] {len(hist_codes)}")
    print(f"[最终更新] {len(codes)}")

    for code in tqdm(codes):
        try:
            update_single_stock_reports(client, code, pool)
            time.sleep(random.uniform(0.8, 1.5))
        except Exception as e:
            FAILED_LIST.append(code)
            print(f"[失败] {code}: {e}")

    # ---- 日志 ----
    if FAILED_LIST:
        pd.DataFrame({"code": FAILED_LIST}).to_csv(
            os.path.join(REPORT_DIR, "failed_reports.csv"),
            index=False,
        )

    if EMPTY_LIST:
        pd.DataFrame({"code": EMPTY_LIST}).to_csv(
            os.path.join(REPORT_DIR, "empty_reports.csv"),
            index=False,
        )

if __name__ == "__main__":
    # run_report_download(pool="sz50")
    run_report_download(pool="hs300")
    # run_report_download(pool="zz500")
    # run_report_download(pool="all")
    bs_logout()