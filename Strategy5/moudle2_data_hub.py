# data_hub.py
import os
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from moudle1_gp_config import Config


class DataHub:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.base_dir = cfg.base_dir

        self.hist_dir = os.path.join(self.base_dir, "stock_hist", cfg.pool, cfg.price_freq)
        self.fund_q_dir = os.path.join(self.base_dir, "baostock_fundamental_q", cfg.pool)

    def _list_hist_files(self) -> List[str]:
        if not os.path.exists(self.hist_dir):
            raise FileNotFoundError(f"hist_dir not found: {self.hist_dir}")
        files = [os.path.join(self.hist_dir, f) for f in os.listdir(self.hist_dir) if f.endswith(".csv")]
        files.sort()
        return files

    def _parse_code_from_filename(self, path: str) -> str:
        # hist: sh_600000_d.csv or sh_600000_5.csv 等，你的文件名是 {code_clean}_{freq}.csv
        name = os.path.basename(path).replace(".csv", "")
        # 去掉最后 _d / _5 / _15 ...
        parts = name.split("_")
        # 例如 sh_600000_d -> ['sh','600000','d']
        code = ".".join(parts[:2])  # sh.600000
        return code

    def load_price_panel(self) -> pd.DataFrame:
        files = self._list_hist_files()
        if self.cfg.max_stocks_for_debug is not None:
            files = files[: self.cfg.max_stocks_for_debug]

        dfs = []
        for fp in files:
            code = self._parse_code_from_filename(fp)
            df = pd.read_csv(fp)
            if df.empty:
                continue
            df["date"] = pd.to_datetime(df["date"])
            df["code"] = code

            # 只保留需要的列
            keep_cols = ["date", "code"] + [c for c in self.cfg.price_features if c in df.columns]
            df = df[keep_cols].copy()

            # 类型
            for c in self.cfg.price_features:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            dfs.append(df)

        if not dfs:
            raise RuntimeError("no price data loaded.")

        panel = pd.concat(dfs, ignore_index=True)
        panel = panel.sort_values(["date", "code"])
        panel = panel[(panel["date"] >= pd.to_datetime(self.cfg.start_date))]
        if self.cfg.end_date:
            panel = panel[(panel["date"] <= pd.to_datetime(self.cfg.end_date))]

        panel = panel.set_index(["date", "code"]).sort_index()
        return panel

    def _load_one_fund_category(self, category: str, codes: List[str]) -> pd.DataFrame:
        d = os.path.join(self.fund_q_dir, category)
        if not os.path.exists(d):
            return pd.DataFrame()

        dfs = []
        for code in codes:
            code_clean = code.replace(".", "_")
            fp = os.path.join(d, f"{code_clean}.csv")
            if not os.path.exists(fp):
                continue
            df = pd.read_csv(fp)
            if df.empty:
                continue

            # 核心字段：statDate + code + 若干指标
            if "statDate" not in df.columns:
                continue
            df["statDate"] = pd.to_datetime(df["statDate"], errors="coerce")
            df["code"] = code

            dfs.append(df)

        if not dfs:
            return pd.DataFrame()
        out = pd.concat(dfs, ignore_index=True)
        return out

    def load_fundamental_daily_aligned(self, price_panel: pd.DataFrame) -> pd.DataFrame:
        # 取 codes
        codes = sorted({c for _, c in price_panel.index})
        # 读若干类别表：profit/operation/growth/balance/cash_flow/dupont
        cats = ["profit", "operation", "growth", "balance", "cash_flow", "dupont"]
        cat_dfs = []
        for cat in cats:
            df = self._load_one_fund_category(cat, codes)
            if df.empty:
                continue
            cat_dfs.append(df)

        if not cat_dfs:
            # 没有基本面也允许跑，只做量价因子挖掘
            return pd.DataFrame(index=price_panel.index)

        fund = pd.concat(cat_dfs, ignore_index=True)

        # 只保留你关心的字段
        cols = ["statDate", "code"] + [c for c in self.cfg.fundamental_features if c in fund.columns]
        fund = fund[cols].copy()

        for c in self.cfg.fundamental_features:
            if c in fund.columns:
                fund[c] = pd.to_numeric(fund[c], errors="coerce")

        fund = fund.dropna(subset=["statDate"])
        fund = fund.sort_values(["code", "statDate"])

        # 对齐到日频：对每只股票，按 price 的 date index forward-fill
        idx = price_panel.reset_index()[["date", "code"]]
        idx = idx.sort_values(["code", "date"])

        # merge_asof 需要按 code 分组做，这里用 groupby 逐股（稳定但偏慢；先保证正确性）
        aligned_list = []
        for code, g in idx.groupby("code", sort=False):
            f = fund[fund["code"] == code].sort_values("statDate")
            if f.empty:
                gg = g.copy()
                for c in self.cfg.fundamental_features:
                    gg[c] = np.nan
                aligned_list.append(gg)
                continue

            gg = g.copy().sort_values("date")
            # asof: date 对齐到 <= date 的最近 statDate
            tmp = pd.merge_asof(
                gg,
                f.rename(columns={"statDate": "date"}).sort_values("date"),
                on="date",
                direction="backward",
                allow_exact_matches=True,
            )
            aligned_list.append(tmp)

        aligned = pd.concat(aligned_list, ignore_index=True)
        aligned = aligned.set_index(["date", "code"]).sort_index()

        # 只返回基本面列
        keep = [c for c in self.cfg.fundamental_features if c in aligned.columns]
        return aligned[keep]

    def make_forward_return_label(self, price_panel: pd.DataFrame) -> pd.Series:
        # 用 close 做未来N日收益
        close = price_panel["close"].copy()
        # panel index: (date, code)
        # shift(-N) 是未来N日的close
        fwd = close.groupby(level=1).shift(-self.cfg.forward_ret_days)
        ret = (fwd / close) - 1.0
        ret.name = f"fwd_ret_{self.cfg.forward_ret_days}d"
        return ret

    def build_panel(self) -> pd.DataFrame:
        price = self.load_price_panel()
        fund = self.load_fundamental_daily_aligned(price)
        y = self.make_forward_return_label(price)

        panel = price.join(fund, how="left")
        panel = panel.join(y, how="left")

        # 清理：去掉close/label缺失
        panel = panel.dropna(subset=["close", y.name])

        return panel
