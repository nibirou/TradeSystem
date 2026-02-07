# module_gnn_factorengine.py
import os
from pathlib import Path
import pandas as pd
import numpy as np

from moudle2_factorengine import FactorEngine, winsorize, standardize


# ==========================================================
#  1) AkShare 基本面加载（你之前那套：可选）
# ==========================================================
class FundamentalLoaderAkShare:
    """
    读 AkShare 三接口落盘（你后续可继续完善字段映射）
    约定目录（示例）：
      {base_dir}/fundamental_akshare/
        financial_abstract_sina/{pool}/*.csv
        financial_indicator_em/{pool}/*.csv
        financial_indicator_sina/{pool}/*.csv
    输出：日频对齐 [date, code, G_xxx...]
    """
    def __init__(self, base_dir: str, pool: str):
        self.base_dir = base_dir
        self.pool = pool
        self.root = Path(base_dir) / "fundamental_akshare"

        self.dir_abs = self.root / "financial_abstract_sina" / pool
        self.dir_em = self.root / "financial_indicator_em" / pool
        self.dir_sina = self.root / "financial_indicator_sina" / pool

    @staticmethod
    def _code_from_filename(fp: Path) -> str:
        name = fp.stem
        name = name.replace("_", ".")
        if name.startswith(("sh.", "sz.")):
            return name
        return name

    def _load_indicator_em(self) -> pd.DataFrame:
        if not self.dir_em.exists():
            return pd.DataFrame()

        dfs = []
        for fp in self.dir_em.glob("*.csv"):
            df = pd.read_csv(fp)
            if df.empty or "REPORT_DATE" not in df.columns:
                continue
            code = self._code_from_filename(fp)
            df["code"] = code
            df["report_date"] = pd.to_datetime(df["REPORT_DATE"], errors="coerce")

            pick = {
                "ROEJQ": "G_em_roe",
                "XSJLL": "G_em_npm",
                "XSMLL": "G_em_gpm",
                "TOTALOPERATEREVETZ": "G_em_yoy_rev",
                "PARENTNETPROFITTZ": "G_em_yoy_ni",
                "ZCFZL": "G_em_debt_asset",
                "LD": "G_em_current_ratio",
                "SD": "G_em_quick_ratio",
                "JYXJLYYSR": "G_em_cfo_or",
            }

            out = df[["code", "report_date"]].copy()
            for k, v in pick.items():
                if k in df.columns:
                    out[v] = pd.to_numeric(df[k], errors="coerce")
            dfs.append(out)

        if not dfs:
            return pd.DataFrame()

        res = pd.concat(dfs, ignore_index=True)
        res = res.sort_values(["code", "report_date"]).groupby(["code", "report_date"], as_index=False).last()
        return res

    def _load_indicator_sina(self) -> pd.DataFrame:
        if not self.dir_sina.exists():
            return pd.DataFrame()

        dfs = []
        for fp in self.dir_sina.glob("*.csv"):
            df = pd.read_csv(fp)
            if df.empty or "日期" not in df.columns:
                continue
            code = self._code_from_filename(fp)
            df["code"] = code
            df["report_date"] = pd.to_datetime(df["日期"], errors="coerce")

            pick = {
                "净资产收益率(%)": "G_sina_roe",
                "销售净利率(%)": "G_sina_npm",
                "销售毛利率(%)": "G_sina_gpm",
                "资产负债率(%)": "G_sina_debt_asset",
                "流动比率": "G_sina_current_ratio",
                "速动比率": "G_sina_quick_ratio",
                "经营现金净流量与净利润的比率(%)": "G_sina_cfo_np",
                "经营现金净流量对销售收入比率(%)": "G_sina_cfo_or",
            }

            out = df[["code", "report_date"]].copy()
            for k, v in pick.items():
                if k in df.columns:
                    out[v] = pd.to_numeric(df[k], errors="coerce")
            dfs.append(out)

        if not dfs:
            return pd.DataFrame()

        res = pd.concat(dfs, ignore_index=True)
        res = res.sort_values(["code", "report_date"]).groupby(["code", "report_date"], as_index=False).last()
        return res

    @staticmethod
    def align_to_trade_dates(fund_q: pd.DataFrame, trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
        if fund_q.empty:
            return pd.DataFrame()

        factor_cols = [c for c in fund_q.columns if c not in ("code", "report_date")]
        pieces = []
        for code, g in fund_q.groupby("code"):
            g = g.sort_values("report_date").set_index("report_date")[factor_cols]
            daily = g.reindex(trade_dates, method="ffill")
            daily["date"] = trade_dates
            daily["code"] = code
            pieces.append(daily.reset_index(drop=True))
        return pd.concat(pieces, ignore_index=True)

    def load_daily_fundamental(self, trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
        em = self._load_indicator_em()
        sina = self._load_indicator_sina()
        if em.empty and sina.empty:
            return pd.DataFrame()

        fund_q = pd.concat([em, sina], ignore_index=True)
        fund_q = fund_q.sort_values(["code", "report_date"]).groupby(["code", "report_date"], as_index=False).last()

        return self.align_to_trade_dates(fund_q, trade_dates)


# ==========================================================
#  2) Baostock 季频基本面加载（你最新这套：必须支持）
# ==========================================================
class FundamentalLoaderBaoStockQ:
    """
    读你落盘的 baostock 季频基本面目录：
      {base_dir}/baostock_fundamental_q/{pool}/{category}/sh_600000.csv
    categories:
      profit/operation/growth/balance/cash_flow/dupont/perf_express/forecast
    输出：日频对齐 [date, code, BQ_xxx...]
    """
    def __init__(self, base_dir: str, pool: str):
        self.base_dir = base_dir
        self.pool = pool
        self.root = Path(base_dir) / "baostock_fundamental_q" / pool

        self.yq_categories = ["profit", "operation", "growth", "balance", "cash_flow", "dupont"]
        self.date_categories = ["perf_express", "forecast"]

    @staticmethod
    def _code_from_filename(fp: Path) -> str:
        # 文件名 sh_600000.csv -> sh.600000
        return fp.stem.replace("_", ".")

    @staticmethod
    def _coerce_numeric_cols(df: pd.DataFrame, exclude: set) -> pd.DataFrame:
        for c in df.columns:
            if c in exclude:
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _load_one_code_yq(self, code: str) -> pd.DataFrame:
        """
        汇总一只股票在 yq_categories 下的所有表，按 statDate 聚合成一张季频宽表：
          index = statDate
          columns = 多表字段（加前缀避免冲突）
        """
        pieces = []
        for cat in self.yq_categories:
            fp = self.root / cat / f"{code.replace('.', '_')}.csv"
            if not fp.exists():
                continue
            df = pd.read_csv(fp)
            if df.empty or "statDate" not in df.columns:
                continue

            # statDate 作为季频锚点（财报期末）
            df["statDate"] = pd.to_datetime(df["statDate"], errors="coerce")
            df = df.dropna(subset=["statDate"])
            if df.empty:
                continue

            exclude = {"code", "pubDate", "statDate"}
            df = self._coerce_numeric_cols(df, exclude=exclude)

            # 加前缀避免不同表字段同名
            rename = {}
            for c in df.columns:
                if c in exclude:
                    continue
                rename[c] = f"BQ_{cat}_{c}"
            df = df.rename(columns=rename)

            keep_cols = ["statDate"] + list(rename.values())
            df = df[keep_cols].sort_values("statDate").drop_duplicates(subset=["statDate"], keep="last")
            pieces.append(df.set_index("statDate"))

        if not pieces:
            return pd.DataFrame()

        q = pd.concat(pieces, axis=1).sort_index()
        q["code"] = code
        q = q.reset_index().rename(columns={"statDate": "report_date"})
        return q

    def _load_one_code_date_tables(self, code: str) -> pd.DataFrame:
        """
        perf_express/forecast：没有严格季报每季都有，且字段变化大。
        我们把它们也对齐到 report_date（优先用 statDate 类字段）
        - perf_express: performanceExpStatDate（统计日期）
        - forecast: profitForcastExpStatDate（统计日期）
        """
        pieces = []
        for cat in self.date_categories:
            fp = self.root / cat / f"{code.replace('.', '_')}.csv"
            if not fp.exists():
                continue
            df = pd.read_csv(fp)
            if df.empty:
                continue

            if cat == "perf_express":
                key = "performanceExpStatDate" if "performanceExpStatDate" in df.columns else None
            else:
                key = "profitForcastExpStatDate" if "profitForcastExpStatDate" in df.columns else None

            if key is None:
                continue

            df["report_date"] = pd.to_datetime(df[key], errors="coerce")
            df = df.dropna(subset=["report_date"])
            if df.empty:
                continue

            exclude = {"code", "report_date"}
            df = self._coerce_numeric_cols(df, exclude=exclude)

            rename = {}
            for c in df.columns:
                if c in exclude or c in (key,):
                    continue
                rename[c] = f"BQ_{cat}_{c}"
            df = df.rename(columns=rename)

            keep_cols = ["report_date"] + list(rename.values())
            df = df[keep_cols].sort_values("report_date").drop_duplicates(subset=["report_date"], keep="last")
            df["code"] = code
            pieces.append(df)

        if not pieces:
            return pd.DataFrame()

        out = pd.concat(pieces, ignore_index=True)
        out = out.sort_values(["code", "report_date"]).groupby(["code", "report_date"], as_index=False).last()
        return out

    @staticmethod
    def align_to_trade_dates(fund_q: pd.DataFrame, trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        fund_q: [code, report_date, ...]
        -> 日频 [date, code, ...] forward-fill
        """
        if fund_q.empty:
            return pd.DataFrame()

        trade_dates = pd.DatetimeIndex(pd.to_datetime(trade_dates)).sort_values()
        factor_cols = [c for c in fund_q.columns if c not in ("code", "report_date")]

        pieces = []
        for code, g in fund_q.groupby("code"):
            g = g.sort_values("report_date").set_index("report_date")[factor_cols]
            daily = g.reindex(trade_dates, method="ffill")
            daily["date"] = trade_dates
            daily["code"] = code
            pieces.append(daily.reset_index(drop=True))

        return pd.concat(pieces, ignore_index=True)

    def load_daily_fundamental(self, trade_dates: pd.DatetimeIndex, code_universe: list[str]) -> pd.DataFrame:
        """
        只加载你当前股票宇宙 code_universe（避免把全池全文件都扫爆）
        """
        qs = []
        for code in code_universe:
            q1 = self._load_one_code_yq(code)
            q2 = self._load_one_code_date_tables(code)

            if q1.empty and q2.empty:
                continue
            if q1.empty:
                q = q2
            elif q2.empty:
                q = q1
            else:
                q = pd.merge(q1, q2, on=["code", "report_date"], how="outer")

            # 同一 report_date 合并
            q = q.sort_values(["code", "report_date"]).groupby(["code", "report_date"], as_index=False).last()
            qs.append(q)

        if not qs:
            return pd.DataFrame()

        fund_q = pd.concat(qs, ignore_index=True)
        fund_q = fund_q.sort_values(["code", "report_date"]).groupby(["code", "report_date"], as_index=False).last()

        fund_daily = self.align_to_trade_dates(fund_q, trade_dates)
        return fund_daily


# ==========================================================
#  3) GNNFactorEngine：融合 技术因子 + AkShare + Baostock季频
# ==========================================================
class GNNFactorEngine(FactorEngine):
    def __init__(self, daily_df, minute5_df, trade_dates, base_dir: str, pool: str):
        super().__init__(daily_df, minute5_df, trade_dates)
        self.base_dir = base_dir
        self.pool = pool
        self.ak_loader = FundamentalLoaderAkShare(base_dir=base_dir, pool=pool)
        self.bs_q_loader = FundamentalLoaderBaoStockQ(base_dir=base_dir, pool=pool)

    def _post_process_new_cols(self, factor_df: pd.DataFrame, new_cols: list[str]) -> pd.DataFrame:
        """
        对新增基本面列：按 date 横截面 winsorize + zscore（保持你因子模块一致风格）
        """
        for col in new_cols:
            if col not in factor_df.columns:
                continue
            # 全空列跳过
            if factor_df[col].notna().sum() == 0:
                continue
            factor_df[col] = factor_df.groupby("date")[col].transform(winsorize)
            factor_df[col] = factor_df.groupby("date")[col].transform(standardize)
        return factor_df

    def compute_all_factors_with_fundamental(self, use_akshare: bool = True, use_baostock_q: bool = True):
        # 1) 原技术/量价因子（L1..R1）
        factor_df = super().compute_all_factors()

        # 代码宇宙（只加载出现过的 code，避免扫全目录）
        code_universe = sorted(factor_df["code"].astype(str).unique().tolist())

        # 2) AkShare 基本面（可选）
        if use_akshare:
            ak_daily = self.ak_loader.load_daily_fundamental(self.trade_dates)
            if not ak_daily.empty:
                new_cols = [c for c in ak_daily.columns if c not in ("date", "code")]
                factor_df = factor_df.merge(ak_daily, on=["date", "code"], how="left")
                factor_df = self._post_process_new_cols(factor_df, new_cols)

        # 3) Baostock 季频基本面（你最新的落盘：强烈建议开）
        if use_baostock_q:
            bs_daily = self.bs_q_loader.load_daily_fundamental(self.trade_dates, code_universe=code_universe)
            if not bs_daily.empty:
                new_cols = [c for c in bs_daily.columns if c not in ("date", "code")]
                factor_df = factor_df.merge(bs_daily, on=["date", "code"], how="left")
                factor_df = self._post_process_new_cols(factor_df, new_cols)

        return factor_df
