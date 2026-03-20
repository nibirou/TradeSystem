# 模块 3：数据源接口 + 基本面加载统一 quant_preprocess/sources_fundamental.py
# 现有的 FundamentalLoaderAkShare、FundamentalLoaderBaoStockQ 变成“统一 DataSource”，方便随时插新源（例如：文本因子、你算法挖掘出的因子 parquet 等）。

# quant_preprocess/sources_fundamental.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Optional

import pandas as pd
import numpy as np


class DataSource(Protocol):
    """
    统一数据源接口：输出必须是日频对齐的 [date, code, ...]
    """
    def load_daily(self, trade_dates: pd.DatetimeIndex, code_universe: list[str]) -> pd.DataFrame: ...


# -----------------------------
# AkShare 落盘基本面（你旧版）
# -----------------------------
@dataclass
class AkShareFundamentalSource:
    base_dir: str
    pool: str

    def __post_init__(self):
        self.root = Path(self.base_dir) / "fundamental_akshare"
        self.dir_abs = self.root / "financial_abstract_sina" / self.pool
        self.dir_em = self.root / "financial_indicator_em" / self.pool
        self.dir_sina = self.root / "financial_indicator_sina" / self.pool

    @staticmethod
    def _code_from_filename(fp: Path) -> str:
        name = fp.stem.replace("_", ".")
        return name

    def _load_indicator_em(self) -> pd.DataFrame:
        if not self.dir_em.exists():
            return pd.DataFrame()
        dfs = []
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
        for fp in self.dir_em.glob("*.csv"):
            df = pd.read_csv(fp)
            if df.empty or "REPORT_DATE" not in df.columns:
                continue
            code = self._code_from_filename(fp)
            out = pd.DataFrame({"code": code, "report_date": pd.to_datetime(df["REPORT_DATE"], errors="coerce")})
            for k, v in pick.items():
                if k in df.columns:
                    out[v] = pd.to_numeric(df[k], errors="coerce")
            out = out.dropna(subset=["report_date"])
            if not out.empty:
                dfs.append(out)
        if not dfs:
            return pd.DataFrame()
        res = pd.concat(dfs, ignore_index=True)
        return res.sort_values(["code", "report_date"]).groupby(["code", "report_date"], as_index=False).last()

    def _load_indicator_sina(self) -> pd.DataFrame:
        if not self.dir_sina.exists():
            return pd.DataFrame()
        dfs = []
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
        for fp in self.dir_sina.glob("*.csv"):
            df = pd.read_csv(fp)
            if df.empty or "日期" not in df.columns:
                continue
            code = self._code_from_filename(fp)
            out = pd.DataFrame({"code": code, "report_date": pd.to_datetime(df["日期"], errors="coerce")})
            for k, v in pick.items():
                if k in df.columns:
                    out[v] = pd.to_numeric(df[k], errors="coerce")
            out = out.dropna(subset=["report_date"])
            if not out.empty:
                dfs.append(out)
        if not dfs:
            return pd.DataFrame()
        res = pd.concat(dfs, ignore_index=True)
        return res.sort_values(["code", "report_date"]).groupby(["code", "report_date"], as_index=False).last()

    @staticmethod
    def _align_q_to_trade_dates(fund_q: pd.DataFrame, trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
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

    def load_daily(self, trade_dates: pd.DatetimeIndex, code_universe: list[str]) -> pd.DataFrame:
        em = self._load_indicator_em()
        sina = self._load_indicator_sina()
        if em.empty and sina.empty:
            return pd.DataFrame()
        fund_q = pd.concat([em, sina], ignore_index=True)
        fund_q = fund_q.sort_values(["code", "report_date"]).groupby(["code", "report_date"], as_index=False).last()
        # 只保留宇宙
        fund_q = fund_q[fund_q["code"].isin(code_universe)]
        return self._align_q_to_trade_dates(fund_q, trade_dates)


# --------------------------------
# BaoStock 季频基本面（你新版）
# --------------------------------
@dataclass
class BaoStockQFundamentalSource:
    base_dir: str
    pool: str

    BAOSTOCK_FIELD_CONFIG = {
        "profit": {
            "roeAvg": "BQ_profit_roe",
            "npMargin": "BQ_profit_npm",
            "gpMargin": "BQ_profit_gpm",
            "epsTTM": "BQ_profit_eps_ttm",
        },
        "operation": {
            "AssetTurnRatio": "BQ_op_asset_turn",
            "NRTurnDays": "BQ_op_ar_turn_days",
            "INVTurnDays": "BQ_op_inv_turn_days",
        },
        "growth": {
            "YOYEquity": "BQ_grow_yoy_equity",
            "YOYAsset": "BQ_grow_yoy_asset",
            "YOYNI": "BQ_grow_yoy_ni",
            "YOYPNI": "BQ_grow_yoy_pni",
        },
        "balance": {
            "liabilityToAsset": "BQ_bal_debt_asset",
            "assetToEquity": "BQ_bal_equity_mult",
            "currentRatio": "BQ_bal_current",
            "quickRatio": "BQ_bal_quick",
        },
        "cash_flow": {
            "CFOToNP": "BQ_cf_cfo_np",
            "CFOToOR": "BQ_cf_cfo_or",
            "ebitToInterest": "BQ_cf_int_cover",
        },
        "dupont": {
            "dupontROE": "BQ_dup_roe",
            "dupontAssetTurn": "BQ_dup_asset_turn",
            "dupontAssetStoEquity": "BQ_dup_equity_mult",
            "dupontNitogr": "BQ_dup_net_margin",
        },
        "perf_express": {
            "performanceExpressROEWa": "BQ_pe_roe_wa",
            "performanceExpressEPSChgPct": "BQ_pe_eps_chg",
            "performanceExpressGRYOY": "BQ_pe_rev_yoy",
            "performanceExpressOPYOY": "BQ_pe_op_yoy",
        },
        "forecast": {
            "profitForcastChgPctUp": "BQ_fc_np_yoy_up",
            "profitForcastChgPctDwn": "BQ_fc_np_yoy_down",
        }
    }

    def __post_init__(self):
        self.root = Path(self.base_dir) / "baostock_fundamental_q" / self.pool
        self.yq_categories = ["profit", "operation", "growth", "balance", "cash_flow", "dupont"]
        self.date_categories = ["perf_express", "forecast"]

    @staticmethod
    def _fp_for(code: str) -> str:
        return f"{code.replace('.', '_')}.csv"

    @staticmethod
    def _align_q_to_trade_dates(fund_q: pd.DataFrame, trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
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

    def _load_one_code(self, code: str) -> pd.DataFrame:
        pieces = []

        # yq 类：statDate
        for cat in self.yq_categories:
            fp = self.root / cat / self._fp_for(code)
            if not fp.exists():
                continue
            df = pd.read_csv(fp)
            if df.empty or "statDate" not in df.columns:
                continue
            out = pd.DataFrame({"report_date": pd.to_datetime(df["statDate"], errors="coerce")})
            out = out.dropna(subset=["report_date"])
            pick = self.BAOSTOCK_FIELD_CONFIG.get(cat, {})
            for raw, new in pick.items():
                if raw in df.columns:
                    out[new] = pd.to_numeric(df[raw], errors="coerce")
            out["code"] = code
            out = out.sort_values("report_date").drop_duplicates(["code", "report_date"], keep="last")
            if not out.empty:
                pieces.append(out)

        # date 表：perf_express / forecast
        for cat in self.date_categories:
            fp = self.root / cat / self._fp_for(code)
            if not fp.exists():
                continue
            df = pd.read_csv(fp)
            if df.empty:
                continue

            if cat == "perf_express":
                k = "performanceExpStatDate"
                if k not in df.columns:
                    continue
                report_date = pd.to_datetime(df[k], errors="coerce")
            else:
                k = "profitForcastExpStatDate"
                if k not in df.columns:
                    continue
                report_date = pd.to_datetime(df[k], errors="coerce")

            out = pd.DataFrame({"report_date": report_date}).dropna(subset=["report_date"])
            pick = self.BAOSTOCK_FIELD_CONFIG.get(cat, {})
            for raw, new in pick.items():
                if raw in df.columns and df[raw].dtype != object:
                    out[new] = pd.to_numeric(df[raw], errors="coerce")

            if cat == "forecast":
                up, dn = "BQ_fc_np_yoy_up", "BQ_fc_np_yoy_down"
                if up in out.columns and dn in out.columns:
                    out["BQ_fc_np_yoy_mid"] = 0.5 * (out[up] + out[dn])

            out["code"] = code
            out = out.sort_values("report_date").drop_duplicates(["code", "report_date"], keep="last")
            if not out.empty:
                pieces.append(out)

        if not pieces:
            return pd.DataFrame()

        q = pd.concat(pieces, ignore_index=True)
        return q.sort_values(["code", "report_date"]).groupby(["code", "report_date"], as_index=False).last()

    def load_daily(self, trade_dates: pd.DatetimeIndex, code_universe: list[str]) -> pd.DataFrame:
        qs = []
        for code in code_universe:
            q = self._load_one_code(code)
            if not q.empty:
                qs.append(q)
        if not qs:
            return pd.DataFrame()
        fund_q = pd.concat(qs, ignore_index=True)
        fund_q = fund_q.sort_values(["code", "report_date"]).groupby(["code", "report_date"], as_index=False).last()
        return self._align_q_to_trade_dates(fund_q, trade_dates)
