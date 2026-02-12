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
    # ============================
    # Baostock 季频字段白名单（可随时调整）
    # ============================
    BAOSTOCK_FIELD_CONFIG = {
        "profit": {
            "roeAvg": "BQ_profit_roe",
            "npMargin": "BQ_profit_npm",
            "gpMargin": "BQ_profit_gpm",
            "epsTTM": "BQ_profit_eps_ttm",
            # --- 可选扩展（需要可比性处理/对数化时再启用）---
            # "netProfit": "BQ_profit_netprofit",
            # "MBRevenue": "BQ_profit_mbrevenue",
        },
        "operation": {
            "AssetTurnRatio": "BQ_op_asset_turn",
            "NRTurnDays": "BQ_op_ar_turn_days",
            "INVTurnDays": "BQ_op_inv_turn_days",
            # 可选
            # "NRTurnRatio": "BQ_op_ar_turn",
            # "INVTurnRatio": "BQ_op_inv_turn",
            # "CATurnRatio": "BQ_op_ca_turn",
        },
        "growth": {
            "YOYEquity": "BQ_grow_yoy_equity",
            "YOYAsset": "BQ_grow_yoy_asset",
            "YOYNI": "BQ_grow_yoy_ni",
            "YOYPNI": "BQ_grow_yoy_pni",
            # 可选
            # "YOYEPSBasic": "BQ_grow_yoy_eps_basic",
        },
        "balance": {
            "liabilityToAsset": "BQ_bal_debt_asset",
            "assetToEquity": "BQ_bal_equity_mult",
            "currentRatio": "BQ_bal_current",
            "quickRatio": "BQ_bal_quick",
            # 可选
            # "cashRatio": "BQ_bal_cash_ratio",
            # "YOYLiability": "BQ_bal_yoy_liab",
        },
        "cash_flow": {
            "CFOToNP": "BQ_cf_cfo_np",
            "CFOToOR": "BQ_cf_cfo_or",
            "ebitToInterest": "BQ_cf_int_cover",
            # 可选
            # "CFOToGr": "BQ_cf_cfo_gr",
            # "CAToAsset": "BQ_cf_ca_asset",
            # "NCAToAsset": "BQ_cf_nca_asset",
            # "tangibleAssetToAsset": "BQ_cf_tangible_asset",
        },
        "dupont": {
            "dupontROE": "BQ_dup_roe",
            "dupontAssetTurn": "BQ_dup_asset_turn",
            "dupontAssetStoEquity": "BQ_dup_equity_mult",
            "dupontNitogr": "BQ_dup_net_margin",
            # 可选
            # "dupontTaxBurden": "BQ_dup_tax_burden",
            # "dupontIntburden": "BQ_dup_int_burden",
            # "dupontEbittogr": "BQ_dup_ebit_margin",
            # "dupontPnitoni": "BQ_dup_pni_to_ni",
        },
        # ============================
        # B 类：可选（缺失多，默认不开）
        # ============================
        "perf_express": {
            "performanceExpressROEWa": "BQ_pe_roe_wa",
            "performanceExpressEPSChgPct": "BQ_pe_eps_chg",
            "performanceExpressGRYOY": "BQ_pe_rev_yoy",
            "performanceExpressOPYOY": "BQ_pe_op_yoy",
            # 可选规模项
            # "performanceExpressTotalAsset": "BQ_pe_total_asset",
            # "performanceExpressNetAsset": "BQ_pe_net_asset",
        },
        "forecast": {
            "profitForcastChgPctUp": "BQ_fc_np_yoy_up",
            "profitForcastChgPctDwn": "BQ_fc_np_yoy_down",
            # 文本字段默认不入（要 NLP/类别处理）
            # "profitForcastType": "BQ_fc_type",
            # "profitForcastAbstract": "BQ_fc_abstract",
        }
    }

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
        pieces = []
        for cat in self.yq_categories:
            if cat not in self.BAOSTOCK_FIELD_CONFIG:
                continue

            fp = self.root / cat / f"{code.replace('.', '_')}.csv"
            if not fp.exists():
                continue

            df = pd.read_csv(fp)
            if df.empty or "statDate" not in df.columns:
                continue

            df["report_date"] = pd.to_datetime(df["statDate"], errors="coerce")
            df = df.dropna(subset=["report_date"])
            if df.empty:
                continue

            pick = self.BAOSTOCK_FIELD_CONFIG[cat]
            cols = ["report_date"]
            out = df[cols].copy()

            for raw_col, new_col in pick.items():
                if raw_col in df.columns:
                    out[new_col] = pd.to_numeric(df[raw_col], errors="coerce")

            # 同一 report_date 取最后
            out["code"] = code
            out = out.sort_values("report_date").drop_duplicates(subset=["code", "report_date"], keep="last")
            pieces.append(out)

        if not pieces:
            return pd.DataFrame()

        q = pd.concat(pieces, ignore_index=True)
        q = q.sort_values(["code", "report_date"]).groupby(["code", "report_date"], as_index=False).last()
        return q


    def _load_one_code_date_tables(self, code: str) -> pd.DataFrame:
        pieces = []

        for cat in self.date_categories:
            if cat not in self.BAOSTOCK_FIELD_CONFIG:
                continue

            fp = self.root / cat / f"{code.replace('.', '_')}.csv"
            if not fp.exists():
                continue

            df = pd.read_csv(fp)
            if df.empty:
                continue

            # 选“统计日期”对齐到 report_date
            if cat == "perf_express":
                stat_key = "performanceExpStatDate"
                if stat_key not in df.columns:
                    continue
                df["report_date"] = pd.to_datetime(df[stat_key], errors="coerce")

            elif cat == "forecast":
                stat_key = "profitForcastExpStatDate"
                if stat_key not in df.columns:
                    continue
                df["report_date"] = pd.to_datetime(df[stat_key], errors="coerce")

            df = df.dropna(subset=["report_date"])
            if df.empty:
                continue

            pick = self.BAOSTOCK_FIELD_CONFIG[cat]
            out = df[["report_date"]].copy()

            for raw_col, new_col in pick.items():
                if raw_col in df.columns:
                    # forecast 的 type/abstract 先不处理（若你打开了它们，这里会变成 object，需要你自己后续编码）
                    if df[raw_col].dtype == object:
                        continue
                    out[new_col] = pd.to_numeric(df[raw_col], errors="coerce")

            # 额外：forecast 上下限合成一个“中心值”（可选但很实用）
            if cat == "forecast":
                up = "BQ_fc_np_yoy_up"
                dn = "BQ_fc_np_yoy_down"
                if up in out.columns and dn in out.columns:
                    out["BQ_fc_np_yoy_mid"] = 0.5 * (out[up] + out[dn])

            out["code"] = code
            out = out.sort_values("report_date").drop_duplicates(subset=["code", "report_date"], keep="last")
            pieces.append(out)

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
        print(factor_df)

        # 代码宇宙（只加载出现过的 code，避免扫全目录）
        code_universe = sorted(factor_df["code"].astype(str).unique().tolist())

        # 2) AkShare 基本面（可选）
        if use_akshare:
            ak_daily = self.ak_loader.load_daily_fundamental(self.trade_dates)
            if not ak_daily.empty:
                new_cols = [c for c in ak_daily.columns if c not in ("date", "code")]
                factor_df = factor_df.merge(ak_daily, on=["date", "code"], how="left")
                factor_df = self._post_process_new_cols(factor_df, new_cols)

        # 3) Baostock 季频基本面（最新的落盘：强烈建议开）
        if use_baostock_q:
            bs_daily = self.bs_q_loader.load_daily_fundamental(self.trade_dates, code_universe=code_universe)
            if not bs_daily.empty:
                new_cols = [c for c in bs_daily.columns if c not in ("date", "code")]
                factor_df = factor_df.merge(bs_daily, on=["date", "code"], how="left")
                factor_df = self._post_process_new_cols(factor_df, new_cols)

        return factor_df
