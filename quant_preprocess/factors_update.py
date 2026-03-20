from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .preprocess import winsorize, zscore, neutralize_simple


FactorFunc = Callable[["FactorContext"], pd.DataFrame]


@dataclass
class FactorDef:
    name: str
    func: FactorFunc
    requires: Sequence[str] = ()   # e.g. ("daily",) or ("minute5", "daily")


class FactorRegistry:
    def __init__(self):
        self._factors: Dict[str, FactorDef] = {}

    def register(self, name: str, func: FactorFunc, requires: Sequence[str] = ()):
        self._factors[name] = FactorDef(name=name, func=func, requires=requires)

    def names(self) -> List[str]:
        return sorted(self._factors.keys())

    def get(self, name: str) -> FactorDef:
        if name not in self._factors:
            raise KeyError(f"Factor not found: {name}")
        return self._factors[name]


@dataclass
class FactorContext:
    # 你已有：日频
    daily: pd.DataFrame

    # 盘中各频率（你已预先整合好就直接传）
    minute5: Optional[pd.DataFrame] = None
    minute15: Optional[pd.DataFrame] = None
    minute30: Optional[pd.DataFrame] = None
    minute60: Optional[pd.DataFrame] = None
    minute120: Optional[pd.DataFrame] = None

    # 周/月（你若已整合好可传；否则可以只用 daily 近似/推导）
    weekly: Optional[pd.DataFrame] = None
    monthly: Optional[pd.DataFrame] = None

    date_col: str = "date"
    datetime_col: str = "datetime"
    code_col: str = "code"


class FactorEngine:
    """
    freq:
      "5min","15min","30min","60min","120min","D","W","M","CUSTOM"
    """
    def __init__(
        self,
        daily: pd.DataFrame,
        freq: str = "D",
        minute5: Optional[pd.DataFrame] = None,
        minute15: Optional[pd.DataFrame] = None,
        minute30: Optional[pd.DataFrame] = None,
        minute60: Optional[pd.DataFrame] = None,
        minute120: Optional[pd.DataFrame] = None,
        weekly: Optional[pd.DataFrame] = None,
        monthly: Optional[pd.DataFrame] = None,
    ):
        self.freq = freq

        self.ctx = FactorContext(
            daily=daily.copy(),
            minute5=None if minute5 is None else minute5.copy(),
            minute15=None if minute15 is None else minute15.copy(),
            minute30=None if minute30 is None else minute30.copy(),
            minute60=None if minute60 is None else minute60.copy(),
            minute120=None if minute120 is None else minute120.copy(),
            weekly=None if weekly is None else weekly.copy(),
            monthly=None if monthly is None else monthly.copy(),
        )

        self.registry = FactorRegistry()

        # ---- 统一规范化 date/datetime（你数据字段：code/datetime/date/time/open/high/low/close/volume/amount...） ----
        self.ctx.daily = self._norm_daily(self.ctx.daily)

        for attr in ["minute5", "minute15", "minute30", "minute60", "minute120"]:
            df = getattr(self.ctx, attr)
            if df is not None:
                setattr(self.ctx, attr, self._norm_intraday(df))

        if self.ctx.weekly is not None:
            self.ctx.weekly = self._norm_period(self.ctx.weekly)
        if self.ctx.monthly is not None:
            self.ctx.monthly = self._norm_period(self.ctx.monthly)

        self._register_builtin_by_freq()

    # -------------------------
    # normalize helpers
    # -------------------------
    def _norm_daily(self, d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        if "date" not in d.columns:
            raise ValueError("daily missing column: date")
        d["date"] = pd.to_datetime(d["date"]).dt.normalize()
        if "code" not in d.columns:
            raise ValueError("daily missing column: code")
        return d.sort_values(["code", "date"]).reset_index(drop=True)

    def _norm_intraday(self, m: pd.DataFrame) -> pd.DataFrame:
        m = m.copy()
        if "datetime" not in m.columns:
            raise ValueError("intraday df missing column: datetime")
        m["datetime"] = pd.to_datetime(m["datetime"])
        # date 若存在就 normalize；不存在就从 datetime 推
        if "date" in m.columns:
            m["date"] = pd.to_datetime(m["date"]).dt.normalize()
        else:
            m["date"] = m["datetime"].dt.normalize()
        if "code" not in m.columns:
            raise ValueError("intraday df missing column: code")
        return m.sort_values(["code", "datetime"]).reset_index(drop=True)

    def _norm_period(self, p: pd.DataFrame) -> pd.DataFrame:
        p = p.copy()
        if "date" not in p.columns:
            raise ValueError("period df missing column: date")
        if "code" not in p.columns:
            raise ValueError("period df missing column: code")
        p["date"] = pd.to_datetime(p["date"]).dt.normalize()
        return p.sort_values(["code", "date"]).reset_index(drop=True)

    def _get_intraday(self, ctx: FactorContext, freq: str) -> pd.DataFrame:
        # 你明确不做1min
        if freq == "1min":
            raise ValueError("1min factors are not supported without true 1min bars.")
        mp = {
            "5min": ctx.minute5,
            "15min": ctx.minute15,
            "30min": ctx.minute30,
            "60min": ctx.minute60,
            "120min": ctx.minute120,
        }
        df = mp.get(freq)
        if df is None:
            raise ValueError(f"{freq} data is None. Please pass {freq} dataframe into FactorEngine.")
        return df

    # -------------------------
    # register factors by freq
    # -------------------------
    def _register_builtin_by_freq(self):
        f = self.freq

        # 月频量价因子（M）
        if f == "M":
            self._register_monthly_factors()
            return

        # 周频量价因子（W）
        if f == "W":
            self._register_weekly_factors()
            return

        # 日频量价因子（D）
        if f == "D":
            self._register_daily_factors()
            return

        # 盘中频率量价因子
        if f in ("5min", "15min", "30min", "60min", "120min"):
            self._register_intraday_factors(freq=f)
            return

        # CUSTOM：留给你自由扩展
        if f == "CUSTOM":
            self._register_custom_factors()
            return

        raise ValueError(f"Unknown freq: {f}")

    # =========================
    # D: 日频量价因子（用 daily 或 minute5 都可；这里优先 minute5 的信息含量）
    # 输出：code, date, factor
    # =========================
    def _register_daily_factors(self):
        # 1) D_VWAPDev：日收盘相对“日内VWAP(用5min amount/volume)”偏离
        def D_VWAPDev(ctx: FactorContext) -> pd.DataFrame:
            if ctx.minute5 is None:
                raise ValueError("D_VWAPDev requires minute5 (for intraday vwap).")
            m = ctx.minute5.copy()
            # 用 amount/volume 更标准
            v = (m.groupby(["code", "date"])
                   .agg({"amount": "sum", "volume": "sum"})
                   .reset_index())
            v["vwap"] = v["amount"] / (v["volume"] + 1e-9)
            d = ctx.daily.merge(v[["code", "date", "vwap"]], on=["code", "date"], how="left")
            d["D_VWAPDev"] = (d["close"] - d["vwap"]) / (d["vwap"] + 1e-9)
            return d[["code", "date", "D_VWAPDev"]]

        # 2) D_RV：日内实现波动率（5min close return 平方和开方）
        def D_RV(ctx: FactorContext) -> pd.DataFrame:
            if ctx.minute5 is None:
                raise ValueError("D_RV requires minute5.")
            m = ctx.minute5.sort_values(["code", "datetime"]).copy()
            m["r"] = m.groupby("code")["close"].pct_change()
            rv = (m.groupby(["code", "date"])["r"]
                    .apply(lambda x: float(np.sqrt(np.nansum((x.values ** 2)))))
                    .reset_index(name="D_RV"))
            return rv[["code", "date", "D_RV"]]

        # 3) D_UpVolRatio：上涨bar成交量占比（量价配合）
        def D_UpVolRatio(ctx: FactorContext) -> pd.DataFrame:
            if ctx.minute5 is None:
                raise ValueError("D_UpVolRatio requires minute5.")
            m = ctx.minute5.sort_values(["code", "datetime"]).copy()
            m["r"] = m.groupby("code")["close"].pct_change()
            m["upv"] = m["volume"] * (m["r"] > 0).astype(float)
            agg = m.groupby(["code", "date"], as_index=False).agg({"upv": "sum", "volume": "sum"})
            agg["D_UpVolRatio"] = agg["upv"] / (agg["volume"] + 1e-9)
            return agg[["code", "date", "D_UpVolRatio"]]

        # 4) D_Amihud：日内非流动性（用 amount 更合理）：sum(|r|)/sum(amount)
        def D_Amihud(ctx: FactorContext) -> pd.DataFrame:
            if ctx.minute5 is None:
                raise ValueError("D_Amihud requires minute5.")
            m = ctx.minute5.sort_values(["code", "datetime"]).copy()
            m["r"] = m.groupby("code")["close"].pct_change().abs()
            agg = (m.groupby(["code", "date"], as_index=False)
                     .agg({"r": "sum", "amount": "sum"}))
            agg["D_Amihud"] = agg["r"] / (agg["amount"] + 1e-9)
            return agg[["code", "date", "D_Amihud"]]

        self.registry.register("D_VWAPDev", D_VWAPDev, requires=("daily", "minute5"))
        self.registry.register("D_RV", D_RV, requires=("minute5",))
        self.registry.register("D_UpVolRatio", D_UpVolRatio, requires=("minute5",))
        self.registry.register("D_Amihud", D_Amihud, requires=("minute5",))

    # =========================
    # W: 周频量价因子
    # 你可以：优先用 ctx.weekly（若你已整合好）；否则用 daily 推导并抽取周末点
    # 输出：code, date, factor
    # =========================
    def _register_weekly_factors(self):
        def _get_w(ctx: FactorContext) -> pd.DataFrame:
            if ctx.weekly is not None:
                return ctx.weekly.copy()
            # fallback：从 daily 推导周频（取周最后一个交易日作为 date）
            d = ctx.daily.sort_values(["code", "date"]).copy()
            d["_week"] = d["date"].dt.to_period("W")
            g = d.groupby(["code", "_week"], as_index=False)
            w = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum"})
            # date 用该周最后一个交易日（用最后一条daily的date）
            last_date = d.groupby(["code", "_week"], as_index=False)["date"].last()
            w = w.merge(last_date, on=["code", "_week"], how="left")
            return w.drop(columns=["_week"])

        def W_TrendOC(ctx: FactorContext) -> pd.DataFrame:
            w = _get_w(ctx)
            w["W_TrendOC"] = (w["close"] / (w["open"] + 1e-9)) - 1.0
            return w[["code", "date", "W_TrendOC"]]

        def W_RangePct(ctx: FactorContext) -> pd.DataFrame:
            w = _get_w(ctx)
            w["W_RangePct"] = (w["high"] - w["low"]) / (w["open"] + 1e-9)
            return w[["code", "date", "W_RangePct"]]

        def W_VolSpike(ctx: FactorContext) -> pd.DataFrame:
            w = _get_w(ctx).sort_values(["code", "date"]).copy()
            ma4 = w.groupby("code")["volume"].rolling(4).mean().reset_index(0, drop=True)
            w["W_VolSpike"] = w["volume"] / (ma4 + 1e-9)
            return w[["code", "date", "W_VolSpike"]]

        self.registry.register("W_TrendOC", W_TrendOC, requires=("daily",))
        self.registry.register("W_RangePct", W_RangePct, requires=("daily",))
        self.registry.register("W_VolSpike", W_VolSpike, requires=("daily",))

    # =========================
    # M: 月频量价因子（同W）
    # 输出：code, date, factor
    # =========================
    def _register_monthly_factors(self):
        def _get_m(ctx: FactorContext) -> pd.DataFrame:
            if ctx.monthly is not None:
                return ctx.monthly.copy()
            d = ctx.daily.sort_values(["code", "date"]).copy()
            d["_mon"] = d["date"].dt.to_period("M")
            g = d.groupby(["code", "_mon"], as_index=False)
            m = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum"})
            last_date = d.groupby(["code", "_mon"], as_index=False)["date"].last()
            m = m.merge(last_date, on=["code", "_mon"], how="left")
            return m.drop(columns=["_mon"])

        def M_TrendOC(ctx: FactorContext) -> pd.DataFrame:
            m = _get_m(ctx)
            m["M_TrendOC"] = (m["close"] / (m["open"] + 1e-9)) - 1.0
            return m[["code", "date", "M_TrendOC"]]

        def M_RangePct(ctx: FactorContext) -> pd.DataFrame:
            m = _get_m(ctx)
            m["M_RangePct"] = (m["high"] - m["low"]) / (m["open"] + 1e-9)
            return m[["code", "date", "M_RangePct"]]

        def M_Amihud(ctx: FactorContext) -> pd.DataFrame:
            m = _get_m(ctx)
            # 月收益绝对值 / 月成交额（粗略版）
            m["ret_abs"] = m.groupby("code")["close"].pct_change().abs()
            m["M_Amihud"] = m["ret_abs"] / (m["amount"] + 1e-9)
            return m[["code", "date", "M_Amihud"]]

        self.registry.register("M_TrendOC", M_TrendOC, requires=("daily",))
        self.registry.register("M_RangePct", M_RangePct, requires=("daily",))
        self.registry.register("M_Amihud", M_Amihud, requires=("daily",))

    # =========================
    # 盘中：5/15/30/60/120min 量价因子（使用你已整合好的对应频率DF）
    # 输出：code, datetime, date, factor
    # =========================
    def _register_intraday_factors(self, freq: str):
        # 1) BarMomK：k根bar动量（k按该freq的bar计数）
        def BarMomK(ctx: FactorContext, k: int, name: str) -> pd.DataFrame:
            b = self._get_intraday(ctx, freq).sort_values(["code", "datetime"]).copy()
            b[name] = b.groupby("code")["close"].pct_change(k)
            return b[["code", "datetime", "date", name]]

        # 2) BarRVK：k根bar实现波动率
        def BarRVK(ctx: FactorContext, k: int, name: str) -> pd.DataFrame:
            b = self._get_intraday(ctx, freq).sort_values(["code", "datetime"]).copy()
            r = b.groupby("code")["close"].pct_change()
            b[name] = (r ** 2).groupby(b["code"]).rolling(k).sum().reset_index(0, drop=True) ** 0.5
            return b[["code", "datetime", "date", name]]

        # 3) BarRangePos：收盘在区间位置
        def BarRangePos(ctx: FactorContext, name: str) -> pd.DataFrame:
            b = self._get_intraday(ctx, freq).copy()
            b[name] = (b["close"] - b["low"]) / (b["high"] - b["low"] + 1e-9)
            return b[["code", "datetime", "date", name]]

        # 4) BarVWAPDev：该bar收盘相对该bar VWAP 偏离（用 amount/volume）
        def BarVWAPDev(ctx: FactorContext, name: str) -> pd.DataFrame:
            b = self._get_intraday(ctx, freq).copy()
            vwap_bar = b["amount"] / (b["volume"] + 1e-9)
            b[name] = (b["close"] - vwap_bar) / (vwap_bar + 1e-9)
            return b[["code", "datetime", "date", name]]

        # 5) BarAmihudK：k根bar的简化非流动性（|r|/amount）
        def BarAmihudK(ctx: FactorContext, k: int, name: str) -> pd.DataFrame:
            b = self._get_intraday(ctx, freq).sort_values(["code", "datetime"]).copy()
            rabs = b.groupby("code")["close"].pct_change().abs()
            illiq = rabs / (b["amount"] + 1e-9)
            b[name] = illiq.groupby(b["code"]).rolling(k).mean().reset_index(0, drop=True)
            return b[["code", "datetime", "date", name]]

        # ---- 这里“分开输出”：只注册当前 self.freq 对应的一组 ----
        # k 的选择：给你一套默认（偏稳健），后续你可以按策略改
        # 5min: k=12≈1小时；15min: k=4≈1小时；30min: k=2≈1小时；60min: k=4≈半天；120min: k=3≈1.5天
        if freq == "5min":
            self.registry.register("I5_Mom12", lambda ctx: BarMomK(ctx, 12, "I5_Mom12"), requires=("minute5",))
            self.registry.register("I5_RV12", lambda ctx: BarRVK(ctx, 12, "I5_RV12"), requires=("minute5",))
            self.registry.register("I5_RangePos", lambda ctx: BarRangePos(ctx, "I5_RangePos"), requires=("minute5",))
            self.registry.register("I5_VWAPDev", lambda ctx: BarVWAPDev(ctx, "I5_VWAPDev"), requires=("minute5",))
            self.registry.register("I5_Amihud12", lambda ctx: BarAmihudK(ctx, 12, "I5_Amihud12"), requires=("minute5",))

        elif freq == "15min":
            self.registry.register("I15_Mom4", lambda ctx: BarMomK(ctx, 4, "I15_Mom4"), requires=("minute15",))
            self.registry.register("I15_RV4", lambda ctx: BarRVK(ctx, 4, "I15_RV4"), requires=("minute15",))
            self.registry.register("I15_RangePos", lambda ctx: BarRangePos(ctx, "I15_RangePos"), requires=("minute15",))
            self.registry.register("I15_VWAPDev", lambda ctx: BarVWAPDev(ctx, "I15_VWAPDev"), requires=("minute15",))
            self.registry.register("I15_Amihud8", lambda ctx: BarAmihudK(ctx, 8, "I15_Amihud8"), requires=("minute15",))

        elif freq == "30min":
            self.registry.register("I30_Mom2", lambda ctx: BarMomK(ctx, 2, "I30_Mom2"), requires=("minute30",))
            self.registry.register("I30_RV4", lambda ctx: BarRVK(ctx, 4, "I30_RV4"), requires=("minute30",))
            self.registry.register("I30_RangePos", lambda ctx: BarRangePos(ctx, "I30_RangePos"), requires=("minute30",))
            self.registry.register("I30_VWAPDev", lambda ctx: BarVWAPDev(ctx, "I30_VWAPDev"), requires=("minute30",))
            self.registry.register("I30_Amihud6", lambda ctx: BarAmihudK(ctx, 6, "I30_Amihud6"), requires=("minute30",))

        elif freq == "60min":
            self.registry.register("I60_Mom2", lambda ctx: BarMomK(ctx, 2, "I60_Mom2"), requires=("minute60",))
            self.registry.register("I60_RV4", lambda ctx: BarRVK(ctx, 4, "I60_RV4"), requires=("minute60",))
            self.registry.register("I60_RangePos", lambda ctx: BarRangePos(ctx, "I60_RangePos"), requires=("minute60",))
            self.registry.register("I60_VWAPDev", lambda ctx: BarVWAPDev(ctx, "I60_VWAPDev"), requires=("minute60",))
            self.registry.register("I60_Amihud4", lambda ctx: BarAmihudK(ctx, 4, "I60_Amihud4"), requires=("minute60",))

        elif freq == "120min":
            self.registry.register("I120_Mom2", lambda ctx: BarMomK(ctx, 2, "I120_Mom2"), requires=("minute120",))
            self.registry.register("I120_RV3", lambda ctx: BarRVK(ctx, 3, "I120_RV3"), requires=("minute120",))
            self.registry.register("I120_RangePos", lambda ctx: BarRangePos(ctx, "I120_RangePos"), requires=("minute120",))
            self.registry.register("I120_VWAPDev", lambda ctx: BarVWAPDev(ctx, "I120_VWAPDev"), requires=("minute120",))
            self.registry.register("I120_Amihud3", lambda ctx: BarAmihudK(ctx, 3, "I120_Amihud3"), requires=("minute120",))

        else:
            raise ValueError(f"Unsupported intraday freq: {freq}")

    # =========================
    # CUSTOM：你自行扩展（比如开盘30min、尾盘30min冲击等）
    # =========================
    def _register_custom_factors(self):
        # 示例：尾盘最后 N 根 5min bar 的收益（CUSTOM_CloseShock）
        def CUSTOM_CloseShock(ctx: FactorContext) -> pd.DataFrame:
            if ctx.minute5 is None:
                raise ValueError("CUSTOM_CloseShock requires minute5.")
            m = ctx.minute5.sort_values(["code", "datetime"]).copy()
            # 取每天最后 6 根 5min（约30分钟）
            tail = m.groupby(["code", "date"]).tail(6).copy()
            # 尾盘收益：最后close / 第一个close - 1
            g = tail.groupby(["code", "date"])
            out = g.agg(first_close=("close", "first"), last_close=("close", "last")).reset_index()
            out["CUSTOM_CloseShock"] = (out["last_close"] / (out["first_close"] + 1e-9)) - 1.0
            return out[["code", "date", "CUSTOM_CloseShock"]]

        self.registry.register("CUSTOM_CloseShock", CUSTOM_CloseShock, requires=("minute5",))

    # -------------------------
    # compute(): 基本不动，只加一个 requires 检查 & 输出字段规范不破坏
    # -------------------------
    def compute(
        self,
        selected: Optional[Sequence[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        post_winsor_zscore: bool = True,
        post_neutralize: bool = False,
    ) -> pd.DataFrame:
        names = list(selected) if selected is not None else self.registry.names()

        # 时间裁剪
        def _clip_date(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if df is None:
                return None
            x = df
            if "date" in x.columns:
                if start is not None:
                    x = x[x["date"] >= pd.to_datetime(start)]
                if end is not None:
                    x = x[x["date"] <= pd.to_datetime(end)]
            return x

        d = _clip_date(self.ctx.daily)
        m5 = _clip_date(self.ctx.minute5)
        m15 = _clip_date(self.ctx.minute15)
        m30 = _clip_date(self.ctx.minute30)
        m60 = _clip_date(self.ctx.minute60)
        m120 = _clip_date(self.ctx.minute120)
        w = _clip_date(self.ctx.weekly)
        mth = _clip_date(self.ctx.monthly)

        ctx = FactorContext(daily=d, minute5=m5, minute15=m15, minute30=m30, minute60=m60, minute120=m120, weekly=w, monthly=mth)

        panel = None
        factor_cols: List[str] = []

        for nm in names:
            fdef = self.registry.get(nm)

            # requires 检查（关键：避免 silent NaN）
            for req in fdef.requires:
                if req == "daily" and ctx.daily is None:
                    raise ValueError(f"{nm} requires daily but daily is None")
                if req == "minute5" and ctx.minute5 is None:
                    raise ValueError(f"{nm} requires minute5 but minute5 is None")
                if req == "minute15" and ctx.minute15 is None:
                    raise ValueError(f"{nm} requires minute15 but minute15 is None")
                if req == "minute30" and ctx.minute30 is None:
                    raise ValueError(f"{nm} requires minute30 but minute30 is None")
                if req == "minute60" and ctx.minute60 is None:
                    raise ValueError(f"{nm} requires minute60 but minute60 is None")
                if req == "minute120" and ctx.minute120 is None:
                    raise ValueError(f"{nm} requires minute120 but minute120 is None")
                if req == "weekly" and ctx.weekly is None:
                    raise ValueError(f"{nm} requires weekly but weekly is None")
                if req == "monthly" and ctx.monthly is None:
                    raise ValueError(f"{nm} requires monthly but monthly is None")

            out = fdef.func(ctx)

            # 统一 keys：必须含 code/date；盘中因子通常还有 datetime（你要求带上必要字段）
            if not set(["code", "date"]).issubset(out.columns):
                raise ValueError(f"Factor {nm} must return columns including ['code','date'], got {list(out.columns)}")

            # 找出新增列
            new_cols = [c for c in out.columns if c not in ("code", "date", "datetime")]
            if len(new_cols) < 1:
                raise ValueError(f"Factor {nm} returned no factor columns.")
            # 允许多列，但建议你保持每函数1列最可控

            if panel is None:
                panel = out.copy()
            else:
                # merge keys：如果双方都有 datetime，就用 (code, datetime)；否则用 (code, date)
                if ("datetime" in panel.columns) and ("datetime" in out.columns):
                    panel = panel.merge(out, on=["code", "datetime", "date"], how="left")
                else:
                    panel = panel.merge(out, on=["code", "date"], how="left")

            for c in new_cols:
                if c not in factor_cols:
                    factor_cols.append(c)

        if panel is None:
            return pd.DataFrame()

        # 横截面后处理（按 date）
        if post_winsor_zscore:
            for c in factor_cols:
                if panel[c].notna().sum() == 0:
                    continue
                panel[c] = panel.groupby("date")[c].transform(winsorize)
                panel[c] = panel.groupby("date")[c].transform(zscore)

        if post_neutralize:
            for c in factor_cols:
                if panel[c].notna().sum() == 0:
                    continue
                panel[c] = panel.groupby("date", group_keys=False).apply(lambda g: neutralize_simple(g, c))

        # 排序：盘中优先 datetime，否则 date
        sort_cols = ["date", "code"]
        if "datetime" in panel.columns:
            sort_cols = ["datetime", "code"]
        return panel.sort_values(sort_cols).reset_index(drop=True)