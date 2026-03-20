# quant_preprocess/factors.py
# FactorEngine.compute_all_factors() 是“写死列表”。这里我改成：
# FactorRegistry：随时 register("S2", func, requires=["minute5"])
# 统一后处理：winsorize + zscore + neutralize（可选）

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
    daily: pd.DataFrame
    minute5: Optional[pd.DataFrame]

    date_col: str = "date"
    datetime_col: str = "datetime"
    code_col: str = "code"


class FactorEngine:
    def __init__(self, daily: pd.DataFrame, minute5: Optional[pd.DataFrame]):
        self.ctx = FactorContext(daily=daily.copy(), minute5=None if minute5 is None else minute5.copy())
        self.registry = FactorRegistry()

        # minute5 补 date（如果缺失）
        if self.ctx.minute5 is not None:
            m = self.ctx.minute5
            if "date" not in m.columns and "datetime" in m.columns:
                m["date"] = pd.to_datetime(m["datetime"]).dt.normalize()
            else:
                m["date"] = pd.to_datetime(m["date"])
            self.ctx.minute5 = m

        self._register_builtin()

    def _register_builtin(self):
        # -------- 你现有的因子：示例注册 --------

        # 月频量价因子（M）

        # 周频量价因子（W）

        # 日频量价因子（D）可能用日频历史行情数据计算、也可能用5min历史行情数据计算，也可能都用）
        def L1(ctx: FactorContext) -> pd.DataFrame:
            df = ctx.daily.copy()
            df["low_20"] = df.groupby("code")["close"].rolling(20).min().reset_index(0, drop=True)
            df["high_20"] = df.groupby("code")["close"].rolling(20).max().reset_index(0, drop=True)
            df["L1"] = (df["close"] - df["low_20"]) / (df["high_20"] - df["low_20"] + 1e-9)
            # print(df[["date", "code", "low_20", "high_20", "L1"]])
            return df[["date", "code", "L1"]]

        def L2(ctx: FactorContext) -> pd.DataFrame:
            df = ctx.daily.copy()
            df["MA5"] = df.groupby("code")["close"].rolling(5).mean().reset_index(0, drop=True)
            df["MA20"] = df.groupby("code")["close"].rolling(20).mean().reset_index(0, drop=True)
            df["L2"] = df["MA5"] / (df["MA20"] + 1e-9) - 1
            return df[["date", "code", "L2"]]

        def S2(ctx: FactorContext) -> pd.DataFrame:
            if ctx.minute5 is None:
                raise ValueError("S2 requires minute5")
            m = ctx.minute5.copy()
            m["pv"] = m["close"] * m["volume"]
            vwap = (m.groupby(["code", "date"]).agg({"pv": "sum", "volume": "sum"}).reset_index())
            vwap["vwap_5min"] = vwap["pv"] / (vwap["volume"] + 1e-9)
            vwap = vwap[["code", "date", "vwap_5min"]]
            d = ctx.daily.merge(vwap, on=["code", "date"], how="left")
            d["S2"] = (d["close"] - d["vwap_5min"]) / (d["vwap_5min"] + 1e-9)
            return d[["date", "code", "S2"]]

        def R1(ctx: FactorContext) -> pd.DataFrame:
            df = ctx.daily.copy()
            df["ret"] = df.groupby("code")["close"].pct_change()
            df["std5"] = df.groupby("code")["ret"].rolling(5).std().reset_index(0, drop=True)
            df["std20"] = df.groupby("code")["ret"].rolling(20).std().reset_index(0, drop=True)
            df["R1"] = df["std5"] / (df["std20"] + 1e-9)
            return df[["date", "code", "R1"]]
        
        # 1min频率量价因子

        # 5min频率量价因子

        # 15min频率量价因子

        # 30min频率量价因子

        # 60min频率量价因子

        # 120min频率量价因子

        self.registry.register("L1", L1, requires=("daily",))
        self.registry.register("L2", L2, requires=("daily",))
        self.registry.register("S2", S2, requires=("daily", "minute5"))
        self.registry.register("R1", R1, requires=("daily",))

        # 你其他 S1/S3/S4/F1/F2…照同样方式 register 即可

    def compute(
        self,
        selected: Optional[Sequence[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        post_winsor_zscore: bool = True,
        post_neutralize: bool = False,
    ) -> pd.DataFrame:
        names = list(selected) if selected is not None else self.registry.names()

        # 时间裁剪（给“因子选取开始结束时间”）
        d = self.ctx.daily
        if start is not None:
            d = d[d["date"] >= pd.to_datetime(start)]
        if end is not None:
            d = d[d["date"] <= pd.to_datetime(end)]

        m = self.ctx.minute5
        if m is not None:
            if start is not None:
                m = m[m["date"] >= pd.to_datetime(start)]
            if end is not None:
                m = m[m["date"] <= pd.to_datetime(end)]

        ctx = FactorContext(daily=d, minute5=m)

        # 逐因子计算 + merge
        panel = None
        factor_cols = []
        for nm in names:
            fdef = self.registry.get(nm)
            out = fdef.func(ctx)

            # 统一 keys
            if not set(["date", "code"]).issubset(out.columns):
                raise ValueError(f"Factor {nm} must return columns ['date','code',...], got {out.columns}")

            # 找出新增列（除 date/code 外）
            new_cols = [c for c in out.columns if c not in ("date", "code")]
            if len(new_cols) != 1:
                # 允许一个函数产多个因子，但要可控
                pass

            if panel is None:
                panel = out.copy()
            else:
                panel = panel.merge(out, on=["date", "code"], how="left")

            factor_cols.extend([c for c in new_cols if c not in factor_cols])

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

        return panel.sort_values(["date", "code"]).reset_index(drop=True)
