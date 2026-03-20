import pandas as pd
import numpy as np
from typing import Optional, List

# ========== 5min -> Xmin (15/30/60/120) ==========

def resample_from_5min_keep_all(
    minute5: pd.DataFrame,
    freq: str,                 # "15min","30min","60min","120min"
    code_col: str = "code",
    dt_col: str = "datetime",
) -> pd.DataFrame:
    if minute5 is None or minute5.empty:
        return pd.DataFrame()

    df = minute5.copy()
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values([code_col, dt_col])

    o, h, l, c = "open", "high", "low", "close"
    v, amt = "volume", "amount"

    required = [code_col, dt_col, o, h, l, c, v, amt, "adjustflag", "source_pool"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    out_parts = []
    for code, g in df.groupby(code_col, sort=False):
        g = g.set_index(dt_col)

        rs = g.resample(freq, label="right", closed="right").agg({
            o: "first",
            h: "max",
            l: "min",
            c: "last",
            v: "sum",
            amt: "sum",
            "adjustflag": "last",
            "source_pool": "last",
        })

        # 去掉午休/停牌产生的空桶
        rs = rs.dropna(subset=[c]).reset_index()
        rs[code_col] = code

        # 生成 5min 风格的 date/time 字段
        rs["date"] = rs[dt_col].dt.normalize()
        rs["time"] = rs[dt_col].dt.strftime("%Y%m%d%H%M%S") + "00000"  # 17位尾巴

        # 输出列顺序完全按 5min 原始字段
        rs = rs[[code_col, dt_col, "date", "time", o, h, l, c, v, amt, "adjustflag", "source_pool"]]
        out_parts.append(rs)

    out = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()
    out = out.sort_values([code_col, dt_col]).reset_index(drop=True)
    return out


# ========== Daily -> W/M ==========

def resample_from_daily_keep_all(
    daily: pd.DataFrame,
    freq: str,                 # "W" or "M"
    code_col: str = "code",
    date_col: str = "date",
) -> pd.DataFrame:
    if daily is None or daily.empty:
        return pd.DataFrame()

    df = daily.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([code_col, date_col])

    o, h, l, c = "open", "high", "low", "close"
    pre = "preclose"
    v, amt = "volume", "amount"

    rule = "W-FRI" if freq == "W" else "M"

    desired = [
        date_col, code_col, o, h, l, c, pre, v, amt,
        "adjustflag", "turn", "tradestatus", "pctChg",
        "peTTM", "psTTM", "pcfNcfTTM", "pbMRQ", "isST", "source_pool"
    ]

    out_parts = []
    for code, g in df.groupby(code_col, sort=False):
        g = g.set_index(date_col)

        agg = {
            o: "first",
            h: "max",
            l: "min",
            c: "last",
            pre: "first",
            v: "sum",
            amt: "sum",
            "adjustflag": "last",
            "turn": "sum",
            "tradestatus": "last",
            "peTTM": "last",
            "psTTM": "last",
            "pcfNcfTTM": "last",
            "pbMRQ": "last",
            "isST": "last",
            "source_pool": "last",
        }
        # 只聚合存在的列
        agg = {k: v for k, v in agg.items() if k in g.columns}

        rs = g.resample(rule, label="right", closed="right").agg(agg)
        rs = rs.dropna(subset=[c]).reset_index()
        rs[code_col] = code

        # pctChg 重算（更一致）
        if pre in rs.columns:
            rs["pctChg"] = (rs[c] / rs[pre] - 1.0) * 100.0

        cols = [x for x in desired if x in rs.columns]
        rs = rs[cols]
        out_parts.append(rs)

    out = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()
    out = out.sort_values([code_col, date_col]).reset_index(drop=True)
    return out


# ========== Unified entry ==========

def synthesize_by_freq(
    daily: pd.DataFrame,
    minute5: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    if freq == "D":
        return daily
    if freq == "5min":
        return minute5

    if freq in ["15min", "30min", "60min", "120min"]:
        return resample_from_5min_keep_all(minute5=minute5, freq=freq)

    if freq in ["W", "M"]:
        return resample_from_daily_keep_all(daily=daily, freq=freq)

    raise ValueError(f"Unsupported freq={freq}")