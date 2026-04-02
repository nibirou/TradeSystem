"""Frequency transformation utilities."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from ..core.constants import INTRADAY_FREQS
from ..core.utils import infer_board_type, infer_industry_bucket


def resample_intraday(minute_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample minute bars into target intraday frequency."""
    if minute_df.empty:
        return minute_df.copy()
    if freq not in INTRADAY_FREQS:
        raise ValueError(f"unsupported intraday freq: {freq}")
    if freq == "5min":
        out = minute_df.copy()
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
        out["date"] = out["datetime"].dt.normalize()
        return out.sort_values(["code", "datetime"]).reset_index(drop=True)

    rule_map = {
        "15min": "15min",
        "30min": "30min",
        "60min": "60min",
        "120min": "120min",
    }
    rule = rule_map[freq]

    m = minute_df.copy()
    m["datetime"] = pd.to_datetime(m["datetime"], errors="coerce")
    m = m.dropna(subset=["code", "datetime"]).sort_values(["code", "datetime"])
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")

    pieces = []
    for code, g in m.groupby("code"):
        gg = g.set_index("datetime")
        rs = gg.resample(rule, label="right", closed="right").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "amount": "sum",
            }
        )
        rs = rs.dropna(subset=["open", "high", "low", "close"], how="any")
        if rs.empty:
            continue
        rs = rs.reset_index()
        rs["code"] = str(code)
        rs["date"] = rs["datetime"].dt.normalize()
        pieces.append(rs)
    if not pieces:
        return pd.DataFrame(columns=["datetime", "date", "code", "open", "high", "low", "close", "volume", "amount"])
    return pd.concat(pieces, ignore_index=True).sort_values(["code", "datetime"]).reset_index(drop=True)


def resample_daily_to_period(daily_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample daily bars to weekly/monthly bars."""
    if daily_df.empty:
        return daily_df.copy()
    if freq not in {"W", "M"}:
        raise ValueError(f"unsupported period freq: {freq}")

    d = daily_df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
    for c in ["open", "high", "low", "close", "volume", "amount", "turn"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    rule = "W-FRI" if freq == "W" else "ME"
    pieces = []
    for code, g in d.groupby("code"):
        gg = g.sort_values("date").set_index("date")
        rs = gg.resample(rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "amount": "sum",
                "turn": "sum",
            }
        )
        rs = rs.dropna(subset=["open", "high", "low", "close"])
        if rs.empty:
            continue
        rs = rs.reset_index()
        rs["code"] = str(code)
        pieces.append(rs)
    if not pieces:
        return pd.DataFrame(columns=["date", "code", "open", "high", "low", "close", "volume", "amount", "turn"])
    out = pd.concat(pieces, ignore_index=True).sort_values(["code", "date"]).reset_index(drop=True)
    out["board_type"] = out["code"].astype(str).map(infer_board_type)
    out["industry_bucket"] = out["code"].astype(str).map(infer_industry_bucket)
    return out


def build_frequency_views(daily_df: pd.DataFrame, minute5_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    views: Dict[str, pd.DataFrame] = {"D": daily_df}
    if minute5_df is not None and not minute5_df.empty:
        for f in ["5min", "15min", "30min", "60min", "120min"]:
            views[f] = resample_intraday(minute5_df, f)
    else:
        for f in ["5min", "15min", "30min", "60min", "120min"]:
            views[f] = pd.DataFrame()
    views["W"] = resample_daily_to_period(daily_df, "W")
    views["M"] = resample_daily_to_period(daily_df, "M")
    return views


def add_generic_micro_structure_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Compute generic rolling micro-structure features for intraday/period bars."""
    if df.empty:
        return df.copy()
    out = df.copy().sort_values(["code", time_col]).reset_index(drop=True)
    g = out.groupby("code")

    out["ret_1"] = g["close"].pct_change(1)
    out["ret_3"] = g["close"].pct_change(3)
    out["ret_6"] = g["close"].pct_change(6)
    out["ret_12"] = g["close"].pct_change(12)
    out["vol_chg_1"] = g["volume"].pct_change(1)

    out["ma_6"] = g["close"].transform(lambda s: s.rolling(6, min_periods=6).mean())
    out["ma_12"] = g["close"].transform(lambda s: s.rolling(12, min_periods=12).mean())
    out["ma_gap_6"] = out["close"] / (out["ma_6"] + 1e-12) - 1.0
    out["ma_gap_12"] = out["close"] / (out["ma_12"] + 1e-12) - 1.0

    out["rv_12"] = g["ret_1"].transform(lambda s: s.rolling(12, min_periods=12).std())
    out["range_norm"] = (out["high"] - out["low"]) / (out["close"].abs() + 1e-12)
    out["amount_ma12"] = g["amount"].transform(lambda s: s.rolling(12, min_periods=12).mean())
    out["amount_ratio_12"] = out["amount"] / (out["amount_ma12"] + 1e-12)

    out["board_type"] = out["code"].astype(str).map(infer_board_type)
    out["industry_bucket"] = out["code"].astype(str).map(infer_industry_bucket)
    out["barra_size_proxy"] = np.log(out["amount_ma12"].clip(lower=0.0) + 1.0)
    out["barra_momentum_proxy"] = out["ret_12"]
    out["barra_volatility_proxy"] = out["rv_12"]
    out["barra_liquidity_proxy"] = out["amount_ratio_12"]
    out["barra_beta_proxy"] = out["vol_chg_1"]
    out["crowding_proxy_raw"] = 0.5 * out["amount_ratio_12"].abs() + 0.5 * out["vol_chg_1"].abs()
    return out
