"""Frequency transformation utilities."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ..core.constants import (
    FREQ_ORDER,
    INTRADAY_FREQS,
    MULTIFREQ_BRIDGE_AGGS,
    MULTIFREQ_BRIDGE_BASE_COLS,
)
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

    # pandas>=2.2 prefers "ME", while some older versions only support "M".
    if freq == "W":
        rule = "W-FRI"
    else:
        rule = _month_end_rule()
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


def _month_end_rule() -> str:
    try:
        pd.tseries.frequencies.to_offset("ME")
        return "ME"
    except Exception:
        return "M"


def finer_source_freqs(target_freq: str, available_freqs: Sequence[str]) -> List[str]:
    target = str(target_freq)
    if target not in FREQ_ORDER:
        return []
    idx = FREQ_ORDER.index(target)
    available = {str(x) for x in available_freqs}
    return [f for f in FREQ_ORDER[:idx] if f in available]


def _flatten_agg_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        cols = []
        for c0, c1 in out.columns.to_flat_index():
            c0s = str(c0)
            c1s = str(c1)
            if c1s in {"", "None"}:
                cols.append(c0s)
            else:
                cols.append(f"{prefix}_{c1s}_{c0s}")
        out.columns = cols
    return out


def _bridge_candidate_cols(df: pd.DataFrame, key_cols: Sequence[str], preferred: Sequence[str]) -> List[str]:
    key_set = set(key_cols)
    cols_pref = [
        c
        for c in preferred
        if c in df.columns and c not in key_set and pd.api.types.is_numeric_dtype(df[c])
    ]
    if cols_pref:
        return sorted(set(cols_pref))
    cols_num = [
        c
        for c in df.columns
        if c not in key_set and pd.api.types.is_numeric_dtype(df[c])
    ]
    return sorted(cols_num[:24])


def _aggregate_source_to_target(
    source_df: pd.DataFrame,
    source_freq: str,
    target_freq: str,
    *,
    value_cols: Sequence[str],
    agg_list: Sequence[str],
) -> pd.DataFrame:
    if source_df.empty:
        return pd.DataFrame()
    src = source_df.copy()
    src["code"] = src["code"].astype(str).str.strip()
    source_freq = str(source_freq)
    target_freq = str(target_freq)
    aggs = [str(a) for a in agg_list if str(a) in {"mean", "std", "max", "min", "last"}]
    if not aggs:
        aggs = ["mean", "std", "max", "min", "last"]

    prefix = f"hf_{source_freq}_to_{target_freq}"
    val_cols = [c for c in value_cols if c in src.columns]
    if not val_cols:
        return pd.DataFrame()

    if source_freq in INTRADAY_FREQS:
        if "datetime" not in src.columns:
            return pd.DataFrame()
        src["datetime"] = pd.to_datetime(src["datetime"], errors="coerce")
        src["date"] = pd.to_datetime(src.get("date", src["datetime"]), errors="coerce").dt.normalize()
    else:
        if "date" not in src.columns:
            return pd.DataFrame()
        src["date"] = pd.to_datetime(src["date"], errors="coerce").dt.normalize()

    src = src.dropna(subset=["code"])
    if source_freq in INTRADAY_FREQS:
        src = src.dropna(subset=["datetime"])
    else:
        src = src.dropna(subset=["date"])
    if src.empty:
        return pd.DataFrame()

    # Target intraday: only intraday source can bridge to intraday target.
    if target_freq in INTRADAY_FREQS:
        if source_freq not in INTRADAY_FREQS:
            return pd.DataFrame()
        if target_freq == "5min":
            return pd.DataFrame()

        rule = {
            "15min": "15min",
            "30min": "30min",
            "60min": "60min",
            "120min": "120min",
        }.get(target_freq, "")
        if not rule:
            return pd.DataFrame()

        pieces = []
        agg_map = {c: aggs for c in val_cols}
        for code, g in src.groupby("code"):
            gg = g.sort_values("datetime").set_index("datetime")
            rs = gg.resample(rule, label="right", closed="right").agg(agg_map)
            rs = _flatten_agg_columns(rs, prefix=prefix).reset_index()
            if rs.empty:
                continue
            rs["code"] = str(code)
            rs["date"] = rs["datetime"].dt.normalize()
            pieces.append(rs)
        if not pieces:
            return pd.DataFrame()
        out = pd.concat(pieces, ignore_index=True)
        return out.sort_values(["code", "datetime"]).reset_index(drop=True)

    # Target daily/weekly/monthly.
    daily_src = src.copy()
    if source_freq in INTRADAY_FREQS:
        daily_src["date"] = daily_src["datetime"].dt.normalize()
    daily_src = daily_src.dropna(subset=["date", "code"])
    if daily_src.empty:
        return pd.DataFrame()
    agg_map = {c: aggs for c in val_cols}

    if target_freq == "D":
        out = (
            daily_src.groupby(["date", "code"], as_index=False)
            .agg(agg_map)
        )
        out = _flatten_agg_columns(out, prefix=prefix)
        return out.sort_values(["code", "date"]).reset_index(drop=True)

    if target_freq in {"W", "M"}:
        rule = "W-FRI" if target_freq == "W" else _month_end_rule()
        pieces = []
        for code, g in daily_src.groupby("code"):
            gg = g.sort_values("date").set_index("date")
            rs = gg.resample(rule).agg(agg_map)
            rs = _flatten_agg_columns(rs, prefix=prefix).reset_index()
            if rs.empty:
                continue
            rs["code"] = str(code)
            pieces.append(rs)
        if not pieces:
            return pd.DataFrame()
        out = pd.concat(pieces, ignore_index=True)
        return out.sort_values(["code", "date"]).reset_index(drop=True)

    return pd.DataFrame()


def add_multifreq_bridge_features(
    views: Dict[str, pd.DataFrame],
    bridge_base_cols: Sequence[str] | None = None,
    bridge_aggs: Sequence[str] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Attach finer-frequency aggregated features onto coarser target-frequency views."""
    out: Dict[str, pd.DataFrame] = {k: v.copy() for k, v in views.items()}
    frozen_sources: Dict[str, pd.DataFrame] = {k: v.copy() for k, v in views.items()}
    base_cols = list(bridge_base_cols) if bridge_base_cols is not None else list(MULTIFREQ_BRIDGE_BASE_COLS)
    aggs = list(bridge_aggs) if bridge_aggs is not None else list(MULTIFREQ_BRIDGE_AGGS)

    for target in FREQ_ORDER:
        if target not in out:
            continue
        tdf = out[target]
        if tdf is None or tdf.empty:
            continue

        target_keys = ["code", "datetime"] if target in INTRADAY_FREQS else ["code", "date"]
        if not all(k in tdf.columns for k in target_keys):
            continue

        for source in finer_source_freqs(target, frozen_sources.keys()):
            sdf = frozen_sources.get(source)
            if sdf is None or sdf.empty:
                continue
            source_keys = ["code", "datetime", "date"] if source in INTRADAY_FREQS else ["code", "date"]
            bridge_cols = _bridge_candidate_cols(sdf, key_cols=source_keys, preferred=base_cols)
            if not bridge_cols:
                continue

            bdf = _aggregate_source_to_target(
                source_df=sdf,
                source_freq=source,
                target_freq=target,
                value_cols=bridge_cols,
                agg_list=aggs,
            )
            if bdf.empty:
                continue

            for k in target_keys:
                if k in tdf.columns and k in bdf.columns:
                    if k == "datetime":
                        tdf[k] = pd.to_datetime(tdf[k], errors="coerce")
                        bdf[k] = pd.to_datetime(bdf[k], errors="coerce")
                    if k == "date":
                        tdf[k] = pd.to_datetime(tdf[k], errors="coerce").dt.normalize()
                        bdf[k] = pd.to_datetime(bdf[k], errors="coerce").dt.normalize()
                    if k == "code":
                        tdf[k] = tdf[k].astype(str).str.strip()
                        bdf[k] = bdf[k].astype(str).str.strip()

            add_cols = [c for c in bdf.columns if c not in target_keys and c != "date"]
            if not add_cols:
                continue
            tdf = tdf.merge(bdf[target_keys + add_cols], on=target_keys, how="left")
        out[target] = tdf

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
