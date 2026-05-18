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
        keep_cols = [c for c in ["datetime", "date", "code", "open", "high", "low", "close", "volume", "amount"] if c in minute_df.columns]
        out = minute_df[keep_cols].copy()
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

    keep_cols = [c for c in ["datetime", "date", "code", "open", "high", "low", "close", "volume", "amount"] if c in minute_df.columns]
    m = minute_df[keep_cols].copy()
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


def _merge_bridge_frame(
    target_df: pd.DataFrame,
    bridge_df: pd.DataFrame,
    *,
    target_freq: str,
) -> pd.DataFrame:
    """Merge a pre-aggregated bridge frame onto a target-frequency panel."""
    if target_df.empty or bridge_df.empty:
        return target_df
    target_keys = ["code", "datetime"] if str(target_freq) in INTRADAY_FREQS else ["code", "date"]
    if not all(k in target_df.columns for k in target_keys):
        return target_df
    if not all(k in bridge_df.columns for k in target_keys):
        return target_df

    tdf = target_df.copy()
    bdf = bridge_df.copy()
    for k in target_keys:
        if k == "datetime":
            tdf[k] = pd.to_datetime(tdf[k], errors="coerce")
            bdf[k] = pd.to_datetime(bdf[k], errors="coerce")
        elif k == "date":
            tdf[k] = pd.to_datetime(tdf[k], errors="coerce").dt.normalize()
            bdf[k] = pd.to_datetime(bdf[k], errors="coerce").dt.normalize()
        elif k == "code":
            tdf[k] = tdf[k].astype(str).str.strip()
            bdf[k] = bdf[k].astype(str).str.strip()
    add_cols = [c for c in bdf.columns if c not in target_keys and c != "date"]
    if not add_cols:
        return tdf
    return tdf.merge(bdf[target_keys + add_cols], on=target_keys, how="left")


def merge_preaggregated_bridge_features(
    target_df: pd.DataFrame,
    bridge_df: pd.DataFrame,
    *,
    target_freq: str,
) -> pd.DataFrame:
    """Public wrapper for memory-friendly bridge aggregation callers."""
    return _merge_bridge_frame(target_df, bridge_df, target_freq=target_freq)


def add_multifreq_bridge_features(
    views: Dict[str, pd.DataFrame],
    bridge_base_cols: Sequence[str] | None = None,
    bridge_aggs: Sequence[str] | None = None,
    target_freqs: Sequence[str] | None = None,
    source_freqs: Sequence[str] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Attach finer-frequency aggregated features onto coarser target-frequency views."""
    out: Dict[str, pd.DataFrame] = dict(views)
    frozen_sources: Dict[str, pd.DataFrame] = dict(views)
    base_cols = list(bridge_base_cols) if bridge_base_cols is not None else list(MULTIFREQ_BRIDGE_BASE_COLS)
    aggs = list(bridge_aggs) if bridge_aggs is not None else list(MULTIFREQ_BRIDGE_AGGS)
    targets = [str(t) for t in target_freqs] if target_freqs is not None else list(FREQ_ORDER)
    source_allow = {str(s) for s in source_freqs} if source_freqs is not None else None

    for target in targets:
        if target not in out:
            continue
        tdf = out[target]
        if tdf is None or tdf.empty:
            continue
        tdf = tdf.copy()

        target_keys = ["code", "datetime"] if target in INTRADAY_FREQS else ["code", "date"]
        if not all(k in tdf.columns for k in target_keys):
            continue

        for source in finer_source_freqs(target, frozen_sources.keys()):
            if source_allow is not None and source not in source_allow:
                continue
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

            tdf = _merge_bridge_frame(tdf, bdf, target_freq=target)
        out[target] = tdf

    return out


def _intraday_source_bars_for_code(code_df: pd.DataFrame, source_freq: str) -> pd.DataFrame:
    """Build one stock's source-frequency bars without materializing the full market view."""
    keep_cols = [c for c in ["datetime", "date", "code", "open", "high", "low", "close", "volume", "amount"] if c in code_df.columns]
    if not keep_cols or "datetime" not in keep_cols:
        return pd.DataFrame(columns=["datetime", "date", "code", "open", "high", "low", "close", "volume", "amount"])

    m = code_df[keep_cols].copy()
    m["datetime"] = pd.to_datetime(m["datetime"], errors="coerce")
    m["date"] = pd.to_datetime(m.get("date", m["datetime"]), errors="coerce").dt.normalize()
    m["code"] = m["code"].astype(str).str.strip()
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=["datetime", "date", "code", "open", "high", "low", "close"]).sort_values("datetime")
    if m.empty:
        return pd.DataFrame(columns=["datetime", "date", "code", "open", "high", "low", "close", "volume", "amount"])

    if source_freq == "5min":
        return m[["datetime", "date", "code", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)

    rule = {
        "15min": "15min",
        "30min": "30min",
        "60min": "60min",
        "120min": "120min",
    }.get(str(source_freq), "")
    if not rule:
        return pd.DataFrame(columns=["datetime", "date", "code", "open", "high", "low", "close", "volume", "amount"])

    code_value = str(m["code"].iloc[0])
    rs = (
        m.set_index("datetime")
        .resample(rule, label="right", closed="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "amount": "sum",
            }
        )
    )
    rs = rs.dropna(subset=["open", "high", "low", "close"], how="any")
    if rs.empty:
        return pd.DataFrame(columns=["datetime", "date", "code", "open", "high", "low", "close", "volume", "amount"])
    rs = rs.reset_index()
    rs["code"] = code_value
    rs["date"] = rs["datetime"].dt.normalize()
    return rs[["datetime", "date", "code", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)


def aggregate_intraday_bridge_to_target(
    minute5_df: pd.DataFrame,
    *,
    source_freq: str,
    target_freq: str,
    bridge_base_cols: Sequence[str] | None = None,
    bridge_aggs: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate intraday bridge features one stock at a time.

    The legacy path first built a full-market source view (for example 5min
    with rolling features) and then aggregated it to D/W/M. On all-market
    windows this can require tens of GB. This helper keeps only one stock's
    source-frequency bars in memory while producing the same `hf_*` columns.
    """
    source = str(source_freq)
    target = str(target_freq)
    if source not in INTRADAY_FREQS or target in INTRADAY_FREQS:
        return pd.DataFrame()
    if minute5_df is None or minute5_df.empty or "code" not in minute5_df.columns:
        return pd.DataFrame()

    base_cols = list(bridge_base_cols) if bridge_base_cols is not None else list(MULTIFREQ_BRIDGE_BASE_COLS)
    aggs = list(bridge_aggs) if bridge_aggs is not None else list(MULTIFREQ_BRIDGE_AGGS)
    pieces: List[pd.DataFrame] = []
    for _, g in minute5_df.groupby("code", sort=False, observed=True):
        src = _intraday_source_bars_for_code(g, source)
        if src.empty:
            continue
        src = add_generic_micro_structure_features(src, time_col="datetime", add_static_context=False)
        source_keys = ["code", "datetime", "date"]
        bridge_cols = _bridge_candidate_cols(src, key_cols=source_keys, preferred=base_cols)
        if not bridge_cols:
            continue
        bdf = _aggregate_source_to_target(
            source_df=src,
            source_freq=source,
            target_freq=target,
            value_cols=bridge_cols,
            agg_list=aggs,
        )
        if not bdf.empty:
            pieces.append(bdf)
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, ignore_index=True)
    for c in out.columns:
        if c in {"date", "datetime", "code"}:
            continue
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    key_cols = ["code", "datetime"] if target in INTRADAY_FREQS else ["code", "date"]
    return out.sort_values(key_cols).reset_index(drop=True)


def build_frequency_views(
    daily_df: pd.DataFrame,
    minute5_df: pd.DataFrame,
    required_freqs: Sequence[str] | None = None,
) -> Dict[str, pd.DataFrame]:
    if required_freqs is None:
        wanted = {"5min", "15min", "30min", "60min", "120min", "D", "W", "M"}
    else:
        wanted = {str(f) for f in required_freqs if str(f).strip()}

    views: Dict[str, pd.DataFrame] = {}
    if "D" in wanted:
        views["D"] = daily_df

    intraday_list = ["5min", "15min", "30min", "60min", "120min"]
    if any(f in wanted for f in intraday_list):
        if minute5_df is not None and not minute5_df.empty:
            for f in intraday_list:
                if f in wanted:
                    views[f] = resample_intraday(minute5_df, f)
        else:
            for f in intraday_list:
                if f in wanted:
                    views[f] = pd.DataFrame()

    if "W" in wanted:
        views["W"] = resample_daily_to_period(daily_df, "W")
    if "M" in wanted:
        views["M"] = resample_daily_to_period(daily_df, "M")
    return views


def add_generic_micro_structure_features(
    df: pd.DataFrame,
    time_col: str,
    *,
    add_static_context: bool = True,
) -> pd.DataFrame:
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

    out["barra_size_proxy"] = np.log(out["amount_ma12"].clip(lower=0.0) + 1.0)
    out["barra_momentum_proxy"] = out["ret_12"]
    out["barra_volatility_proxy"] = out["rv_12"]
    out["barra_liquidity_proxy"] = out["amount_ratio_12"]
    out["barra_beta_proxy"] = out["vol_chg_1"]
    out["crowding_proxy_raw"] = 0.5 * out["amount_ratio_12"].abs() + 0.5 * out["vol_chg_1"].abs()
    if add_static_context:
        out["board_type"] = out["code"].astype(str).map(infer_board_type)
        out["industry_bucket"] = out["code"].astype(str).map(infer_industry_bucket)
    return out
