"""Label engineering with no-future-leakage alignment."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from ..core.constants import EPS, EXECUTION_SCHEMES, INTRADAY_FREQS


def attach_execution_prices(df: pd.DataFrame, price_table: pd.DataFrame, buy_col: str, sell_col: str) -> pd.DataFrame:
    buy = price_table[["code", "date", buy_col]].rename(columns={"date": "entry_date", buy_col: "entry_price"})
    sell = price_table[["code", "date", sell_col]].rename(columns={"date": "exit_date", sell_col: "exit_price"})
    out = df.merge(buy, on=["code", "entry_date"], how="left")
    out = out.merge(sell, on=["code", "exit_date"], how="left")
    return out


def _add_daily_label(
    panel: pd.DataFrame,
    horizon: int,
    execution_scheme: str,
    price_table: pd.DataFrame,
) -> pd.DataFrame:
    out = panel.copy()
    g = out.groupby("code")
    out["entry_date"] = g["date"].shift(-1)
    out["exit_date"] = g["date"].shift(-(horizon + 1))

    conf = EXECUTION_SCHEMES[execution_scheme]
    out = attach_execution_prices(out, price_table, buy_col=conf["buy_col"], sell_col=conf["sell_col"])
    out["future_ret_n"] = out["exit_price"] / (out["entry_price"] + EPS) - 1.0
    out["target_return"] = out["future_ret_n"]
    out["target_up"] = (out["future_ret_n"] > 0).astype(int)

    # realized volatility label from next horizon daily close path
    out["fwd_vol_label"] = (
        g["ret_1d"].shift(-1).rolling(horizon, min_periods=max(2, min(5, horizon))).std().reset_index(level=0, drop=True)
    )
    out["target_volatility"] = out["fwd_vol_label"]
    out["target_date"] = out["exit_date"]
    out["signal_ts"] = out["date"]
    out["entry_ts"] = out["entry_date"]
    out["exit_ts"] = out["exit_date"]
    out["time_freq"] = "D"
    return out


def _add_generic_bar_label(panel: pd.DataFrame, horizon: int, time_col: str, freq: str) -> pd.DataFrame:
    out = panel.copy()
    out = out.sort_values(["code", time_col]).copy()
    g = out.groupby("code")
    out["entry_ts"] = g[time_col].shift(-1)
    out["exit_ts"] = g[time_col].shift(-(horizon + 1))

    # use next bars close for generic intraday/period task labels
    out["entry_price"] = g["close"].shift(-1)
    out["exit_price"] = g["close"].shift(-(horizon + 1))
    out["future_ret_n"] = out["exit_price"] / (out["entry_price"] + EPS) - 1.0
    out["target_return"] = out["future_ret_n"]
    out["target_up"] = (out["future_ret_n"] > 0).astype(int)

    out["ret_1_bar"] = g["close"].pct_change(1)
    out["target_volatility"] = (
        g["ret_1_bar"].shift(-1).rolling(horizon, min_periods=max(2, min(5, horizon))).std().reset_index(level=0, drop=True)
    )
    out["signal_ts"] = out[time_col]
    out["target_date"] = out["exit_ts"]
    out["time_freq"] = freq
    return out


def add_labels(
    panel: pd.DataFrame,
    horizon: int,
    execution_scheme: str,
    price_table_daily: pd.DataFrame,
    factor_freq: str,
) -> pd.DataFrame:
    """Build labels for selected factor frequency."""
    if factor_freq == "D":
        return _add_daily_label(panel, horizon=horizon, execution_scheme=execution_scheme, price_table=price_table_daily)
    if factor_freq in {"W", "M"}:
        return _add_generic_bar_label(panel, horizon=horizon, time_col="date", freq=factor_freq)
    if factor_freq in INTRADAY_FREQS:
        return _add_generic_bar_label(panel, horizon=horizon, time_col="datetime", freq=factor_freq)
    raise ValueError(f"unsupported factor_freq for labels: {factor_freq}")


def pick_target_column(label_task: str) -> str:
    if label_task == "direction":
        return "target_up"
    if label_task == "return":
        return "target_return"
    if label_task == "volatility":
        return "target_volatility"
    if label_task == "multi_task":
        return "target_up"
    raise ValueError(f"unknown label_task: {label_task}")


def split_train_test(
    panel: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    factor_freq: str,
    label_task: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = panel.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    target_col = pick_target_column(label_task)
    base_required: List[str] = ["entry_ts", "exit_ts", "future_ret_n", target_col]
    out = out.dropna(subset=[c for c in base_required if c in out.columns]).copy()

    time_col = "date" if factor_freq in {"D", "W", "M"} else "datetime"
    if time_col not in out.columns:
        raise ValueError(f"missing time column {time_col} for factor_freq={factor_freq}")

    t = pd.to_datetime(out[time_col], errors="coerce")
    train_mask = (t >= train_start) & (t <= train_end)
    test_mask = (t >= test_start) & (t <= test_end)

    if factor_freq == "D":
        if "target_date" in out.columns:
            td = pd.to_datetime(out["target_date"], errors="coerce")
            train_mask = train_mask & (td <= train_end)
            test_mask = test_mask & (td <= test_end)
    return out[train_mask].copy(), out[test_mask].copy()

