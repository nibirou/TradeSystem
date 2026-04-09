"""Time/frequency utilities shared across Strategy7 modules."""

from __future__ import annotations

import math
from typing import Tuple

import pandas as pd

from .constants import INTRADAY_FREQS, TRADING_DAYS_PER_YEAR


_INTRADAY_BARS_PER_DAY = {
    "5min": 48.0,
    "15min": 16.0,
    "30min": 8.0,
    "60min": 4.0,
    "120min": 2.0,
}


def base_periods_per_year(factor_freq: str) -> float:
    """Base signal count per year under the given frequency (stride=1)."""
    f = str(factor_freq).strip()
    if f == "D":
        return float(TRADING_DAYS_PER_YEAR)
    if f == "W":
        return 52.0
    if f == "M":
        return 12.0
    if f in _INTRADAY_BARS_PER_DAY:
        return float(TRADING_DAYS_PER_YEAR) * float(_INTRADAY_BARS_PER_DAY[f])
    return float(TRADING_DAYS_PER_YEAR)


def infer_periods_per_year(factor_freq: str, stride: int = 1) -> float:
    """Effective annualized periods after rebalancing stride adjustment."""
    s = max(int(stride), 1)
    return max(base_periods_per_year(factor_freq) / float(s), 1e-9)


def horizon_to_calendar_days(factor_freq: str, horizon: int, buffer_days: int = 10) -> int:
    """Convert horizon bars under the target frequency into conservative calendar days."""
    h = max(int(horizon), 1)
    f = str(factor_freq).strip()
    if f in INTRADAY_FREQS:
        # Convert bars -> trading days first.
        td = int(math.ceil(h / max(float(_INTRADAY_BARS_PER_DAY.get(f, 1.0)), 1e-9)))
    elif f == "D":
        td = h
    elif f == "W":
        td = h * 5
    elif f == "M":
        td = h * 21
    else:
        td = h

    # Trading days -> calendar days with safety buffer for weekends/holidays.
    cal_days = int(math.ceil(td * 7.0 / 5.0))
    return max(cal_days + int(buffer_days), int(buffer_days))


def shift_trade_date(trade_dates: pd.DatetimeIndex, anchor: pd.Timestamp, n: int) -> pd.Timestamp:
    td = pd.DatetimeIndex(pd.to_datetime(trade_dates)).sort_values()
    if len(td) == 0:
        raise ValueError("trade_dates is empty.")
    anchor = pd.to_datetime(anchor)
    if anchor <= td[0]:
        idx = 0
    elif anchor >= td[-1]:
        idx = len(td) - 1
    else:
        idx = int(td.searchsorted(anchor, side="left"))
        if idx >= len(td):
            idx = len(td) - 1
    j = max(0, min(len(td) - 1, idx + int(n)))
    return pd.Timestamp(td[j]).normalize()


def compute_load_window(
    train_start: pd.Timestamp,
    test_end: pd.Timestamp,
    lookback_days: int,
    horizon: int,
    factor_freq: str = "D",
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    load_start = pd.Timestamp(train_start) - pd.Timedelta(days=int(lookback_days))
    fwd_days = horizon_to_calendar_days(factor_freq=factor_freq, horizon=horizon, buffer_days=10)
    load_end = pd.Timestamp(test_end) + pd.Timedelta(days=int(fwd_days))
    return load_start.normalize(), load_end.normalize()
