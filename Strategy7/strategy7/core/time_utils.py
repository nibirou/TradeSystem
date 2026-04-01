"""Time window utilities."""

from __future__ import annotations

from typing import Tuple

import pandas as pd


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
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    load_start = pd.Timestamp(train_start) - pd.Timedelta(days=int(lookback_days))
    load_end = pd.Timestamp(test_end) + pd.Timedelta(days=int(horizon) + 5)
    return load_start.normalize(), load_end.normalize()

