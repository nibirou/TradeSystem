"""Global constants and defaults."""

from __future__ import annotations

from typing import Dict, List

EPS = 1e-12
TRADING_DAYS_PER_YEAR = 252

SUPPORTED_FREQS: List[str] = ["5min", "15min", "30min", "60min", "120min", "D", "W", "M"]
INTRADAY_FREQS = {"5min", "15min", "30min", "60min", "120min"}
FREQ_ORDER: List[str] = ["5min", "15min", "30min", "60min", "120min", "D", "W", "M"]

# Multi-frequency bridge settings:
# finer-frequency views can be aggregated onto a coarser target-frequency view.
# Users can extend this column list to add new cross-frequency bridge factors.
MULTIFREQ_BRIDGE_BASE_COLS: List[str] = [
    # price / volume base
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    # generic intraday features
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "vol_chg_1",
    "ma_gap_6",
    "ma_gap_12",
    "rv_12",
    "range_norm",
    "amount_ratio_12",
    "crowding_proxy_raw",
    # daily/micro context
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "realized_vol_20",
    "amihud_20",
    "ret_vol_corr_20",
    "close_to_vwap_day",
    "morning_momentum_30m",
    "last30_momentum",
]
MULTIFREQ_BRIDGE_AGGS: List[str] = ["mean", "std", "last"]

EXECUTION_SCHEMES: Dict[str, Dict[str, str]] = {
    "open5_open5": {
        "buy_col": "px_open5",
        "sell_col": "px_open5",
        "description": "Signal day close -> next day first 5m open buy, hold n days, first 5m open sell.",
    },
    "vwap30_vwap30": {
        "buy_col": "px_vwap30",
        "sell_col": "px_vwap30",
        "description": "Signal day close -> next day first 30m VWAP buy/sell with holding horizon.",
    },
    "open5_twap_last30": {
        "buy_col": "px_open5",
        "sell_col": "px_twap_last30",
        "description": "Signal day close -> next day first 5m open buy, exit day last 30m TWAP sell.",
    },
    "daily_close_daily_close": {
        "buy_col": "px_daily_close",
        "sell_col": "px_daily_close",
        "description": "Signal day close -> next day daily close buy/sell with holding horizon.",
    },
}
