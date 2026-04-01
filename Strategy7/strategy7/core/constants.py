"""Global constants and defaults."""

from __future__ import annotations

from typing import Dict, List

EPS = 1e-12
TRADING_DAYS_PER_YEAR = 252

SUPPORTED_FREQS: List[str] = ["5min", "15min", "30min", "60min", "120min", "D", "W", "M"]
INTRADAY_FREQS = {"5min", "15min", "30min", "60min", "120min"}

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

