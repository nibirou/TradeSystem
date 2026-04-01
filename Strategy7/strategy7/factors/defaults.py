"""Built-in factor sets for multiple frequencies."""

from __future__ import annotations

from typing import Dict, List

from .base import FactorLibrary


DEFAULT_FACTOR_SET_BY_FREQ: Dict[str, List[str]] = {
    "D": [
        "mom_5",
        "mom_10",
        "mom_20",
        "rev_1",
        "rev_3",
        "ma_gap_5",
        "ma_gap_20",
        "breakout_20",
        "vol_ratio_20",
        "amount_ratio_20",
        "turn_ratio_5",
        "atr_norm_14",
        "realized_vol_20",
        "downside_vol_ratio_20",
        "amihud_20",
        "ret_vol_corr_20",
        "close_to_vwap_day",
        "open_to_close_intraday",
        "morning_momentum_30m",
        "last30_momentum",
        "minute_up_ratio_5m",
        "minute_ret_skew_5m",
        "signed_vol_imbalance_5m",
        "jump_ratio_5m",
    ],
    "5min": ["ret_1", "ret_3", "ret_6", "ma_gap_6", "rv_12", "amount_ratio_12", "range_norm", "vol_chg_1"],
    "15min": ["ret_1", "ret_3", "ret_6", "ma_gap_6", "rv_12", "amount_ratio_12", "range_norm", "vol_chg_1"],
    "30min": ["ret_1", "ret_3", "ret_6", "ma_gap_6", "rv_12", "amount_ratio_12", "range_norm", "vol_chg_1"],
    "60min": ["ret_1", "ret_3", "ret_6", "ma_gap_6", "rv_12", "amount_ratio_12", "range_norm", "vol_chg_1"],
    "120min": ["ret_1", "ret_3", "ret_6", "ma_gap_6", "rv_12", "amount_ratio_12", "range_norm", "vol_chg_1"],
    "W": ["ret_1", "ret_3", "ma_gap_6", "rv_12", "range_norm", "amount_ratio_12"],
    "M": ["ret_1", "ret_3", "ma_gap_6", "rv_12", "range_norm", "amount_ratio_12"],
}


def register_default_daily_factors(library: FactorLibrary) -> None:
    library.register("mom_5", "trend", "5-day momentum", lambda d: d["ret_5d"], freq="D")
    library.register("mom_10", "trend", "10-day momentum", lambda d: d["ret_10d"], freq="D")
    library.register("mom_20", "trend", "20-day momentum", lambda d: d["ret_20d"], freq="D")
    library.register("rev_1", "reversal", "1-day reversal", lambda d: -d["ret_1d"], freq="D")
    library.register("rev_3", "reversal", "3-day reversal", lambda d: -d["ret_3d"], freq="D")

    library.register("ma_gap_5", "trend", "close/ma5 - 1", lambda d: d["ma_gap_5"], freq="D")
    library.register("ma_gap_10", "trend", "close/ma10 - 1", lambda d: d["ma_gap_10"], freq="D")
    library.register("ma_gap_20", "trend", "close/ma20 - 1", lambda d: d["ma_gap_20"], freq="D")
    library.register("ma_cross_5_20", "trend", "ma5/ma20 - 1", lambda d: d["ma_cross_5_20"], freq="D")
    library.register("breakout_20", "trend", "close/rolling high20 - 1", lambda d: d["breakout_20"], freq="D")

    library.register("vol_ratio_5", "liquidity", "volume/vol_ma5", lambda d: d["vol_ratio_5"], freq="D")
    library.register("vol_ratio_20", "liquidity", "volume/vol_ma20", lambda d: d["vol_ratio_20"], freq="D")
    library.register("amount_ratio_20", "liquidity", "amount/amount_ma20", lambda d: d["amount_ratio_20"], freq="D")
    library.register("turn_ratio_5", "liquidity", "turn/turn_ma5", lambda d: d["turn_ratio_5"], freq="D")
    library.register("amihud_20", "liquidity", "negative amihud20", lambda d: -d["amihud_20"], freq="D")
    library.register("ret_vol_corr_20", "flow", "ret-volume corr20", lambda d: d["ret_vol_corr_20"], freq="D")

    library.register("intraday_range", "volatility", "intraday range negative", lambda d: -d["intraday_range"], freq="D")
    library.register("body_ratio", "price_action", "body ratio", lambda d: d["body_ratio"], freq="D")
    library.register("close_pos", "price_action", "close position in range", lambda d: d["close_pos"], freq="D")
    library.register("atr_norm_14", "volatility", "atr14 norm negative", lambda d: -d["atr_norm_14"], freq="D")
    library.register("realized_vol_20", "volatility", "realized vol negative", lambda d: -d["realized_vol_20"], freq="D")
    library.register("downside_vol_ratio_20", "volatility", "downside ratio negative", lambda d: -d["downside_vol_ratio_20"], freq="D")
    library.register("rsi14", "oscillator", "rsi14", lambda d: d["rsi14"] / 100.0, freq="D")

    library.register("close_to_vwap_day", "intraday_micro", "close-vwap", lambda d: d["close_to_vwap_day"], freq="D")
    library.register("morning_momentum_30m", "intraday_micro", "morning momentum", lambda d: d["morning_momentum_30m"], freq="D")
    library.register("last30_momentum", "intraday_micro", "last30 momentum", lambda d: d["last30_momentum"], freq="D")
    library.register("vwap30_vs_day", "intraday_micro", "vwap30-vwap day", lambda d: d["vwap30_vs_day"], freq="D")
    library.register("minute_realized_vol_5m", "intraday_micro", "5m rv negative", lambda d: -d["minute_realized_vol_5m"], freq="D")
    library.register("minute_up_ratio_5m", "intraday_micro", "5m up ratio", lambda d: d["minute_up_ratio_5m"], freq="D")
    library.register("minute_ret_skew_5m", "intraday_micro", "5m skew", lambda d: d["minute_ret_skew_5m"], freq="D")
    library.register("minute_ret_kurt_5m", "intraday_micro", "5m kurt negative", lambda d: -d["minute_ret_kurt_5m"], freq="D")
    library.register("signed_vol_imbalance_5m", "intraday_micro", "signed volume imbalance", lambda d: d["signed_vol_imbalance_5m"], freq="D")
    library.register("jump_ratio_5m", "intraday_micro", "jump ratio negative", lambda d: -d["jump_ratio_5m"], freq="D")
    library.register("open_to_close_intraday", "intraday_micro", "open-close momentum", lambda d: d["open_to_close_intraday"], freq="D")
    library.register("overnight_gap", "overnight", "overnight gap", lambda d: d["overnight_gap"], freq="D")


def register_generic_freq_factors(library: FactorLibrary, freq: str) -> None:
    library.register("ret_1", "trend", "1-bar return", lambda d: d["ret_1"], freq=freq)
    library.register("ret_3", "trend", "3-bar return", lambda d: d["ret_3"], freq=freq)
    library.register("ret_6", "trend", "6-bar return", lambda d: d["ret_6"], freq=freq)
    library.register("ret_12", "trend", "12-bar return", lambda d: d["ret_12"], freq=freq)
    library.register("ma_gap_6", "trend", "close/ma6 - 1", lambda d: d["ma_gap_6"], freq=freq)
    library.register("ma_gap_12", "trend", "close/ma12 - 1", lambda d: d["ma_gap_12"], freq=freq)
    library.register("rv_12", "volatility", "12-bar rv negative", lambda d: -d["rv_12"], freq=freq)
    library.register("range_norm", "volatility", "range norm negative", lambda d: -d["range_norm"], freq=freq)
    library.register("amount_ratio_12", "liquidity", "amount ratio", lambda d: d["amount_ratio_12"], freq=freq)
    library.register("vol_chg_1", "flow", "volume change", lambda d: d["vol_chg_1"], freq=freq)
    library.register("crowding_proxy_raw", "crowding", "crowding proxy negative", lambda d: -d["crowding_proxy_raw"], freq=freq)


def register_default_factors(library: FactorLibrary) -> None:
    register_default_daily_factors(library)
    for freq in ["5min", "15min", "30min", "60min", "120min", "W", "M"]:
        register_generic_freq_factors(library, freq)

