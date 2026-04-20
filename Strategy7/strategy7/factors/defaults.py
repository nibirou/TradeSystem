"""Built-in factor sets for multiple frequencies.

Goals:
1. Keep compatibility with existing factor names.
2. Add richer, frequency-differentiated price-volume factors.
3. Support high->low frequency bridge factors via `hf_{src}_to_{tgt}_*`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ..core.constants import INTRADAY_FREQS
from .base import FactorLibrary

EPS = 1e-12


def _uniq(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _nan_like(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.nan, index=df.index, dtype=float)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return _nan_like(df)
    return pd.to_numeric(df[name], errors="coerce")


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return pd.to_numeric(numer, errors="coerce") / (pd.to_numeric(denom, errors="coerce") + EPS)


def _hf_name(source_freq: str, target_freq: str, agg: str, base_col: str) -> str:
    return f"hf_{source_freq}_to_{target_freq}_{agg}_{base_col}"


def _hf_col(df: pd.DataFrame, source_freq: str, target_freq: str, agg: str, base_col: str) -> pd.Series:
    return _col(df, _hf_name(source_freq, target_freq, agg, base_col))


def _coalesce(df: pd.DataFrame, series_list: Sequence[pd.Series]) -> pd.Series:
    out: pd.Series | None = None
    for s in series_list:
        out = s if out is None else out.combine_first(s)
    return out if out is not None else _nan_like(df)


def _hf_any(
    df: pd.DataFrame,
    source_freq: str,
    target_freq: str,
    agg: str,
    base_candidates: Sequence[str],
) -> pd.Series:
    return _coalesce(df, [_hf_col(df, source_freq, target_freq, agg, c) for c in base_candidates])


DAILY_BASE_FACTORS: List[str] = [
    "mom_5",
    "mom_10",
    "mom_20",
    "rev_1",
    "rev_3",
    "ma_gap_5",
    "ma_gap_10",
    "ma_gap_20",
    "ma_cross_5_20",
    "breakout_20",
    "vol_ratio_5",
    "vol_ratio_20",
    "amount_ratio_20",
    "turn_ratio_5",
    "amihud_20",
    "ret_vol_corr_20",
    "intraday_range",
    "body_ratio",
    "close_pos",
    "atr_norm_14",
    "realized_vol_20",
    "downside_vol_ratio_20",
    "rsi14",
    "close_to_vwap_day",
    "morning_momentum_30m",
    "last30_momentum",
    "vwap30_vs_day",
    "minute_realized_vol_5m",
    "minute_up_ratio_5m",
    "minute_ret_skew_5m",
    "minute_ret_kurt_5m",
    "signed_vol_imbalance_5m",
    "jump_ratio_5m",
    "open_to_close_intraday",
    "overnight_gap",
]

DAILY_ALPHA_EXTENSION: List[str] = [
    "trend_shock_5_20",
    "momentum_quality_20",
    "breakout_strength_20",
    "volume_price_confirmation_20",
    "turnover_pressure_reversal",
    "amihud_reversal_20",
    "vol_asymmetry_alpha",
    "candle_imbalance_alpha",
    "overnight_intraday_rotation",
    "opening_drive_vs_close",
    "vwap_reversion_day",
    "intraday_sentiment_divergence",
    "jump_risk_discount",
    "skew_carry_intraday",
    "crowding_unwind_daily",
    "liquidity_regime_switch",
    "trend_vol_regime",
    "micro_macro_conflict",
    "hf_ultra_noise_5m_day",
    "hf_fast_slow_liquidity_diff_day",
    "hf_fast_slow_trend_diff_day",
]

GENERIC_CORE_FACTORS: List[str] = [
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "ma_gap_6",
    "ma_gap_12",
    "rv_12",
    "range_norm",
    "amount_ratio_12",
    "vol_chg_1",
    "crowding_proxy_raw",
    "ret_spread_1_6",
    "ret_accel_3_12",
    "momentum_quality_12",
    "reversal_pressure_1",
    "trend_efficiency_6",
    "range_break_signal",
    "crowding_unwind",
    "flow_momentum_sync",
    "liquidity_stress",
    "trend_liquidity_combo",
    "vol_adjusted_gap",
    "ret_curve_312",
    "vol_flow_dislocation",
    "context_trend_20d",
    "context_quality_20d",
    "context_liquidity_20d",
    "context_intraday_mood",
]

INTRADAY_SIGNATURE_FACTORS: List[str] = [
    "intraday_signal_density",
    "intraday_flow_fragility",
    "intraday_adaptive_breakout",
    "intraday_crowding_stress",
    "intraday_volatility_drag",
    "intraday_trend_persistence",
    "intraday_liquidity_recovery",
    "intraday_breakout_quality",
]

MICRO_5MIN_EXTRA_FACTORS: List[str] = [
    "micro_trend_convexity",
    "micro_reversal_flow",
    "micro_break_efficiency",
    "micro_vol_breakout",
    "micro_flow_accel",
    "micro_noise_filter",
    "micro_liquidity_shock",
    "micro_crowding_pressure",
]

PERIOD_SIGNATURE_FACTORS: List[str] = [
    "period_trend_vol_balance",
    "period_macro_rotation",
    "period_liquidity_cycle",
    "period_micro_macro_resonance",
    "period_downside_guard",
    "period_breakout_decay",
]

BRIDGE_PRIMARY_FACTORS: List[str] = [
    "hf_noise_to_signal",
    "hf_liquidity_pulse",
    "hf_trend_carry",
    "hf_vol_compression",
    "hf_crowding_unwind",
    "hf_close_dispersion",
    "hf_hilo_spread",
    "hf_close_vs_mean",
    "hf_liquidity_trend_sync",
    "hf_micro_regime_shift",
]

BRIDGE_MULTISCALE_FACTORS: List[str] = [
    "hf_fast_slow_trend_diff",
    "hf_fast_slow_liquidity_diff",
    "hf_fast_slow_noise_diff",
]

DAILY_FLOW_FACTORS: List[str] = [
    "ret_vol_corr_20",
    "volume_price_confirmation_20",
    "turnover_pressure_reversal",
]

GENERIC_FLOW_FACTORS: List[str] = [
    "vol_chg_1",
    "flow_momentum_sync",
    "vol_flow_dislocation",
]

DAILY_CROWDING_FACTORS: List[str] = [
    "crowding_unwind_daily",
]

GENERIC_CROWDING_FACTORS: List[str] = [
    "crowding_proxy_raw",
    "crowding_unwind",
]

DAILY_PRICE_ACTION_FACTORS: List[str] = [
    "body_ratio",
    "close_pos",
    "candle_imbalance_alpha",
]

GENERIC_PRICE_ACTION_FACTORS: List[str] = [
    "range_break_signal",
]

DAILY_INTRADAY_MICRO_FACTORS: List[str] = [
    "close_to_vwap_day",
    "morning_momentum_30m",
    "last30_momentum",
    "vwap30_vs_day",
    "minute_realized_vol_5m",
    "minute_up_ratio_5m",
    "minute_ret_skew_5m",
    "minute_ret_kurt_5m",
    "signed_vol_imbalance_5m",
    "jump_ratio_5m",
    "open_to_close_intraday",
    "opening_drive_vs_close",
    "vwap_reversion_day",
    "intraday_sentiment_divergence",
    "jump_risk_discount",
    "skew_carry_intraday",
    "micro_macro_conflict",
]

DAILY_OVERNIGHT_FACTORS: List[str] = [
    "overnight_gap",
    "overnight_intraday_rotation",
]

DAILY_OSCILLATOR_FACTORS: List[str] = [
    "rsi14",
]

DAILY_MULTI_FREQ_EXTRA_FACTORS: List[str] = [
    "hf_ultra_noise_5m_day",
    "hf_fast_slow_liquidity_diff_day",
    "hf_fast_slow_trend_diff_day",
]

FUNDAMENTAL_PACK_TO_CATEGORY: Dict[str, str] = {
    "fund_growth": "growth",
    "fund_valuation": "valuation",
    "fund_profitability": "profitability",
    "fund_quality": "quality",
    "fund_leverage": "leverage",
    "fund_cashflow": "cashflow",
    "fund_efficiency": "efficiency",
    "fund_expectation": "expectation",
}

FUND_HF_METRICS: List[str] = ["flow", "jump", "intraday", "vwap", "sentiment"]


def _fundamental_raw_count(freq: str) -> int:
    return 12 if str(freq) in {"D", "W", "M"} else 5


def _fundamental_cols_for_category(freq: str, cat: str) -> List[str]:
    raw_n = _fundamental_raw_count(freq)
    cols = [f"fd_{cat}_raw_{i:02d}" for i in range(1, raw_n + 1)]
    cols.extend([f"fd_{cat}_score", f"fd_{cat}_trend", f"fd_{cat}_disp"])
    if str(freq) in INTRADAY_FREQS:
        cols.extend([f"fd_hf_{cat}_{m}" for m in FUND_HF_METRICS])
    return _uniq(cols)


def _fundamental_hf_fusion_cols(freq: str) -> List[str]:
    cols: List[str] = []
    for cat in FUNDAMENTAL_PACK_TO_CATEGORY.values():
        cols.extend([f"fd_hf_{cat}_{m}" for m in FUND_HF_METRICS])
    cols.extend(["fd_hf_fusion_score", "fd_hf_fusion_disp"])
    # Add some explicit bridge columns so this pack can combine valuation/fundamental
    # context with intraday micro-structure aggregation.
    cols.extend(
        [
            _hf_name("5min", str(freq), "mean", "ret_1"),
            _hf_name("5min", str(freq), "std", "ret_1"),
            _hf_name("5min", str(freq), "mean", "amount_ratio_12"),
            _hf_name("120min", str(freq), "last", "ret_3"),
            _hf_name("120min", str(freq), "mean", "amount_ratio_12"),
            "hf_fast_slow_trend_diff",
            "hf_fast_slow_liquidity_diff",
            "hf_fast_slow_noise_diff",
        ]
    )
    return _uniq(cols)


TEXT_PACKAGES: Sequence[str] = (
    "text_sentiment",
    "text_attention",
    "text_event",
    "text_topic",
    "text_fusion",
)


def _text_cols_for_package(freq: str, package: str) -> List[str]:
    # Text features are derived from daily panels and then broadcast to non-daily
    # views as daily context fields, so the same feature namespace is valid for
    # all frequencies.
    _ = str(freq)
    p = str(package)
    mapping: Dict[str, List[str]] = {
        "text_sentiment": [
            "txt_sentiment_mean",
            "txt_sentiment_std",
            "txt_pos_share",
            "txt_neg_share",
            "txt_sent_5",
            "txt_sent_20",
            "txt_sent_trend_5",
            "txt_sent_vol_20",
            "txt_novelty_mean",
            "txt_novelty_20",
        ],
        "text_attention": [
            "txt_event_count",
            "txt_source_coverage",
            "txt_attention_mean",
            "txt_attention_5",
            "txt_attention_20",
            "txt_attention_shock",
            "txt_count_5",
            "txt_count_20",
            "txt_news_burst",
            "txt_notice_burst",
            "txt_em_report_burst",
            "txt_iwencai_burst",
        ],
        "text_event": [
            "txt_risk_mean",
            "txt_uncertainty_mean",
            "txt_action_mean",
            "txt_risk_5",
            "txt_risk_20",
            "txt_uncertainty_5",
            "txt_uncertainty_20",
            "txt_news_count",
            "txt_notice_count",
            "txt_em_report_count",
            "txt_iwencai_count",
            "txt_rating_delta_mean",
        ],
        "text_topic": [
            "txt_topic_earnings",
            "txt_topic_policy",
            "txt_topic_mna",
            "txt_topic_risk",
            "txt_topic_operation",
            "txt_topic_capital",
            "txt_topic_risk_vs_earnings",
            "txt_topic_policy_mna_mix",
            "txt_news_share",
            "txt_notice_share",
            "txt_em_report_share",
            "txt_iwencai_share",
            "txt_source_dispersion",
        ],
        "text_fusion": [
            "txt_hf_sent_flow",
            "txt_hf_attention_jump",
            "txt_hf_risk_vol",
            "txt_hf_momentum_resonance",
            "txt_hf_liquidity_pulse",
            "txt_fd_growth_resonance",
            "txt_fd_quality_guard",
            "txt_fd_expectation_spread",
            "txt_fd_cashflow_conflict",
            "txt_fd_valuation_contrast",
            "txt_fusion_score",
            "txt_fusion_disp",
        ],
    }
    return _uniq(mapping.get(p, []))


DEFAULT_FACTOR_SET_BY_FREQ: Dict[str, List[str]] = {
    "D": _uniq(DAILY_BASE_FACTORS + DAILY_ALPHA_EXTENSION + BRIDGE_PRIMARY_FACTORS + BRIDGE_MULTISCALE_FACTORS),
    "5min": _uniq(GENERIC_CORE_FACTORS + INTRADAY_SIGNATURE_FACTORS + MICRO_5MIN_EXTRA_FACTORS),
    "15min": _uniq(GENERIC_CORE_FACTORS + INTRADAY_SIGNATURE_FACTORS + BRIDGE_PRIMARY_FACTORS),
    "30min": _uniq(GENERIC_CORE_FACTORS + INTRADAY_SIGNATURE_FACTORS + BRIDGE_PRIMARY_FACTORS + BRIDGE_MULTISCALE_FACTORS),
    "60min": _uniq(GENERIC_CORE_FACTORS + INTRADAY_SIGNATURE_FACTORS + BRIDGE_PRIMARY_FACTORS + BRIDGE_MULTISCALE_FACTORS),
    "120min": _uniq(GENERIC_CORE_FACTORS + INTRADAY_SIGNATURE_FACTORS + BRIDGE_PRIMARY_FACTORS + BRIDGE_MULTISCALE_FACTORS),
    "W": _uniq(GENERIC_CORE_FACTORS + PERIOD_SIGNATURE_FACTORS + BRIDGE_PRIMARY_FACTORS + BRIDGE_MULTISCALE_FACTORS),
    "M": _uniq(GENERIC_CORE_FACTORS + PERIOD_SIGNATURE_FACTORS + BRIDGE_PRIMARY_FACTORS + BRIDGE_MULTISCALE_FACTORS),
}


def register_default_daily_factors(library: FactorLibrary) -> None:
    r = library.register
    r("mom_5", "trend", "5-day momentum", lambda d: _col(d, "ret_5d"), freq="D")
    r("mom_10", "trend", "10-day momentum", lambda d: _col(d, "ret_10d"), freq="D")
    r("mom_20", "trend", "20-day momentum", lambda d: _col(d, "ret_20d"), freq="D")
    r("rev_1", "reversal", "1-day reversal", lambda d: -_col(d, "ret_1d"), freq="D")
    r("rev_3", "reversal", "3-day reversal", lambda d: -_col(d, "ret_3d"), freq="D")

    r("ma_gap_5", "trend", "close/ma5 - 1", lambda d: _col(d, "ma_gap_5"), freq="D")
    r("ma_gap_10", "trend", "close/ma10 - 1", lambda d: _col(d, "ma_gap_10"), freq="D")
    r("ma_gap_20", "trend", "close/ma20 - 1", lambda d: _col(d, "ma_gap_20"), freq="D")
    r("ma_cross_5_20", "trend", "ma5/ma20 - 1", lambda d: _col(d, "ma_cross_5_20"), freq="D")
    r("breakout_20", "trend", "close/rolling high20 - 1", lambda d: _col(d, "breakout_20"), freq="D")

    r("vol_ratio_5", "liquidity", "volume/vol_ma5", lambda d: _col(d, "vol_ratio_5"), freq="D")
    r("vol_ratio_20", "liquidity", "volume/vol_ma20", lambda d: _col(d, "vol_ratio_20"), freq="D")
    r("amount_ratio_20", "liquidity", "amount/amount_ma20", lambda d: _col(d, "amount_ratio_20"), freq="D")
    r("turn_ratio_5", "liquidity", "turn/turn_ma5", lambda d: _col(d, "turn_ratio_5"), freq="D")
    r("amihud_20", "liquidity", "negative amihud20", lambda d: -_col(d, "amihud_20"), freq="D")
    r("ret_vol_corr_20", "flow", "ret-volume corr20", lambda d: _col(d, "ret_vol_corr_20"), freq="D")

    r("intraday_range", "volatility", "intraday range negative", lambda d: -_col(d, "intraday_range"), freq="D")
    r("body_ratio", "price_action", "body ratio", lambda d: _col(d, "body_ratio"), freq="D")
    r("close_pos", "price_action", "close position in range", lambda d: _col(d, "close_pos"), freq="D")
    r("atr_norm_14", "volatility", "atr14 norm negative", lambda d: -_col(d, "atr_norm_14"), freq="D")
    r("realized_vol_20", "volatility", "realized vol negative", lambda d: -_col(d, "realized_vol_20"), freq="D")
    r("downside_vol_ratio_20", "volatility", "downside ratio negative", lambda d: -_col(d, "downside_vol_ratio_20"), freq="D")
    r("rsi14", "oscillator", "rsi14", lambda d: _col(d, "rsi14") / 100.0, freq="D")

    r("close_to_vwap_day", "intraday_micro", "close-vwap", lambda d: _col(d, "close_to_vwap_day"), freq="D")
    r("morning_momentum_30m", "intraday_micro", "morning momentum", lambda d: _col(d, "morning_momentum_30m"), freq="D")
    r("last30_momentum", "intraday_micro", "last30 momentum", lambda d: _col(d, "last30_momentum"), freq="D")
    r("vwap30_vs_day", "intraday_micro", "vwap30-vwap day", lambda d: _col(d, "vwap30_vs_day"), freq="D")
    r("minute_realized_vol_5m", "intraday_micro", "5m rv negative", lambda d: -_col(d, "minute_realized_vol_5m"), freq="D")
    r("minute_up_ratio_5m", "intraday_micro", "5m up ratio", lambda d: _col(d, "minute_up_ratio_5m"), freq="D")
    r("minute_ret_skew_5m", "intraday_micro", "5m skew", lambda d: _col(d, "minute_ret_skew_5m"), freq="D")
    r("minute_ret_kurt_5m", "intraday_micro", "5m kurt negative", lambda d: -_col(d, "minute_ret_kurt_5m"), freq="D")
    r("signed_vol_imbalance_5m", "intraday_micro", "signed volume imbalance", lambda d: _col(d, "signed_vol_imbalance_5m"), freq="D")
    r("jump_ratio_5m", "intraday_micro", "jump ratio negative", lambda d: -_col(d, "jump_ratio_5m"), freq="D")
    r("open_to_close_intraday", "intraday_micro", "open-close momentum", lambda d: _col(d, "open_to_close_intraday"), freq="D")
    r("overnight_gap", "overnight", "overnight gap", lambda d: _col(d, "overnight_gap"), freq="D")

    r("trend_shock_5_20", "trend", "ret5-ret20 trend shock", lambda d: _col(d, "ret_5d") - _col(d, "ret_20d"), freq="D")
    r(
        "momentum_quality_20",
        "trend",
        "ret20 adjusted by vol20",
        lambda d: _safe_div(_col(d, "ret_20d"), _col(d, "realized_vol_20").abs() + EPS),
        freq="D",
    )
    r(
        "breakout_strength_20",
        "trend",
        "breakout normalized by ATR",
        lambda d: _safe_div(_col(d, "breakout_20"), _col(d, "atr_norm_14").abs() + EPS),
        freq="D",
    )
    r(
        "volume_price_confirmation_20",
        "flow",
        "ret5 confirmed by vol ratio",
        lambda d: _col(d, "vol_ratio_20") * _col(d, "ret_5d"),
        freq="D",
    )
    r(
        "turnover_pressure_reversal",
        "flow",
        "turnover pressure reversal",
        lambda d: -_col(d, "turn_ratio_5") * _col(d, "ret_1d"),
        freq="D",
    )
    r(
        "amihud_reversal_20",
        "liquidity",
        "illiquidity reversal",
        lambda d: -_col(d, "amihud_20") * _col(d, "ret_1d"),
        freq="D",
    )
    r("vol_asymmetry_alpha", "volatility", "downside/upside vol asymmetry", lambda d: -(_col(d, "downside_vol_ratio_20") - 1.0), freq="D")
    r("candle_imbalance_alpha", "price_action", "body*close-position imbalance", lambda d: _col(d, "body_ratio") * (_col(d, "close_pos") - 0.5), freq="D")
    r("overnight_intraday_rotation", "overnight", "overnight minus intraday trend", lambda d: _col(d, "overnight_gap") - _col(d, "open_to_close_intraday"), freq="D")
    r("opening_drive_vs_close", "intraday_micro", "morning drive minus last30", lambda d: _col(d, "morning_momentum_30m") - _col(d, "last30_momentum"), freq="D")
    r("vwap_reversion_day", "intraday_micro", "close-vwap reversion weighted by vwap spread", lambda d: -_col(d, "close_to_vwap_day") * _col(d, "vwap30_vs_day").abs(), freq="D")
    r("intraday_sentiment_divergence", "intraday_micro", "signed volume minus up-ratio", lambda d: _col(d, "signed_vol_imbalance_5m") - _col(d, "minute_up_ratio_5m"), freq="D")
    r("jump_risk_discount", "intraday_micro", "jump risk discount", lambda d: -_col(d, "jump_ratio_5m") * _col(d, "realized_vol_20"), freq="D")
    r("skew_carry_intraday", "intraday_micro", "intraday skew carry", lambda d: _safe_div(_col(d, "minute_ret_skew_5m"), _col(d, "minute_realized_vol_5m").abs() + EPS), freq="D")
    r("crowding_unwind_daily", "crowding", "crowding unwind with ret3", lambda d: -_col(d, "crowding_proxy_raw") * _col(d, "ret_3d"), freq="D")
    r("liquidity_regime_switch", "liquidity", "vol ratio short-long spread", lambda d: _col(d, "vol_ratio_5") - _col(d, "vol_ratio_20"), freq="D")
    r("trend_vol_regime", "trend", "trend over vol regime", lambda d: _safe_div(_col(d, "ma_gap_20"), _col(d, "realized_vol_20").abs() + EPS), freq="D")
    r("micro_macro_conflict", "intraday_micro", "morning trend minus daily ret1", lambda d: _col(d, "morning_momentum_30m") - _col(d, "ret_1d"), freq="D")
    r(
        "hf_ultra_noise_5m_day",
        "multi_freq",
        "5min->D noise-to-signal",
        lambda d: -_safe_div(_hf_col(d, "5min", "D", "std", "ret_1"), _hf_col(d, "5min", "D", "mean", "ret_1").abs() + EPS),
        freq="D",
    )
    r(
        "hf_fast_slow_liquidity_diff_day",
        "multi_freq",
        "liquidity pulse: 5min->D minus 120min->D",
        lambda d: (
            _safe_div(
                _hf_col(d, "5min", "D", "last", "amount_ratio_12") - _hf_col(d, "5min", "D", "mean", "amount_ratio_12"),
                _hf_col(d, "5min", "D", "mean", "amount_ratio_12").abs() + EPS,
            )
            - _safe_div(
                _hf_col(d, "120min", "D", "last", "amount_ratio_12") - _hf_col(d, "120min", "D", "mean", "amount_ratio_12"),
                _hf_col(d, "120min", "D", "mean", "amount_ratio_12").abs() + EPS,
            )
        ),
        freq="D",
    )
    r(
        "hf_fast_slow_trend_diff_day",
        "multi_freq",
        "trend carry: 5min->D minus 120min->D",
        lambda d: (
            (_hf_col(d, "5min", "D", "last", "ret_3") - _hf_col(d, "5min", "D", "mean", "ret_3"))
            - (_hf_col(d, "120min", "D", "last", "ret_3") - _hf_col(d, "120min", "D", "mean", "ret_3"))
        ),
        freq="D",
    )


def register_generic_freq_factors(library: FactorLibrary, freq: str) -> None:
    r = library.register
    r("ret_1", "trend", "1-bar return", lambda d: _col(d, "ret_1"), freq=freq)
    r("ret_3", "trend", "3-bar return", lambda d: _col(d, "ret_3"), freq=freq)
    r("ret_6", "trend", "6-bar return", lambda d: _col(d, "ret_6"), freq=freq)
    r("ret_12", "trend", "12-bar return", lambda d: _col(d, "ret_12"), freq=freq)
    r("ma_gap_6", "trend", "close/ma6 - 1", lambda d: _col(d, "ma_gap_6"), freq=freq)
    r("ma_gap_12", "trend", "close/ma12 - 1", lambda d: _col(d, "ma_gap_12"), freq=freq)
    r("rv_12", "volatility", "12-bar rv negative", lambda d: -_col(d, "rv_12"), freq=freq)
    r("range_norm", "volatility", "range norm negative", lambda d: -_col(d, "range_norm"), freq=freq)
    r("amount_ratio_12", "liquidity", "amount ratio", lambda d: _col(d, "amount_ratio_12"), freq=freq)
    r("vol_chg_1", "flow", "volume change", lambda d: _col(d, "vol_chg_1"), freq=freq)
    r("crowding_proxy_raw", "crowding", "crowding proxy negative", lambda d: -_col(d, "crowding_proxy_raw"), freq=freq)

    r("ret_spread_1_6", "trend", "ret1-ret6 spread", lambda d: _col(d, "ret_1") - _col(d, "ret_6"), freq=freq)
    r("ret_accel_3_12", "trend", "ret3-ret12 acceleration", lambda d: _col(d, "ret_3") - _col(d, "ret_12"), freq=freq)
    r("momentum_quality_12", "trend", "ret12 adjusted by rv12", lambda d: _safe_div(_col(d, "ret_12"), _col(d, "rv_12").abs() + EPS), freq=freq)
    r("reversal_pressure_1", "reversal", "ret1 reversal with amount", lambda d: -_col(d, "ret_1") * _col(d, "amount_ratio_12"), freq=freq)
    r("trend_efficiency_6", "trend", "|ret6| / rv12", lambda d: _safe_div(_col(d, "ret_6").abs(), _col(d, "rv_12").abs() + EPS), freq=freq)
    r("range_break_signal", "price_action", "ma_gap6/range", lambda d: _safe_div(_col(d, "ma_gap_6"), _col(d, "range_norm").abs() + EPS), freq=freq)
    r("crowding_unwind", "crowding", "crowding unwind", lambda d: -_col(d, "crowding_proxy_raw") * _col(d, "ret_1"), freq=freq)
    r("flow_momentum_sync", "flow", "vol change * ret3", lambda d: _col(d, "vol_chg_1") * _col(d, "ret_3"), freq=freq)
    r("liquidity_stress", "liquidity", "-rv12*|amount_ratio-1|", lambda d: -_col(d, "rv_12").abs() * (_col(d, "amount_ratio_12") - 1.0).abs(), freq=freq)
    r("trend_liquidity_combo", "trend", "ma_gap12 * amount_ratio", lambda d: _col(d, "ma_gap_12") * _col(d, "amount_ratio_12"), freq=freq)
    r("vol_adjusted_gap", "trend", "ma_gap12/rv12", lambda d: _safe_div(_col(d, "ma_gap_12"), _col(d, "rv_12").abs() + EPS), freq=freq)
    r("ret_curve_312", "trend", "0.6*ret3+0.4*ret6-ret12", lambda d: 0.6 * _col(d, "ret_3") + 0.4 * _col(d, "ret_6") - _col(d, "ret_12"), freq=freq)
    r("vol_flow_dislocation", "flow", "vol change minus amount ratio", lambda d: _col(d, "vol_chg_1") - _col(d, "amount_ratio_12"), freq=freq)

    r("context_trend_20d", "context", "daily ret20 context", lambda d: _col(d, "ret_20d"), freq=freq)
    r("context_quality_20d", "context", "daily ret20 over vol20", lambda d: _safe_div(_col(d, "ret_20d"), _col(d, "realized_vol_20").abs() + EPS), freq=freq)
    r("context_liquidity_20d", "context", "negative daily amihud20", lambda d: -_col(d, "amihud_20"), freq=freq)
    r("context_intraday_mood", "context", "up-ratio minus jump ratio", lambda d: _col(d, "minute_up_ratio_5m") - _col(d, "jump_ratio_5m"), freq=freq)


def register_intraday_signature_factors(library: FactorLibrary, freq: str) -> None:
    scale_map = {"5min": 1.0, "15min": 1.2, "30min": 1.5, "60min": 2.0, "120min": 2.5}
    sf = float(scale_map.get(str(freq), 1.0))
    r = library.register

    r("intraday_signal_density", "intraday_signature", "ret1/(rv12*scale)", lambda d, s=sf: _safe_div(_col(d, "ret_1"), _col(d, "rv_12").abs() * s + EPS), freq=freq)
    r("intraday_flow_fragility", "intraday_signature", "-|vol_chg|*|range|*scale", lambda d, s=sf: -_col(d, "vol_chg_1").abs() * _col(d, "range_norm").abs() * s, freq=freq)
    r("intraday_adaptive_breakout", "intraday_signature", "ma_gap6*(1+amount_ratio/scale)", lambda d, s=sf: _col(d, "ma_gap_6") * (1.0 + _col(d, "amount_ratio_12") / (s + EPS)), freq=freq)
    r("intraday_crowding_stress", "intraday_signature", "-crowding*ret3*scale", lambda d, s=sf: -_col(d, "crowding_proxy_raw") * _col(d, "ret_3") * s, freq=freq)
    r("intraday_volatility_drag", "intraday_signature", "ret6-rv12*scale", lambda d, s=sf: _col(d, "ret_6") - _col(d, "rv_12") * s, freq=freq)
    r("intraday_trend_persistence", "intraday_signature", "ret12-ret3", lambda d: _col(d, "ret_12") - _col(d, "ret_3"), freq=freq)
    r("intraday_liquidity_recovery", "intraday_signature", "(amount_ratio-1)-|vol_chg|", lambda d: (_col(d, "amount_ratio_12") - 1.0) - _col(d, "vol_chg_1").abs(), freq=freq)
    r("intraday_breakout_quality", "intraday_signature", "ma_gap12 normalized by range", lambda d: _safe_div(_col(d, "ma_gap_12"), _col(d, "range_norm").abs() + EPS), freq=freq)


def register_5min_specific_factors(library: FactorLibrary) -> None:
    freq = "5min"
    r = library.register
    r("micro_trend_convexity", "intraday_micro", "ret1-0.7*ret3+0.3*ret6", lambda d: _col(d, "ret_1") - 0.7 * _col(d, "ret_3") + 0.3 * _col(d, "ret_6"), freq=freq)
    r("micro_reversal_flow", "intraday_micro", "-ret1*vol_chg1", lambda d: -_col(d, "ret_1") * _col(d, "vol_chg_1"), freq=freq)
    r("micro_break_efficiency", "intraday_micro", "ma_gap6/(rv12+range)", lambda d: _safe_div(_col(d, "ma_gap_6"), _col(d, "rv_12").abs() + _col(d, "range_norm").abs() + EPS), freq=freq)
    r("micro_vol_breakout", "intraday_micro", "range*amount_ratio", lambda d: _col(d, "range_norm") * _col(d, "amount_ratio_12"), freq=freq)
    r("micro_flow_accel", "intraday_micro", "vol_chg*(ret3-ret1)", lambda d: _col(d, "vol_chg_1") * (_col(d, "ret_3") - _col(d, "ret_1")), freq=freq)
    r("micro_noise_filter", "intraday_micro", "-rv12/|ret6|", lambda d: -_safe_div(_col(d, "rv_12"), _col(d, "ret_6").abs() + EPS), freq=freq)
    r("micro_liquidity_shock", "intraday_micro", "(amount_ratio-1)*range", lambda d: (_col(d, "amount_ratio_12") - 1.0) * _col(d, "range_norm"), freq=freq)
    r("micro_crowding_pressure", "intraday_micro", "-crowding*vol_chg", lambda d: -_col(d, "crowding_proxy_raw") * _col(d, "vol_chg_1"), freq=freq)


def register_period_signature_factors(library: FactorLibrary, freq: str) -> None:
    sf = 1.0 if str(freq) == "W" else 1.6
    r = library.register
    r("period_trend_vol_balance", "period_signature", "ret12/(rv12+|ma_gap12|)", lambda d: _safe_div(_col(d, "ret_12"), _col(d, "rv_12").abs() + _col(d, "ma_gap_12").abs() + EPS), freq=freq)
    r("period_macro_rotation", "period_signature", "context trend minus ret3", lambda d: _col(d, "context_trend_20d") - _col(d, "ret_3"), freq=freq)
    r("period_liquidity_cycle", "period_signature", "context liquidity * amount ratio", lambda d: _col(d, "context_liquidity_20d") * _col(d, "amount_ratio_12"), freq=freq)
    r("period_micro_macro_resonance", "period_signature", "intraday mood + ret6*scale", lambda d, s=sf: _col(d, "context_intraday_mood") + _col(d, "ret_6") * s, freq=freq)
    r("period_downside_guard", "period_signature", "-rv12*|range|", lambda d: -_col(d, "rv_12").abs() * _col(d, "range_norm").abs(), freq=freq)
    r("period_breakout_decay", "period_signature", "ma_gap6-ma_gap12", lambda d: _col(d, "ma_gap_6") - _col(d, "ma_gap_12"), freq=freq)


def register_primary_bridge_factors(library: FactorLibrary, target_freq: str, source_freq: str) -> None:
    r = library.register
    pref = f"{source_freq}->{target_freq}"
    r(
        "hf_noise_to_signal",
        "multi_freq",
        f"{pref} return noise-to-signal",
        lambda d, s=source_freq, t=target_freq: -_safe_div(
            _hf_any(d, s, t, "std", ["ret_1", "ret_1d"]),
            _hf_any(d, s, t, "mean", ["ret_1", "ret_1d"]).abs() + EPS,
        ),
        freq=target_freq,
    )
    r(
        "hf_liquidity_pulse",
        "multi_freq",
        f"{pref} liquidity pulse",
        lambda d, s=source_freq, t=target_freq: _safe_div(
            _hf_any(d, s, t, "last", ["amount_ratio_12", "amount"])
            - _hf_any(d, s, t, "mean", ["amount_ratio_12", "amount"]),
            _hf_any(d, s, t, "mean", ["amount_ratio_12", "amount"]).abs() + EPS,
        ),
        freq=target_freq,
    )
    r(
        "hf_trend_carry",
        "multi_freq",
        f"{pref} trend carry",
        lambda d, s=source_freq, t=target_freq: _hf_any(d, s, t, "last", ["ret_3", "ret_3d", "ret_5d"])
        - _hf_any(d, s, t, "mean", ["ret_3", "ret_3d", "ret_5d"]),
        freq=target_freq,
    )
    r(
        "hf_vol_compression",
        "multi_freq",
        f"{pref} volatility compression",
        lambda d, s=source_freq, t=target_freq: -_hf_any(d, s, t, "std", ["rv_12", "realized_vol_20"]),
        freq=target_freq,
    )
    r(
        "hf_crowding_unwind",
        "multi_freq",
        f"{pref} crowding unwind",
        lambda d, s=source_freq, t=target_freq: -_hf_any(d, s, t, "last", ["crowding_proxy_raw"]) * _hf_any(d, s, t, "last", ["ret_1", "ret_1d"]),
        freq=target_freq,
    )
    r(
        "hf_close_dispersion",
        "multi_freq",
        f"{pref} close dispersion",
        lambda d, s=source_freq, t=target_freq: -_safe_div(_hf_any(d, s, t, "std", ["close"]), _hf_any(d, s, t, "mean", ["close"]).abs() + EPS),
        freq=target_freq,
    )
    r(
        "hf_hilo_spread",
        "multi_freq",
        f"{pref} high-low spread",
        lambda d, s=source_freq, t=target_freq: _safe_div(
            _hf_any(d, s, t, "mean", ["high"]) - _hf_any(d, s, t, "mean", ["low"]),
            _hf_any(d, s, t, "mean", ["close"]).abs() + EPS,
        ),
        freq=target_freq,
    )
    r(
        "hf_close_vs_mean",
        "multi_freq",
        f"{pref} last close vs mean close",
        lambda d, s=source_freq, t=target_freq: _safe_div(
            _hf_any(d, s, t, "last", ["close"]) - _hf_any(d, s, t, "mean", ["close"]),
            _hf_any(d, s, t, "mean", ["close"]).abs() + EPS,
        ),
        freq=target_freq,
    )
    r(
        "hf_liquidity_trend_sync",
        "multi_freq",
        f"{pref} liquidity pulse * trend",
        lambda d, s=source_freq, t=target_freq: _safe_div(
            _hf_any(d, s, t, "last", ["amount_ratio_12", "amount"]) - _hf_any(d, s, t, "mean", ["amount_ratio_12", "amount"]),
            _hf_any(d, s, t, "mean", ["amount_ratio_12", "amount"]).abs() + EPS,
        )
        * _hf_any(d, s, t, "last", ["ret_1", "ret_1d"]),
        freq=target_freq,
    )
    r(
        "hf_micro_regime_shift",
        "multi_freq",
        f"{pref} medium-short noise spread",
        lambda d, s=source_freq, t=target_freq: _hf_any(d, s, t, "std", ["ret_3", "ret_3d", "ret_5d"])
        - _hf_any(d, s, t, "std", ["ret_1", "ret_1d"]),
        freq=target_freq,
    )


def register_multiscale_bridge_factors(library: FactorLibrary, target_freq: str, fast_source: str, slow_source: str) -> None:
    r = library.register
    pref = f"fast={fast_source}, slow={slow_source}, target={target_freq}"
    r(
        "hf_fast_slow_trend_diff",
        "multi_freq",
        f"{pref}: trend difference",
        lambda d, f=fast_source, s=slow_source, t=target_freq: _hf_any(d, f, t, "last", ["ret_1", "ret_1d"])
        - _hf_any(d, s, t, "last", ["ret_1", "ret_1d"]),
        freq=target_freq,
    )
    r(
        "hf_fast_slow_liquidity_diff",
        "multi_freq",
        f"{pref}: liquidity pulse difference",
        lambda d, f=fast_source, s=slow_source, t=target_freq: (
            _safe_div(
                _hf_any(d, f, t, "last", ["amount_ratio_12", "amount"]) - _hf_any(d, f, t, "mean", ["amount_ratio_12", "amount"]),
                _hf_any(d, f, t, "mean", ["amount_ratio_12", "amount"]).abs() + EPS,
            )
            - _safe_div(
                _hf_any(d, s, t, "last", ["amount_ratio_12", "amount"]) - _hf_any(d, s, t, "mean", ["amount_ratio_12", "amount"]),
                _hf_any(d, s, t, "mean", ["amount_ratio_12", "amount"]).abs() + EPS,
            )
        ),
        freq=target_freq,
    )
    r(
        "hf_fast_slow_noise_diff",
        "multi_freq",
        f"{pref}: slow-noise minus fast-noise",
        lambda d, f=fast_source, s=slow_source, t=target_freq: (
            _safe_div(_hf_any(d, s, t, "std", ["ret_1", "ret_1d"]), _hf_any(d, s, t, "mean", ["ret_1", "ret_1d"]).abs() + EPS)
            - _safe_div(_hf_any(d, f, t, "std", ["ret_1", "ret_1d"]), _hf_any(d, f, t, "mean", ["ret_1", "ret_1d"]).abs() + EPS)
        ),
        freq=target_freq,
    )


def _template_generated_factor_names(
    *,
    prefix: str,
    cols: Sequence[str],
    pair_limit: int = 6,
    anchors: Sequence[str] | None = None,
) -> List[str]:
    names: List[str] = []
    cols_u = _uniq([str(c) for c in cols if str(c)])
    for c in cols_u:
        names.extend(
            [
                f"{prefix}_lvl_{c}",
                f"{prefix}_neg_{c}",
                f"{prefix}_tanh_{c}",
            ]
        )

    use_cols = cols_u[: max(int(pair_limit), 2)]
    for i in range(len(use_cols)):
        for j in range(i + 1, len(use_cols)):
            a = use_cols[i]
            b = use_cols[j]
            names.extend(
                [
                    f"{prefix}_diff_{a}__{b}",
                    f"{prefix}_ratio_{a}__{b}",
                    f"{prefix}_sym_{a}__{b}",
                ]
            )

    anchor_list = [x for x in _uniq(list(anchors or [])) if x in cols_u]
    for anc in anchor_list:
        for c in cols_u:
            if c == anc:
                continue
            names.append(f"{prefix}_rel_{c}_to_{anc}")
    return _uniq(names)


def _register_template_generated_package(
    *,
    library: FactorLibrary,
    freq: str,
    package: str,
    category: str,
    cols: Sequence[str],
    pair_limit: int = 6,
    anchors: Sequence[str] | None = None,
) -> List[str]:
    names = _template_generated_factor_names(
        prefix=package,
        cols=cols,
        pair_limit=pair_limit,
        anchors=anchors,
    )
    for name in names:
        if name.startswith(f"{package}_lvl_"):
            c = name[len(f"{package}_lvl_") :]
            library.register(name, category, f"{package} level {c}", lambda d, col=c: _col(d, col), freq=freq)
            continue
        if name.startswith(f"{package}_neg_"):
            c = name[len(f"{package}_neg_") :]
            library.register(name, category, f"{package} negative {c}", lambda d, col=c: -_col(d, col), freq=freq)
            continue
        if name.startswith(f"{package}_tanh_"):
            c = name[len(f"{package}_tanh_") :]
            library.register(name, category, f"{package} tanh {c}", lambda d, col=c: np.tanh(_col(d, col)), freq=freq)
            continue
        if name.startswith(f"{package}_diff_"):
            tail = name[len(f"{package}_diff_") :]
            a, b = tail.split("__", 1)
            library.register(
                name,
                category,
                f"{package} diff {a}-{b}",
                lambda d, ca=a, cb=b: _col(d, ca) - _col(d, cb),
                freq=freq,
            )
            continue
        if name.startswith(f"{package}_ratio_"):
            tail = name[len(f"{package}_ratio_") :]
            a, b = tail.split("__", 1)
            library.register(
                name,
                category,
                f"{package} ratio {a}/{b}",
                lambda d, ca=a, cb=b: _safe_div(_col(d, ca), _col(d, cb).abs() + EPS),
                freq=freq,
            )
            continue
        if name.startswith(f"{package}_sym_"):
            tail = name[len(f"{package}_sym_") :]
            a, b = tail.split("__", 1)
            library.register(
                name,
                category,
                f"{package} symmetric spread {a},{b}",
                lambda d, ca=a, cb=b: _safe_div(_col(d, ca) - _col(d, cb), _col(d, ca).abs() + _col(d, cb).abs() + EPS),
                freq=freq,
            )
            continue
        if name.startswith(f"{package}_rel_") and "_to_" in name:
            tail = name[len(f"{package}_rel_") :]
            c, anc = tail.split("_to_", 1)
            library.register(
                name,
                category,
                f"{package} rel {c} to {anc}",
                lambda d, cc=c, aa=anc: _safe_div(_col(d, cc) - _col(d, aa), _col(d, aa).abs() + EPS),
                freq=freq,
            )
    return names


def _bridge_sources_for_target(target_freq: str) -> List[str]:
    mapping: Dict[str, List[str]] = {
        "5min": [],
        "15min": ["5min"],
        "30min": ["5min", "15min"],
        "60min": ["5min", "15min", "30min"],
        "120min": ["5min", "15min", "30min", "60min"],
        "D": ["5min", "15min", "30min", "60min", "120min"],
        "W": ["5min", "15min", "30min", "60min", "120min", "D"],
        "M": ["5min", "15min", "30min", "60min", "120min", "D", "W"],
    }
    return mapping.get(str(target_freq), [])


def _bridge_factor_names(target_freq: str) -> List[str]:
    names: List[str] = []
    for s in _bridge_sources_for_target(target_freq):
        names.extend(
            [
                f"bridge_{s}_noise",
                f"bridge_{s}_trend_carry",
                f"bridge_{s}_liq_pulse",
                f"bridge_{s}_vol_comp",
                f"bridge_{s}_close_disp",
                f"bridge_{s}_close_dev",
                f"bridge_{s}_hilo_spread",
                f"bridge_{s}_crowd_unwind",
                f"bridge_{s}_liq_trend_sync",
                f"bridge_{s}_micro_shift",
                f"bridge_{s}_ret3_std",
                f"bridge_{s}_liq_pressure",
            ]
        )
    return _uniq(names)


def _register_bridge_factor_pack(library: FactorLibrary, target_freq: str) -> List[str]:
    names = _bridge_factor_names(target_freq)
    for name in names:
        source = name.split("_", 2)[1]
        if name.endswith("_noise"):
            library.register(
                name,
                "bridge",
                f"{source}->{target_freq} noise-to-signal",
                lambda d, s=source, t=target_freq: -_safe_div(
                    _hf_any(d, s, t, "std", ["ret_1", "ret_1d"]),
                    _hf_any(d, s, t, "mean", ["ret_1", "ret_1d"]).abs() + EPS,
                ),
                freq=target_freq,
            )
        elif name.endswith("_trend_carry"):
            library.register(
                name,
                "bridge",
                f"{source}->{target_freq} trend carry",
                lambda d, s=source, t=target_freq: _hf_any(d, s, t, "last", ["ret_3", "ret_3d", "ret_5d"])
                - _hf_any(d, s, t, "mean", ["ret_3", "ret_3d", "ret_5d"]),
                freq=target_freq,
            )
        elif name.endswith("_liq_pulse"):
            library.register(
                name,
                "bridge",
                f"{source}->{target_freq} liquidity pulse",
                lambda d, s=source, t=target_freq: _safe_div(
                    _hf_any(d, s, t, "last", ["amount_ratio_12", "amount"]) - _hf_any(d, s, t, "mean", ["amount_ratio_12", "amount"]),
                    _hf_any(d, s, t, "mean", ["amount_ratio_12", "amount"]).abs() + EPS,
                ),
                freq=target_freq,
            )
        elif name.endswith("_vol_comp"):
            library.register(
                name,
                "bridge",
                f"{source}->{target_freq} volatility compression",
                lambda d, s=source, t=target_freq: -_hf_any(d, s, t, "std", ["rv_12", "realized_vol_20"]),
                freq=target_freq,
            )
        elif name.endswith("_close_disp"):
            library.register(
                name,
                "bridge",
                f"{source}->{target_freq} close dispersion",
                lambda d, s=source, t=target_freq: -_safe_div(
                    _hf_any(d, s, t, "std", ["close"]),
                    _hf_any(d, s, t, "mean", ["close"]).abs() + EPS,
                ),
                freq=target_freq,
            )
        elif name.endswith("_close_dev"):
            library.register(
                name,
                "bridge",
                f"{source}->{target_freq} close deviation",
                lambda d, s=source, t=target_freq: _safe_div(
                    _hf_any(d, s, t, "last", ["close"]) - _hf_any(d, s, t, "mean", ["close"]),
                    _hf_any(d, s, t, "mean", ["close"]).abs() + EPS,
                ),
                freq=target_freq,
            )
        elif name.endswith("_hilo_spread"):
            library.register(
                name,
                "bridge",
                f"{source}->{target_freq} high-low spread",
                lambda d, s=source, t=target_freq: _safe_div(
                    _hf_any(d, s, t, "mean", ["high"]) - _hf_any(d, s, t, "mean", ["low"]),
                    _hf_any(d, s, t, "mean", ["close"]).abs() + EPS,
                ),
                freq=target_freq,
            )
        elif name.endswith("_crowd_unwind"):
            library.register(
                name,
                "bridge",
                f"{source}->{target_freq} crowding unwind",
                lambda d, s=source, t=target_freq: -_hf_any(d, s, t, "last", ["crowding_proxy_raw"]) * _hf_any(d, s, t, "last", ["ret_1", "ret_1d"]),
                freq=target_freq,
            )
        elif name.endswith("_liq_trend_sync"):
            library.register(
                name,
                "bridge",
                f"{source}->{target_freq} liquidity-trend sync",
                lambda d, s=source, t=target_freq: _safe_div(
                    _hf_any(d, s, t, "last", ["amount_ratio_12", "amount"]) - _hf_any(d, s, t, "mean", ["amount_ratio_12", "amount"]),
                    _hf_any(d, s, t, "mean", ["amount_ratio_12", "amount"]).abs() + EPS,
                )
                * _hf_any(d, s, t, "last", ["ret_1", "ret_1d"]),
                freq=target_freq,
            )
        elif name.endswith("_micro_shift"):
            library.register(
                name,
                "bridge",
                f"{source}->{target_freq} micro regime shift",
                lambda d, s=source, t=target_freq: _hf_any(d, s, t, "std", ["ret_3", "ret_3d", "ret_5d"])
                - _hf_any(d, s, t, "std", ["ret_1", "ret_1d"]),
                freq=target_freq,
            )
        elif name.endswith("_ret3_std"):
            library.register(
                name,
                "bridge",
                f"{source}->{target_freq} ret3 std",
                lambda d, s=source, t=target_freq: -_hf_any(d, s, t, "std", ["ret_3", "ret_3d", "ret_5d"]),
                freq=target_freq,
            )
        elif name.endswith("_liq_pressure"):
            library.register(
                name,
                "bridge",
                f"{source}->{target_freq} liquidity pressure",
                lambda d, s=source, t=target_freq: -(
                    _hf_any(d, s, t, "last", ["amount_ratio_12", "amount"]) - _hf_any(d, s, t, "mean", ["amount_ratio_12", "amount"])
                ).abs(),
                freq=target_freq,
            )
    return names


def _multiscale_factor_names(target_freq: str) -> List[str]:
    src = _bridge_sources_for_target(target_freq)
    out: List[str] = []
    for i in range(len(src)):
        for j in range(i + 1, len(src)):
            fast, slow = src[i], src[j]
            out.extend(
                [
                    f"ms_{fast}_vs_{slow}_trend",
                    f"ms_{fast}_vs_{slow}_liq",
                    f"ms_{fast}_vs_{slow}_noise",
                    f"ms_{fast}_vs_{slow}_close",
                ]
            )
    return _uniq(out)


def _register_multiscale_factor_pack(library: FactorLibrary, target_freq: str) -> List[str]:
    names = _multiscale_factor_names(target_freq)
    for name in names:
        tail = name[len("ms_") :]
        fs, metric = tail.rsplit("_", 1)
        fast, slow = fs.split("_vs_")
        if metric == "trend":
            library.register(
                name,
                "multiscale",
                f"{fast} vs {slow} trend",
                lambda d, f=fast, s=slow, t=target_freq: _hf_any(d, f, t, "last", ["ret_1", "ret_1d"]) - _hf_any(d, s, t, "last", ["ret_1", "ret_1d"]),
                freq=target_freq,
            )
        elif metric == "liq":
            library.register(
                name,
                "multiscale",
                f"{fast} vs {slow} liquidity",
                lambda d, f=fast, s=slow, t=target_freq: (
                    _safe_div(
                        _hf_any(d, f, t, "last", ["amount_ratio_12", "amount"]) - _hf_any(d, f, t, "mean", ["amount_ratio_12", "amount"]),
                        _hf_any(d, f, t, "mean", ["amount_ratio_12", "amount"]).abs() + EPS,
                    )
                    - _safe_div(
                        _hf_any(d, s, t, "last", ["amount_ratio_12", "amount"]) - _hf_any(d, s, t, "mean", ["amount_ratio_12", "amount"]),
                        _hf_any(d, s, t, "mean", ["amount_ratio_12", "amount"]).abs() + EPS,
                    )
                ),
                freq=target_freq,
            )
        elif metric == "noise":
            library.register(
                name,
                "multiscale",
                f"{fast} vs {slow} noise",
                lambda d, f=fast, s=slow, t=target_freq: (
                    _safe_div(_hf_any(d, s, t, "std", ["ret_1", "ret_1d"]), _hf_any(d, s, t, "mean", ["ret_1", "ret_1d"]).abs() + EPS)
                    - _safe_div(_hf_any(d, f, t, "std", ["ret_1", "ret_1d"]), _hf_any(d, f, t, "mean", ["ret_1", "ret_1d"]).abs() + EPS)
                ),
                freq=target_freq,
            )
        elif metric == "close":
            library.register(
                name,
                "multiscale",
                f"{fast} vs {slow} close deviation",
                lambda d, f=fast, s=slow, t=target_freq: (
                    _safe_div(
                        _hf_any(d, f, t, "last", ["close"]) - _hf_any(d, f, t, "mean", ["close"]),
                        _hf_any(d, f, t, "mean", ["close"]).abs() + EPS,
                    )
                    - _safe_div(
                        _hf_any(d, s, t, "last", ["close"]) - _hf_any(d, s, t, "mean", ["close"]),
                        _hf_any(d, s, t, "mean", ["close"]).abs() + EPS,
                    )
                ),
                freq=target_freq,
            )
    return names


def _generated_columns_by_freq(freq: str) -> Dict[str, List[str]]:
    f = str(freq)
    if f == "D":
        return {
            "trend": [
                "ret_1d",
                "ret_3d",
                "ret_5d",
                "ret_10d",
                "ret_20d",
                "ma_gap_5",
                "ma_gap_10",
                "ma_gap_20",
                "ma_cross_5_20",
                "breakout_20",
                "open_to_close_intraday",
                "close_to_vwap_day",
            ],
            "reversal": [
                "ret_1d",
                "ret_3d",
                "ret_5d",
                "overnight_gap",
                "body_ratio",
                "close_pos",
                "morning_momentum_30m",
                "last30_momentum",
                "vwap30_vs_day",
                "signed_vol_imbalance_5m",
            ],
            "liquidity": [
                "vol_ratio_5",
                "vol_ratio_20",
                "amount_ratio_20",
                "turn_ratio_5",
                "amihud_20",
                "ret_vol_corr_20",
                "crowding_proxy_raw",
                "signed_vol_imbalance_5m",
                "barra_liquidity_proxy",
            ],
            "volatility": [
                "atr_norm_14",
                "realized_vol_20",
                "downside_vol_ratio_20",
                "intraday_range",
                "jump_ratio_5m",
                "minute_realized_vol_5m",
                "minute_ret_kurt_5m",
                "barra_volatility_proxy",
            ],
            "structure": [
                "close_to_vwap_day",
                "vwap30_vs_day",
                "morning_momentum_30m",
                "last30_momentum",
                "minute_up_ratio_5m",
                "minute_ret_skew_5m",
                "open_to_close_intraday",
                "overnight_gap",
                "rsi14",
            ],
            "context": [
                "barra_size_proxy",
                "barra_momentum_proxy",
                "barra_volatility_proxy",
                "barra_liquidity_proxy",
                "barra_beta_proxy",
                "ret_20d",
                "realized_vol_20",
                "amihud_20",
                "crowding_proxy_raw",
            ],
        }
    base = [
        "ret_1",
        "ret_3",
        "ret_6",
        "ret_12",
        "ma_gap_6",
        "ma_gap_12",
        "rv_12",
        "range_norm",
        "amount_ratio_12",
        "vol_chg_1",
        "crowding_proxy_raw",
        "context_trend_20d",
        "context_quality_20d",
        "context_liquidity_20d",
        "context_intraday_mood",
    ]
    return {
        "trend": base[:10],
        "reversal": [base[i] for i in [0, 1, 2, 5, 9, 10, 14, 11]],
        "liquidity": [base[i] for i in [8, 9, 10, 13, 14, 11, 12]],
        "volatility": [base[i] for i in [6, 7, 12, 14, 2, 3, 10]],
        "structure": [base[i] for i in [0, 1, 4, 5, 7, 8, 9, 10, 14]],
        "context": [base[i] for i in [11, 12, 13, 14, 3, 6, 8]],
    }


def _core_price_volume_package_raw_names_for_freq(freq: str) -> Dict[str, List[str]]:
    f = str(freq)
    if f == "D":
        return {
            "trend": [
                "mom_5",
                "mom_10",
                "mom_20",
                "ma_gap_5",
                "ma_gap_10",
                "ma_gap_20",
                "ma_cross_5_20",
                "breakout_20",
                "trend_shock_5_20",
                "momentum_quality_20",
                "breakout_strength_20",
                "trend_vol_regime",
            ],
            "reversal": [
                "rev_1",
                "rev_3",
                "amihud_reversal_20",
            ],
            "liquidity": [
                "vol_ratio_5",
                "vol_ratio_20",
                "amount_ratio_20",
                "turn_ratio_5",
                "amihud_20",
                "liquidity_regime_switch",
            ],
            "volatility": [
                "intraday_range",
                "atr_norm_14",
                "realized_vol_20",
                "downside_vol_ratio_20",
                "vol_asymmetry_alpha",
            ],
            "flow": DAILY_FLOW_FACTORS,
            "crowding": DAILY_CROWDING_FACTORS,
            "price_action": DAILY_PRICE_ACTION_FACTORS,
            "intraday_micro": DAILY_INTRADAY_MICRO_FACTORS,
            "oscillator": DAILY_OSCILLATOR_FACTORS,
            "overnight": DAILY_OVERNIGHT_FACTORS,
            "multi_freq": DAILY_MULTI_FREQ_EXTRA_FACTORS + BRIDGE_PRIMARY_FACTORS + BRIDGE_MULTISCALE_FACTORS,
            "context": [
                "ret_20d",
                "realized_vol_20",
                "amihud_20",
            ],
        }
    if f in {"5min", "15min", "30min", "60min", "120min", "W", "M"}:
        return {
            "trend": [
                "ret_1",
                "ret_3",
                "ret_6",
                "ret_12",
                "ma_gap_6",
                "ma_gap_12",
                "ret_spread_1_6",
                "ret_accel_3_12",
                "momentum_quality_12",
                "trend_efficiency_6",
                "trend_liquidity_combo",
                "vol_adjusted_gap",
                "ret_curve_312",
            ],
            "reversal": [
                "reversal_pressure_1",
            ],
            "liquidity": [
                "amount_ratio_12",
                "liquidity_stress",
            ],
            "volatility": [
                "rv_12",
                "range_norm",
            ],
            "flow": GENERIC_FLOW_FACTORS,
            "crowding": GENERIC_CROWDING_FACTORS,
            "price_action": GENERIC_PRICE_ACTION_FACTORS,
            "intraday_signature": INTRADAY_SIGNATURE_FACTORS if f in INTRADAY_FREQS else [],
            "intraday_micro": MICRO_5MIN_EXTRA_FACTORS if f == "5min" else [],
            "period_signature": PERIOD_SIGNATURE_FACTORS if f in {"W", "M"} else [],
            "multi_freq": BRIDGE_PRIMARY_FACTORS + BRIDGE_MULTISCALE_FACTORS,
            "context": [
                "context_trend_20d",
                "context_quality_20d",
                "context_liquidity_20d",
                "context_intraday_mood",
            ],
        }
    return {}


def _flow_package_names_for_freq(freq: str) -> List[str]:
    f = str(freq)
    if f == "D":
        return _uniq(DAILY_FLOW_FACTORS)
    if f in {"5min", "15min", "30min", "60min", "120min", "W", "M"}:
        return _uniq(GENERIC_FLOW_FACTORS)
    return []


def _crowding_package_names_for_freq(freq: str) -> List[str]:
    f = str(freq)
    if f == "D":
        return _uniq(DAILY_CROWDING_FACTORS)
    if f in {"5min", "15min", "30min", "60min", "120min", "W", "M"}:
        return _uniq(GENERIC_CROWDING_FACTORS)
    return []


def _price_action_package_names_for_freq(freq: str) -> List[str]:
    f = str(freq)
    if f == "D":
        return _uniq(DAILY_PRICE_ACTION_FACTORS)
    if f in {"5min", "15min", "30min", "60min", "120min", "W", "M"}:
        return _uniq(GENERIC_PRICE_ACTION_FACTORS)
    return []


def _intraday_signature_package_names_for_freq(freq: str) -> List[str]:
    return _uniq(INTRADAY_SIGNATURE_FACTORS) if str(freq) in INTRADAY_FREQS else []


def _intraday_micro_package_names_for_freq(freq: str) -> List[str]:
    f = str(freq)
    if f == "D":
        return _uniq(DAILY_INTRADAY_MICRO_FACTORS)
    if f == "5min":
        return _uniq(MICRO_5MIN_EXTRA_FACTORS)
    return []


def _period_signature_package_names_for_freq(freq: str) -> List[str]:
    return _uniq(PERIOD_SIGNATURE_FACTORS) if str(freq) in {"W", "M"} else []


def _oscillator_package_names_for_freq(freq: str) -> List[str]:
    return _uniq(DAILY_OSCILLATOR_FACTORS) if str(freq) == "D" else []


def _overnight_package_names_for_freq(freq: str) -> List[str]:
    return _uniq(DAILY_OVERNIGHT_FACTORS) if str(freq) == "D" else []


def _multi_freq_package_names_for_freq(freq: str) -> List[str]:
    f = str(freq)
    names: List[str] = []
    if f in {"15min", "30min", "60min", "120min", "D", "W", "M"}:
        names.extend(BRIDGE_PRIMARY_FACTORS)
    if f in {"30min", "60min", "120min", "D", "W", "M"}:
        names.extend(BRIDGE_MULTISCALE_FACTORS)
    if f == "D":
        names.extend(DAILY_MULTI_FREQ_EXTRA_FACTORS)
    return _uniq(names)


def _build_default_factor_packs_by_freq() -> Dict[str, Dict[str, List[str]]]:
    freqs = ["5min", "15min", "30min", "60min", "120min", "D", "W", "M"]
    out: Dict[str, Dict[str, List[str]]] = {}
    for f in freqs:
        cols_map = _generated_columns_by_freq(f)
        packs: Dict[str, List[str]] = {}
        for pkg in ["trend", "reversal", "liquidity", "volatility", "structure", "context"]:
            raw_names = _core_price_volume_package_raw_names_for_freq(f).get(pkg, [])
            cols = cols_map.get(pkg, [])
            anchors = cols[:2]
            template_names = _template_generated_factor_names(prefix=f"g_{pkg}", cols=cols, pair_limit=6, anchors=anchors)
            packs[pkg] = _uniq(list(raw_names) + list(template_names))
        extra_pack_builders = {
            "flow": _flow_package_names_for_freq,
            "crowding": _crowding_package_names_for_freq,
            "price_action": _price_action_package_names_for_freq,
            "intraday_signature": _intraday_signature_package_names_for_freq,
            "intraday_micro": _intraday_micro_package_names_for_freq,
            "period_signature": _period_signature_package_names_for_freq,
            "oscillator": _oscillator_package_names_for_freq,
            "overnight": _overnight_package_names_for_freq,
            "multi_freq": _multi_freq_package_names_for_freq,
        }
        for pkg, fn in extra_pack_builders.items():
            names = list(fn(f))
            names.extend(_core_price_volume_package_raw_names_for_freq(f).get(pkg, []))
            if names:
                packs[pkg] = _uniq(names)
        packs["bridge"] = _bridge_factor_names(f)
        packs["multiscale"] = _multiscale_factor_names(f)
        fund_pair_limit = 9 if str(f) in {"D", "W", "M"} else 8
        for pack_name, cat in FUNDAMENTAL_PACK_TO_CATEGORY.items():
            cols = _fundamental_cols_for_category(f, cat)
            packs[pack_name] = _template_generated_factor_names(
                prefix=pack_name,
                cols=cols,
                pair_limit=fund_pair_limit,
                anchors=cols[:3],
            )
        fusion_cols = _fundamental_hf_fusion_cols(f)
        packs["fund_hf_fusion"] = _template_generated_factor_names(
            prefix="fund_hf_fusion",
            cols=fusion_cols,
            pair_limit=fund_pair_limit,
            anchors=fusion_cols[:3],
        )
        text_pair_limit = 8
        for pack_name in TEXT_PACKAGES:
            cols = _text_cols_for_package(f, pack_name)
            if not cols:
                continue
            packs[pack_name] = _template_generated_factor_names(
                prefix=pack_name,
                cols=cols,
                pair_limit=text_pair_limit,
                anchors=cols[:3],
            )
        out[f] = {k: _uniq(v) for k, v in packs.items()}
    return out


DEFAULT_FACTOR_PACKS_BY_FREQ: Dict[str, Dict[str, List[str]]] = _build_default_factor_packs_by_freq()
DEFAULT_FACTOR_SET_BY_FREQ = {
    f: _uniq([x for _, v in DEFAULT_FACTOR_PACKS_BY_FREQ[f].items() for x in v])
    for f in DEFAULT_FACTOR_PACKS_BY_FREQ
}


def list_default_factor_packages(freq: str) -> List[str]:
    return sorted(DEFAULT_FACTOR_PACKS_BY_FREQ.get(str(freq), {}).keys())


def resolve_default_factor_set(freq: str, package_expr: str = "") -> List[str]:
    packs = DEFAULT_FACTOR_PACKS_BY_FREQ.get(str(freq), {})
    if not packs:
        return []
    expr = str(package_expr).strip()
    if not expr:
        return _uniq([x for _, vals in packs.items() for x in vals])

    req = [x.strip() for x in expr.split(",") if x.strip()]
    req_norm = [x.lower() for x in req]
    if "all" in req_norm:
        return _uniq([x for _, vals in packs.items() for x in vals])

    avail_map = {k.lower(): k for k in packs.keys()}
    missing = [x for x in req_norm if x not in avail_map]
    if missing:
        raise ValueError(f"unknown factor package(s) for freq={freq}: {missing}; available={sorted(packs.keys())}")

    chosen: List[str] = []
    for k in req_norm:
        chosen.extend(packs[avail_map[k]])
    return _uniq(chosen)


_PACKAGE_PRIMARY_PRIORITY: List[str] = [
    "trend",
    "reversal",
    "liquidity",
    "volatility",
    "structure",
    "context",
    "flow",
    "crowding",
    "price_action",
    "intraday_signature",
    "intraday_micro",
    "period_signature",
    "oscillator",
    "overnight",
    "multi_freq",
    "bridge",
    "multiscale",
    "text_sentiment",
    "text_attention",
    "text_event",
    "text_topic",
    "text_fusion",
    "fund_growth",
    "fund_valuation",
    "fund_profitability",
    "fund_quality",
    "fund_leverage",
    "fund_cashflow",
    "fund_efficiency",
    "fund_expectation",
    "fund_hf_fusion",
]


def build_factor_package_index(freq: str) -> Dict[str, List[str]]:
    packs = DEFAULT_FACTOR_PACKS_BY_FREQ.get(str(freq), {})
    out: Dict[str, List[str]] = {}
    for pkg, factors in packs.items():
        for fac in factors:
            out.setdefault(str(fac), []).append(str(pkg))
    return {k: _uniq(v) for k, v in out.items()}


def _fallback_package_from_category(category: str) -> str:
    c = str(category or "").strip()
    if not c:
        return "other_custom"
    if c.startswith("fundamental_"):
        suffix = c[len("fundamental_") :]
        return f"fund_{suffix}"
    if c in {
        "trend",
        "reversal",
        "liquidity",
        "volatility",
        "structure",
        "context",
        "flow",
        "crowding",
        "price_action",
        "intraday_signature",
        "intraday_micro",
        "period_signature",
        "oscillator",
        "overnight",
        "multi_freq",
        "bridge",
        "multiscale",
        "text_sentiment",
        "text_attention",
        "text_event",
        "text_topic",
        "text_fusion",
    }:
        return c
    if c == "mined_factor":
        return "catalog_custom"
    if c == "auto_panel":
        return "auto_panel"
    return "other_custom"


def resolve_primary_factor_package(
    *,
    freq: str,
    factor: str,
    category: str = "",
    package_index: Dict[str, List[str]] | None = None,
) -> tuple[str, List[str]]:
    idx = package_index if package_index is not None else build_factor_package_index(freq)
    memberships = list(idx.get(str(factor), []))
    if memberships:
        pri_map = {p: i for i, p in enumerate(_PACKAGE_PRIMARY_PRIORITY)}
        memberships = sorted(memberships, key=lambda p: pri_map.get(p, 10_000))
        return memberships[0], memberships
    fb = _fallback_package_from_category(category)
    return fb, [fb]


def register_default_factors(library: FactorLibrary) -> None:
    register_default_daily_factors(library)
    for freq in ["5min", "15min", "30min", "60min", "120min", "W", "M"]:
        register_generic_freq_factors(library, freq)

    for freq in sorted(INTRADAY_FREQS):
        register_intraday_signature_factors(library, freq=freq)
    register_5min_specific_factors(library)
    for freq in ["W", "M"]:
        register_period_signature_factors(library, freq=freq)

    primary_source_map: Dict[str, str] = {
        "15min": "5min",
        "30min": "15min",
        "60min": "30min",
        "120min": "60min",
        "D": "120min",
        "W": "D",
        "M": "W",
    }
    for target, source in primary_source_map.items():
        register_primary_bridge_factors(library, target_freq=target, source_freq=source)

    multiscale_map: Dict[str, tuple[str, str]] = {
        "30min": ("5min", "15min"),
        "60min": ("5min", "30min"),
        "120min": ("15min", "60min"),
        "D": ("5min", "120min"),
        "W": ("30min", "D"),
        "M": ("D", "W"),
    }
    for target, (fast_source, slow_source) in multiscale_map.items():
        register_multiscale_bridge_factors(library, target_freq=target, fast_source=fast_source, slow_source=slow_source)

    for freq in ["5min", "15min", "30min", "60min", "120min", "D", "W", "M"]:
        cols_map = _generated_columns_by_freq(freq)
        for pkg in ["trend", "reversal", "liquidity", "volatility", "structure", "context"]:
            cols = cols_map.get(pkg, [])
            _register_template_generated_package(
                library=library,
                freq=freq,
                package=f"g_{pkg}",
                category=pkg,
                cols=cols,
                pair_limit=6,
                anchors=cols[:2],
            )
        fund_pair_limit = 9 if str(freq) in {"D", "W", "M"} else 8
        for pack_name, cat in FUNDAMENTAL_PACK_TO_CATEGORY.items():
            fund_cols = _fundamental_cols_for_category(freq, cat)
            _register_template_generated_package(
                library=library,
                freq=freq,
                package=pack_name,
                category=f"fundamental_{cat}",
                cols=fund_cols,
                pair_limit=fund_pair_limit,
                anchors=fund_cols[:3],
            )
        fusion_cols = _fundamental_hf_fusion_cols(freq)
        _register_template_generated_package(
            library=library,
            freq=freq,
            package="fund_hf_fusion",
            category="fundamental_hf_fusion",
            cols=fusion_cols,
            pair_limit=fund_pair_limit,
            anchors=fusion_cols[:3],
        )
        text_pair_limit = 8
        for pack_name in TEXT_PACKAGES:
            text_cols = _text_cols_for_package(freq, pack_name)
            if not text_cols:
                continue
            _register_template_generated_package(
                library=library,
                freq=freq,
                package=pack_name,
                category=pack_name,
                cols=text_cols,
                pair_limit=text_pair_limit,
                anchors=text_cols[:3],
            )
        _register_bridge_factor_pack(library, target_freq=freq)
        _register_multiscale_factor_pack(library, target_freq=freq)
