"""Template: custom factor plugin for Strategy7.

Usage:
python run_strategy7.py --custom-factor-py ./strategy7/plugins/custom_factor_template.py
"""

from __future__ import annotations


def register_factors(library) -> None:
    # Daily custom factors
    library.register(
        name="custom_ret_accel_5_20",
        category="custom_trend",
        description="ret_5d - ret_20d",
        func=lambda d: d["ret_5d"] - d["ret_20d"],
        freq="D",
    )

    # Intraday generic factors (example: 15min)
    library.register(
        name="custom_intraday_mr",
        category="custom_micro",
        description="-ret_1 + 0.5*ret_3",
        func=lambda d: -d["ret_1"] + 0.5 * d["ret_3"],
        freq="15min",
    )

