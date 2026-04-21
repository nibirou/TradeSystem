"""Template: custom factor plugin for Strategy7.

Usage:
python run_strategy7.py --custom-factor-py ./strategy7/plugins/custom_factor_template.py
"""

from __future__ import annotations

from strategy7.factors.base import register_external_factor_table


def register_factors(library) -> None:
    # Example 1) Manual custom factors.
    # Category should use existing factor-package categories, not a standalone custom category.
    # Name is recommended to start with `custom_`.
    library.register(
        name="custom_ret_accel_5_20",
        category="trend",
        description="ret_5d - ret_20d",
        func=lambda d: d["ret_5d"] - d["ret_20d"],
        freq="D",
    )

    # Example 2) Intraday custom factor.
    library.register(
        name="custom_intraday_mr",
        category="intraday_micro",
        description="-ret_1 + 0.5*ret_3",
        func=lambda d: -d["ret_1"] + 0.5 * d["ret_3"],
        freq="15min",
    )

    # Example 3) Register external table columns as factors.
    # The source table should contain columns: date/code/(or datetime) + numeric factor columns.
    # Replace path/category/freq as needed.
    # register_external_factor_table(
    #     library,
    #     path="D:/PythonProject/Quant/data_baostock/custom_factors/my_daily_factors.csv",
    #     freq="D",
    #     category="fund_quality",
    #     name_prefix="custom_ext",
    #     date_col="date",
    #     code_col="code",
    #     file_format="auto",
    # )
