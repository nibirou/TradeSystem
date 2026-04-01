"""Template: custom external source plugin for Strategy7.

Usage:
python run_strategy7.py --extra-source-module ./strategy7/plugins/custom_source_template.py
"""

from __future__ import annotations

from strategy7.data.sources import TableFileSource


def register_sources(registry) -> None:
    # Example 1: external NLP signal table with columns [date, code, sentiment_score, attention_score]
    registry.register(
        "nlp_sentiment",
        TableFileSource(
            name="nlp_sentiment",
            path="./Data/News/text_factor_daily.csv",
            date_col="date",
            code_col="code",
            prefix="nlp",
            file_format="auto",
        ),
    )

    # Example 2: mined alpha factor table [date, code, alpha_xxx...]
    registry.register(
        "mined_alpha",
        TableFileSource(
            name="mined_alpha",
            path="./Data/Trading/mined_alpha_daily.csv",
            date_col="date",
            code_col="code",
            prefix="mine",
            file_format="auto",
        ),
    )

