"""Artifact persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from ..core.utils import ensure_dir


def build_run_tag(
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    horizon: int,
    execution_scheme: str,
    factor_freq: str,
    board_tag: str,
    portfolio_mode: str,
) -> str:
    return (
        f"tr_{train_start}_{train_end}"
        f"_te_{test_start}_{test_end}"
        f"_f{factor_freq}"
        f"_h{horizon}_{execution_scheme}_{board_tag}_{portfolio_mode}"
    )


def save_dataframe(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_common_artifacts(
    output_dir: Path,
    run_tag: str,
    pred_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    factor_ic_summary_df: pd.DataFrame,
    factor_ic_series_df: pd.DataFrame,
    model_ic_series_df: pd.DataFrame,
    factor_meta_df: pd.DataFrame,
) -> Dict[str, Path]:
    ensure_dir(output_dir)
    files = {
        "predictions_csv": output_dir / f"predictions_{run_tag}.csv",
        "trades_csv": output_dir / f"trades_{run_tag}.csv",
        "positions_csv": output_dir / f"positions_{run_tag}.csv",
        "curve_csv": output_dir / f"curve_{run_tag}.csv",
        "factor_ic_summary_csv": output_dir / f"factor_ic_summary_{run_tag}.csv",
        "factor_ic_series_csv": output_dir / f"factor_ic_series_{run_tag}.csv",
        "model_ic_series_csv": output_dir / f"model_ic_series_{run_tag}.csv",
        "factor_library_csv": output_dir / f"factor_library_{run_tag}.csv",
    }
    save_dataframe(files["predictions_csv"], pred_df)
    save_dataframe(files["trades_csv"], trades_df)
    save_dataframe(files["positions_csv"], positions_df)
    save_dataframe(files["curve_csv"], curve_df)
    save_dataframe(files["factor_ic_summary_csv"], factor_ic_summary_df)
    save_dataframe(files["factor_ic_series_csv"], factor_ic_series_df)
    save_dataframe(files["model_ic_series_csv"], model_ic_series_df)
    save_dataframe(files["factor_library_csv"], factor_meta_df)
    return files

