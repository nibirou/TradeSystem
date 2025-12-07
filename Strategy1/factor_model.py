# factor_model.py
import numpy as np
import pandas as pd
from typing import Dict, List

from config import IC_LOOKBACK, IC_MIN_DATES


def compute_cross_sectional_ic(
    df: pd.DataFrame,
    factor_col: str,
    ret_col: str = "ret_fwd_1",
) -> float:
    """
    计算整个样本期内某个因子的截面IC（Spearman相关）。
    一般用在滚动窗口中。
    """
    sub = df[[factor_col, ret_col]].dropna()
    if len(sub) < 20:
        return np.nan
    corr = sub.corr(method="spearman").iloc[0, 1]
    return corr


def compute_rolling_factor_weights(
    df: pd.DataFrame,
    factor_cols: List[str],
    ret_col: str = "ret_fwd_1",
    lookback: int = IC_LOOKBACK,
    min_dates: int = IC_MIN_DATES,
) -> pd.DataFrame:
    """
    滚动截面IC → 因子权重：
      - 对于每个交易日date_t，用之前 lookback 个截面（date_{t-lookback}..date_{t-1}）的截面样本，
        计算每个因子的Spearman IC；
      - 然后按 |IC| 归一化为权重；
      - 结果返回 DataFrame: [date, w_factor1, w_factor2, ...]
    """
    df = df.copy()
    if ret_col not in df.columns:
        raise ValueError(f"ret_col {ret_col} not in df")

    dates = sorted(df["date"].unique())
    records = []

    for i, d in enumerate(dates):
        # 过去的截面日期
        past_dates = dates[max(0, i - lookback): i]
        if len(past_dates) < min_dates:
            # 历史太短，先不给权重，后面用最近有效权重填充
            continue

        df_past = df[df["date"].isin(past_dates)]
        weights = {}
        for col in factor_cols:
            ic = compute_cross_sectional_ic(df_past, col, ret_col=ret_col)
            weights[col] = ic

        w = pd.Series(weights)
        # 绝对值归一化，避免正负抵消
        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if w.abs().sum() == 0:
            # 若所有IC都接近0，则均匀权重
            w[:] = 1.0 / len(w)
        else:
            w = w / w.abs().sum()

        rec = {"date": d}
        for col in factor_cols:
            rec[f"w_{col}"] = w[col]
        records.append(rec)

    weight_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    # 前期没有权重的日期，可以用第一个有权重的日期的权重向前填充
    if not weight_df.empty:
        first_date = weight_df["date"].min()
        # 对于更早的日期，用 first_date 对应的权重填充
        early_dates = [d for d in dates if d < first_date]
        if early_dates:
            first_row = weight_df.iloc[0].to_dict()
            for d in early_dates:
                r = first_row.copy()
                r["date"] = d
                weight_df = pd.concat([pd.DataFrame([r]), weight_df], ignore_index=True)
        weight_df = weight_df.sort_values("date").reset_index(drop=True)

    return weight_df


def apply_factor_weights(
    df: pd.DataFrame, factor_cols: List[str], weight_df: pd.DataFrame
) -> pd.DataFrame:
    """
    将滚动因子权重与因子值结合，生成 score：
      score_{i,t} = sum_j w_{j,t} * factor_{j,i,t}
    """
    merged = df.merge(weight_df, on="date", how="left")
    merged = merged.sort_values(["date", "code"]).reset_index(drop=True)

    # 线性组合生成 score
    def _compute_score(row):
        s = 0.0
        for col in factor_cols:
            w_col = f"w_{col}"
            w = row.get(w_col, np.nan)
            f = row.get(col, np.nan)
            if np.isnan(w) or np.isnan(f):
                continue
            s += w * f
        return s

    merged["score"] = merged.apply(_compute_score, axis=1)
    return merged