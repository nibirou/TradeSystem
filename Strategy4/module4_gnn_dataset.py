# module_gnn_dataset.py
# 输入：daily, trade_dates, factor_df（就跟你 StatBacktester 用的一样）
# 输出：每个日期一个样本：X/A/y_ret/y_vol/codes

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict

from module4_gnn_graphbuilder import build_corr_graph

@dataclass
class GNNDataConfig:
    lookback_corr: int = 60
    horizon: int = 20
    topk_graph: int = 10
    min_nodes: int = 80

def _logret(close: pd.Series) -> pd.Series:
    return np.log(close).diff()

def build_ret_vol_labels(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    daily: [date, code, close] 至少要有
    输出：新增 ret_fwd, vol_fwd（按 code 计算）
    """
    df = daily[["date","code","close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code","date"])
    df["log_close"] = np.log(df["close"])
    df["ret1"] = df.groupby("code")["log_close"].diff()

    H = horizon
    df["ret_fwd"] = df.groupby("code")["log_close"].shift(-H) - df["log_close"]
    df["vol_fwd"] = (
        df.groupby("code")["ret1"]
          .rolling(H)
          .std()
          .shift(-(H-1))
          .reset_index(level=0, drop=True)
          * math.sqrt(252)
    )
    return df[["date","code","ret1","ret_fwd","vol_fwd"]]

def make_gnn_samples(
    trade_dates: pd.DatetimeIndex,
    factor_df: pd.DataFrame,
    daily: pd.DataFrame,
    cfg: GNNDataConfig
) -> List[Dict]:
    """
    每个交易日 t 生成一个图样本
    """
    trade_dates = pd.DatetimeIndex(pd.to_datetime(trade_dates)).sort_values()

    # labels（收益/波动）
    labels = build_ret_vol_labels(daily, cfg.horizon)

    # 合并到因子面板（保持你习惯的 [date, code, factors...]）
    panel = factor_df.merge(labels, on=["date","code"], how="left")
    panel = panel.sort_values(["date","code"])

    # 因子列
    feat_cols = [c for c in panel.columns if c not in ("date","code","ret1","ret_fwd","vol_fwd")]
    feat_cols = [c for c in feat_cols if panel[c].notna().any()]

    # pivot：ret1 用于构图
    ret_pv = panel.pivot_table(index="date", columns="code", values="ret1").reindex(trade_dates)

    samples = []
    L = cfg.lookback_corr
    H = cfg.horizon

    for i in range(L, len(trade_dates) - H):
        t = trade_dates[i]
        win_dates = trade_dates[i-L:i]

        xt = panel[panel["date"] == t].set_index("code")
        codes = sorted(xt.index.unique().tolist())
        if len(codes) < cfg.min_nodes:
            continue

        R = ret_pv.loc[win_dates, codes].T.values  # [N,L]
        A = build_corr_graph(R, cfg.topk_graph)

        X = xt.loc[codes, feat_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)

        y_ret = xt.loc[codes, "ret_fwd"].values.astype(np.float32)
        y_vol = xt.loc[codes, "vol_fwd"].values.astype(np.float32)
        mask = np.isfinite(y_ret) & np.isfinite(y_vol)
        if mask.sum() < max(30, int(0.3 * len(codes))):
            continue

        samples.append({
            "date": t,
            "codes": codes,
            "X": X,
            "A": A,
            "y_ret": y_ret,
            "y_vol": y_vol,
            "mask": mask.astype(np.float32),
            "feat_cols": feat_cols
        })

    return samples
