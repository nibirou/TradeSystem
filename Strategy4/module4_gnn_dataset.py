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

import math
import numpy as np
import pandas as pd

def build_ret_vol_labels(
    daily: pd.DataFrame,
    trade_dates: pd.DatetimeIndex,
    horizon: int,
    min_valid_ratio: float = 0.8,
    ffill_limit: int | None = None,   # 连续停牌天数很长时可限制 ffill
) -> pd.DataFrame:
    """
    输入:
      daily: 至少含 [date, code, close]
      trade_dates: 交易日序列（DatetimeIndex）
    输出:
      [date, code, ret1, ret_fwd, vol_fwd]

    核心处理:
      1) 交易日网格补齐 (date x code)
      2) close<=0 或缺失 -> NaN -> 按 code ffill（停牌期间价格保持不变）
      3) ret1/ret_fwd/vol_fwd 用 log close 与未来窗口 t+1..t+H 对齐
      4) min_periods 控制未来窗口的有效占比
    """
    df = daily[["date", "code", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    trade_dates = pd.DatetimeIndex(pd.to_datetime(trade_dates)).sort_values()

    # 只保留 trade_dates 范围内的数据（避免网格爆炸）
    df = df[df["date"].isin(trade_dates)]

    # close 清洗：非法值先置 NaN（包括 <=0, inf）
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df.loc[df["close"] <= 0, "close"] = np.nan
    df["close"] = df["close"].replace([np.inf, -np.inf], np.nan)

    # --- 1) 交易日网格补齐 ---
    codes = df["code"].astype(str).unique()
    full_index = pd.MultiIndex.from_product([trade_dates, codes], names=["date", "code"])
    df = df.set_index(["date", "code"]).reindex(full_index).reset_index()

    # --- 2) 对每只股票 forward-fill close ---
    # 语义：停牌/缺数 -> 价格保持不变 -> 当天收益=0
    df["close"] = df.groupby("code")["close"].ffill(limit=ffill_limit)

    # 注意：如果某股票最开始若干天都缺失，ffill 仍是 NaN，这种股票会自然在 mask 中被过滤
    # 你也可以选择 bfill 一次，但会引入“未来信息”，不建议用于训练标签
    # df["close"] = df.groupby("code")["close"].apply(lambda s: s.ffill().bfill())  # ❌ 不建议

    # --- 3) 计算 ret1 / ret_fwd / vol_fwd ---
    df = df.sort_values(["code", "date"])

    df["log_close"] = np.log(df["close"])
    df["ret1"] = df.groupby("code")["log_close"].diff()
    df["ret1"] = df["ret1"].replace([np.inf, -np.inf], np.nan)

    H = int(horizon)
    min_periods = max(2, int(math.ceil(H * min_valid_ratio)))

    # 未来窗口：t+1..t+H
    fut = df.groupby("code")["ret1"].shift(-1)

    df["ret_fwd"] = (
        fut.groupby(df["code"])
           .rolling(H, min_periods=min_periods)
           .sum()
           .shift(-(H - 1))
           .reset_index(level=0, drop=True)
    )

    df["vol_fwd"] = (
        fut.groupby(df["code"])
           .rolling(H, min_periods=min_periods)
           .std(ddof=0)
           .shift(-(H - 1))
           .reset_index(level=0, drop=True)
           * math.sqrt(252)
    )

    return df[["date", "code", "ret1", "ret_fwd", "vol_fwd"]]


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
    labels = build_ret_vol_labels(
        daily=daily,
        trade_dates=trade_dates,
        horizon=cfg.horizon,
        min_valid_ratio=0.8,
        ffill_limit=None,   # 或者 30/60，限制超长停牌的“价格不变”天数
    )

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
