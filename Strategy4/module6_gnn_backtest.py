# module_gnn_backtest.py（保持你输出风格：给每日选股列表 + 简洁回测）
import math
import numpy as np
import pandas as pd
from typing import Optional

# 收益率/波动率打分选股 的持有期收益回测（标准、和GNN目标一致）
# 复用你的 LabelBuilder：把“选中的股票”丢给 LabelBuilder.build_labels()，
# 统计胜率（跟你现在 StatBacktester 很像）

def make_rebalance_dates(trade_dates: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    td = pd.DatetimeIndex(pd.to_datetime(trade_dates)).sort_values()
    if freq == "M":
        return pd.DatetimeIndex(pd.Series(td).groupby(pd.Series(td).dt.to_period("M")).max().values)
    if freq == "W":
        return pd.DatetimeIndex(pd.Series(td).groupby(pd.Series(td).dt.to_period("W")).max().values)
    raise ValueError(freq)

def backtest_score_long_only(
    daily: pd.DataFrame,
    pred: pd.DataFrame,
    trade_dates: pd.DatetimeIndex,
    start_date: str,
    end_date: str,
    rebalance: str = "M",
    topN: int = 50,
    lam_vol: float = 0.3,
    vol_filter_q: float = 0.9,
):
    px = daily[["date","code","close"]].copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["date","code"])
    close_pv = px.pivot_table(index="date", columns="code", values="close")

    td = pd.DatetimeIndex(pd.to_datetime(trade_dates)).sort_values()
    rebal = make_rebalance_dates(td, rebalance)
    rebal = rebal[(rebal >= pd.to_datetime(start_date)) & (rebal <= pd.to_datetime(end_date))]
    if len(rebal) < 2:
        return pd.DataFrame()

    pred = pred.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    pred["score"] = pred["pred_ret"] - lam_vol * pred["pred_vol"]

    rows = []
    for i in range(len(rebal)-1):
        t, t2 = rebal[i], rebal[i+1]
        if t not in close_pv.index or t2 not in close_pv.index:
            continue

        sig = pred[pred["date"] == t].copy()
        if sig.empty:
            continue

        vcut = sig["pred_vol"].quantile(vol_filter_q) if sig["pred_vol"].notna().any() else np.inf
        sig = sig[sig["pred_vol"] <= vcut]
        sig = sig.sort_values("score", ascending=False).head(topN)

        picks = sig["code"].tolist()
        print("picks", picks)
        if not picks:
            continue

        p0 = close_pv.loc[t, picks]
        p1 = close_pv.loc[t2, picks]
        valid = p0.notna() & p1.notna()
        if valid.sum() == 0:
            continue

        ret = (p1[valid] / p0[valid] - 1.0).mean()
        rows.append({"date": t2, "ret": float(ret), "n": int(valid.sum())})

    bt = pd.DataFrame(rows)
    print(bt)
    if bt.empty:
        return bt
    bt["cum"] = (1 + bt["ret"]).cumprod()

    ann = bt["ret"].mean() * (12 if rebalance == "M" else 52)
    vol = bt["ret"].std() * math.sqrt(12 if rebalance == "M" else 52)
    sharpe = ann / (vol + 1e-12)
    print(f"[BT] ann={ann:.4f} vol={vol:.4f} sharpe={sharpe:.3f} periods={len(bt)}")
    return bt

def stat_winrate_with_labelbuilder(
    pred: pd.DataFrame,
    label_builder,
    trade_dates: pd.DatetimeIndex,
    start_date: str,
    end_date: str,
    topk: int = 30,
    horizon_n: int = 10,
    stop_loss: float = -0.10,
    lam_vol: float = 0.3,
):
    """
    用你的 LabelBuilder 评估：每个调仓日选 topk，看 y 的均值/胜率
    """
    td = pd.DatetimeIndex(pd.to_datetime(trade_dates)).sort_values()
    dates = td[(td >= pd.to_datetime(start_date)) & (td <= pd.to_datetime(end_date))]

    pred = pred.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    pred["score"] = pred["pred_ret"] - lam_vol * pred["pred_vol"]

    res = []
    for t in dates:
        sig = pred[pred["date"] == t].sort_values("score", ascending=False).head(topk)
        if sig.empty:
            continue
        picks = sig["code"].tolist()

        y_df = label_builder.build_labels(t, horizon_n=horizon_n, stop_loss=stop_loss)
        if y_df.empty:
            continue

        hit = y_df.loc[y_df.index.intersection(picks)]
        if hit.empty:
            continue

        res.append({
            "date": t,
            "TopK_mean_y": float(hit["y"].mean()),
            "TopK_winrate": float((hit["y"] > 0).mean()),
            "N": int(len(hit)),
        })

    if not res:
        return pd.DataFrame()
    return pd.DataFrame(res).set_index("date").sort_index()
