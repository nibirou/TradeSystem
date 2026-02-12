# module_gnn_backtest_v2.py
import math
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import os


# -------------------------
# AkShare index loader
# -------------------------
def load_ak_index_close(path: str, date_col: str = "date", close_col: str = "close") -> pd.Series:
    """
    读取你保存的 AkShare 指数数据（csv 或 parquet），返回 Series(index=date, value=close)
    文件结构示例：
        date, open, high, low, close, volume
    """
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if date_col not in df.columns or close_col not in df.columns:
        raise ValueError(f"Index file missing columns: need [{date_col}, {close_col}], got {df.columns.tolist()}")

    s = df[[date_col, close_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col]).dt.normalize()
    s[close_col] = pd.to_numeric(s[close_col], errors="coerce")
    s = s.dropna(subset=[date_col, close_col]).sort_values(date_col)
    out = pd.Series(s[close_col].values, index=pd.DatetimeIndex(s[date_col].values), name=close_col)
    # 去重：同一天保留最后
    out = out[~out.index.duplicated(keep="last")]
    return out


# -------------------------
# Rebalance dates
# -------------------------
def make_rebalance_dates(trade_dates: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    td = pd.DatetimeIndex(pd.to_datetime(trade_dates)).sort_values().normalize()
    if freq == "D":
        return td
    if freq == "M":
        return pd.DatetimeIndex(pd.Series(td).groupby(pd.Series(td).dt.to_period("M")).max().values).sort_values()
    if freq == "W":
        return pd.DatetimeIndex(pd.Series(td).groupby(pd.Series(td).dt.to_period("W")).max().values).sort_values()
    raise ValueError(f"Unknown freq={freq}")


def _pivot_close(daily: pd.DataFrame) -> pd.DataFrame:
    px = daily[["date", "code", "close"]].copy()
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    px["code"] = px["code"].astype(str)
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px = px.dropna(subset=["date", "code", "close"])
    px = px.sort_values(["date", "code"])
    return px.pivot_table(index="date", columns="code", values="close")


def _spearman(a: pd.Series, b: pd.Series) -> float:
    if a.nunique() <= 1 or b.nunique() <= 1:
        return np.nan
    return float(a.rank().corr(b.rank()))


def _periods_per_year(freq: str) -> int:
    if freq == "D":
        return 252
    if freq == "W":
        return 52
    if freq == "M":
        return 12
    raise ValueError(freq)


def _perf_summary(ret: pd.Series, freq: str) -> Dict[str, float]:
    ret = ret.dropna()
    if ret.empty:
        return {"ann": np.nan, "vol": np.nan, "sharpe": np.nan}
    ppy = _periods_per_year(freq)
    ann = float(ret.mean() * ppy)
    vol = float(ret.std(ddof=1) * math.sqrt(ppy))
    sharpe = float(ann / (vol + 1e-12))
    return {"ann": ann, "vol": vol, "sharpe": sharpe}


# -------------------------
# Plotting
# -------------------------
def _safe_mkdir(d: str):
    os.makedirs(d, exist_ok=True)


def plot_nav_curves(
    bt: pd.DataFrame,
    bench_nav: pd.DataFrame,
    title: str = "NAV curves",
    save_dir: str = "./plots",
    fname: str = "nav_curves.png",
    dpi: int = 160,
    show: bool = False,
):
    """
    保存：组合净值 vs 指数净值
    bt: columns include [date, nav]
    bench_nav: index=date, columns like ["HS300","ZZ500","ZZ1000"]
    """
    if bt is None or bt.empty:
        print("[plot] bt is empty, skip nav plot")
        return

    _safe_mkdir(save_dir)

    d = pd.to_datetime(bt["date"]).dt.normalize()
    nav = pd.Series(bt["nav"].values, index=d, name="PORT")

    df = pd.concat([nav, bench_nav], axis=1)
    df = df.dropna(subset=["PORT"])
    if df.empty:
        print("[plot] nav align empty, skip")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["PORT"], label="PORT")
    for c in bench_nav.columns:
        if c in df.columns and df[c].notna().any():
            plt.plot(df.index, df[c], label=c)
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel("net value (normalized)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(save_dir, fname)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[plot saved] {out_path}")

    if show:
        plt.show()
    plt.close()


def plot_excess_nav(
    bt: pd.DataFrame,
    bench_names=("HS300", "ZZ500", "ZZ1000"),
    title: str = "Excess NAV",
    save_dir: str = "./plots",
    fname: str = "excess_nav.png",
    dpi: int = 160,
    show: bool = False,
):
    """
    保存：组合对各基准的累计超额净值：(1+excess).cumprod()
    bt: need columns [date, nav, ex_HS300/ex_ZZ500/ex_ZZ1000 ...]
    """
    if bt is None or bt.empty:
        print("[plot] bt is empty, skip excess plot")
        return

    _safe_mkdir(save_dir)

    x = pd.to_datetime(bt["date"]).dt.normalize()

    plt.figure(figsize=(12, 6))
    # 组合净值
    if "nav" in bt.columns:
        plt.plot(x, bt["nav"].values, label="PORT_NAV")

    for name in bench_names:
        ex_col = f"ex_{name}"
        if ex_col in bt.columns and bt[ex_col].notna().any():
            ex_nav = (1.0 + bt[ex_col].fillna(0.0)).cumprod()
            plt.plot(x, ex_nav.values, label=f"EX_{name}")

    plt.title(title)
    plt.xlabel("date")
    plt.ylabel("net value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(save_dir, fname)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[plot saved] {out_path}")

    if show:
        plt.show()
    plt.close()


# -------------------------
# Main unified backtest
# -------------------------
def gnn_score_backtest(
    daily: pd.DataFrame,
    pred: pd.DataFrame,
    trade_dates: pd.DatetimeIndex,
    start_date: str,
    end_date: str,
    freq: str = "M",
    topN: int = 50,
    lam_vol: float = 0.3,
    vol_filter_q: float = 0.9,
    # AkShare index files (csv/parquet)
    ak_index_paths: Optional[Dict[str, str]] = None,
    # optional label-based evaluation
    label_builder=None,
    label_horizon_n: int = 10,
    label_stop_loss: float = -0.10,
    label_topk: int = 30,
    # plotting
    plot: bool = True,
    verbose: bool = True,
    plot_dir: str = "/workspace/Quant/plots/gnn_bt",
    plot_show: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    """
    ak_index_paths: {"HS300": ".../hs300_price.csv", "ZZ500": "...", "ZZ1000": "..."}
    """

    # ---------- defaults ----------
    if ak_index_paths is None:
        ak_index_paths = {}

    # ---------- prepare ----------
    td = pd.DatetimeIndex(pd.to_datetime(trade_dates)).sort_values().normalize()
    rebal = make_rebalance_dates(td, freq=freq)
    rebal = rebal[(rebal >= pd.to_datetime(start_date)) & (rebal <= pd.to_datetime(end_date))]
    if len(rebal) < 2:
        return pd.DataFrame(), {}

    close_pv = _pivot_close(daily)

    pred = pred.copy()
    pred["date"] = pd.to_datetime(pred["date"]).dt.normalize()
    pred["code"] = pred["code"].astype(str)
    pred["pred_ret"] = pd.to_numeric(pred["pred_ret"], errors="coerce")
    pred["pred_vol"] = pd.to_numeric(pred["pred_vol"], errors="coerce")
    pred["score"] = pred["pred_ret"] - lam_vol * pred["pred_vol"]

    # ---------- load AkShare indices ----------
    bench_close: Dict[str, Optional[pd.Series]] = {}
    for name, path in ak_index_paths.items():
        try:
            bench_close[name] = load_ak_index_close(path)
        except Exception as e:
            print(f"[WARN] load index {name} failed: {e}")
            bench_close[name] = None

    # 预先构造基准净值（用日频 close 转日收益再 cumprod），用于画图对齐
    bench_nav = pd.DataFrame(index=td)
    for name, s in bench_close.items():
        if s is None or s.empty:
            continue
        s = s.reindex(td).ffill()
        r = s.pct_change()
        nav = (1.0 + r.fillna(0.0)).cumprod()
        bench_nav[name] = nav

    # ---------- loop ----------
    rows = []
    picks_prev = None

    for i in range(len(rebal) - 1):
        t, t2 = rebal[i], rebal[i + 1]

        if t not in close_pv.index or t2 not in close_pv.index:
            continue

        sig = pred[pred["date"] == t].copy()
        if sig.empty:
            continue

        # 1) vol filter + score sort
        if sig["pred_vol"].notna().any():
            vcut = float(sig["pred_vol"].quantile(vol_filter_q))
            sig = sig[sig["pred_vol"] <= vcut]
        sig = sig.sort_values("score", ascending=False)

        # 2) realized return for RankIC (sig universe)
        codes_all = sig["code"].tolist()
        p0_all = close_pv.loc[t, codes_all]
        p1_all = close_pv.loc[t2, codes_all]
        valid_all = p0_all.notna() & p1_all.notna() & (p0_all > 0) & (p1_all > 0)
        r_all = (p1_all[valid_all] / p0_all[valid_all] - 1.0)
        score_all = sig.set_index("code").loc[r_all.index, "score"]
        rankic = _spearman(score_all, r_all)

        # 3) TopN portfolio return
        sig_top = sig.head(topN)
        picks = sig_top["code"].tolist()
        if verbose:
            print(f"[{t.date()}] picks(top{topN})={picks[:20]}{' ...' if len(picks)>20 else ''}")

        p0 = close_pv.loc[t, picks]
        p1 = close_pv.loc[t2, picks]
        valid = p0.notna() & p1.notna() & (p0 > 0) & (p1 > 0)
        if valid.sum() == 0:
            continue
        port_ret = float((p1[valid] / p0[valid] - 1.0).mean())

        # 4) turnover
        turnover = np.nan
        if picks_prev is not None:
            s1, s2 = set(picks_prev), set(picks)
            union = len(s1 | s2)
            inter = len(s1 & s2)
            turnover = float(1.0 - (inter / union if union > 0 else 0.0))
        picks_prev = picks

        # 5) benchmark return & excess (AkShare close)
        bench_ret = {}
        excess = {}
        for name, s in bench_close.items():
            if s is None or s.empty:
                bench_ret[name] = np.nan
                excess[name] = np.nan
                continue
            # 用 t/t2 在指数 close 上取值
            if (t not in s.index) or (t2 not in s.index):
                # 尝试用 ffill 对齐到最近交易日（更稳）
                s2 = s.reindex(td).ffill()
                if (t not in s2.index) or (t2 not in s2.index):
                    bench_ret[name] = np.nan
                    excess[name] = np.nan
                    continue
                b0, b1 = s2.loc[t], s2.loc[t2]
            else:
                b0, b1 = s.loc[t], s.loc[t2]

            if pd.isna(b0) or pd.isna(b1) or b0 <= 0 or b1 <= 0:
                bench_ret[name] = np.nan
                excess[name] = np.nan
                continue

            r = float(b1 / b0 - 1.0)
            bench_ret[name] = r
            excess[name] = float(port_ret - r)

        # 6) LabelBuilder stats (optional)
        topk_mean_y = np.nan
        topk_winrate = np.nan
        n_hit = 0
        if label_builder is not None:
            y_df = label_builder.build_labels(t, horizon_n=label_horizon_n, stop_loss=label_stop_loss)
            if y_df is not None and not y_df.empty:
                hit = y_df.loc[y_df.index.intersection(picks[:label_topk])]
                if not hit.empty:
                    topk_mean_y = float(hit["y"].mean())
                    topk_winrate = float((hit["y"] > 0).mean())
                    n_hit = int(len(hit))

        rows.append({
            "t": t,
            "date": t2,  # 记账到下一期
            "ret": port_ret,
            "rankIC": rankic,
            "turnover": turnover,
            "n_valid": int(valid.sum()),
            "TopK_mean_y": topk_mean_y,
            "TopK_winrate": topk_winrate,
            "TopK_labelN": n_hit,
            **{f"bm_{k}": v for k, v in bench_ret.items()},
            **{f"ex_{k}": v for k, v in excess.items()},
        })

    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt, {}

    bt = bt.sort_values("date").reset_index(drop=True)
    bt["nav"] = (1.0 + bt["ret"].fillna(0.0)).cumprod()

    # add excess nav
    for name in bench_close.keys():
        ex_col = f"ex_{name}"
        if ex_col in bt.columns:
            bt[f"ex_nav_{name}"] = (1.0 + bt[ex_col].fillna(0.0)).cumprod()

    # ---------- summary ----------
    rk = bt["rankIC"].dropna()
    rk_mean = float(rk.mean()) if not rk.empty else np.nan
    rk_std = float(rk.std(ddof=1)) if len(rk) > 1 else np.nan
    icir = float(rk_mean / (rk_std + 1e-12)) if np.isfinite(rk_mean) and np.isfinite(rk_std) else np.nan
    ic_pos = float((rk > 0).mean()) if not rk.empty else np.nan

    perf = _perf_summary(bt["ret"], freq=freq)
    avg_turn = float(bt["turnover"].dropna().mean()) if bt["turnover"].notna().any() else np.nan

    summary = {
        "freq": freq,
        "topN": topN,
        "lam_vol": lam_vol,
        "vol_filter_q": vol_filter_q,
        "periods": int(len(bt)),
        "rankIC_mean": rk_mean,
        "rankIC_std": rk_std,
        "ICIR": icir,
        "IC>0_ratio": ic_pos,
        "avg_turnover": avg_turn,
        **{f"port_{k}": v for k, v in perf.items()},
    }

    # excess perf summary for each benchmark
    for name in bench_close.keys():
        ex_col = f"ex_{name}"
        if ex_col in bt.columns and bt[ex_col].notna().any():
            ex_perf = _perf_summary(bt[ex_col], freq=freq)
            summary.update({f"ex_{name}_{k}": v for k, v in ex_perf.items()})

    # label summary
    if label_builder is not None and bt["TopK_mean_y"].notna().any():
        summary["label_TopK_mean_y"] = float(bt["TopK_mean_y"].dropna().mean())
        summary["label_TopK_winrate"] = float(bt["TopK_winrate"].dropna().mean())

    if verbose:
        print(bt.tail(5))
        print("[SUMMARY]")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")

    # ---------- plots ----------
    if plot and not bench_nav.empty:
        tag = f"{freq}_top{topN}_lam{lam_vol}_vq{vol_filter_q}_{pd.to_datetime(start_date).date()}_{pd.to_datetime(end_date).date()}"
        plot_nav_curves(
            bt=bt[["date", "nav"]].copy(),
            bench_nav=bench_nav.reindex(pd.to_datetime(bt["date"]).dt.normalize()).ffill(),
            title=f"NAV: PORT vs Indices ({freq})",
            save_dir=plot_dir,
            fname=f"nav_{tag}.png",
            show=plot_show,
        )

        plot_excess_nav(
            bt=bt[["date", "nav"] + [c for c in bt.columns if c.startswith("ex_")]].copy(),
            bench_names=tuple(bench_close.keys()),
            title=f"Excess NAV: PORT - Benchmark ({freq})",
            save_dir=plot_dir,
            fname=f"excess_{tag}.png",
            show=plot_show,
        )

    return bt, summary
