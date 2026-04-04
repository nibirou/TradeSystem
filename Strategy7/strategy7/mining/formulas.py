"""Parametric factor formulas (fundamental + minute-level) for mining framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from ..core.constants import EPS


@dataclass
class FundamentalFormulaSpec:
    y_field: str
    x_field: str
    y_log: bool
    x_log: bool
    y_tr: bool
    x_tr: bool
    y_tr_period: str
    x_tr_period: str
    y_tr_form: str
    x_tr_form: str
    mode: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class MinuteFormulaSpec:
    a_field: str
    b_field: str
    window: int
    slice_pos: float | None
    mask_field: str
    mask_rule: str
    mode: int
    op_name: str
    cross_op_name: str
    b_shift_lag: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def winsorize_mad_cs(s: pd.Series, group: pd.Series, limit: float = 3.0) -> pd.Series:
    def _clip(v: pd.Series) -> pd.Series:
        med = float(v.median())
        mad = float((v - med).abs().median())
        if mad <= EPS:
            return v
        lo = med - limit * 1.4826 * mad
        hi = med + limit * 1.4826 * mad
        return v.clip(lower=lo, upper=hi)

    return s.groupby(group).transform(_clip)


def cs_zscore(s: pd.Series, group: pd.Series) -> pd.Series:
    def _z(v: pd.Series) -> pd.Series:
        std = float(v.std(ddof=0))
        if std <= EPS:
            return pd.Series(np.zeros(len(v), dtype=float), index=v.index)
        return (v - float(v.mean())) / (std + EPS)

    return s.groupby(group).transform(_z)


def neutralize_series(
    y: pd.Series,
    df: pd.DataFrame,
    group_col: str = "date",
    size_col: str = "barra_size_proxy",
    industry_col: str | None = None,
) -> pd.Series:
    out = pd.Series(np.nan, index=y.index, dtype=float)

    for _t, g in df.groupby(group_col):
        idx = g.index
        yy = pd.to_numeric(y.loc[idx], errors="coerce")
        valid = yy.notna()

        if size_col in g.columns:
            sx = pd.to_numeric(g[size_col], errors="coerce")
            valid = valid & sx.notna()
        else:
            sx = pd.Series(np.nan, index=g.index)

        if valid.sum() < 8:
            out.loc[idx] = yy - yy.mean()
            continue

        X_parts: List[np.ndarray] = [np.ones((int(valid.sum()), 1), dtype=float)]
        s_valid = sx.loc[valid].astype(float)
        s_std = float(s_valid.std(ddof=0))
        if np.isfinite(s_std) and s_std > EPS:
            s_norm = ((s_valid - float(s_valid.mean())) / (s_std + EPS)).to_numpy().reshape(-1, 1)
            X_parts.append(s_norm)

        if industry_col and industry_col in g.columns:
            inds = g.loc[valid, industry_col].astype(str)
            dummies = pd.get_dummies(inds, drop_first=True)
            if not dummies.empty:
                X_parts.append(dummies.to_numpy(dtype=float))

        X = np.concatenate(X_parts, axis=1)
        yv = yy.loc[valid].to_numpy(dtype=float)
        try:
            beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
            resid = yv - X.dot(beta)
        except Exception:
            resid = yv - np.nanmean(yv)

        tmp = pd.Series(np.nan, index=idx, dtype=float)
        tmp.loc[valid.index[valid]] = resid
        tmp = tmp.fillna(tmp.mean())
        out.loc[idx] = tmp

    return out


def build_minute_feature_matrix(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Build minute-level feature pool used by parametric minute formula."""
    if minute_df.empty:
        return minute_df.copy()

    m = minute_df.copy()
    m["datetime"] = pd.to_datetime(m["datetime"], errors="coerce")
    m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.normalize()
    m["code"] = m["code"].astype(str).str.strip()
    for c in ["open", "high", "low", "close", "volume", "amount", "num_trades"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")

    m = m.dropna(subset=["date", "datetime", "code", "close"]).sort_values(["code", "date", "datetime"]).copy()
    g_day = m.groupby(["code", "date"]) 

    m["return"] = g_day["close"].pct_change().fillna(0.0)
    m["twap"] = g_day["close"].expanding().mean().reset_index(level=[0, 1], drop=True)

    signed_vol = np.sign(g_day["close"].diff().fillna(0.0)) * m["volume"].fillna(0.0)
    m["obv"] = signed_vol.groupby([m["code"], m["date"]]).cumsum()
    m["pvt"] = (m["return"].fillna(0.0) * m["volume"].fillna(0.0)).groupby([m["code"], m["date"]]).cumsum()

    if "num_trades" in m.columns:
        m["vol_per_trade"] = m["volume"] / (m["num_trades"] + EPS)
        m["amt_per_trade"] = m["amount"] / (m["num_trades"] + EPS)
    else:
        m["vol_per_trade"] = np.nan
        m["amt_per_trade"] = np.nan

    for base in ["close", "return", "volume", "vol_per_trade"]:
        for w in [5, 15, 30]:
            col = f"{base}_ma{w}"
            m[col] = g_day[base].transform(lambda s: s.rolling(w, min_periods=max(2, w // 3)).mean())

    return m


def _transform_ts_feature(
    s: pd.Series,
    code: pd.Series,
    use_log: bool,
    use_tr: bool,
    tr_period: str,
    tr_form: str,
) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if use_log:
        x = np.sign(x) * np.log1p(np.abs(x))

    if not use_tr:
        return x

    lag = 63 if str(tr_period).lower() in {"q", "qoq", "quarter", "season"} else 252
    g = x.groupby(code)
    prev = g.shift(lag)

    form = str(tr_form).lower()
    if form == "diff":
        return x - prev
    if form == "pct":
        return x / (prev + EPS) - 1.0
    if form == "std":
        return g.transform(lambda v: v.rolling(lag, min_periods=max(5, lag // 6)).std())
    return x


def compute_fundamental_factor(panel: pd.DataFrame, spec: FundamentalFormulaSpec) -> pd.Series:
    if spec.y_field not in panel.columns or spec.x_field not in panel.columns:
        return pd.Series(np.nan, index=panel.index, dtype=float)

    code = panel["code"].astype(str)
    date = pd.to_datetime(panel["date"], errors="coerce").dt.normalize() if "date" in panel.columns else pd.Series(pd.NaT, index=panel.index)

    y = _transform_ts_feature(
        s=panel[spec.y_field],
        code=code,
        use_log=bool(spec.y_log),
        use_tr=bool(spec.y_tr),
        tr_period=str(spec.y_tr_period),
        tr_form=str(spec.y_tr_form),
    )
    x = _transform_ts_feature(
        s=panel[spec.x_field],
        code=code,
        use_log=bool(spec.x_log),
        use_tr=bool(spec.x_tr),
        tr_period=str(spec.x_tr_period),
        tr_form=str(spec.x_tr_form),
    )

    mode = str(spec.mode).lower()
    if mode == "ratio":
        raw = y / (x + EPS)
    elif mode == "diff":
        raw = y - x
    elif mode == "sum":
        raw = y + x
    elif mode == "prod":
        raw = y * x
    elif mode == "mean":
        raw = 0.5 * (y + x)
    elif mode == "corr20":
        raw = (
            pd.DataFrame({"code": code, "y": y, "x": x})
            .groupby("code", group_keys=False)
            .apply(lambda t: t["y"].rolling(20, min_periods=10).corr(t["x"]))
            .reset_index(level=0, drop=True)
        )
    elif mode == "beta20":
        tmp = pd.DataFrame({"code": code, "y": y, "x": x})
        cov = (
            tmp.groupby("code", group_keys=False)
            .apply(lambda t: t["y"].rolling(20, min_periods=10).cov(t["x"]))
            .reset_index(level=0, drop=True)
        )
        var = tmp.groupby("code")["x"].transform(lambda s: s.rolling(20, min_periods=10).var())
        raw = cov / (var + EPS)
    else:
        raw = y - x

    out = pd.to_numeric(raw, errors="coerce")
    out = winsorize_mad_cs(out, group=date, limit=3.0)
    out = neutralize_series(out, panel, group_col="date", size_col="barra_size_proxy", industry_col=None)
    out = cs_zscore(out, group=date)
    return pd.to_numeric(out, errors="coerce")


def _parse_mask_rule(mask_rule: str) -> tuple[str, float]:
    rule = str(mask_rule).strip().lower()
    if rule in {"", "none", "all"}:
        return "none", 0.0
    if "_" not in rule:
        return "none", 0.0
    side, q = rule.split("_", 1)
    try:
        qq = float(q)
    except Exception:
        qq = 0.0
    qq = float(np.clip(qq, 0.0, 1.0))
    if side not in {"high", "low"}:
        side = "none"
    return side, qq


def _slice_range(n: int, window: int, slice_pos: float | None) -> tuple[int, int]:
    if n <= 0:
        return 0, 0
    w = max(1, int(window))
    if w >= n:
        return 0, n
    if slice_pos is None:
        return max(0, n - w), n

    sp = float(np.clip(slice_pos, 0.0, 1.0))
    center = int(round((n - 1) * sp))
    half = w // 2
    left = center - half
    right = left + w
    if left < 0:
        right -= left
        left = 0
    if right > n:
        left -= right - n
        right = n
    return max(0, left), min(n, right)


def _unary_op(v: np.ndarray, op_name: str) -> float:
    if v.size == 0:
        return float("nan")
    x = v[np.isfinite(v)]
    if x.size == 0:
        return float("nan")

    op = str(op_name).lower()
    if op == "mean":
        return float(np.mean(x))
    if op == "std":
        return float(np.std(x))
    if op == "skew":
        return float(pd.Series(x).skew()) if x.size > 2 else float("nan")
    if op == "kurt":
        return float(pd.Series(x).kurt()) if x.size > 3 else float("nan")
    if op == "max":
        return float(np.max(x))
    if op == "min":
        return float(np.min(x))
    if op == "abs_mean":
        return float(np.mean(np.abs(x)))
    if op == "rank":
        return float(pd.Series(x).rank(pct=True).iloc[-1])
    if op == "slope":
        if x.size < 3:
            return float("nan")
        t = np.arange(x.size, dtype=float)
        b = np.polyfit(t, x, deg=1)[0]
        return float(b)
    if op == "r2":
        if x.size < 3:
            return float("nan")
        t = np.arange(x.size, dtype=float)
        b1, b0 = np.polyfit(t, x, deg=1)
        y_hat = b1 * t + b0
        ss_res = float(np.sum((x - y_hat) ** 2))
        ss_tot = float(np.sum((x - np.mean(x)) ** 2))
        return float(1.0 - ss_res / (ss_tot + EPS))
    return float(np.mean(x))


def _binary_op(a: np.ndarray, b: np.ndarray, op_name: str) -> float:
    n = min(a.size, b.size)
    if n <= 0:
        return float("nan")
    x = a[:n]
    y = b[:n]
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 3:
        return float("nan")

    op = str(op_name).lower()
    if op == "corr":
        return float(pd.Series(x).corr(pd.Series(y), method="pearson"))
    if op == "cov":
        return float(np.cov(x, y, ddof=1)[0, 1])
    if op == "beta":
        return float(np.cov(x, y, ddof=1)[0, 1] / (np.var(y, ddof=1) + EPS))
    if op == "euc_dist":
        return float(np.linalg.norm(x - y) / np.sqrt(len(x)))
    if op == "spread_mean":
        return float(np.mean(x - y))
    if op == "ols_intercept":
        b1, b0 = np.polyfit(y, x, deg=1)
        _ = b1
        return float(b0)
    if op == "r2":
        b1, b0 = np.polyfit(y, x, deg=1)
        y_hat = b1 * y + b0
        ss_res = float(np.sum((x - y_hat) ** 2))
        ss_tot = float(np.sum((x - np.mean(x)) ** 2))
        return float(1.0 - ss_res / (ss_tot + EPS))
    return float(pd.Series(x).corr(pd.Series(y), method="pearson"))


def compute_minute_factor_daily(
    minute_feature_df: pd.DataFrame,
    daily_context_df: pd.DataFrame,
    spec: MinuteFormulaSpec,
) -> pd.Series:
    if minute_feature_df.empty:
        return pd.Series(dtype=float)

    needed = {spec.a_field, spec.mask_field}
    if int(spec.mode) == 2:
        needed.add(spec.b_field)
    for c in needed:
        if c not in minute_feature_df.columns:
            return pd.Series(dtype=float)

    recs: List[Dict[str, object]] = []
    by = minute_feature_df.groupby(["date", "code"], sort=False)
    side, q = _parse_mask_rule(spec.mask_rule)

    for (dt, code), g in by:
        g = g.sort_values("datetime")
        a = pd.to_numeric(g[spec.a_field], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(g[spec.b_field], errors="coerce").to_numpy(dtype=float) if spec.b_field in g.columns else np.full(len(g), np.nan)
        m = pd.to_numeric(g[spec.mask_field], errors="coerce").to_numpy(dtype=float)

        l, r = _slice_range(len(g), window=int(spec.window), slice_pos=spec.slice_pos)
        if r <= l:
            continue

        a_seg = a[l:r]
        b_seg = b[l:r]
        m_seg = m[l:r]

        keep = np.isfinite(m_seg)
        if side != "none" and np.any(keep):
            rank = pd.Series(m_seg[keep]).rank(pct=True).to_numpy(dtype=float)
            local_keep = np.zeros(np.sum(keep), dtype=bool)
            if side == "high":
                local_keep = rank >= q
            elif side == "low":
                local_keep = rank <= q
            keep_idx = np.where(keep)[0]
            keep = np.zeros_like(keep, dtype=bool)
            keep[keep_idx[local_keep]] = True

        a_sel = a_seg[keep]
        b_sel = b_seg[keep]

        if int(spec.mode) == 1:
            value = _unary_op(a_sel, spec.op_name)
        else:
            lag = int(spec.b_shift_lag)
            if lag != 0 and b_sel.size > 0:
                b_sel = np.roll(b_sel, shift=lag)
            value = _binary_op(a_sel, b_sel, spec.cross_op_name)

        recs.append({"date": pd.Timestamp(dt), "code": str(code), "_raw": float(value) if np.isfinite(value) else np.nan})

    if not recs:
        return pd.Series(dtype=float)

    raw = pd.DataFrame(recs)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw["code"] = raw["code"].astype(str)

    ctx_cols = [c for c in ["date", "code", "barra_size_proxy", "industry_bucket"] if c in daily_context_df.columns]
    ctx = daily_context_df[ctx_cols].copy().drop_duplicates(["date", "code"], keep="last")
    df = raw.merge(ctx, on=["date", "code"], how="left")

    fac = pd.to_numeric(df["_raw"], errors="coerce")
    fac = winsorize_mad_cs(fac, group=df["date"], limit=3.0)
    fac = neutralize_series(
        fac,
        df,
        group_col="date",
        size_col="barra_size_proxy",
        industry_col="industry_bucket",
    )
    fac = cs_zscore(fac, group=df["date"])
    df["_factor"] = pd.to_numeric(fac, errors="coerce")

    out = df.set_index(["date", "code"])["_factor"].sort_index()
    return out
