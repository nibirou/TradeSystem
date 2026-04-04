"""End-to-end factor mining runner (fundamental/mminute/custom) with catalog persistence."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import numpy as np
import pandas as pd

from ..core.utils import dump_json, ensure_dir
from .catalog import upsert_catalog_entries
from .custom import CustomFactorSpec, evaluate_custom_specs
from .evaluation import (
    check_admission,
    evaluate_factor_panel,
    objectives_from_metrics,
    periods_per_year_from_freq,
    resolve_admission_standard,
)
from .formulas import (
    FundamentalFormulaSpec,
    MinuteFormulaSpec,
    build_minute_feature_matrix,
    compute_fundamental_factor,
    compute_minute_factor_daily,
)
from .nsga import apply_dynamic_shortboard_penalty, nsga2_select, nsga3_select


@dataclass
class FactorMiningConfig:
    framework: str = "fundamental_multiobj"
    factor_freq: str = "D"
    horizon: int = 5
    train_start: pd.Timestamp = pd.Timestamp("2021-01-01")
    train_end: pd.Timestamp = pd.Timestamp("2023-12-31")
    valid_start: pd.Timestamp = pd.Timestamp("2024-01-01")
    valid_end: pd.Timestamp = pd.Timestamp("2024-12-31")
    population_size: int = 128
    generations: int = 20
    elite_size: int = 12
    mutation_rate: float = 0.25
    crossover_rate: float = 0.70
    top_n: int = 20
    corr_threshold: float = 0.60
    min_cross_section: int = 30
    top_frac: float = 0.10
    random_state: int = 42
    factor_store_root: str = ""
    catalog_path: str = ""
    save_format: str = "parquet"
    min_abs_ic_mean: float | None = None
    min_ic_win_rate: float | None = None
    min_ic_ir: float | None = None
    min_long_excess_annualized: float | None = None
    min_long_sharpe: float | None = None
    min_long_win_rate: float | None = None
    min_coverage: float | None = None


def _seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _hash_name(prefix: str, payload: Dict[str, object], length: int = 10) -> str:
    txt = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    h = hashlib.sha1(txt.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{h}"


def _to_key(payload: Dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _safe_obj(vals: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(vals), dtype=float)
    arr = np.where(np.isfinite(arr), arr, -1e30)
    return arr


def _save_table(df: pd.DataFrame, path: Path, fmt: str) -> Path:
    ensure_dir(path.parent)
    f = str(fmt).lower().strip()
    if f == "csv":
        fp = path.with_suffix(".csv")
        df.to_csv(fp, index=False, encoding="utf-8")
        return fp

    # parquet first, fallback csv
    fp_parquet = path.with_suffix(".parquet")
    try:
        df.to_parquet(fp_parquet, index=False)
        return fp_parquet
    except Exception:
        fp_csv = path.with_suffix(".csv")
        df.to_csv(fp_csv, index=False, encoding="utf-8")
        return fp_csv


def _split_mask(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, time_col: str = "date") -> pd.Series:
    t = pd.to_datetime(df[time_col], errors="coerce")
    return (t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end))


def _discover_daily_feature_pool(panel: pd.DataFrame) -> List[str]:
    exclude = {
        "date",
        "datetime",
        "code",
        "entry_date",
        "exit_date",
        "entry_ts",
        "exit_ts",
        "target_date",
        "future_ret_n",
        "target_up",
        "target_return",
        "target_volatility",
        "signal_ts",
        "time_freq",
    }
    prefer_contains = ("roe", "profit", "cash", "equity", "pe", "pb", "ev", "debt", "asset", "growth", "margin")

    numeric_cols: List[str] = []
    for c in panel.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(panel[c]):
            numeric_cols.append(c)

    preferred = [c for c in numeric_cols if any(k in c.lower() for k in prefer_contains)]
    if len(preferred) >= 12:
        return sorted(preferred)
    return sorted(numeric_cols)


def _discover_minute_feature_pool(minute_feat: pd.DataFrame) -> List[str]:
    exclude = {"date", "datetime", "code"}
    cols = [c for c in minute_feat.columns if c not in exclude and pd.api.types.is_numeric_dtype(minute_feat[c])]
    return sorted(cols)


def _fundamental_random_spec(rng: np.random.Generator, fields: List[str]) -> FundamentalFormulaSpec:
    periods = ["qoq", "yoy"]
    forms = ["diff", "pct", "std"]
    modes = ["ratio", "diff", "sum", "prod", "mean", "corr20", "beta20"]
    return FundamentalFormulaSpec(
        y_field=str(rng.choice(fields)),
        x_field=str(rng.choice(fields)),
        y_log=bool(rng.integers(0, 2)),
        x_log=bool(rng.integers(0, 2)),
        y_tr=bool(rng.integers(0, 2)),
        x_tr=bool(rng.integers(0, 2)),
        y_tr_period=str(rng.choice(periods)),
        x_tr_period=str(rng.choice(periods)),
        y_tr_form=str(rng.choice(forms)),
        x_tr_form=str(rng.choice(forms)),
        mode=str(rng.choice(modes)),
    )


def _fundamental_mutate(spec: FundamentalFormulaSpec, rng: np.random.Generator, fields: List[str]) -> FundamentalFormulaSpec:
    s = copy.deepcopy(spec)
    attr = str(rng.choice([
        "y_field",
        "x_field",
        "y_log",
        "x_log",
        "y_tr",
        "x_tr",
        "y_tr_period",
        "x_tr_period",
        "y_tr_form",
        "x_tr_form",
        "mode",
    ]))
    if attr in {"y_field", "x_field"}:
        setattr(s, attr, str(rng.choice(fields)))
    elif attr in {"y_log", "x_log", "y_tr", "x_tr"}:
        setattr(s, attr, bool(rng.integers(0, 2)))
    elif attr in {"y_tr_period", "x_tr_period"}:
        setattr(s, attr, str(rng.choice(["qoq", "yoy"])))
    elif attr in {"y_tr_form", "x_tr_form"}:
        setattr(s, attr, str(rng.choice(["diff", "pct", "std"])))
    elif attr == "mode":
        setattr(s, attr, str(rng.choice(["ratio", "diff", "sum", "prod", "mean", "corr20", "beta20"])))
    return s


def _fundamental_crossover(a: FundamentalFormulaSpec, b: FundamentalFormulaSpec, rng: np.random.Generator) -> FundamentalFormulaSpec:
    out = copy.deepcopy(a)
    for k in out.to_dict().keys():
        if bool(rng.integers(0, 2)):
            setattr(out, k, getattr(b, k))
    return out


def _minute_random_spec(rng: np.random.Generator, fields: List[str]) -> MinuteFormulaSpec:
    unary_ops = ["mean", "std", "skew", "kurt", "rank", "slope", "r2", "max", "min", "abs_mean"]
    cross_ops = ["corr", "cov", "beta", "euc_dist", "ols_intercept", "spread_mean", "r2"]
    windows = [20, 30, 45, 60, 90, 120]
    slices = [None, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    masks = ["none", "high_0.7", "high_0.8", "low_0.2", "low_0.3"]

    mode = int(rng.choice([1, 2]))
    return MinuteFormulaSpec(
        a_field=str(rng.choice(fields)),
        b_field=str(rng.choice(fields)),
        window=int(rng.choice(windows)),
        slice_pos=rng.choice(slices),
        mask_field=str(rng.choice(fields)),
        mask_rule=str(rng.choice(masks)),
        mode=mode,
        op_name=str(rng.choice(unary_ops)),
        cross_op_name=str(rng.choice(cross_ops)),
        b_shift_lag=int(rng.integers(-5, 6)),
    )


def _minute_mutate(spec: MinuteFormulaSpec, rng: np.random.Generator, fields: List[str]) -> MinuteFormulaSpec:
    s = copy.deepcopy(spec)
    attr = str(rng.choice([
        "a_field",
        "b_field",
        "window",
        "slice_pos",
        "mask_field",
        "mask_rule",
        "mode",
        "op_name",
        "cross_op_name",
        "b_shift_lag",
    ]))
    if attr in {"a_field", "b_field", "mask_field"}:
        setattr(s, attr, str(rng.choice(fields)))
    elif attr == "window":
        setattr(s, attr, int(rng.choice([20, 30, 45, 60, 90, 120])))
    elif attr == "slice_pos":
        setattr(s, attr, rng.choice([None, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]))
    elif attr == "mask_rule":
        setattr(s, attr, str(rng.choice(["none", "high_0.7", "high_0.8", "low_0.2", "low_0.3"])))
    elif attr == "mode":
        setattr(s, attr, int(rng.choice([1, 2])))
    elif attr == "op_name":
        setattr(s, attr, str(rng.choice(["mean", "std", "skew", "kurt", "rank", "slope", "r2", "max", "min", "abs_mean"])))
    elif attr == "cross_op_name":
        setattr(s, attr, str(rng.choice(["corr", "cov", "beta", "euc_dist", "ols_intercept", "spread_mean", "r2"])))
    elif attr == "b_shift_lag":
        setattr(s, attr, int(rng.integers(-5, 6)))
    return s


def _minute_crossover(a: MinuteFormulaSpec, b: MinuteFormulaSpec, rng: np.random.Generator) -> MinuteFormulaSpec:
    out = copy.deepcopy(a)
    for k in out.to_dict().keys():
        if bool(rng.integers(0, 2)):
            setattr(out, k, getattr(b, k))
    return out


def _collect_metrics(
    df: pd.DataFrame,
    factor_col: str,
    cfg: FactorMiningConfig,
    framework: str,
) -> Dict[str, float]:
    ppy = periods_per_year_from_freq(cfg.factor_freq, cfg.horizon)
    return evaluate_factor_panel(
        panel=df,
        factor_col=factor_col,
        ret_col="future_ret_n",
        group_col="date",
        top_frac=cfg.top_frac,
        min_cross_section=cfg.min_cross_section,
        periods_per_year=ppy,
    )


def _ic_score(metrics_valid: Dict[str, float]) -> float:
    icir = float(metrics_valid.get("ic_ir", float("nan")))
    if not np.isfinite(icir):
        return -1e9
    return icir


def _series_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 50:
        return float("nan")
    c = x[valid].corr(y[valid], method="spearman")
    return float(c) if pd.notna(c) else float("nan")


def _greedy_low_corr_select(
    candidates: List[Dict[str, object]],
    series_getter: Callable[[Dict[str, object]], pd.Series],
    top_n: int,
    corr_threshold: float,
) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    selected_series: Dict[str, pd.Series] = {}

    for c in candidates:
        if len(selected) >= int(top_n):
            break
        s = series_getter(c)
        if s.empty:
            continue
        good = True
        for kk, ss in selected_series.items():
            cc = _series_corr(s, ss)
            if np.isfinite(cc) and abs(cc) >= float(corr_threshold):
                _ = kk
                good = False
                break
        if not good:
            continue
        selected.append(c)
        selected_series[str(c["key"])] = s

    return selected


def _make_run_root(cfg: FactorMiningConfig) -> Path:
    ts = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d_%H%M%S")
    fw = str(cfg.framework).replace("_", "")
    root = Path(cfg.factor_store_root).expanduser() / "factor_mining" / fw / ts
    return ensure_dir(root)


def run_factor_mining(
    cfg: FactorMiningConfig,
    daily_panel_with_label: pd.DataFrame,
    minute_df: pd.DataFrame | None = None,
    custom_specs: List[CustomFactorSpec] | None = None,
) -> Dict[str, object]:
    """Run mining and persist admitted factors into factor catalog.

    Required columns in daily_panel_with_label:
    - date, code
    - future_ret_n
    - (optional) barra_size_proxy, industry_bucket
    """
    rng = _seed(cfg.random_state)

    panel = daily_panel_with_label.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["code"] = panel["code"].astype(str).str.strip()
    panel = panel.dropna(subset=["date", "code", "future_ret_n"]).sort_values(["date", "code"]).reset_index(drop=True)

    train_mask = _split_mask(panel, cfg.train_start, cfg.train_end, time_col="date")
    valid_mask = _split_mask(panel, cfg.valid_start, cfg.valid_end, time_col="date")
    if int(train_mask.sum()) <= 0:
        raise RuntimeError("factor mining train set is empty")
    if int(valid_mask.sum()) <= 0:
        raise RuntimeError("factor mining valid set is empty")

    framework = str(cfg.framework).strip().lower()
    standard = resolve_admission_standard(cfg.factor_freq, framework=framework)
    for attr in [
        "min_abs_ic_mean",
        "min_ic_win_rate",
        "min_ic_ir",
        "min_long_excess_annualized",
        "min_long_sharpe",
        "min_long_win_rate",
        "min_coverage",
    ]:
        v = getattr(cfg, attr, None)
        if v is not None and np.isfinite(float(v)):
            setattr(standard, attr, float(v))

    cache: Dict[str, Dict[str, object]] = {}

    if framework == "custom":
        specs = custom_specs or []
        if not specs:
            raise RuntimeError("custom framework requires at least one custom factor spec")

        fac_df = evaluate_custom_specs(panel=panel, specs=specs, time_col="date", code_col="code")
        results: List[Dict[str, object]] = []
        for spec in specs:
            factor_col = spec.name
            tmp = panel[["date", "code", "future_ret_n"]].copy()
            tmp[factor_col] = pd.to_numeric(fac_df[factor_col], errors="coerce")
            m_tr = _collect_metrics(tmp.loc[train_mask], factor_col=factor_col, cfg=cfg, framework=framework)
            m_va = _collect_metrics(tmp.loc[valid_mask], factor_col=factor_col, cfg=cfg, framework=framework)
            passed, admission = check_admission(m_va, standard)
            key = _to_key(spec.to_dict())
            result = {
                "key": key,
                "spec": spec,
                "name": factor_col,
                "metrics_train": m_tr,
                "metrics_valid": m_va,
                "objectives": objectives_from_metrics(m_va, framework=framework),
                "score": _ic_score(m_va),
                "passed": bool(passed),
                "admission": admission,
            }
            cache[key] = result
            results.append(result)

    elif framework == "fundamental_multiobj":
        fields = _discover_daily_feature_pool(panel)
        if len(fields) < 2:
            raise RuntimeError("insufficient daily feature columns for fundamental mining")

        def _evaluate(spec: FundamentalFormulaSpec) -> Dict[str, object]:
            key = _to_key(spec.to_dict())
            if key in cache:
                return cache[key]

            fac = compute_fundamental_factor(panel, spec)
            tmp = panel[["date", "code", "future_ret_n"]].copy()
            tmp["_factor"] = pd.to_numeric(fac, errors="coerce")

            m_tr = _collect_metrics(tmp.loc[train_mask], factor_col="_factor", cfg=cfg, framework=framework)
            m_va = _collect_metrics(tmp.loc[valid_mask], factor_col="_factor", cfg=cfg, framework=framework)
            obj = 0.5 * (_safe_obj(objectives_from_metrics(m_tr, framework)) + _safe_obj(objectives_from_metrics(m_va, framework)))

            passed, admission = check_admission(m_va, standard)
            score = _ic_score(m_va)
            result = {
                "key": key,
                "spec": spec,
                "metrics_train": m_tr,
                "metrics_valid": m_va,
                "objectives": obj.tolist(),
                "score": score,
                "passed": bool(passed),
                "admission": admission,
            }
            cache[key] = result
            return result

        pop: List[FundamentalFormulaSpec] = [_fundamental_random_spec(rng, fields) for _ in range(cfg.population_size)]
        archive: Dict[str, Dict[str, object]] = {}

        for gen in range(cfg.generations):
            res = [_evaluate(s) for s in pop]
            for r in res:
                k = str(r["key"])
                if k not in archive or float(r["score"]) > float(archive[k]["score"]):
                    archive[k] = r

            objs = [r["objectives"] for r in res]
            parent_idx = nsga2_select(objs, n_select=max(8, cfg.population_size // 2))
            parent_pool = [pop[i] for i in parent_idx]
            elites = sorted(res, key=lambda r: float(r["score"]), reverse=True)[: max(1, cfg.elite_size)]

            best = elites[0]
            print(
                f"[fundamental][gen={gen:02d}] best_score={best['score']:.4f} "
                f"absIC={best['metrics_valid'].get('abs_ic_mean', float('nan')):.4f} "
                f"win={best['metrics_valid'].get('ic_win_rate', float('nan')):.4f}"
            )

            new_pop: List[FundamentalFormulaSpec] = [copy.deepcopy(e["spec"]) for e in elites]
            while len(new_pop) < cfg.population_size:
                p1 = copy.deepcopy(parent_pool[int(rng.integers(0, len(parent_pool)))])
                child = p1
                if rng.random() < cfg.crossover_rate and len(parent_pool) > 1:
                    p2 = copy.deepcopy(parent_pool[int(rng.integers(0, len(parent_pool)))])
                    child = _fundamental_crossover(p1, p2, rng)
                if rng.random() < cfg.mutation_rate:
                    child = _fundamental_mutate(child, rng, fields)
                new_pop.append(child)
            pop = new_pop[: cfg.population_size]

        results = list(archive.values())

    elif framework == "minute_parametric":
        if minute_df is None or minute_df.empty:
            raise RuntimeError("minute_parametric framework requires minute_df")

        minute_feat = build_minute_feature_matrix(minute_df)
        fields = _discover_minute_feature_pool(minute_feat)
        if len(fields) < 2:
            raise RuntimeError("insufficient minute feature columns for minute mining")

        daily_ctx = panel[[c for c in ["date", "code", "future_ret_n", "barra_size_proxy", "industry_bucket"] if c in panel.columns]].copy()

        def _evaluate(spec: MinuteFormulaSpec) -> Dict[str, object]:
            key = _to_key(spec.to_dict())
            if key in cache:
                return cache[key]

            fac = compute_minute_factor_daily(minute_feature_df=minute_feat, daily_context_df=daily_ctx, spec=spec)
            fac_df = fac.rename("_factor").reset_index()
            tmp = daily_ctx[["date", "code", "future_ret_n"]].merge(fac_df, on=["date", "code"], how="left")

            m_tr = _collect_metrics(tmp.loc[train_mask], factor_col="_factor", cfg=cfg, framework=framework)
            m_va = _collect_metrics(tmp.loc[valid_mask], factor_col="_factor", cfg=cfg, framework=framework)
            obj = 0.5 * (_safe_obj(objectives_from_metrics(m_tr, framework)) + _safe_obj(objectives_from_metrics(m_va, framework)))

            passed, admission = check_admission(m_va, standard)
            score = _ic_score(m_va)
            result = {
                "key": key,
                "spec": spec,
                "metrics_train": m_tr,
                "metrics_valid": m_va,
                "objectives": obj.tolist(),
                "score": score,
                "passed": bool(passed),
                "admission": admission,
            }
            cache[key] = result
            return result

        pop: List[MinuteFormulaSpec] = [_minute_random_spec(rng, fields) for _ in range(cfg.population_size)]
        archive: Dict[str, Dict[str, object]] = {}

        for gen in range(cfg.generations):
            res = [_evaluate(s) for s in pop]
            for r in res:
                k = str(r["key"])
                if k not in archive or float(r["score"]) > float(archive[k]["score"]):
                    archive[k] = r

            raw_obj = np.asarray([r["objectives"] for r in res], dtype=float)
            penalized_obj, penalty = apply_dynamic_shortboard_penalty(raw_obj, floor_quantile=0.30, penalty_strength=0.20)
            parent_idx = nsga3_select(penalized_obj, n_select=max(8, cfg.population_size // 2), ref_divisions=8)
            parent_pool = [pop[i] for i in parent_idx]
            elites = sorted(res, key=lambda r: float(r["score"]), reverse=True)[: max(1, cfg.elite_size)]

            best = elites[0]
            mean_penalty = float(np.mean(penalty)) if len(penalty) > 0 else float("nan")
            print(
                f"[minute][gen={gen:02d}] best_score={best['score']:.4f} "
                f"absIC={best['metrics_valid'].get('abs_ic_mean', float('nan')):.4f} "
                f"win={best['metrics_valid'].get('ic_win_rate', float('nan')):.4f} "
                f"penalty={mean_penalty:.4f}"
            )

            new_pop: List[MinuteFormulaSpec] = [copy.deepcopy(e["spec"]) for e in elites]
            while len(new_pop) < cfg.population_size:
                p1 = copy.deepcopy(parent_pool[int(rng.integers(0, len(parent_pool)))])
                child = p1
                if rng.random() < cfg.crossover_rate and len(parent_pool) > 1:
                    p2 = copy.deepcopy(parent_pool[int(rng.integers(0, len(parent_pool)))])
                    child = _minute_crossover(p1, p2, rng)
                if rng.random() < cfg.mutation_rate:
                    child = _minute_mutate(child, rng, fields)
                new_pop.append(child)
            pop = new_pop[: cfg.population_size]

        results = list(archive.values())

    else:
        raise ValueError(f"unsupported mining framework: {cfg.framework}")

    # sort candidates: admitted first, then by validation ICIR score
    ranked = sorted(results, key=lambda r: (int(bool(r.get("passed", False))), float(r.get("score", -1e9))), reverse=True)
    minute_feat_cached = build_minute_feature_matrix(minute_df) if framework == "minute_parametric" and minute_df is not None else None

    def _series_getter(cand: Dict[str, object]) -> pd.Series:
        spec = cand["spec"]
        sign_ref = float(cand["metrics_valid"].get("rank_ic_mean", float("nan")))
        sign = -1.0 if np.isfinite(sign_ref) and sign_ref < 0.0 else 1.0

        if framework == "fundamental_multiobj":
            s = compute_fundamental_factor(panel, spec)
            fac = pd.DataFrame({"date": panel["date"], "code": panel["code"], "v": s * sign})
            return fac.set_index(["date", "code"])["v"].sort_index()
        if framework == "minute_parametric":
            minute_feat = minute_feat_cached if minute_feat_cached is not None else pd.DataFrame()
            daily_ctx = panel[[c for c in ["date", "code", "future_ret_n", "barra_size_proxy", "industry_bucket"] if c in panel.columns]].copy()
            s2 = compute_minute_factor_daily(minute_feature_df=minute_feat, daily_context_df=daily_ctx, spec=spec)
            return (s2 * sign).sort_index()

        # custom
        fac_df = evaluate_custom_specs(panel=panel, specs=[spec], time_col="date", code_col="code")
        fac = pd.DataFrame({"date": panel["date"], "code": panel["code"], "v": fac_df[spec.name] * sign})
        return fac.set_index(["date", "code"])["v"].sort_index()

    ranked_passed = [r for r in ranked if bool(r.get("passed", False))]
    selected = _greedy_low_corr_select(
        candidates=ranked_passed,
        series_getter=_series_getter,
        top_n=int(cfg.top_n),
        corr_threshold=float(cfg.corr_threshold),
    )

    run_root = _make_run_root(cfg)

    # build factor value table
    base_tbl = panel[["date", "code"]].drop_duplicates(["date", "code"]).sort_values(["date", "code"]).reset_index(drop=True)
    catalog_entries: List[Dict[str, object]] = []

    for i, cand in enumerate(selected, start=1):
        spec = cand["spec"]
        sign_ref = float(cand["metrics_valid"].get("rank_ic_mean", float("nan")))
        score_sign = -1.0 if np.isfinite(sign_ref) and sign_ref < 0.0 else 1.0

        if framework == "custom":
            fac_name = str(spec.name)
        elif framework == "fundamental_multiobj":
            fac_name = _hash_name("fund_mo", spec.to_dict())
        else:
            fac_name = _hash_name("min_mo", spec.to_dict())

        s = _series_getter(cand)
        fac_df = s.rename(fac_name).reset_index()
        base_tbl = base_tbl.merge(fac_df, on=["date", "code"], how="left")

        catalog_entries.append(
            {
                "name": fac_name,
                "freq": cfg.factor_freq,
                "category": "mined_factor" if framework != "custom" else "custom_factor",
                "description": (
                    f"{framework} admitted factor #{i}; "
                    f"profile={cand['admission'].get('profile', '')}"
                ),
                "source": "custom" if framework == "custom" else "mined",
                "framework": framework,
                "status": "active",
                "table_path": "",
                "value_col": fac_name,
                "score_sign": score_sign,
                "metrics_train": cand["metrics_train"],
                "metrics_valid": cand["metrics_valid"],
                "admission": cand["admission"],
                "formula": spec.to_dict(),
                "created_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            }
        )

    # persist factor table
    data_path = _save_table(
        df=base_tbl,
        path=run_root / f"factor_values_{cfg.factor_freq}",
        fmt=cfg.save_format,
    )

    for e in catalog_entries:
        e["table_path"] = str(data_path)

    catalog_path = Path(cfg.catalog_path)
    upsert_catalog_entries(catalog_path, catalog_entries)

    summary = {
        "framework": framework,
        "factor_freq": cfg.factor_freq,
        "train_period": [str(pd.Timestamp(cfg.train_start).date()), str(pd.Timestamp(cfg.train_end).date())],
        "valid_period": [str(pd.Timestamp(cfg.valid_start).date()), str(pd.Timestamp(cfg.valid_end).date())],
        "population_size": int(cfg.population_size),
        "generations": int(cfg.generations),
        "candidate_count": int(len(results)),
        "selected_count": int(len(selected)),
        "catalog_path": str(catalog_path),
        "factor_table_path": str(data_path),
        "selected_factors": [e["name"] for e in catalog_entries],
        "admission_profile": standard.profile,
        "top_candidates": [
            {
                "name": catalog_entries[i]["name"] if i < len(catalog_entries) else "",
                "score": float(selected[i]["score"]),
                "passed": bool(selected[i]["passed"]),
                "metrics_valid": selected[i]["metrics_valid"],
            }
            for i in range(len(selected))
        ],
    }

    summary_path = run_root / "mining_summary.json"
    dump_json(summary_path, summary)
    return summary
