"""Minute parametric mining model."""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd


def run_minute_parametric_model(
    *,
    panel: pd.DataFrame,
    minute_df: pd.DataFrame | None,
    factor_freq: str,
    time_col: str,
    framework: str,
    eval_group_col: str,
    cfg: Any,
    rng: Any,
    cache: Dict[str, Dict[str, object]],
    standard: Any,
    build_minute_feature_matrix: Callable[[pd.DataFrame], pd.DataFrame],
    discover_minute_feature_pool: Callable[[pd.DataFrame], List[str]],
    discover_daily_feature_pool: Callable[[pd.DataFrame], List[str]],
    compute_minute_factor_panel: Callable[..., pd.Series],
    compute_minute_factor_daily: Callable[..., pd.Series],
    split_train_valid: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]],
    collect_metrics: Callable[[pd.DataFrame, str, str], Dict[str, float]],
    objectives_from_metrics: Callable[[Dict[str, float], str], List[float]],
    safe_obj: Callable[[List[float]], np.ndarray],
    check_admission: Callable[[Dict[str, float], Any], tuple[bool, Dict[str, object]]],
    to_key: Callable[[Dict[str, object]], str],
    ic_score: Callable[[Dict[str, float]], float],
    nsga3_select: Callable[..., List[int]],
    apply_dynamic_shortboard_penalty: Callable[..., Tuple[np.ndarray, np.ndarray]],
    random_spec_fn: Callable[..., Any],
    mutate_spec_fn: Callable[..., Any],
    crossover_spec_fn: Callable[..., Any],
    log_progress: Callable[..., None],
) -> Tuple[List[Dict[str, object]], bool]:
    minute_panel_mode = str(factor_freq).upper() != "D"
    if minute_panel_mode:
        minute_feat = panel.copy()
        metric_group_col = eval_group_col
    else:
        if minute_df is None or minute_df.empty:
            raise RuntimeError(f"{framework} framework requires minute_df")
        minute_feat = build_minute_feature_matrix(minute_df)
        metric_group_col = "date"

    if minute_panel_mode:
        fields = discover_minute_feature_pool(minute_feat)
        daily_ctx = panel[[c for c in ["date", "code", "future_ret_n", "barra_size_proxy", "industry_bucket"] if c in panel.columns]].copy()
        daily_material_count = 0
    else:
        minute_fields = discover_minute_feature_pool(minute_feat)
        daily_fields = discover_daily_feature_pool(panel)
        fields = sorted(set(minute_fields) | set(daily_fields))

        # For daily-mode minute mining, keep numeric daily context so daily factors
        # can be used as constant-within-day materials inside minute formulas.
        keep_cols: List[str] = []
        for c in panel.columns:
            if c in {"date", "code", "future_ret_n", "barra_size_proxy", "industry_bucket"}:
                keep_cols.append(c)
                continue
            if pd.api.types.is_numeric_dtype(panel[c]):
                keep_cols.append(c)
        daily_ctx = panel[sorted(set(keep_cols))].copy()
        daily_material_count = int(len(daily_fields))

    if len(fields) < 2:
        raise RuntimeError(f"insufficient minute feature columns for {framework}")

    pop_size = int(cfg.population_size)
    generations = int(cfg.generations)
    log_progress(
        f"Start minute evolution: framework={framework}, feature_pool={len(fields)}, "
        f"population={pop_size}, generations={generations}, panel_mode={int(minute_panel_mode)}, "
        f"daily_material_fields={daily_material_count}.",
        module="mining",
    )

    def _evaluate(spec: Any) -> Dict[str, object]:
        key = to_key(spec.to_dict())
        if key in cache:
            return cache[key]

        if minute_panel_mode:
            fac = compute_minute_factor_panel(
                panel_feature_df=minute_feat,
                spec=spec,
                time_col=time_col,
                session_col="date" if time_col == "datetime" else None,
            )
            tmp = panel[[time_col, "code", "future_ret_n"]].copy()
            tmp["_factor"] = pd.to_numeric(fac, errors="coerce")
        else:
            fac = compute_minute_factor_daily(minute_feature_df=minute_feat, daily_context_df=daily_ctx, spec=spec)
            fac_df = fac.rename("_factor").reset_index()
            tmp = daily_ctx[["date", "code", "future_ret_n"]].merge(fac_df, on=["date", "code"], how="left")

        tmp_tr, tmp_va = split_train_valid(tmp)
        m_tr = collect_metrics(tmp_tr, "_factor", metric_group_col)
        m_va = collect_metrics(tmp_va, "_factor", metric_group_col)
        obj = 0.5 * (safe_obj(objectives_from_metrics(m_tr, framework)) + safe_obj(objectives_from_metrics(m_va, framework)))

        passed, admission = check_admission(m_va, standard)
        score = ic_score(m_va)
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

    pop: List[Any] = [random_spec_fn(rng, fields) for _ in range(pop_size)]
    archive: Dict[str, Dict[str, object]] = {}

    for gen in range(generations):
        res = [_evaluate(s) for s in pop]
        for r in res:
            k = str(r["key"])
            if k not in archive or float(r["score"]) > float(archive[k]["score"]):
                archive[k] = r

        raw_obj = np.asarray([r["objectives"] for r in res], dtype=float)
        penalized_obj, penalty = apply_dynamic_shortboard_penalty(raw_obj, floor_quantile=0.30, penalty_strength=0.20)
        parent_idx = nsga3_select(penalized_obj, n_select=max(8, pop_size // 2), ref_divisions=8)
        parent_pool = [pop[i] for i in parent_idx]
        elites = sorted(res, key=lambda r: float(r["score"]), reverse=True)[: max(1, int(cfg.elite_size))]

        best = elites[0]
        mean_penalty = float(np.mean(penalty)) if len(penalty) > 0 else float("nan")
        log_progress(
            (
                f"[minute][gen={gen:02d}] best_score={best['score']:.4f} "
                f"absIC={best['metrics_valid'].get('abs_ic_mean', float('nan')):.4f} "
                f"win={best['metrics_valid'].get('ic_win_rate', float('nan')):.4f} "
                f"penalty={mean_penalty:.4f}"
            ),
            module="mining",
            level="debug",
        )

        new_pop: List[Any] = [copy.deepcopy(e["spec"]) for e in elites]
        while len(new_pop) < pop_size:
            p1 = copy.deepcopy(parent_pool[int(rng.integers(0, len(parent_pool)))])
            child = p1
            if rng.random() < float(cfg.crossover_rate) and len(parent_pool) > 1:
                p2 = copy.deepcopy(parent_pool[int(rng.integers(0, len(parent_pool)))])
                child = crossover_spec_fn(p1, p2, rng)
            if rng.random() < float(cfg.mutation_rate):
                child = mutate_spec_fn(child, rng, fields)
            new_pop.append(child)
        pop = new_pop[:pop_size]

    results = list(archive.values())
    log_progress(f"Minute evolution finished: candidate_count={len(results)}.", module="mining")
    return results, minute_panel_mode

