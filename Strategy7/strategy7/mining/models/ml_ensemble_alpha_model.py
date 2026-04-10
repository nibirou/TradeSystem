"""ML ensemble alpha mining model."""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List

import pandas as pd


def run_ml_ensemble_alpha_model(
    *,
    panel: pd.DataFrame,
    train_mask: pd.Series,
    time_col: str,
    eval_group_col: str,
    framework: str,
    cfg: Any,
    rng: Any,
    cache: Dict[str, Dict[str, object]],
    standard: Any,
    discover_daily_feature_pool: Callable[[pd.DataFrame], List[str]],
    valid_nonconstant_features: Callable[[pd.DataFrame, List[str]], List[str]],
    prefilter_features_by_ic: Callable[[pd.DataFrame, List[str], str, int], List[str]],
    fit_ml_factor_series: Callable[..., pd.Series],
    split_train_valid: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]],
    collect_metrics: Callable[[pd.DataFrame, str, str], Dict[str, float]],
    objectives_from_metrics: Callable[[Dict[str, float], str], List[float]],
    safe_obj: Callable[[List[float]], Any],
    check_admission: Callable[[Dict[str, float], Any], tuple[bool, Dict[str, object]]],
    to_key: Callable[[Dict[str, object]], str],
    ic_score: Callable[[Dict[str, float]], float],
    nsga2_select: Callable[..., List[int]],
    random_spec_fn: Callable[..., Any],
    mutate_spec_fn: Callable[..., Any],
    crossover_spec_fn: Callable[..., Any],
    log_progress: Callable[..., None],
) -> List[Dict[str, object]]:
    fields_all = discover_daily_feature_pool(panel)
    train_frame = panel.loc[train_mask].copy()
    fields_valid = valid_nonconstant_features(train_frame, fields_all)
    if len(fields_valid) < 5:
        raise RuntimeError("insufficient daily features for ml_ensemble_alpha")

    fields_pref = prefilter_features_by_ic(
        train_frame=train_frame,
        candidate_cols=fields_valid,
        target_col="future_ret_n",
        topk=int(cfg.ml_prefilter_topk),
    )
    feature_pool = fields_pref if len(fields_pref) >= 5 else fields_valid
    if len(feature_pool) < 5:
        raise RuntimeError("ml_ensemble_alpha prefiltered feature pool too small")
    log_progress(
        f"开始 ML 集成进化挖掘：raw_features={len(fields_all)}, valid_features={len(fields_valid)}, "
        f"prefilter_features={len(feature_pool)}, population={max(8, int(cfg.ml_population_size))}, "
        f"generations={max(1, int(cfg.ml_generations))}。",
        module="mining",
    )

    pop_size = max(8, int(cfg.ml_population_size))
    generations = max(1, int(cfg.ml_generations))
    archive: Dict[str, Dict[str, object]] = {}

    def _evaluate(spec: Any) -> Dict[str, object]:
        key = to_key(spec.to_dict())
        if key in cache:
            return cache[key]

        fac = fit_ml_factor_series(
            panel=panel,
            train_mask=train_mask,
            spec=spec,
            cfg=cfg,
            group_col=eval_group_col,
        )
        tmp = panel[[time_col, "code", "future_ret_n"]].copy()
        tmp["_factor"] = pd.to_numeric(fac, errors="coerce")

        tmp_tr, tmp_va = split_train_valid(tmp)
        m_tr = collect_metrics(tmp_tr, "_factor", eval_group_col)
        m_va = collect_metrics(tmp_va, "_factor", eval_group_col)
        obj = 0.5 * (safe_obj(objectives_from_metrics(m_tr, framework)) + safe_obj(objectives_from_metrics(m_va, framework)))

        passed, admission = check_admission(m_va, standard)
        score = ic_score(m_va)
        result = {
            "key": key,
            "spec": spec,
            "series": pd.DataFrame({time_col: panel[time_col], "code": panel["code"], "v": fac})
            .set_index([time_col, "code"])["v"]
            .sort_index(),
            "metrics_train": m_tr,
            "metrics_valid": m_va,
            "objectives": obj.tolist(),
            "score": score,
            "passed": bool(passed),
            "admission": admission,
        }
        cache[key] = result
        return result

    pop: List[Any] = [random_spec_fn(rng, feature_pool=feature_pool, cfg=cfg) for _ in range(pop_size)]
    for gen in range(generations):
        res = [_evaluate(s) for s in pop]
        for r in res:
            k = str(r["key"])
            if k not in archive or float(r["score"]) > float(archive[k]["score"]):
                archive[k] = r

        objs = [r["objectives"] for r in res]
        parent_idx = nsga2_select(objs, n_select=max(8, pop_size // 2))
        parent_pool = [pop[i] for i in parent_idx]
        elites = sorted(res, key=lambda r: float(r["score"]), reverse=True)[: max(1, int(cfg.elite_size))]

        best = elites[0]
        log_progress(
            (
                f"[ml_ensemble][gen={gen:02d}] best_score={best['score']:.4f} "
                f"absIC={best['metrics_valid'].get('abs_ic_mean', float('nan')):.4f} "
                f"win={best['metrics_valid'].get('ic_win_rate', float('nan')):.4f} "
                f"featN={len(best['spec'].feature_cols)} "
                f"model={best['spec'].model_name}"
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
                child = crossover_spec_fn(p1, p2, rng, cfg=cfg)
            if rng.random() < float(cfg.mutation_rate):
                child = mutate_spec_fn(child, rng, feature_pool=feature_pool, cfg=cfg)
            new_pop.append(child)
        pop = new_pop[:pop_size]

    results = list(archive.values())
    log_progress(f"ML 集成进化完成：candidate_count={len(results)}。", module="mining")
    return results

