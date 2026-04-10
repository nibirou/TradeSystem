"""Fundamental multi-objective mining model."""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List

import pandas as pd


def run_fundamental_multiobj_model(
    *,
    panel: pd.DataFrame,
    fields: List[str],
    cfg: Any,
    rng: Any,
    framework: str,
    eval_group_col: str,
    cache: Dict[str, Dict[str, object]],
    standard: Any,
    compute_fundamental_factor: Callable[..., pd.Series],
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
    if len(fields) < 2:
        raise RuntimeError("insufficient daily feature columns for fundamental mining")
    log_progress(
        f"开始基本面进化挖掘：feature_pool={len(fields)}, population={cfg.population_size}, "
        f"generations={cfg.generations}。",
        module="mining",
    )

    def _evaluate(spec: Any) -> Dict[str, object]:
        key = to_key(spec.to_dict())
        if key in cache:
            return cache[key]

        fac = compute_fundamental_factor(panel, spec, group_col=eval_group_col)
        tmp = panel[[eval_group_col, "code", "future_ret_n"]].copy()
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
            "metrics_train": m_tr,
            "metrics_valid": m_va,
            "objectives": obj.tolist(),
            "score": score,
            "passed": bool(passed),
            "admission": admission,
        }
        cache[key] = result
        return result

    pop: List[Any] = [random_spec_fn(rng, fields) for _ in range(int(cfg.population_size))]
    archive: Dict[str, Dict[str, object]] = {}

    for gen in range(int(cfg.generations)):
        res = [_evaluate(s) for s in pop]
        for r in res:
            k = str(r["key"])
            if k not in archive or float(r["score"]) > float(archive[k]["score"]):
                archive[k] = r

        objs = [r["objectives"] for r in res]
        parent_idx = nsga2_select(objs, n_select=max(8, int(cfg.population_size) // 2))
        parent_pool = [pop[i] for i in parent_idx]
        elites = sorted(res, key=lambda r: float(r["score"]), reverse=True)[: max(1, int(cfg.elite_size))]

        best = elites[0]
        log_progress(
            (
                f"[fundamental][gen={gen:02d}] best_score={best['score']:.4f} "
                f"absIC={best['metrics_valid'].get('abs_ic_mean', float('nan')):.4f} "
                f"win={best['metrics_valid'].get('ic_win_rate', float('nan')):.4f}"
            ),
            module="mining",
            level="debug",
        )

        new_pop: List[Any] = [copy.deepcopy(e["spec"]) for e in elites]
        while len(new_pop) < int(cfg.population_size):
            p1 = copy.deepcopy(parent_pool[int(rng.integers(0, len(parent_pool)))])
            child = p1
            if rng.random() < float(cfg.crossover_rate) and len(parent_pool) > 1:
                p2 = copy.deepcopy(parent_pool[int(rng.integers(0, len(parent_pool)))])
                child = crossover_spec_fn(p1, p2, rng)
            if rng.random() < float(cfg.mutation_rate):
                child = mutate_spec_fn(child, rng, fields)
            new_pop.append(child)
        pop = new_pop[: int(cfg.population_size)]

    results = list(archive.values())
    log_progress(f"基本面进化完成：candidate_count={len(results)}。", module="mining")
    return results

