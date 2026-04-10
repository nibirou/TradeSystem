"""GP library (gplearn) symbolic alpha mining model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class GPLearnProgramSpec:
    run_idx: int
    component_idx: int
    expression: str
    length: int
    depth: int
    raw_fitness: float
    random_state: int
    metric: str
    feature_cols: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _safe_expr(program: object, idx: int) -> str:
    try:
        txt = str(program)
        return txt if txt else f"program_{idx}"
    except Exception:
        return f"program_{idx}"


def _extract_program_stats(program: object) -> tuple[int, int, float]:
    length = int(getattr(program, "length_", 0)) if program is not None else 0
    depth = int(getattr(program, "depth_", 0)) if program is not None else 0
    raw_fit = float(getattr(program, "raw_fitness_", float("nan"))) if program is not None else float("nan")
    if not np.isfinite(raw_fit):
        raw_fit = 0.0
    return length, depth, raw_fit


def _resolve_function_set(raw: str) -> tuple[str, ...]:
    allowed = {"add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "inv", "max", "min"}
    parts = [str(x).strip().lower() for x in str(raw).split(",")]
    out = [x for x in parts if x in allowed]
    if not out:
        out = ["add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "max", "min"]
    return tuple(out)


def _resolve_gp_probs(crossover_rate: float, mutation_rate: float) -> tuple[float, float, float, float]:
    p_cross = float(np.clip(float(crossover_rate), 0.05, 0.95))
    mut_total = float(np.clip(float(mutation_rate), 0.01, 0.70))
    p_subtree = mut_total * 0.55
    p_hoist = mut_total * 0.15
    p_point = mut_total * 0.30
    total = p_cross + p_subtree + p_hoist + p_point
    if total >= 0.98:
        scale = 0.98 / total
        p_cross *= scale
        p_subtree *= scale
        p_hoist *= scale
        p_point *= scale
    return p_cross, p_subtree, p_hoist, p_point


def run_gplearn_symbolic_alpha_model(
    *,
    panel: pd.DataFrame,
    train_mask: pd.Series,
    time_col: str,
    eval_group_col: str,
    framework: str,
    cfg: Any,
    cache: Dict[str, Dict[str, object]],
    standard: Any,
    discover_feature_pool: Callable[[pd.DataFrame], List[str]],
    valid_nonconstant_features: Callable[[pd.DataFrame, Sequence[str]], List[str]],
    prefilter_features_by_ic: Callable[[pd.DataFrame, Sequence[str], str, int], List[str]],
    split_train_valid: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]],
    collect_metrics: Callable[[pd.DataFrame, str, str], Dict[str, float]],
    objectives_from_metrics: Callable[[Dict[str, float], str], List[float]],
    safe_obj: Callable[[Sequence[float]], np.ndarray],
    check_admission: Callable[[Dict[str, float], Any], tuple[bool, Dict[str, object]]],
    to_key: Callable[[Dict[str, object]], str],
    ic_score: Callable[[Dict[str, float]], float],
    winsorize_mad_cs: Callable[[pd.Series, pd.Series, float], pd.Series],
    neutralize_series: Callable[..., pd.Series],
    cs_zscore: Callable[[pd.Series, pd.Series], pd.Series],
    log_progress: Callable[..., None],
) -> List[Dict[str, object]]:
    try:
        from gplearn.genetic import SymbolicTransformer
    except Exception as exc:
        raise RuntimeError(
            "framework=gplearn_symbolic_alpha requires `gplearn`. "
            "Please install in your env_quant environment first, e.g. `pip install gplearn`."
        ) from exc

    fields_all = discover_feature_pool(panel)
    train_frame = panel.loc[train_mask].copy()
    fields_valid = valid_nonconstant_features(train_frame, fields_all)
    if len(fields_valid) < 5:
        raise RuntimeError("insufficient features for gplearn_symbolic_alpha")

    fields_pref = prefilter_features_by_ic(
        train_frame=train_frame,
        candidate_cols=fields_valid,
        target_col="future_ret_n",
        topk=int(cfg.gp_prefilter_topk),
    )
    feature_pool = fields_pref if len(fields_pref) >= 5 else fields_valid
    if len(feature_pool) < 5:
        raise RuntimeError("gplearn_symbolic_alpha prefiltered feature pool too small")

    pop_size = max(100, int(cfg.gp_population_size))
    generations = max(3, int(cfg.gp_generations))
    n_runs = max(1, int(cfg.gp_num_runs))
    hall_of_fame = int(np.clip(int(cfg.gp_hall_of_fame), 2, pop_size))
    n_components = int(np.clip(int(cfg.gp_n_components), 1, hall_of_fame))
    function_set = _resolve_function_set(cfg.gp_function_set)
    p_cross, p_subtree, p_hoist, p_point = _resolve_gp_probs(cfg.crossover_rate, cfg.mutation_rate)

    log_progress(
        f"Start GP-library mining: features={len(feature_pool)}, runs={n_runs}, "
        f"population={pop_size}, generations={generations}, components={n_components}.",
        module="mining",
    )

    all_x = panel[feature_pool].copy()
    for c in feature_pool:
        all_x[c] = pd.to_numeric(all_x[c], errors="coerce")
    all_x = all_x.replace([np.inf, -np.inf], np.nan)

    train_df = panel.loc[train_mask, feature_pool + ["future_ret_n"]].copy()
    for c in feature_pool:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
    train_df["future_ret_n"] = pd.to_numeric(train_df["future_ret_n"], errors="coerce")
    train_df = train_df.replace([np.inf, -np.inf], np.nan)

    med = train_df[feature_pool].median(numeric_only=True)
    valid_cols = [c for c in feature_pool if c in med.index and np.isfinite(float(med[c]))]
    if len(valid_cols) < 5:
        raise RuntimeError("gplearn_symbolic_alpha valid feature columns too few after median check")

    x_train = train_df[valid_cols].fillna(med[valid_cols])
    y_train = train_df["future_ret_n"]
    keep = y_train.notna()
    x_train = x_train.loc[keep]
    y_train = y_train.loc[keep]
    if len(x_train) < 3000:
        raise RuntimeError("gplearn_symbolic_alpha train rows too few (need >=3000)")

    sample_frac = float(np.clip(float(cfg.gp_train_sample_frac), 0.10, 1.0))
    max_rows = max(3000, int(cfg.gp_max_train_rows))
    n_target = int(min(max_rows, max(3000, round(len(x_train) * sample_frac))))
    if len(x_train) > n_target:
        sample_idx = x_train.sample(n=n_target, random_state=int(cfg.random_state)).index
        x_train = x_train.loc[sample_idx]
        y_train = y_train.loc[sample_idx]

    x_all = all_x[valid_cols].fillna(med[valid_cols])
    x_all_np = x_all.to_numpy(dtype=float)
    y_train_np = y_train.to_numpy(dtype=float)
    x_train_np = x_train.to_numpy(dtype=float)

    results: List[Dict[str, object]] = []
    metric_name = str(cfg.gp_metric).strip().lower()
    metric_name = metric_name if metric_name in {"spearman", "pearson"} else "spearman"

    for run_idx in range(n_runs):
        seed = int(cfg.random_state) + run_idx * 7919
        model = SymbolicTransformer(
            population_size=pop_size,
            generations=generations,
            tournament_size=int(np.clip(int(cfg.gp_tournament_size), 2, pop_size)),
            stopping_criteria=0.999,
            hall_of_fame=hall_of_fame,
            n_components=n_components,
            function_set=function_set,
            metric=metric_name,
            parsimony_coefficient=float(cfg.gp_parsimony),
            p_crossover=p_cross,
            p_subtree_mutation=p_subtree,
            p_hoist_mutation=p_hoist,
            p_point_mutation=p_point,
            p_point_replace=0.10,
            max_samples=float(np.clip(float(cfg.gp_max_samples), 0.20, 1.0)),
            verbose=0,
            random_state=seed,
            feature_names=valid_cols,
            init_depth=(2, max(3, int(cfg.gp_max_depth))),
            init_method="half and half",
            const_range=(-1.0, 1.0),
            n_jobs=(int(cfg.gp_num_jobs) if int(cfg.gp_num_jobs) != 0 else 1),
        )
        try:
            model.fit(x_train_np, y_train_np)
            z_all = model.transform(x_all_np)
        except Exception as exc:
            log_progress(f"gplearn run#{run_idx} failed: {exc}", module="mining", level="debug")
            continue

        best_programs = getattr(model, "_best_programs", None)
        if not isinstance(best_programs, (list, tuple)):
            best_programs = [None] * int(z_all.shape[1])

        for comp_idx in range(int(z_all.shape[1])):
            raw_series = pd.Series(z_all[:, comp_idx], index=panel.index, dtype=float)
            fac = winsorize_mad_cs(raw_series, group=panel[eval_group_col], limit=3.0)
            fac = neutralize_series(
                fac,
                panel,
                group_col=eval_group_col,
                size_col="barra_size_proxy",
                industry_col="industry_bucket" if "industry_bucket" in panel.columns else None,
            )
            fac = cs_zscore(fac, group=panel[eval_group_col])
            fac = pd.to_numeric(fac, errors="coerce")

            program = best_programs[comp_idx] if comp_idx < len(best_programs) else None
            length, depth, raw_fit = _extract_program_stats(program)
            spec = GPLearnProgramSpec(
                run_idx=int(run_idx),
                component_idx=int(comp_idx),
                expression=_safe_expr(program, comp_idx),
                length=int(length),
                depth=int(depth),
                raw_fitness=float(raw_fit),
                random_state=int(seed),
                metric=metric_name,
                feature_cols=[str(x) for x in valid_cols],
            )

            key = to_key(spec.to_dict())
            if key in cache:
                continue

            tmp = panel[[time_col, "code", "future_ret_n"]].copy()
            tmp["_factor"] = fac
            tmp_tr, tmp_va = split_train_valid(tmp)
            m_tr = collect_metrics(tmp_tr, "_factor", eval_group_col)
            m_va = collect_metrics(tmp_va, "_factor", eval_group_col)

            base_obj = 0.5 * (safe_obj(objectives_from_metrics(m_tr, framework)) + safe_obj(objectives_from_metrics(m_va, framework)))
            tr_ic = float(m_tr.get("abs_ic_mean", float("nan")))
            va_ic = float(m_va.get("abs_ic_mean", float("nan")))
            stability = -abs(tr_ic - va_ic) if np.isfinite(tr_ic) and np.isfinite(va_ic) else -1.0
            simplicity = -float(np.log1p(max(length, 0)))
            obj = np.concatenate([base_obj, np.asarray([stability, simplicity], dtype=float)], axis=0)

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
            results.append(result)

    if not results:
        raise RuntimeError("gplearn_symbolic_alpha produced no valid candidates")

    # Keep a manageable candidate set before global diversification stage.
    results = sorted(results, key=lambda r: float(r.get("score", -1e9)), reverse=True)
    keep_top = max(int(cfg.top_n) * 8, 64)
    results = results[:keep_top]
    log_progress(f"GP-library mining finished: candidate_count={len(results)}.", module="mining")
    return results

