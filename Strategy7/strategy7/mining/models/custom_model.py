"""Custom-expression factor mining model."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

import pandas as pd


def run_custom_model(
    *,
    panel: pd.DataFrame,
    specs: Sequence[Any],
    time_col: str,
    framework: str,
    eval_group_col: str,
    cache: Dict[str, Dict[str, object]],
    standard: Any,
    evaluate_custom_specs: Callable[..., pd.DataFrame],
    split_train_valid: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]],
    collect_metrics: Callable[[pd.DataFrame, str, str], Dict[str, float]],
    check_admission: Callable[[Dict[str, float], Any], tuple[bool, Dict[str, object]]],
    objectives_from_metrics: Callable[[Dict[str, float], str], List[float]],
    to_key: Callable[[Dict[str, object]], str],
    ic_score: Callable[[Dict[str, float]], float],
    log_progress: Callable[..., None],
) -> List[Dict[str, object]]:
    if not specs:
        raise RuntimeError("custom framework requires at least one custom factor spec")
    log_progress(f"开始评估 custom 因子：spec_count={len(specs)}。", module="mining")

    results: List[Dict[str, object]] = []
    for i, spec in enumerate(specs, start=1):
        fac_df = evaluate_custom_specs(panel=panel, specs=[spec], time_col=time_col, code_col="code")
        factor_col = str(spec.name)
        if factor_col not in fac_df.columns:
            log_progress(
                f"跳过 custom 规格（未生成列）：index={i}, factor={factor_col}",
                module="mining",
                level="debug",
            )
            continue

        tmp = panel[[time_col, "code", "future_ret_n"]].copy()
        tmp[factor_col] = pd.to_numeric(fac_df[factor_col], errors="coerce")
        tmp_tr, tmp_va = split_train_valid(tmp)
        m_tr = collect_metrics(tmp_tr, factor_col, eval_group_col)
        m_va = collect_metrics(tmp_va, factor_col, eval_group_col)
        passed, admission = check_admission(m_va, standard)
        key = to_key(spec.to_dict())
        result = {
            "key": key,
            "spec": spec,
            "name": factor_col,
            "metrics_train": m_tr,
            "metrics_valid": m_va,
            "objectives": objectives_from_metrics(m_va, framework=framework),
            "score": ic_score(m_va),
            "passed": bool(passed),
            "admission": admission,
        }
        cache[key] = result
        results.append(result)

    log_progress(f"custom 因子评估完成：candidate_count={len(results)}。", module="mining")
    return results

