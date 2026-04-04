"""Multi-objective selection helpers: NSGA-II / NSGA-III and short-board penalty."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


NEG_INF = -1e30


def _sanitize_objectives(objectives: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(objectives, dtype=float)
    if arr.ndim != 2:
        raise ValueError("objectives must be 2D")
    arr = np.where(np.isfinite(arr), arr, NEG_INF)
    return arr


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a >= b) and np.any(a > b))


def non_dominated_sort(objectives: Sequence[Sequence[float]]) -> List[List[int]]:
    obj = _sanitize_objectives(objectives)
    n = obj.shape[0]
    if n == 0:
        return []

    dominated_count = np.zeros(n, dtype=int)
    dominates_set: List[List[int]] = [[] for _ in range(n)]
    fronts: List[List[int]] = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(obj[i], obj[j]):
                dominates_set[i].append(j)
                dominated_count[j] += 1
            elif dominates(obj[j], obj[i]):
                dominates_set[j].append(i)
                dominated_count[i] += 1

    for i in range(n):
        if dominated_count[i] == 0:
            fronts[0].append(i)

    f = 0
    while f < len(fronts) and fronts[f]:
        next_front: List[int] = []
        for i in fronts[f]:
            for j in dominates_set[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        f += 1

    return fronts


def crowding_distance(objectives: Sequence[Sequence[float]], front: Sequence[int]) -> Dict[int, float]:
    obj = _sanitize_objectives(objectives)
    idx = list(front)
    if not idx:
        return {}
    m = obj.shape[1]
    dist = {i: 0.0 for i in idx}
    if len(idx) <= 2:
        for i in idx:
            dist[i] = float("inf")
        return dist

    for k in range(m):
        sorted_idx = sorted(idx, key=lambda i: obj[i, k])
        dist[sorted_idx[0]] = float("inf")
        dist[sorted_idx[-1]] = float("inf")
        lo = obj[sorted_idx[0], k]
        hi = obj[sorted_idx[-1], k]
        scale = hi - lo
        if scale <= 1e-12:
            continue
        for p in range(1, len(sorted_idx) - 1):
            i_prev = sorted_idx[p - 1]
            i_next = sorted_idx[p + 1]
            i_cur = sorted_idx[p]
            if np.isinf(dist[i_cur]):
                continue
            dist[i_cur] += float((obj[i_next, k] - obj[i_prev, k]) / scale)

    return dist


def nsga2_select(objectives: Sequence[Sequence[float]], n_select: int) -> List[int]:
    obj = _sanitize_objectives(objectives)
    n = obj.shape[0]
    if n_select >= n:
        return list(range(n))

    fronts = non_dominated_sort(obj)
    selected: List[int] = []
    for front in fronts:
        if len(selected) + len(front) <= n_select:
            selected.extend(front)
            continue
        dist = crowding_distance(obj, front)
        ranked = sorted(front, key=lambda i: dist.get(i, 0.0), reverse=True)
        selected.extend(ranked[: max(0, n_select - len(selected))])
        break
    return selected


def _generate_reference_points(n_obj: int, p: int) -> np.ndarray:
    points: List[List[float]] = []

    def rec(remaining: int, left: int, prefix: List[int]) -> None:
        if remaining == 1:
            points.append(prefix + [left])
            return
        for i in range(left + 1):
            rec(remaining - 1, left - i, prefix + [i])

    rec(n_obj, p, [])
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return np.ones((1, n_obj), dtype=float) / max(n_obj, 1)
    arr = arr / max(float(p), 1.0)
    return arr


def _normalize(obj: np.ndarray) -> np.ndarray:
    out = obj.copy()
    for k in range(out.shape[1]):
        col = out[:, k]
        lo = np.min(col)
        hi = np.max(col)
        if hi - lo <= 1e-12:
            out[:, k] = 0.5
        else:
            out[:, k] = (col - lo) / (hi - lo)
    return out


def _associate_to_refs(norm_obj: np.ndarray, ref_points: np.ndarray, idxs: Sequence[int]) -> Dict[int, Tuple[int, float]]:
    assoc: Dict[int, Tuple[int, float]] = {}
    for i in idxs:
        x = norm_obj[i]
        best_r = 0
        best_d = float("inf")
        for r_id, r in enumerate(ref_points):
            r_norm = np.linalg.norm(r)
            if r_norm <= 1e-12:
                d = float(np.linalg.norm(x))
            else:
                proj = (np.dot(x, r) / (r_norm**2)) * r
                d = float(np.linalg.norm(x - proj))
            if d < best_d:
                best_d = d
                best_r = r_id
        assoc[i] = (best_r, best_d)
    return assoc


def nsga3_select(objectives: Sequence[Sequence[float]], n_select: int, ref_divisions: int = 8) -> List[int]:
    obj = _sanitize_objectives(objectives)
    n, m = obj.shape
    if n_select >= n:
        return list(range(n))

    fronts = non_dominated_sort(obj)
    selected: List[int] = []
    last_front: List[int] = []
    for front in fronts:
        if len(selected) + len(front) <= n_select:
            selected.extend(front)
        else:
            last_front = list(front)
            break

    if len(selected) == n_select:
        return selected

    ref_points = _generate_reference_points(m, p=max(2, int(ref_divisions)))
    norm_obj = _normalize(obj)

    assoc_all = _associate_to_refs(norm_obj, ref_points, selected + last_front)
    niche_count = np.zeros(len(ref_points), dtype=int)
    for i in selected:
        r_id, _ = assoc_all[i]
        niche_count[r_id] += 1

    remain = max(0, n_select - len(selected))
    candidates = set(last_front)
    while remain > 0 and candidates:
        by_ref: Dict[int, List[int]] = {}
        for i in candidates:
            r_id, _d = assoc_all[i]
            by_ref.setdefault(r_id, []).append(i)

        available_refs = [r for r, arr in by_ref.items() if arr]
        if not available_refs:
            break
        min_niche = int(np.min([niche_count[r] for r in available_refs]))
        ref_candidates = [r for r in available_refs if niche_count[r] == min_niche]

        # deterministic tie-break: smaller ref id first
        r_pick = sorted(ref_candidates)[0]
        pool = by_ref[r_pick]
        pool_sorted = sorted(pool, key=lambda i: assoc_all[i][1])
        pick = pool_sorted[0]

        selected.append(pick)
        candidates.remove(pick)
        niche_count[r_pick] += 1
        remain -= 1

    if remain > 0 and candidates:
        # fallback by non-dominated crowding in last front
        dist = crowding_distance(norm_obj, list(candidates))
        ranked = sorted(candidates, key=lambda i: dist.get(i, 0.0), reverse=True)
        selected.extend(ranked[:remain])

    return selected[:n_select]


def apply_dynamic_shortboard_penalty(
    objectives: Sequence[Sequence[float]],
    floor_quantile: float = 0.30,
    penalty_strength: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply dynamic weak-dimension penalty (for NSGA-III high-dim objective stability)."""
    obj = _sanitize_objectives(objectives)
    if len(obj) == 0:
        return obj, np.asarray([], dtype=float)

    norm = _normalize(obj)
    q = float(np.clip(floor_quantile, 0.0, 1.0))
    floor = np.quantile(norm, q=q, axis=0)
    deficit = np.maximum(0.0, floor.reshape(1, -1) - norm)
    penalty = float(max(penalty_strength, 0.0)) * deficit.mean(axis=1)

    penalized = norm - penalty.reshape(-1, 1)
    return penalized, penalty
