"""End-to-end factor mining runner (fundamental/mminute/custom) with catalog persistence."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from ..core.constants import INTRADAY_FREQS
from ..core.utils import dump_json, ensure_dir, log_progress
from ..factors.labeling import validate_label_frequency_alignment
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
    cs_zscore,
    compute_fundamental_factor,
    compute_minute_factor_daily,
    compute_minute_factor_panel,
    neutralize_series,
    winsorize_mad_cs,
)
from .nsga import apply_dynamic_shortboard_penalty, nsga2_select, nsga3_select


@dataclass
class MLEnsembleFormulaSpec:
    model_name: str
    feature_cols: List[str]
    n_estimators: int
    max_depth: int
    min_samples_leaf: int
    learning_rate: float
    max_leaf_nodes: int
    subsample: float
    random_state: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


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
    # ml_ensemble_alpha 专用参数（仅该框架读取，不影响旧框架）
    ml_population_size: int = 48
    ml_generations: int = 10
    ml_model_pool: str = "rf,et,hgbt"
    ml_prefilter_topk: int = 80
    ml_feature_min: int = 10
    ml_feature_max: int = 36
    ml_train_sample_frac: float = 0.40
    ml_max_train_rows: int = 220000
    ml_num_jobs: int = -1


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


def _primary_time_col(freq: str) -> str:
    return "datetime" if str(freq).lower() in INTRADAY_FREQS else "date"


def _time_anchor(ts: pd.Series, freq: str) -> pd.Series:
    out = pd.to_datetime(ts, errors="coerce")
    if str(freq).lower() in INTRADAY_FREQS:
        return out.dt.normalize()
    return out


def _split_mask_by_freq(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    time_col: str,
    freq: str,
) -> pd.Series:
    t = _time_anchor(df[time_col], freq=freq)
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


_MINUTE_UNARY_OPS_V1 = ["mean", "std", "skew", "kurt", "rank", "slope", "r2", "max", "min", "abs_mean"]
_MINUTE_CROSS_OPS_V1 = ["corr", "cov", "beta", "euc_dist", "ols_intercept", "spread_mean", "r2"]
_MINUTE_WINDOWS_V1 = [20, 30, 45, 60, 90, 120]
_MINUTE_SLICES_V1 = [None, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
_MINUTE_MASKS_V1 = ["none", "high_0.7", "high_0.8", "low_0.2", "low_0.3"]

# minute_parametric_plus 使用增强算子池；旧框架参数空间保持不变。
_MINUTE_UNARY_OPS_V2 = _MINUTE_UNARY_OPS_V1 + [
    "median",
    "mad",
    "iqr",
    "up_ratio",
    "down_ratio",
    "tail_ratio",
    "energy",
    "entropy",
    "autocorr1",
]
_MINUTE_CROSS_OPS_V2 = _MINUTE_CROSS_OPS_V1 + [
    "spearman_corr",
    "kendall_corr",
    "cosine_sim",
    "zspread_mean",
    "downside_beta",
    "corr_diff1",
]
_MINUTE_WINDOWS_V2 = [15, 20, 30, 45, 60, 90, 120, 180, 240]
_MINUTE_SLICES_V2 = [None, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
_MINUTE_MASKS_V2 = _MINUTE_MASKS_V1 + ["high_0.9", "low_0.1"]


def _minute_random_spec_with_space(
    rng: np.random.Generator,
    fields: List[str],
    unary_ops: Sequence[str],
    cross_ops: Sequence[str],
    windows: Sequence[int],
    slices: Sequence[float | None],
    masks: Sequence[str],
    shift_lag: tuple[int, int],
) -> MinuteFormulaSpec:
    lo, hi = int(shift_lag[0]), int(shift_lag[1])
    if lo > hi:
        lo, hi = hi, lo

    mode = int(rng.choice([1, 2]))
    return MinuteFormulaSpec(
        a_field=str(rng.choice(fields)),
        b_field=str(rng.choice(fields)),
        window=int(rng.choice(list(windows))),
        slice_pos=rng.choice(list(slices)),
        mask_field=str(rng.choice(fields)),
        mask_rule=str(rng.choice(list(masks))),
        mode=mode,
        op_name=str(rng.choice(list(unary_ops))),
        cross_op_name=str(rng.choice(list(cross_ops))),
        b_shift_lag=int(rng.integers(lo, hi + 1)),
    )


def _minute_mutate_with_space(
    spec: MinuteFormulaSpec,
    rng: np.random.Generator,
    fields: List[str],
    unary_ops: Sequence[str],
    cross_ops: Sequence[str],
    windows: Sequence[int],
    slices: Sequence[float | None],
    masks: Sequence[str],
    shift_lag: tuple[int, int],
) -> MinuteFormulaSpec:
    lo, hi = int(shift_lag[0]), int(shift_lag[1])
    if lo > hi:
        lo, hi = hi, lo

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
        setattr(s, attr, int(rng.choice(list(windows))))
    elif attr == "slice_pos":
        setattr(s, attr, rng.choice(list(slices)))
    elif attr == "mask_rule":
        setattr(s, attr, str(rng.choice(list(masks))))
    elif attr == "mode":
        setattr(s, attr, int(rng.choice([1, 2])))
    elif attr == "op_name":
        setattr(s, attr, str(rng.choice(list(unary_ops))))
    elif attr == "cross_op_name":
        setattr(s, attr, str(rng.choice(list(cross_ops))))
    elif attr == "b_shift_lag":
        setattr(s, attr, int(rng.integers(lo, hi + 1)))
    return s


def _minute_random_spec(rng: np.random.Generator, fields: List[str]) -> MinuteFormulaSpec:
    return _minute_random_spec_with_space(
        rng=rng,
        fields=fields,
        unary_ops=_MINUTE_UNARY_OPS_V1,
        cross_ops=_MINUTE_CROSS_OPS_V1,
        windows=_MINUTE_WINDOWS_V1,
        slices=_MINUTE_SLICES_V1,
        masks=_MINUTE_MASKS_V1,
        shift_lag=(-5, 5),
    )


def _minute_mutate(spec: MinuteFormulaSpec, rng: np.random.Generator, fields: List[str]) -> MinuteFormulaSpec:
    return _minute_mutate_with_space(
        spec=spec,
        rng=rng,
        fields=fields,
        unary_ops=_MINUTE_UNARY_OPS_V1,
        cross_ops=_MINUTE_CROSS_OPS_V1,
        windows=_MINUTE_WINDOWS_V1,
        slices=_MINUTE_SLICES_V1,
        masks=_MINUTE_MASKS_V1,
        shift_lag=(-5, 5),
    )


def _minute_plus_random_spec(rng: np.random.Generator, fields: List[str]) -> MinuteFormulaSpec:
    return _minute_random_spec_with_space(
        rng=rng,
        fields=fields,
        unary_ops=_MINUTE_UNARY_OPS_V2,
        cross_ops=_MINUTE_CROSS_OPS_V2,
        windows=_MINUTE_WINDOWS_V2,
        slices=_MINUTE_SLICES_V2,
        masks=_MINUTE_MASKS_V2,
        shift_lag=(-10, 10),
    )


def _minute_plus_mutate(spec: MinuteFormulaSpec, rng: np.random.Generator, fields: List[str]) -> MinuteFormulaSpec:
    return _minute_mutate_with_space(
        spec=spec,
        rng=rng,
        fields=fields,
        unary_ops=_MINUTE_UNARY_OPS_V2,
        cross_ops=_MINUTE_CROSS_OPS_V2,
        windows=_MINUTE_WINDOWS_V2,
        slices=_MINUTE_SLICES_V2,
        masks=_MINUTE_MASKS_V2,
        shift_lag=(-10, 10),
    )


def _minute_crossover(a: MinuteFormulaSpec, b: MinuteFormulaSpec, rng: np.random.Generator) -> MinuteFormulaSpec:
    out = copy.deepcopy(a)
    for k in out.to_dict().keys():
        if bool(rng.integers(0, 2)):
            setattr(out, k, getattr(b, k))
    return out


def _valid_nonconstant_features(frame: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    out: List[str] = []
    for c in cols:
        if c not in frame.columns:
            continue
        s = pd.to_numeric(frame[c], errors="coerce")
        if int(s.notna().sum()) < 100:
            continue
        std = float(s.std(ddof=0))
        if np.isfinite(std) and std > 1e-12:
            out.append(str(c))
    return sorted(set(out))


def _prefilter_features_by_ic(
    train_frame: pd.DataFrame,
    candidate_cols: Sequence[str],
    target_col: str,
    topk: int,
) -> List[str]:
    scores: List[tuple[str, float]] = []
    y = pd.to_numeric(train_frame[target_col], errors="coerce")
    for c in candidate_cols:
        x = pd.to_numeric(train_frame[c], errors="coerce")
        valid = x.notna() & y.notna()
        if int(valid.sum()) < 200:
            continue
        if int(x[valid].nunique()) < 5:
            continue
        ic = x[valid].corr(y[valid], method="spearman")
        if pd.notna(ic):
            scores.append((str(c), abs(float(ic))))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    if int(topk) <= 0:
        return [k for k, _ in scores]
    return [k for k, _ in scores[: int(topk)]]


def _bounded_int(v: int, lo: int, hi: int) -> int:
    if lo > hi:
        lo, hi = hi, lo
    return int(np.clip(int(v), lo, hi))


def _random_feature_subset(
    rng: np.random.Generator,
    feature_pool: Sequence[str],
    n_min: int,
    n_max: int,
) -> List[str]:
    pool = list(dict.fromkeys([str(x) for x in feature_pool if str(x)]))
    if not pool:
        return []
    n_min = max(1, int(n_min))
    n_max = max(n_min, int(n_max))
    n_pick = _bounded_int(int(rng.integers(n_min, n_max + 1)), 1, len(pool))
    return sorted([str(x) for x in rng.choice(pool, size=n_pick, replace=False).tolist()])


def _parse_model_pool(raw: str) -> List[str]:
    allowed = {"rf", "et", "hgbt"}
    parts = [str(x).strip().lower() for x in str(raw).split(",")]
    out = [x for x in parts if x in allowed]
    return out if out else ["rf", "et", "hgbt"]


def _ml_random_spec(rng: np.random.Generator, feature_pool: Sequence[str], cfg: FactorMiningConfig) -> MLEnsembleFormulaSpec:
    model_pool = _parse_model_pool(cfg.ml_model_pool)
    return MLEnsembleFormulaSpec(
        model_name=str(rng.choice(model_pool)),
        feature_cols=_random_feature_subset(rng, feature_pool, cfg.ml_feature_min, cfg.ml_feature_max),
        n_estimators=int(rng.choice([120, 160, 200, 260, 320])),
        max_depth=int(rng.choice([3, 4, 5, 6, 8, 10, 12])),
        min_samples_leaf=int(rng.choice([8, 16, 24, 36, 48, 64])),
        learning_rate=float(rng.choice([0.03, 0.05, 0.08, 0.10])),
        max_leaf_nodes=int(rng.choice([15, 31, 63, 127, 255])),
        subsample=float(rng.choice([0.55, 0.65, 0.75, 0.85, 1.0])),
        random_state=int(rng.integers(1, 10_000_000)),
    )


def _ml_mutate(spec: MLEnsembleFormulaSpec, rng: np.random.Generator, feature_pool: Sequence[str], cfg: FactorMiningConfig) -> MLEnsembleFormulaSpec:
    s = copy.deepcopy(spec)
    attrs = [
        "model_name",
        "feature_cols",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "learning_rate",
        "max_leaf_nodes",
        "subsample",
    ]
    attr = str(rng.choice(attrs))
    if attr == "model_name":
        s.model_name = str(rng.choice(_parse_model_pool(cfg.ml_model_pool)))
    elif attr == "feature_cols":
        current = set(s.feature_cols)
        action = str(rng.choice(["add", "drop", "swap", "resample"]))
        if action == "resample" or len(current) < 2:
            s.feature_cols = _random_feature_subset(rng, feature_pool, cfg.ml_feature_min, cfg.ml_feature_max)
        else:
            if action in {"add", "swap"} and len(current) < len(feature_pool):
                remain = [f for f in feature_pool if f not in current]
                if remain:
                    current.add(str(rng.choice(remain)))
            if action in {"drop", "swap"} and len(current) > max(2, int(cfg.ml_feature_min)):
                current.remove(str(rng.choice(sorted(list(current)))))
            s.feature_cols = sorted(current)
    elif attr == "n_estimators":
        s.n_estimators = int(rng.choice([120, 160, 200, 260, 320]))
    elif attr == "max_depth":
        s.max_depth = int(rng.choice([3, 4, 5, 6, 8, 10, 12]))
    elif attr == "min_samples_leaf":
        s.min_samples_leaf = int(rng.choice([8, 16, 24, 36, 48, 64]))
    elif attr == "learning_rate":
        s.learning_rate = float(rng.choice([0.03, 0.05, 0.08, 0.10]))
    elif attr == "max_leaf_nodes":
        s.max_leaf_nodes = int(rng.choice([15, 31, 63, 127, 255]))
    elif attr == "subsample":
        s.subsample = float(rng.choice([0.55, 0.65, 0.75, 0.85, 1.0]))
    s.feature_cols = sorted(set([x for x in s.feature_cols if x in set(feature_pool)]))
    return s


def _ml_crossover(a: MLEnsembleFormulaSpec, b: MLEnsembleFormulaSpec, rng: np.random.Generator, cfg: FactorMiningConfig) -> MLEnsembleFormulaSpec:
    out = copy.deepcopy(a)
    for k in ["model_name", "n_estimators", "max_depth", "min_samples_leaf", "learning_rate", "max_leaf_nodes", "subsample"]:
        if bool(rng.integers(0, 2)):
            setattr(out, k, getattr(b, k))
    union_feats = sorted(set(a.feature_cols) | set(b.feature_cols))
    out.feature_cols = _random_feature_subset(
        rng=rng,
        feature_pool=union_feats if union_feats else a.feature_cols,
        n_min=max(2, int(cfg.ml_feature_min)),
        n_max=max(2, int(cfg.ml_feature_max)),
    )
    return out


def _fit_ml_factor_series(
    panel: pd.DataFrame,
    train_mask: pd.Series,
    spec: MLEnsembleFormulaSpec,
    cfg: FactorMiningConfig,
    *,
    group_col: str,
) -> pd.Series:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

    cols = [c for c in spec.feature_cols if c in panel.columns]
    if len(cols) < 2:
        return pd.Series(np.nan, index=panel.index, dtype=float)

    train_df = panel.loc[train_mask, cols + ["future_ret_n"]].copy()
    for c in cols:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
    train_df["future_ret_n"] = pd.to_numeric(train_df["future_ret_n"], errors="coerce")
    train_df = train_df.replace([np.inf, -np.inf], np.nan)

    med = train_df[cols].median(numeric_only=True)
    valid_cols = [c for c in cols if c in med.index and np.isfinite(float(med[c]))]
    if len(valid_cols) < 2:
        return pd.Series(np.nan, index=panel.index, dtype=float)

    train_x = train_df[valid_cols].fillna(med[valid_cols])
    train_y = train_df["future_ret_n"]
    valid_rows = train_y.notna()
    train_x = train_x.loc[valid_rows]
    train_y = train_y.loc[valid_rows]
    if len(train_x) < 1200:
        return pd.Series(np.nan, index=panel.index, dtype=float)

    sample_frac = float(np.clip(spec.subsample * float(cfg.ml_train_sample_frac), 0.10, 1.0))
    cap = int(max(2000, cfg.ml_max_train_rows))
    n_target = int(min(cap, max(1000, round(len(train_x) * sample_frac))))
    if len(train_x) > n_target:
        sample_idx = train_x.sample(n=n_target, random_state=int(spec.random_state)).index
        train_x = train_x.loc[sample_idx]
        train_y = train_y.loc[sample_idx]

    model_name = str(spec.model_name).strip().lower()
    if model_name == "rf":
        model = RandomForestRegressor(
            n_estimators=int(spec.n_estimators),
            max_depth=int(spec.max_depth),
            min_samples_leaf=int(spec.min_samples_leaf),
            random_state=int(spec.random_state),
            n_jobs=int(cfg.ml_num_jobs),
        )
    elif model_name == "et":
        model = ExtraTreesRegressor(
            n_estimators=int(spec.n_estimators),
            max_depth=int(spec.max_depth),
            min_samples_leaf=int(spec.min_samples_leaf),
            random_state=int(spec.random_state),
            n_jobs=int(cfg.ml_num_jobs),
        )
    else:
        model = HistGradientBoostingRegressor(
            max_depth=int(spec.max_depth),
            max_leaf_nodes=int(spec.max_leaf_nodes),
            min_samples_leaf=int(spec.min_samples_leaf),
            learning_rate=float(spec.learning_rate),
            max_iter=max(80, int(spec.n_estimators)),
            random_state=int(spec.random_state),
        )

    try:
        model.fit(train_x.to_numpy(dtype=float), train_y.to_numpy(dtype=float))
    except Exception:
        return pd.Series(np.nan, index=panel.index, dtype=float)

    all_x = panel[valid_cols].copy()
    for c in valid_cols:
        all_x[c] = pd.to_numeric(all_x[c], errors="coerce")
    all_x = all_x.replace([np.inf, -np.inf], np.nan).fillna(med[valid_cols])
    try:
        pred = model.predict(all_x.to_numpy(dtype=float))
    except Exception:
        return pd.Series(np.nan, index=panel.index, dtype=float)

    if group_col not in panel.columns:
        return pd.Series(np.nan, index=panel.index, dtype=float)

    fac = pd.Series(pred, index=panel.index, dtype=float)
    fac = winsorize_mad_cs(fac, group=panel[group_col], limit=3.0)
    fac = neutralize_series(
        fac,
        panel,
        group_col=group_col,
        size_col="barra_size_proxy",
        industry_col="industry_bucket" if "industry_bucket" in panel.columns else None,
    )
    fac = cs_zscore(fac, group=panel[group_col])
    return pd.to_numeric(fac, errors="coerce")


def _collect_metrics(
    df: pd.DataFrame,
    factor_col: str,
    cfg: FactorMiningConfig,
    framework: str,
    *,
    group_col: str,
) -> Dict[str, float]:
    ppy = periods_per_year_from_freq(cfg.factor_freq, cfg.horizon)
    return evaluate_factor_panel(
        panel=df,
        factor_col=factor_col,
        ret_col="future_ret_n",
        group_col=group_col,
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
    panel_with_label: pd.DataFrame,
    minute_df: pd.DataFrame | None = None,
    custom_specs: List[CustomFactorSpec] | None = None,
) -> Dict[str, object]:
    """Run mining and persist admitted factors into factor catalog.

    Required columns in panel_with_label:
    - code
    - primary time key: datetime (intraday) / date (D/W/M)
    - future_ret_n
    - (optional) barra_size_proxy, industry_bucket
    """
    rng = _seed(cfg.random_state)
    framework = str(cfg.framework).strip().lower()
    factor_freq = str(cfg.factor_freq)
    time_col = _primary_time_col(factor_freq)
    log_progress(
        f"挖掘启动：framework={framework}, factor_freq={cfg.factor_freq}, "
        f"train={pd.Timestamp(cfg.train_start).date()}~{pd.Timestamp(cfg.train_end).date()}, "
        f"valid={pd.Timestamp(cfg.valid_start).date()}~{pd.Timestamp(cfg.valid_end).date()}。",
        module="mining",
    )

    # 1) Normalize and validate panel with frequency-aligned future return labels.
    panel = panel_with_label.copy()
    if time_col not in panel.columns:
        raise RuntimeError(f"panel missing required time column: {time_col} (factor_freq={factor_freq})")
    panel[time_col] = pd.to_datetime(panel[time_col], errors="coerce")
    if "date" in panel.columns:
        panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    elif time_col == "datetime":
        panel["date"] = panel["datetime"].dt.normalize()
    else:
        panel["date"] = pd.to_datetime(panel[time_col], errors="coerce").dt.normalize()
    panel["code"] = panel["code"].astype(str).str.strip()
    panel = panel.dropna(subset=[time_col, "code", "future_ret_n"]).sort_values([time_col, "code"]).reset_index(drop=True)
    align = validate_label_frequency_alignment(
        panel=panel,
        factor_freq=factor_freq,
        horizon=int(cfg.horizon),
        strict=True,
    )

    def _target_anchor(frame: pd.DataFrame) -> pd.Series:
        if "target_date" in frame.columns:
            raw = pd.to_datetime(frame["target_date"], errors="coerce")
        elif "exit_ts" in frame.columns:
            raw = pd.to_datetime(frame["exit_ts"], errors="coerce")
        elif "exit_date" in frame.columns:
            raw = pd.to_datetime(frame["exit_date"], errors="coerce")
        else:
            raw = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
        return _time_anchor(raw, freq=factor_freq)

    def _build_split_mask(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        sig = _split_mask_by_freq(frame, start, end, time_col=time_col, freq=factor_freq)
        tgt = _target_anchor(frame)
        if tgt.notna().any():
            sig = sig & (tgt <= pd.Timestamp(end))
        return sig

    train_mask = _build_split_mask(panel, cfg.train_start, cfg.train_end)
    valid_mask = _build_split_mask(panel, cfg.valid_start, cfg.valid_end)
    if int(train_mask.sum()) <= 0:
        raise RuntimeError("factor mining train set is empty")
    if int(valid_mask.sum()) <= 0:
        raise RuntimeError("factor mining valid set is empty")

    def _split_train_valid(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split by date range on the frame itself.

        We intentionally avoid reusing outer boolean masks by index alignment because
        some candidate-evaluation paths (e.g. merge/join) may rebuild indexes.
        """
        tm = _build_split_mask(frame, cfg.train_start, cfg.train_end)
        vm = _build_split_mask(frame, cfg.valid_start, cfg.valid_end)
        return frame.loc[tm], frame.loc[vm]

    log_progress(
        f"输入面板清洗完成：rows={len(panel)}, time_col={time_col}, "
        f"train_rows={int(train_mask.sum())}, valid_rows={int(valid_mask.sum())}, "
        f"bad_entry_shift={int(align.get('bad_entry_shift', 0))}, bad_exit_shift={int(align.get('bad_exit_shift', 0))}。",
        module="mining",
    )

    # 2) Resolve admission standard (supports CLI threshold overrides).
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
    log_progress(
        f"准入标准已就绪：profile={standard.profile}, min_abs_ic_mean={standard.min_abs_ic_mean}, "
        f"min_ic_win_rate={standard.min_ic_win_rate}, min_ic_ir={standard.min_ic_ir}。",
        module="mining",
    )

    # Candidate cache avoids repeated recomputation for duplicated specs in evolution.
    cache: Dict[str, Dict[str, object]] = {}
    eval_group_col = time_col
    minute_panel_mode = False

    if framework == "custom":
        # 3A) Custom framework: evaluate user-provided expressions directly.
        specs = custom_specs or []
        if not specs:
            raise RuntimeError("custom framework requires at least one custom factor spec")
        log_progress(f"开始评估 custom 因子：spec_count={len(specs)}。", module="mining")

        fac_df = evaluate_custom_specs(panel=panel, specs=specs, time_col=time_col, code_col="code")
        results: List[Dict[str, object]] = []
        for spec in specs:
            factor_col = spec.name
            tmp = panel[[time_col, "code", "future_ret_n"]].copy()
            tmp[factor_col] = pd.to_numeric(fac_df[factor_col], errors="coerce")
            tmp_tr, tmp_va = _split_train_valid(tmp)
            m_tr = _collect_metrics(tmp_tr, factor_col=factor_col, cfg=cfg, framework=framework, group_col=eval_group_col)
            m_va = _collect_metrics(tmp_va, factor_col=factor_col, cfg=cfg, framework=framework, group_col=eval_group_col)
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
        log_progress(f"custom 因子评估完成：candidate_count={len(results)}。", module="mining")

    elif framework == "fundamental_multiobj":
        # 3B) Fundamental framework: NSGA-II over parametric daily formulas.
        fields = _discover_daily_feature_pool(panel)
        if len(fields) < 2:
            raise RuntimeError("insufficient daily feature columns for fundamental mining")
        log_progress(
            f"开始基本面进化挖掘：feature_pool={len(fields)}, population={cfg.population_size}, "
            f"generations={cfg.generations}。",
            module="mining",
        )

        def _evaluate(spec: FundamentalFormulaSpec) -> Dict[str, object]:
            key = _to_key(spec.to_dict())
            if key in cache:
                return cache[key]

            fac = compute_fundamental_factor(panel, spec, group_col=eval_group_col)
            tmp = panel[[time_col, "code", "future_ret_n"]].copy()
            tmp["_factor"] = pd.to_numeric(fac, errors="coerce")

            tmp_tr, tmp_va = _split_train_valid(tmp)
            m_tr = _collect_metrics(tmp_tr, factor_col="_factor", cfg=cfg, framework=framework, group_col=eval_group_col)
            m_va = _collect_metrics(tmp_va, factor_col="_factor", cfg=cfg, framework=framework, group_col=eval_group_col)
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
            log_progress(
                (
                    f"[fundamental][gen={gen:02d}] best_score={best['score']:.4f} "
                    f"absIC={best['metrics_valid'].get('abs_ic_mean', float('nan')):.4f} "
                    f"win={best['metrics_valid'].get('ic_win_rate', float('nan')):.4f}"
                ),
                module="mining",
                level="debug",
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
        log_progress(f"基本面进化完成：candidate_count={len(results)}。", module="mining")

    elif framework in {"minute_parametric", "minute_parametric_plus"}:
        # 3C/3D) Minute framework family: NSGA-III over parametric minute operators.
        minute_panel_mode = str(factor_freq).upper() != "D"
        if minute_panel_mode:
            # Intraday/period frequency: evaluate formulas directly on target-frequency panel.
            minute_feat = panel.copy()
            metric_group_col = eval_group_col
        else:
            # Daily frequency: derive one factor value per [date, code] from minute bars.
            if minute_df is None or minute_df.empty:
                raise RuntimeError(f"{framework} framework requires minute_df")
            minute_feat = build_minute_feature_matrix(minute_df)
            metric_group_col = "date"

        fields = _discover_minute_feature_pool(minute_feat)
        if len(fields) < 2:
            raise RuntimeError(f"insufficient minute feature columns for {framework}")

        is_plus = framework == "minute_parametric_plus"
        pop_size = int(cfg.population_size)
        generations = int(cfg.generations)
        mutate_fn = _minute_plus_mutate if is_plus else _minute_mutate
        random_fn = _minute_plus_random_spec if is_plus else _minute_random_spec
        tag = "minute_plus" if is_plus else "minute"
        log_progress(
            f"开始分钟进化挖掘：framework={framework}, feature_pool={len(fields)}, "
            f"population={pop_size}, generations={generations}, panel_mode={int(minute_panel_mode)}。",
            module="mining",
        )

        daily_ctx = panel[[c for c in ["date", "code", "future_ret_n", "barra_size_proxy", "industry_bucket"] if c in panel.columns]].copy()

        def _evaluate(spec: MinuteFormulaSpec) -> Dict[str, object]:
            key = _to_key(spec.to_dict())
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

            tmp_tr, tmp_va = _split_train_valid(tmp)
            m_tr = _collect_metrics(tmp_tr, factor_col="_factor", cfg=cfg, framework=framework, group_col=metric_group_col)
            m_va = _collect_metrics(tmp_va, factor_col="_factor", cfg=cfg, framework=framework, group_col=metric_group_col)

            base_obj = 0.5 * (_safe_obj(objectives_from_metrics(m_tr, framework)) + _safe_obj(objectives_from_metrics(m_va, framework)))
            if is_plus:
                # 稳定性目标：训练/验证 absIC 差异越小越好
                tr_ic = float(m_tr.get("abs_ic_mean", float("nan")))
                va_ic = float(m_va.get("abs_ic_mean", float("nan")))
                stability = -abs(tr_ic - va_ic) if np.isfinite(tr_ic) and np.isfinite(va_ic) else -1.0
                obj = np.concatenate([base_obj, np.asarray([stability], dtype=float)], axis=0)
            else:
                obj = base_obj

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

        pop: List[MinuteFormulaSpec] = [random_fn(rng, fields) for _ in range(pop_size)]
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
                    f"[{tag}][gen={gen:02d}] best_score={best['score']:.4f} "
                    f"absIC={best['metrics_valid'].get('abs_ic_mean', float('nan')):.4f} "
                    f"win={best['metrics_valid'].get('ic_win_rate', float('nan')):.4f} "
                    f"penalty={mean_penalty:.4f}"
                ),
                module="mining",
                level="debug",
            )

            new_pop: List[MinuteFormulaSpec] = [copy.deepcopy(e["spec"]) for e in elites]
            while len(new_pop) < pop_size:
                p1 = copy.deepcopy(parent_pool[int(rng.integers(0, len(parent_pool)))])
                child = p1
                if rng.random() < cfg.crossover_rate and len(parent_pool) > 1:
                    p2 = copy.deepcopy(parent_pool[int(rng.integers(0, len(parent_pool)))])
                    child = _minute_crossover(p1, p2, rng)
                if rng.random() < cfg.mutation_rate:
                    child = mutate_fn(child, rng, fields)
                new_pop.append(child)
            pop = new_pop[: pop_size]

        results = list(archive.values())
        log_progress(f"分钟进化完成：candidate_count={len(results)}。", module="mining")

    elif framework == "ml_ensemble_alpha":
        # 3E) ML ensemble framework: model + feature-subset co-search.
        fields_all = _discover_daily_feature_pool(panel)
        train_frame = panel.loc[train_mask].copy()
        fields_valid = _valid_nonconstant_features(train_frame, fields_all)
        if len(fields_valid) < 5:
            raise RuntimeError("insufficient daily features for ml_ensemble_alpha")

        fields_pref = _prefilter_features_by_ic(
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

        def _evaluate(spec: MLEnsembleFormulaSpec) -> Dict[str, object]:
            key = _to_key(spec.to_dict())
            if key in cache:
                return cache[key]

            fac = _fit_ml_factor_series(
                panel=panel,
                train_mask=train_mask,
                spec=spec,
                cfg=cfg,
                group_col=eval_group_col,
            )
            tmp = panel[[time_col, "code", "future_ret_n"]].copy()
            tmp["_factor"] = pd.to_numeric(fac, errors="coerce")

            tmp_tr, tmp_va = _split_train_valid(tmp)
            m_tr = _collect_metrics(tmp_tr, factor_col="_factor", cfg=cfg, framework=framework, group_col=eval_group_col)
            m_va = _collect_metrics(tmp_va, factor_col="_factor", cfg=cfg, framework=framework, group_col=eval_group_col)
            obj = 0.5 * (_safe_obj(objectives_from_metrics(m_tr, framework)) + _safe_obj(objectives_from_metrics(m_va, framework)))

            passed, admission = check_admission(m_va, standard)
            score = _ic_score(m_va)
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

        pop: List[MLEnsembleFormulaSpec] = [_ml_random_spec(rng, feature_pool=feature_pool, cfg=cfg) for _ in range(pop_size)]
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

            new_pop: List[MLEnsembleFormulaSpec] = [copy.deepcopy(e["spec"]) for e in elites]
            while len(new_pop) < pop_size:
                p1 = copy.deepcopy(parent_pool[int(rng.integers(0, len(parent_pool)))])
                child = p1
                if rng.random() < cfg.crossover_rate and len(parent_pool) > 1:
                    p2 = copy.deepcopy(parent_pool[int(rng.integers(0, len(parent_pool)))])
                    child = _ml_crossover(p1, p2, rng, cfg=cfg)
                if rng.random() < cfg.mutation_rate:
                    child = _ml_mutate(child, rng, feature_pool=feature_pool, cfg=cfg)
                new_pop.append(child)
            pop = new_pop[: pop_size]

        results = list(archive.values())
        log_progress(f"ML 集成进化完成：candidate_count={len(results)}。", module="mining")

    else:
        raise ValueError(f"unsupported mining framework: {cfg.framework}")

    # 4) Candidate ranking and diversification.
    # Rule: admitted first, then higher validation ICIR, then low pairwise correlation.
    ranked = sorted(results, key=lambda r: (int(bool(r.get("passed", False))), float(r.get("score", -1e9))), reverse=True)
    minute_frameworks = {"minute_parametric", "minute_parametric_plus"}
    if framework in minute_frameworks:
        if minute_panel_mode:
            minute_feat_cached = panel.copy()
        elif minute_df is not None:
            minute_feat_cached = build_minute_feature_matrix(minute_df)
        else:
            minute_feat_cached = None
    else:
        minute_feat_cached = None

    def _series_getter(cand: Dict[str, object]) -> pd.Series:
        spec = cand["spec"]
        sign_ref = float(cand["metrics_valid"].get("rank_ic_mean", float("nan")))
        sign = -1.0 if np.isfinite(sign_ref) and sign_ref < 0.0 else 1.0

        if framework == "fundamental_multiobj":
            s = compute_fundamental_factor(panel, spec, group_col=eval_group_col)
            fac = pd.DataFrame({time_col: panel[time_col], "code": panel["code"], "v": s * sign})
            return fac.set_index([time_col, "code"])["v"].sort_index()
        if framework in minute_frameworks:
            if minute_panel_mode:
                minute_feat = minute_feat_cached if isinstance(minute_feat_cached, pd.DataFrame) else panel
                s2 = compute_minute_factor_panel(
                    panel_feature_df=minute_feat,
                    spec=spec,
                    time_col=time_col,
                    session_col="date" if time_col == "datetime" else None,
                )
                fac = pd.DataFrame({time_col: panel[time_col], "code": panel["code"], "v": s2 * sign})
                return fac.set_index([time_col, "code"])["v"].sort_index()

            minute_feat = minute_feat_cached if minute_feat_cached is not None else pd.DataFrame()
            daily_ctx = panel[[c for c in ["date", "code", "future_ret_n", "barra_size_proxy", "industry_bucket"] if c in panel.columns]].copy()
            s2 = compute_minute_factor_daily(minute_feature_df=minute_feat, daily_context_df=daily_ctx, spec=spec)
            return (s2 * sign).sort_index()
        if framework == "ml_ensemble_alpha":
            s3 = cand.get("series")
            if isinstance(s3, pd.Series):
                return (pd.to_numeric(s3, errors="coerce") * sign).sort_index()
            return pd.Series(dtype=float)

        # custom
        fac_df = evaluate_custom_specs(panel=panel, specs=[spec], time_col=time_col, code_col="code")
        fac = pd.DataFrame({time_col: panel[time_col], "code": panel["code"], "v": fac_df[spec.name] * sign})
        return fac.set_index([time_col, "code"])["v"].sort_index()

    ranked_passed = [r for r in ranked if bool(r.get("passed", False))]
    selected = _greedy_low_corr_select(
        candidates=ranked_passed,
        series_getter=_series_getter,
        top_n=int(cfg.top_n),
        corr_threshold=float(cfg.corr_threshold),
    )
    log_progress(
        f"候选筛选完成：passed_count={len(ranked_passed)}, selected_count={len(selected)}, top_n={cfg.top_n}。",
        module="mining",
    )

    # 5) Persist selected factors and update global catalog.
    run_root = _make_run_root(cfg)

    # build factor value table
    merge_keys = [time_col, "code"]
    base_cols = [time_col, "code"]
    if time_col == "datetime" and "date" in panel.columns:
        base_cols = ["datetime", "date", "code"]
    base_tbl = (
        panel[base_cols]
        .drop_duplicates(merge_keys)
        .sort_values(merge_keys)
        .reset_index(drop=True)
    )
    catalog_entries: List[Dict[str, object]] = []

    for i, cand in enumerate(selected, start=1):
        spec = cand["spec"]
        sign_ref = float(cand["metrics_valid"].get("rank_ic_mean", float("nan")))
        score_sign = -1.0 if np.isfinite(sign_ref) and sign_ref < 0.0 else 1.0

        if framework == "custom":
            fac_name = str(spec.name)
        elif framework == "fundamental_multiobj":
            fac_name = _hash_name("fund_mo", spec.to_dict())
        elif framework == "minute_parametric":
            fac_name = _hash_name("min_mo", spec.to_dict())
        elif framework == "minute_parametric_plus":
            fac_name = _hash_name("minx_mo", spec.to_dict())
        elif framework == "ml_ensemble_alpha":
            fac_name = _hash_name("mla_mo", spec.to_dict())
        else:
            fac_name = _hash_name("min_mo", spec.to_dict())

        s = _series_getter(cand)
        fac_df = s.rename(fac_name).reset_index()
        base_tbl = base_tbl.merge(fac_df, on=merge_keys, how="left")

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
    log_progress(
        f"因子落盘与 catalog 更新完成：factor_table={data_path}, catalog_path={catalog_path}, "
        f"inserted_or_updated={len(catalog_entries)}。",
        module="mining",
    )

    summary = {
        "framework": framework,
        "factor_freq": cfg.factor_freq,
        "train_period": [str(pd.Timestamp(cfg.train_start).date()), str(pd.Timestamp(cfg.train_end).date())],
        "valid_period": [str(pd.Timestamp(cfg.valid_start).date()), str(pd.Timestamp(cfg.valid_end).date())],
        "population_size": int(cfg.ml_population_size) if framework == "ml_ensemble_alpha" else int(cfg.population_size),
        "generations": int(cfg.ml_generations) if framework == "ml_ensemble_alpha" else int(cfg.generations),
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
    log_progress(f"挖掘流程结束，summary 写入：{summary_path}", module="mining")
    return summary
