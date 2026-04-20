"""End-to-end factor mining runner (fundamental/mminute/custom) with catalog persistence."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from ..core.constants import INTRADAY_FREQS
from ..core.utils import dump_json, ensure_dir, log_progress
from ..factors.defaults import DEFAULT_FACTOR_SET_BY_FREQ
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
from .models import (
    run_custom_model,
    run_fundamental_multiobj_model,
    run_minute_parametric_model,
    run_minute_parametric_plus_model,
    run_ml_ensemble_alpha_model,
    run_gplearn_symbolic_alpha_model,
)
from .nsga import apply_dynamic_shortboard_penalty, nsga2_select, nsga3_select

_DEFAULT_FACTOR_NAME_UNIVERSE = frozenset(
    x
    for _, vals in DEFAULT_FACTOR_SET_BY_FREQ.items()
    for x in vals
)


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
    # gplearn_symbolic_alpha 专用参数（仅该框架读取，不影响旧框架）
    gp_population_size: int = 400
    gp_generations: int = 12
    gp_num_runs: int = 3
    gp_n_components: int = 24
    gp_hall_of_fame: int = 64
    gp_tournament_size: int = 20
    gp_parsimony: float = 0.001
    gp_metric: str = "spearman"
    gp_function_set: str = "add,sub,mul,div,sqrt,log,abs,neg,max,min"
    gp_prefilter_topk: int = 80
    gp_train_sample_frac: float = 0.40
    gp_max_train_rows: int = 220000
    gp_max_depth: int = 5
    gp_max_samples: float = 0.90
    gp_num_jobs: int = -1
    # 素材池控制参数（run_factor_mining.py 注入默认因子后透传记录）
    include_default_factor_materials: bool = True
    factor_packages: str = ""
    factor_list: str = ""
    material_factor_count: int = 0
    material_factor_names: List[str] | None = None
    universe: str = ""
    index_context_cols: List[str] | None = None


def _seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _hash_name(prefix: str, payload: Dict[str, object], length: int = 10) -> str:
    txt = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    h = hashlib.sha1(txt.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{h}"


def _to_key(payload: Dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _norm_tag(text: object, *, max_len: int = 24) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    if not s:
        return "na"
    return s[:max_len]


def _framework_alias(framework: str) -> str:
    mp = {
        "fundamental_multiobj": "fmo",
        "minute_parametric": "mmo",
        "minute_parametric_plus": "mmx",
        "ml_ensemble_alpha": "mla",
        "gplearn_symbolic_alpha": "gpl",
        "custom": "cus",
    }
    return mp.get(str(framework).strip().lower(), "unk")


def _extract_spec_material_columns(spec: object) -> List[str]:
    if spec is None or not hasattr(spec, "to_dict"):
        return []
    try:
        payload = spec.to_dict()
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []

    cols: List[str] = []
    for k, v in payload.items():
        key = str(k).strip().lower()
        if key == "feature_cols" and isinstance(v, list):
            cols.extend([str(x).strip() for x in v if str(x).strip()])
            continue
        if key.endswith("_field") or key in {"x_field", "y_field", "a_field", "b_field", "mask_field"}:
            s = str(v).strip()
            if s:
                cols.append(s)
            continue
        if key == "expression":
            tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(v))
            cols.extend(tokens)
    return sorted(set([x for x in cols if x]))


def _infer_mined_factor_type(
    *,
    framework: str,
    material_columns: Sequence[str],
) -> str:
    counts = {
        "price_volume": 0,
        "fundamental": 0,
        "text": 0,
        "fusion": 0,
    }

    for col in material_columns:
        c = str(col).strip().lower()
        if not c:
            continue
        if c.startswith(("fd_", "fdsrc_", "fund_")):
            counts["fundamental"] += 1
            continue
        if c.startswith("txt_"):
            counts["text"] += 1
            continue
        if c.startswith(("hf_", "bridge_", "ms_", "context_", "mkt_")):
            counts["fusion"] += 1
            continue
        if c.startswith(
            (
                "ret_",
                "rv_",
                "ma_",
                "amount_",
                "vol_",
                "range_",
                "close",
                "open",
                "high",
                "low",
                "volume",
                "turn",
                "amihud",
                "signed_",
                "intraday_",
                "period_",
            )
        ):
            counts["price_volume"] += 1

    non_zero = [k for k, v in counts.items() if int(v) > 0]
    if len(non_zero) >= 2:
        return "fusion"
    if len(non_zero) == 1:
        return non_zero[0]

    fw = str(framework).strip().lower()
    if fw == "fundamental_multiobj":
        return "fundamental"
    if fw in {"minute_parametric", "minute_parametric_plus"}:
        return "price_volume"
    if fw == "custom":
        return "custom"
    return "other"


def _mined_factor_packages(
    *,
    factor_type: str,
    framework: str,
    factor_freq: str,
    universe: str,
    material_packages_expr: str,
    has_explicit_factor_list: bool,
) -> List[str]:
    ftype = str(factor_type).strip().lower()
    primary = f"mined_{ftype}" if ftype else "mined_other"
    out = [
        "mined",
        primary,
        f"mined_fw_{_framework_alias(framework)}",
        f"mined_freq_{_norm_tag(factor_freq, max_len=12)}",
        f"mined_universe_{_norm_tag(universe, max_len=16)}",
    ]
    for p in [x.strip() for x in str(material_packages_expr).split(",") if x.strip()]:
        out.append(f"mined_materialpkg_{_norm_tag(p)}")
    if has_explicit_factor_list:
        out.append("mined_materiallist_explicit")
    if str(framework).strip().lower() == "custom":
        out.append("mined_custom")
    dedup: List[str] = []
    seen: set[str] = set()
    for x in out:
        sx = str(x).strip()
        if not sx or sx in seen:
            continue
        seen.add(sx)
        dedup.append(sx)
    return dedup


def _build_mined_factor_name(
    *,
    framework: str,
    factor_type: str,
    factor_freq: str,
    universe: str,
    payload: Dict[str, object],
) -> str:
    freq_tag = _norm_tag(str(factor_freq).replace("min", "m"), max_len=10)
    uni_tag = _norm_tag(universe, max_len=10)
    type_tag = _norm_tag(factor_type, max_len=12)
    fw_tag = _framework_alias(framework)
    sig = _hash_name("sig", payload, length=10).split("_", 1)[1]
    return f"mf_{fw_tag}_{type_tag}_{freq_tag}_{uni_tag}_{sig}"


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
    factor_like = [
        c
        for c in numeric_cols
        if c in _DEFAULT_FACTOR_NAME_UNIVERSE
        or c.startswith("g_")
        or c.startswith("bridge_")
        or c.startswith("ms_")
        or c.startswith("hf_")
    ]
    core = preferred if len(preferred) >= 12 else numeric_cols
    return sorted(set(core) | set(factor_like))


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
    log_progress(
        "素材池配置："
        f"include_default_factor_materials={int(bool(cfg.include_default_factor_materials))}, "
        f"factor_packages='{cfg.factor_packages}', "
        f"factor_list='{cfg.factor_list}', "
        f"material_factor_count={int(cfg.material_factor_count)}。",
        module="mining",
        level="debug",
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
        results = run_custom_model(
            panel=panel,
            specs=custom_specs or [],
            time_col=time_col,
            framework=framework,
            eval_group_col=eval_group_col,
            cache=cache,
            standard=standard,
            evaluate_custom_specs=evaluate_custom_specs,
            split_train_valid=_split_train_valid,
            collect_metrics=lambda df, factor_col, group_col: _collect_metrics(
                df, factor_col=factor_col, cfg=cfg, framework=framework, group_col=group_col
            ),
            check_admission=check_admission,
            objectives_from_metrics=objectives_from_metrics,
            to_key=_to_key,
            ic_score=_ic_score,
            log_progress=log_progress,
        )

    elif framework == "fundamental_multiobj":
        fields = _discover_daily_feature_pool(panel)
        results = run_fundamental_multiobj_model(
            panel=panel,
            fields=fields,
            cfg=cfg,
            rng=rng,
            framework=framework,
            eval_group_col=eval_group_col,
            cache=cache,
            standard=standard,
            compute_fundamental_factor=compute_fundamental_factor,
            split_train_valid=_split_train_valid,
            collect_metrics=lambda df, factor_col, group_col: _collect_metrics(
                df, factor_col=factor_col, cfg=cfg, framework=framework, group_col=group_col
            ),
            objectives_from_metrics=objectives_from_metrics,
            safe_obj=_safe_obj,
            check_admission=check_admission,
            to_key=_to_key,
            ic_score=_ic_score,
            nsga2_select=nsga2_select,
            random_spec_fn=_fundamental_random_spec,
            mutate_spec_fn=_fundamental_mutate,
            crossover_spec_fn=_fundamental_crossover,
            log_progress=log_progress,
        )

    elif framework == "minute_parametric":
        results, minute_panel_mode = run_minute_parametric_model(
            panel=panel,
            minute_df=minute_df,
            factor_freq=factor_freq,
            time_col=time_col,
            framework=framework,
            eval_group_col=eval_group_col,
            cfg=cfg,
            rng=rng,
            cache=cache,
            standard=standard,
            build_minute_feature_matrix=build_minute_feature_matrix,
            discover_minute_feature_pool=_discover_minute_feature_pool,
            discover_daily_feature_pool=_discover_daily_feature_pool,
            compute_minute_factor_panel=compute_minute_factor_panel,
            compute_minute_factor_daily=compute_minute_factor_daily,
            split_train_valid=_split_train_valid,
            collect_metrics=lambda df, factor_col, group_col: _collect_metrics(
                df, factor_col=factor_col, cfg=cfg, framework=framework, group_col=group_col
            ),
            objectives_from_metrics=objectives_from_metrics,
            safe_obj=_safe_obj,
            check_admission=check_admission,
            to_key=_to_key,
            ic_score=_ic_score,
            nsga3_select=nsga3_select,
            apply_dynamic_shortboard_penalty=apply_dynamic_shortboard_penalty,
            random_spec_fn=_minute_random_spec,
            mutate_spec_fn=_minute_mutate,
            crossover_spec_fn=_minute_crossover,
            log_progress=log_progress,
        )

    elif framework == "minute_parametric_plus":
        results, minute_panel_mode = run_minute_parametric_plus_model(
            panel=panel,
            minute_df=minute_df,
            factor_freq=factor_freq,
            time_col=time_col,
            framework=framework,
            eval_group_col=eval_group_col,
            cfg=cfg,
            rng=rng,
            cache=cache,
            standard=standard,
            build_minute_feature_matrix=build_minute_feature_matrix,
            discover_minute_feature_pool=_discover_minute_feature_pool,
            discover_daily_feature_pool=_discover_daily_feature_pool,
            compute_minute_factor_panel=compute_minute_factor_panel,
            compute_minute_factor_daily=compute_minute_factor_daily,
            split_train_valid=_split_train_valid,
            collect_metrics=lambda df, factor_col, group_col: _collect_metrics(
                df, factor_col=factor_col, cfg=cfg, framework=framework, group_col=group_col
            ),
            objectives_from_metrics=objectives_from_metrics,
            safe_obj=_safe_obj,
            check_admission=check_admission,
            to_key=_to_key,
            ic_score=_ic_score,
            nsga3_select=nsga3_select,
            apply_dynamic_shortboard_penalty=apply_dynamic_shortboard_penalty,
            random_spec_fn=_minute_plus_random_spec,
            mutate_spec_fn=_minute_plus_mutate,
            crossover_spec_fn=_minute_crossover,
            log_progress=log_progress,
        )

    elif framework == "ml_ensemble_alpha":
        results = run_ml_ensemble_alpha_model(
            panel=panel,
            train_mask=train_mask,
            time_col=time_col,
            eval_group_col=eval_group_col,
            framework=framework,
            cfg=cfg,
            rng=rng,
            cache=cache,
            standard=standard,
            discover_daily_feature_pool=_discover_daily_feature_pool,
            valid_nonconstant_features=_valid_nonconstant_features,
            prefilter_features_by_ic=_prefilter_features_by_ic,
            fit_ml_factor_series=_fit_ml_factor_series,
            split_train_valid=_split_train_valid,
            collect_metrics=lambda df, factor_col, group_col: _collect_metrics(
                df, factor_col=factor_col, cfg=cfg, framework=framework, group_col=group_col
            ),
            objectives_from_metrics=objectives_from_metrics,
            safe_obj=_safe_obj,
            check_admission=check_admission,
            to_key=_to_key,
            ic_score=_ic_score,
            nsga2_select=nsga2_select,
            random_spec_fn=_ml_random_spec,
            mutate_spec_fn=_ml_mutate,
            crossover_spec_fn=_ml_crossover,
            log_progress=log_progress,
        )

    elif framework == "gplearn_symbolic_alpha":
        results = run_gplearn_symbolic_alpha_model(
            panel=panel,
            train_mask=train_mask,
            time_col=time_col,
            eval_group_col=eval_group_col,
            framework=framework,
            cfg=cfg,
            cache=cache,
            standard=standard,
            discover_feature_pool=_discover_daily_feature_pool,
            valid_nonconstant_features=_valid_nonconstant_features,
            prefilter_features_by_ic=_prefilter_features_by_ic,
            split_train_valid=_split_train_valid,
            collect_metrics=lambda df, factor_col, group_col: _collect_metrics(
                df, factor_col=factor_col, cfg=cfg, framework=framework, group_col=group_col
            ),
            objectives_from_metrics=objectives_from_metrics,
            safe_obj=_safe_obj,
            check_admission=check_admission,
            to_key=_to_key,
            ic_score=_ic_score,
            winsorize_mad_cs=winsorize_mad_cs,
            neutralize_series=neutralize_series,
            cs_zscore=cs_zscore,
            log_progress=log_progress,
        )

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
        if framework in {"ml_ensemble_alpha", "gplearn_symbolic_alpha"}:
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
        spec_payload = spec.to_dict() if hasattr(spec, "to_dict") else {}
        material_cols = _extract_spec_material_columns(spec)
        factor_type = _infer_mined_factor_type(
            framework=framework,
            material_columns=material_cols,
        )
        packages = _mined_factor_packages(
            factor_type=factor_type,
            framework=framework,
            factor_freq=cfg.factor_freq,
            universe=cfg.universe,
            material_packages_expr=cfg.factor_packages,
            has_explicit_factor_list=bool(str(cfg.factor_list).strip()),
        )
        primary_package = packages[1] if len(packages) > 1 else (packages[0] if packages else "mined_other")

        if framework == "custom":
            fac_name = str(spec.name)
        else:
            fac_name = _build_mined_factor_name(
                framework=framework,
                factor_type=factor_type,
                factor_freq=cfg.factor_freq,
                universe=cfg.universe,
                payload=spec_payload if isinstance(spec_payload, dict) else {},
            )

        s = _series_getter(cand)
        fac_df = s.rename(fac_name).reset_index()
        base_tbl = base_tbl.merge(fac_df, on=merge_keys, how="left")

        catalog_entries.append(
            {
                "name": fac_name,
                "freq": cfg.factor_freq,
                "category": primary_package,
                "factor_package": primary_package,
                "factor_packages": ",".join(packages),
                "description": (
                    f"{framework} admitted factor #{i}; "
                    f"profile={cand['admission'].get('profile', '')}"
                ),
                "source": "custom" if framework == "custom" else "mined",
                "framework": framework,
                "mined_factor_type": factor_type,
                "mined_universe": str(cfg.universe),
                "mined_freq": str(cfg.factor_freq),
                "material_factor_packages": str(cfg.factor_packages),
                "material_factor_list": str(cfg.factor_list),
                "material_factor_count": int(cfg.material_factor_count),
                "status": "active",
                "table_path": "",
                "value_col": fac_name,
                "score_sign": score_sign,
                "metrics_train": cand["metrics_train"],
                "metrics_valid": cand["metrics_valid"],
                "admission": cand["admission"],
                "formula": spec_payload,
                "material_columns": material_cols,
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
        "universe": str(cfg.universe),
        "include_default_factor_materials": bool(cfg.include_default_factor_materials),
        "factor_packages": str(cfg.factor_packages),
        "factor_list": str(cfg.factor_list),
        "material_factor_count": int(cfg.material_factor_count),
        "material_factor_names": [str(x) for x in (cfg.material_factor_names or [])],
        "index_context_cols": list(cfg.index_context_cols or []),
        "train_period": [str(pd.Timestamp(cfg.train_start).date()), str(pd.Timestamp(cfg.train_end).date())],
        "valid_period": [str(pd.Timestamp(cfg.valid_start).date()), str(pd.Timestamp(cfg.valid_end).date())],
        "population_size": (
            int(cfg.ml_population_size)
            if framework == "ml_ensemble_alpha"
            else (int(cfg.gp_population_size) if framework == "gplearn_symbolic_alpha" else int(cfg.population_size))
        ),
        "generations": (
            int(cfg.ml_generations)
            if framework == "ml_ensemble_alpha"
            else (int(cfg.gp_generations) if framework == "gplearn_symbolic_alpha" else int(cfg.generations))
        ),
        "candidate_count": int(len(results)),
        "selected_count": int(len(selected)),
        "catalog_path": str(catalog_path),
        "factor_table_path": str(data_path),
        "selected_factors": [e["name"] for e in catalog_entries],
        "selected_factor_packages": sorted(
            set(
                str(x).strip()
                for e in catalog_entries
                for x in str(e.get("factor_packages", "")).split(",")
                if str(x).strip()
            )
        ),
        "admission_profile": standard.profile,
        "top_candidates": [
            {
                "name": catalog_entries[i]["name"] if i < len(catalog_entries) else "",
                "factor_package": catalog_entries[i].get("factor_package", "") if i < len(catalog_entries) else "",
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
