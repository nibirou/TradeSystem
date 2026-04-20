"""Financial-text ingestion and NLP feature engineering utilities.

This module focuses on:
1. loading multiple text datasets (news / notices / reports),
2. normalizing them into unified event records,
3. generating daily text feature panels,
4. creating rolling and cross-domain fusion features.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.constants import EPS
from ..core.utils import split_exchange_code


TEXT_SOURCES: Sequence[str] = ("news", "notice", "em_report", "iwencai")
TEXT_SOURCE_WEIGHTS: Dict[str, float] = {
    "news": 0.85,
    "notice": 1.20,
    "em_report": 1.10,
    "iwencai": 1.00,
}

TEXT_POSITIVE_WORDS: Sequence[str] = (
    "增长",
    "改善",
    "上调",
    "增持",
    "买入",
    "看好",
    "超预期",
    "盈利",
    "突破",
    "回升",
    "修复",
    "受益",
    "提升",
    "景气",
    "利好",
)
TEXT_NEGATIVE_WORDS: Sequence[str] = (
    "下滑",
    "下降",
    "减持",
    "卖出",
    "亏损",
    "承压",
    "不及预期",
    "风险",
    "违约",
    "处罚",
    "减值",
    "诉讼",
    "暴雷",
    "利空",
    "下调",
)
TEXT_UNCERTAINTY_WORDS: Sequence[str] = (
    "或",
    "可能",
    "预计",
    "预期",
    "不确定",
    "视情况",
    "存在",
    "待定",
    "谨慎",
    "波动",
    "仍需",
)
TEXT_RISK_WORDS: Sequence[str] = (
    "风险",
    "诉讼",
    "处罚",
    "问询",
    "立案",
    "违约",
    "减值",
    "退市",
    "停牌",
    "爆仓",
    "流动性压力",
    "偿债压力",
)
TEXT_ACTION_WORDS: Sequence[str] = (
    "回购",
    "增持",
    "减持",
    "并购",
    "重组",
    "中标",
    "签约",
    "定增",
    "分红",
    "募资",
    "投产",
    "扩产",
)

TEXT_TOPIC_KEYWORDS: Dict[str, Sequence[str]] = {
    "earnings": ("业绩", "利润", "净利", "营收", "快报", "季报", "年报", "中报", "预增", "预减"),
    "policy": ("政策", "监管", "证监会", "交易所", "央行", "降准", "降息", "改革", "指导意见", "补贴"),
    "mna": ("并购", "重组", "收购", "增持", "减持", "回购", "定增", "配股", "可转债", "股权激励"),
    "risk": ("风险", "诉讼", "处罚", "违约", "退市", "立案", "问询", "减值", "停牌", "爆仓"),
    "operation": ("订单", "合同", "投产", "扩产", "新品", "产能", "开工", "签约", "中标", "交付"),
    "capital": ("资金", "融资", "债券", "利率", "现金流", "分红", "回款", "质押", "担保", "贷款"),
}

TEXT_CONTEXT_COLUMNS: List[str] = [
    "txt_event_count",
    "txt_source_coverage",
    "txt_sentiment_mean",
    "txt_sentiment_std",
    "txt_pos_share",
    "txt_neg_share",
    "txt_uncertainty_mean",
    "txt_risk_mean",
    "txt_action_mean",
    "txt_attention_mean",
    "txt_novelty_mean",
    "txt_quality_mean",
    "txt_news_count",
    "txt_notice_count",
    "txt_em_report_count",
    "txt_iwencai_count",
    "txt_news_share",
    "txt_notice_share",
    "txt_em_report_share",
    "txt_iwencai_share",
    "txt_topic_earnings",
    "txt_topic_policy",
    "txt_topic_mna",
    "txt_topic_risk",
    "txt_topic_operation",
    "txt_topic_capital",
    "txt_news_sentiment",
    "txt_notice_sentiment",
    "txt_em_report_sentiment",
    "txt_iwencai_sentiment",
    "txt_news_attention",
    "txt_notice_attention",
    "txt_em_report_attention",
    "txt_iwencai_attention",
    "txt_news_risk",
    "txt_notice_risk",
    "txt_em_report_risk",
    "txt_iwencai_risk",
    "txt_rating_delta_mean",
    "txt_count_3",
    "txt_count_5",
    "txt_count_10",
    "txt_count_20",
    "txt_sent_5",
    "txt_sent_20",
    "txt_sent_trend_5",
    "txt_sent_vol_20",
    "txt_attention_5",
    "txt_attention_20",
    "txt_attention_shock",
    "txt_risk_5",
    "txt_risk_20",
    "txt_uncertainty_5",
    "txt_uncertainty_20",
    "txt_novelty_5",
    "txt_novelty_20",
    "txt_news_burst",
    "txt_notice_burst",
    "txt_em_report_burst",
    "txt_iwencai_burst",
    "txt_source_dispersion",
    "txt_topic_risk_vs_earnings",
    "txt_topic_policy_mna_mix",
    "txt_hf_sent_flow",
    "txt_hf_attention_jump",
    "txt_hf_risk_vol",
    "txt_hf_momentum_resonance",
    "txt_hf_liquidity_pulse",
    "txt_fd_growth_resonance",
    "txt_fd_quality_guard",
    "txt_fd_expectation_spread",
    "txt_fd_cashflow_conflict",
    "txt_fd_valuation_contrast",
    "txt_fusion_score",
    "txt_fusion_disp",
]


def _normalize_code(code_or_key: object, *, dotted: bool = True) -> str:
    ex, code = split_exchange_code(str(code_or_key))
    if ex not in {"sh", "sz"} or not code:
        return ""
    return f"{ex}.{code}" if dotted else f"{ex}_{code}"


def _read_table_file(path: Path, file_format: str = "auto") -> pd.DataFrame:
    fmt = str(file_format).strip().lower()
    ext = path.suffix.lower()
    use_parquet = (fmt == "parquet") or (fmt == "auto" and ext == ".parquet")
    if use_parquet:
        try:
            return pd.read_parquet(path)
        except Exception:
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path, low_memory=False)
            raise
    return pd.read_csv(path, low_memory=False)


def _pick_stem_file(folder: Path, stem: str, file_format: str = "auto") -> Optional[Path]:
    if file_format == "parquet":
        cands = [folder / f"{stem}.parquet"]
    elif file_format == "csv":
        cands = [folder / f"{stem}.csv"]
    else:
        cands = [folder / f"{stem}.parquet", folder / f"{stem}.csv"]
    for p in cands:
        if p.exists():
            return p
    return None


def select_text_universe_dirs(base_dir: Path, universe: str) -> List[Path]:
    if not base_dir.exists():
        return []
    order = [str(universe).strip().lower(), "all", "hs300", "zz500", "sz50"]
    out: List[Path] = []
    seen: set[str] = set()
    for u in order:
        if not u or u in seen:
            continue
        seen.add(u)
        cand = base_dir / u
        if cand.exists() and cand.is_dir():
            out.append(cand)
    return out


def pick_symbol_text_file(
    universe_dirs: Sequence[Path],
    symbol_key: str,
    *,
    file_format: str = "auto",
) -> Optional[Path]:
    for folder in universe_dirs:
        fp = _pick_stem_file(folder, symbol_key, file_format=file_format)
        if fp is not None:
            return fp
    return None


def _choose_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = {str(c): str(c).lower() for c in df.columns}
    lower_map = {v: k for k, v in cols.items()}
    for cand in candidates:
        if cand in df.columns:
            return cand
        cl = str(cand).lower()
        if cl in lower_map:
            return lower_map[cl]
    # fallback: choose first date-like column
    if any("date" in str(c).lower() or "日期" in str(c) or "时间" in str(c) for c in candidates):
        for c, cl in cols.items():
            if "date" in cl or "日期" in c or "时间" in c:
                return c
    return None


def _safe_text_series(s: pd.Series) -> pd.Series:
    out = s.fillna("").astype(str)
    out = out.str.replace("\r", " ", regex=False).str.replace("\n", " ", regex=False)
    out = out.str.replace(r"\s+", " ", regex=True).str.strip()
    return out


def _keyword_count(series: pd.Series, keywords: Sequence[str]) -> pd.Series:
    out = pd.Series(0.0, index=series.index, dtype=float)
    for kw in keywords:
        k = str(kw).strip()
        if not k:
            continue
        out = out + series.str.count(re.escape(k))
    return out.astype(float)


def _rating_delta_from_text(series: pd.Series) -> pd.Series:
    s = _safe_text_series(series)
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0.0).astype(float)
    out = pd.Series(0.0, index=s.index, dtype=float)
    pos_mask = s.str.contains("上调|调高|增持|买入|推荐|强烈推荐", regex=True)
    neg_mask = s.str.contains("下调|调低|减持|卖出|回避|谨慎", regex=True)
    out[pos_mask] = 1.0
    out[neg_mask] = -1.0
    return out


def _empty_text_event_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "code",
            "source",
            "title_len",
            "body_len",
            "text_len",
            "digit_ratio",
            "punc_intensity",
            "sentiment",
            "sent_pos",
            "sent_neg",
            "uncertainty_score",
            "risk_score",
            "action_score",
            "attention_score",
            "novelty_score",
            "source_quality",
            "rating_delta",
            "topic_earnings",
            "topic_policy",
            "topic_mna",
            "topic_risk",
            "topic_operation",
            "topic_capital",
        ]
    )


def load_text_source_events_from_file(
    path: Path,
    *,
    symbol_key: str,
    source: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    file_format: str = "auto",
) -> pd.DataFrame:
    src = str(source).strip().lower()
    if src not in set(TEXT_SOURCES):
        return _empty_text_event_frame()
    df = _read_table_file(path, file_format=file_format)
    if df.empty:
        return _empty_text_event_frame()

    date_candidates_map: Dict[str, Sequence[str]] = {
        "news": ("发布时间", "publish_time", "date"),
        "notice": ("公告发布时间", "公告日期", "publish_time", "date"),
        "em_report": ("研报发布时间", "发布日期", "publish_time", "date"),
        "iwencai": ("publish_time", "发布时间", "date"),
    }
    code_candidates_map: Dict[str, Sequence[str]] = {
        "news": ("关键词", "股票代码", "code"),
        "notice": ("股票代码", "code"),
        "em_report": ("股票代码", "code"),
        "iwencai": ("股票代码", "code"),
    }
    title_candidates_map: Dict[str, Sequence[str]] = {
        "news": ("新闻标题", "title", "标题"),
        "notice": ("公告标题", "title", "标题"),
        "em_report": ("研报标题", "title", "标题"),
        "iwencai": ("title", "研报标题", "标题"),
    }
    body_candidates_map: Dict[str, Sequence[str]] = {
        "news": ("新闻正文", "新闻内容", "content", "正文"),
        "notice": ("公告正文", "公告摘要", "content", "摘要"),
        "em_report": ("研报正文", "研报标题", "content"),
        "iwencai": ("content", "正文"),
    }

    date_col = _choose_column(df, date_candidates_map[src])
    if date_col is None:
        return _empty_text_event_frame()

    code_col = _choose_column(df, code_candidates_map[src])
    title_col = _choose_column(df, title_candidates_map[src])
    body_col = _choose_column(df, body_candidates_map[src])
    rating_col = _choose_column(df, ("评级变动", "rating_change", "rating_delta")) if src == "em_report" else None

    out = df.copy()
    out["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    out = out[(out["date"] >= start_date) & (out["date"] <= end_date)]
    out = out.dropna(subset=["date"])
    if out.empty:
        return _empty_text_event_frame()

    default_code = _normalize_code(symbol_key)
    if code_col is not None and code_col in out.columns:
        out["code"] = out[code_col].map(_normalize_code)
    else:
        out["code"] = default_code
    out["code"] = out["code"].where(out["code"].astype(str).ne(""), default_code)
    out = out[out["code"].astype(str).ne("")]
    if out.empty:
        return _empty_text_event_frame()

    title = _safe_text_series(out[title_col]) if (title_col is not None and title_col in out.columns) else pd.Series("", index=out.index, dtype=str)
    body = _safe_text_series(out[body_col]) if (body_col is not None and body_col in out.columns) else pd.Series("", index=out.index, dtype=str)
    text = _safe_text_series(title + " " + body)

    title_len = title.str.len().astype(float)
    body_len = body.str.len().astype(float)
    text_len = text.str.len().astype(float)
    digits = text.str.count(r"\d").astype(float)
    punc = text.str.count(r"[!！?？]").astype(float)

    pos_cnt = _keyword_count(text, TEXT_POSITIVE_WORDS)
    neg_cnt = _keyword_count(text, TEXT_NEGATIVE_WORDS)
    uncertainty_cnt = _keyword_count(text, TEXT_UNCERTAINTY_WORDS)
    risk_cnt = _keyword_count(text, TEXT_RISK_WORDS)
    action_cnt = _keyword_count(text, TEXT_ACTION_WORDS)

    out_evt = pd.DataFrame(
        {
            "date": out["date"],
            "code": out["code"].astype(str),
            "source": src,
            "title_len": title_len,
            "body_len": body_len,
            "text_len": text_len,
            "digit_ratio": digits / (text_len + 1.0),
            "punc_intensity": punc / (text_len + 1.0),
            "sentiment": (pos_cnt - neg_cnt) / (pos_cnt + neg_cnt + 1.0),
            "sent_pos": (pos_cnt > neg_cnt).astype(float),
            "sent_neg": (neg_cnt > pos_cnt).astype(float),
            "uncertainty_score": uncertainty_cnt / (text_len + 1.0),
            "risk_score": risk_cnt / (text_len + 1.0),
            "action_score": action_cnt / (text_len + 1.0),
            "attention_score": np.log1p(text_len) + np.log1p(punc + action_cnt),
            "novelty_score": (~title.duplicated(keep=False)).astype(float),
            "source_quality": float(TEXT_SOURCE_WEIGHTS.get(src, 1.0)),
            "rating_delta": (
                _rating_delta_from_text(out[rating_col]) if (rating_col is not None and rating_col in out.columns) else 0.0
            ),
        }
    )

    for topic, kws in TEXT_TOPIC_KEYWORDS.items():
        out_evt[f"topic_{topic}"] = _keyword_count(text, kws) / (text_len + 1.0)

    out_evt = out_evt.dropna(subset=["date", "code"]).reset_index(drop=True)
    for c in out_evt.columns:
        if c in {"date", "code", "source"}:
            continue
        out_evt[c] = pd.to_numeric(out_evt[c], errors="coerce").astype("float32")
    return out_evt


def _source_count_table(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["date", "code"])
    cnt = (
        events.groupby(["date", "code", "source"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    piv = cnt.pivot_table(index=["date", "code"], columns="source", values="count", aggfunc="sum", fill_value=0.0)
    piv = piv.reset_index()
    rename = {src: f"txt_{src}_count" for src in piv.columns if src not in {"date", "code"}}
    piv = piv.rename(columns=rename)
    for src in TEXT_SOURCES:
        col = f"txt_{src}_count"
        if col not in piv.columns:
            piv[col] = 0.0
    return piv


def _source_metric_table(events: pd.DataFrame, metric_col: str, suffix: str) -> pd.DataFrame:
    if events.empty or metric_col not in events.columns:
        return pd.DataFrame(columns=["date", "code"])
    agg = (
        events.groupby(["date", "code", "source"], as_index=False)[metric_col]
        .mean()
        .rename(columns={metric_col: "v"})
    )
    piv = agg.pivot_table(index=["date", "code"], columns="source", values="v", aggfunc="mean")
    piv = piv.reset_index()
    rename = {src: f"txt_{src}_{suffix}" for src in piv.columns if src not in {"date", "code"}}
    piv = piv.rename(columns=rename)
    return piv


def build_text_daily_features(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["date", "code"])
    x = events.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    x["code"] = x["code"].astype(str)
    x = x.dropna(subset=["date", "code"])
    if x.empty:
        return pd.DataFrame(columns=["date", "code"])

    g = x.groupby(["date", "code"], as_index=False)
    base = g.agg(
        txt_event_count=("source", "size"),
        txt_source_coverage=("source", "nunique"),
        txt_title_len_mean=("title_len", "mean"),
        txt_body_len_mean=("body_len", "mean"),
        txt_text_len_mean=("text_len", "mean"),
        txt_sentiment_mean=("sentiment", "mean"),
        txt_sentiment_std=("sentiment", "std"),
        txt_pos_share=("sent_pos", "mean"),
        txt_neg_share=("sent_neg", "mean"),
        txt_uncertainty_mean=("uncertainty_score", "mean"),
        txt_risk_mean=("risk_score", "mean"),
        txt_action_mean=("action_score", "mean"),
        txt_attention_mean=("attention_score", "mean"),
        txt_novelty_mean=("novelty_score", "mean"),
        txt_digit_ratio_mean=("digit_ratio", "mean"),
        txt_punc_intensity_mean=("punc_intensity", "mean"),
        txt_quality_mean=("source_quality", "mean"),
        txt_rating_delta_mean=("rating_delta", "mean"),
        txt_topic_earnings=("topic_earnings", "mean"),
        txt_topic_policy=("topic_policy", "mean"),
        txt_topic_mna=("topic_mna", "mean"),
        txt_topic_risk=("topic_risk", "mean"),
        txt_topic_operation=("topic_operation", "mean"),
        txt_topic_capital=("topic_capital", "mean"),
    )

    cnt = _source_count_table(x)
    sent = _source_metric_table(x, metric_col="sentiment", suffix="sentiment")
    attn = _source_metric_table(x, metric_col="attention_score", suffix="attention")
    risk = _source_metric_table(x, metric_col="risk_score", suffix="risk")

    out = base.merge(cnt, on=["date", "code"], how="left")
    out = out.merge(sent, on=["date", "code"], how="left")
    out = out.merge(attn, on=["date", "code"], how="left")
    out = out.merge(risk, on=["date", "code"], how="left")

    for src in TEXT_SOURCES:
        c = f"txt_{src}_count"
        if c not in out.columns:
            out[c] = 0.0
    denom = pd.to_numeric(out["txt_event_count"], errors="coerce").fillna(0.0) + EPS
    out["txt_news_share"] = out["txt_news_count"] / denom
    out["txt_notice_share"] = out["txt_notice_count"] / denom
    out["txt_em_report_share"] = out["txt_em_report_count"] / denom
    out["txt_iwencai_share"] = out["txt_iwencai_count"] / denom

    numeric_cols = [c for c in out.columns if c not in {"date", "code"}]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    return out.sort_values(["code", "date"]).reset_index(drop=True)


def add_text_rolling_and_fusion_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "code" not in out.columns or "date" not in out.columns:
        return out
    text_cols = [c for c in out.columns if str(c).startswith("txt_")]
    if not text_cols:
        return out

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["code"] = out["code"].astype(str)
    out = out.sort_values(["code", "date"]).reset_index(drop=True)

    def _num(col: str) -> pd.Series:
        if col not in out.columns:
            return pd.Series(np.nan, index=out.index, dtype=float)
        return pd.to_numeric(out[col], errors="coerce")

    def _roll_mean(s: pd.Series, w: int, minp: int) -> pd.Series:
        return s.rolling(w, min_periods=minp).mean()

    def _roll_std(s: pd.Series, w: int, minp: int) -> pd.Series:
        return s.rolling(w, min_periods=minp).std()

    cnt = _num("txt_event_count").fillna(0.0)
    sent = _num("txt_sentiment_mean").fillna(0.0)
    attn = _num("txt_attention_mean").fillna(0.0)
    risk = _num("txt_risk_mean").fillna(0.0)
    unct = _num("txt_uncertainty_mean").fillna(0.0)
    nov = _num("txt_novelty_mean").fillna(0.0)

    gkey = out["code"]
    out["txt_count_3"] = cnt.groupby(gkey).transform(lambda s: s.rolling(3, min_periods=1).sum())
    out["txt_count_5"] = cnt.groupby(gkey).transform(lambda s: s.rolling(5, min_periods=1).sum())
    out["txt_count_10"] = cnt.groupby(gkey).transform(lambda s: s.rolling(10, min_periods=2).sum())
    out["txt_count_20"] = cnt.groupby(gkey).transform(lambda s: s.rolling(20, min_periods=5).sum())

    out["txt_sent_5"] = sent.groupby(gkey).transform(lambda s: _roll_mean(s, 5, 1))
    out["txt_sent_20"] = sent.groupby(gkey).transform(lambda s: _roll_mean(s, 20, 5))
    out["txt_sent_trend_5"] = out["txt_sent_5"] - out["txt_sent_20"]
    out["txt_sent_vol_20"] = sent.groupby(gkey).transform(lambda s: _roll_std(s, 20, 5))

    out["txt_attention_5"] = attn.groupby(gkey).transform(lambda s: _roll_mean(s, 5, 1))
    out["txt_attention_20"] = attn.groupby(gkey).transform(lambda s: _roll_mean(s, 20, 5))
    out["txt_attention_shock"] = out["txt_attention_5"] - out["txt_attention_20"]

    out["txt_risk_5"] = risk.groupby(gkey).transform(lambda s: _roll_mean(s, 5, 1))
    out["txt_risk_20"] = risk.groupby(gkey).transform(lambda s: _roll_mean(s, 20, 5))
    out["txt_uncertainty_5"] = unct.groupby(gkey).transform(lambda s: _roll_mean(s, 5, 1))
    out["txt_uncertainty_20"] = unct.groupby(gkey).transform(lambda s: _roll_mean(s, 20, 5))
    out["txt_novelty_5"] = nov.groupby(gkey).transform(lambda s: _roll_mean(s, 5, 1))
    out["txt_novelty_20"] = nov.groupby(gkey).transform(lambda s: _roll_mean(s, 20, 5))

    for src in TEXT_SOURCES:
        c = f"txt_{src}_count"
        b = f"txt_{src}_burst"
        if c in out.columns:
            s = _num(c).fillna(0.0)
            ma20 = s.groupby(gkey).transform(lambda x: _roll_mean(x, 20, 5))
            out[b] = s / (ma20 + EPS)
        else:
            out[b] = np.nan

    share_cols = [f"txt_{src}_share" for src in TEXT_SOURCES if f"txt_{src}_share" in out.columns]
    if share_cols:
        out["txt_source_dispersion"] = out[share_cols].std(axis=1, skipna=True)
    else:
        out["txt_source_dispersion"] = np.nan
    out["txt_topic_risk_vs_earnings"] = _num("txt_topic_risk") - _num("txt_topic_earnings")
    out["txt_topic_policy_mna_mix"] = _num("txt_topic_policy") + _num("txt_topic_mna")

    # Text -> price/high-frequency fusion.
    out["txt_hf_sent_flow"] = _num("txt_sentiment_mean") * _num("signed_vol_imbalance_5m")
    out["txt_hf_attention_jump"] = -_num("txt_attention_shock") * _num("jump_ratio_5m")
    out["txt_hf_risk_vol"] = -_num("txt_risk_20") * _num("realized_vol_20")
    out["txt_hf_momentum_resonance"] = _num("txt_sent_trend_5") * _num("ret_3d")
    out["txt_hf_liquidity_pulse"] = _num("txt_news_burst") * _num("amount_ratio_20")

    # Text -> fundamental fusion.
    out["txt_fd_growth_resonance"] = _num("txt_sentiment_mean") * _num("fd_growth_score")
    out["txt_fd_quality_guard"] = -_num("txt_risk_20") * _num("fd_quality_score")
    out["txt_fd_expectation_spread"] = _num("txt_uncertainty_20") * _num("fd_expectation_score")
    out["txt_fd_cashflow_conflict"] = _num("txt_attention_shock") - _num("fd_cashflow_trend")
    out["txt_fd_valuation_contrast"] = _num("txt_sentiment_mean") - _num("fd_valuation_score")

    fusion_cols = [
        "txt_hf_sent_flow",
        "txt_hf_attention_jump",
        "txt_hf_risk_vol",
        "txt_hf_momentum_resonance",
        "txt_hf_liquidity_pulse",
        "txt_fd_growth_resonance",
        "txt_fd_quality_guard",
        "txt_fd_expectation_spread",
        "txt_fd_cashflow_conflict",
        "txt_fd_valuation_contrast",
    ]
    existing_fusion_cols = [c for c in fusion_cols if c in out.columns]
    if existing_fusion_cols:
        out["txt_fusion_score"] = out[existing_fusion_cols].mean(axis=1, skipna=True)
        out["txt_fusion_disp"] = out[existing_fusion_cols].std(axis=1, skipna=True)
    else:
        out["txt_fusion_score"] = np.nan
        out["txt_fusion_disp"] = np.nan

    for c in [x for x in out.columns if str(x).startswith("txt_")]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype("float32")
    return out
