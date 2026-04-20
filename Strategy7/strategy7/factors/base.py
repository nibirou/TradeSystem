"""Pluggable factor library."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List

import pandas as pd

from ..core.utils import import_module_from_file

FactorFunc = Callable[[pd.DataFrame], pd.Series]


@dataclass
class FactorDef:
    name: str
    category: str
    description: str
    func: FactorFunc
    freq: str


class FactorLibrary:
    """Frequency-aware factor registry."""

    def __init__(self) -> None:
        self._factors: Dict[str, FactorDef] = {}

    def _key(self, freq: str, name: str) -> str:
        return f"{freq}:{name}"

    def register(self, name: str, category: str, description: str, func: FactorFunc, freq: str = "D") -> None:
        self._factors[self._key(freq, name)] = FactorDef(name=name, category=category, description=description, func=func, freq=freq)

    def names(self, freq: str) -> List[str]:
        return sorted(v.name for v in self._factors.values() if v.freq == freq)

    def get(self, freq: str, name: str) -> FactorDef:
        k = self._key(freq, name)
        if k not in self._factors:
            raise KeyError(f"factor not found: {name} @ {freq}")
        return self._factors[k]

    def has(self, freq: str, name: str) -> bool:
        return self._key(freq, name) in self._factors

    def metadata(self, freq: str | None = None) -> pd.DataFrame:
        rows = []
        for fd in self._factors.values():
            if freq is not None and fd.freq != freq:
                continue
            rows.append(
                {
                    "factor": fd.name,
                    "freq": fd.freq,
                    "category": fd.category,
                    "description": fd.description,
                }
            )
        return pd.DataFrame(rows).sort_values(["freq", "factor"]) if rows else pd.DataFrame(columns=["factor", "freq", "category", "description"])


_CATEGORY_CN: Dict[str, str] = {
    "trend": "趋势动量",
    "reversal": "反转均值回复",
    "liquidity": "流动性",
    "volatility": "波动率",
    "flow": "成交资金流",
    "context": "市场上下文",
    "structure": "结构形态",
    "bridge": "跨频桥接",
    "multi_freq": "跨频桥接",
    "multiscale": "多尺度",
    "intraday_signature": "日内特征签名",
    "intraday_micro": "日内微观结构",
    "period_signature": "周期签名",
    "price_action": "价格行为",
    "crowding": "拥挤度",
    "oscillator": "摆动指标",
    "overnight": "隔夜",
    "auto_panel": "自动透传",
    "fundamental_growth": "基本面-成长",
    "fundamental_valuation": "基本面-估值",
    "fundamental_profitability": "基本面-盈利能力",
    "fundamental_quality": "基本面-质量",
    "fundamental_leverage": "基本面-杠杆偿债",
    "fundamental_cashflow": "基本面-现金流",
    "fundamental_efficiency": "基本面-运营效率",
    "fundamental_expectation": "基本面-预期",
    "fundamental_hf_fusion": "基本面-高频融合",
    "mined_price_volume": "挖掘因子-量价",
    "mined_fundamental": "挖掘因子-基本面",
    "mined_text": "挖掘因子-文本",
    "mined_fusion": "挖掘因子-融合",
    "mined_other": "挖掘因子-其他",
    "mined_custom": "挖掘因子-自定义",
    "catalog_custom": "Catalog-自定义",
}

_CATEGORY_CN.update(
    {
        "text_sentiment": "金融文本-情绪",
        "text_attention": "金融文本-关注度",
        "text_event": "金融文本-事件",
        "text_topic": "金融文本-主题",
        "text_fusion": "金融文本-融合",
    }
)

_PACKAGE_CN: Dict[str, str] = {
    "g_trend": "量价趋势包",
    "g_reversal": "量价反转包",
    "g_liquidity": "量价流动性包",
    "g_volatility": "量价波动率包",
    "g_structure": "量价结构包",
    "g_context": "量价上下文包",
    "fund_growth": "基本面成长包",
    "fund_valuation": "基本面估值包",
    "fund_profitability": "基本面盈利能力包",
    "fund_quality": "基本面质量包",
    "fund_leverage": "基本面杠杆偿债包",
    "fund_cashflow": "基本面现金流包",
    "fund_efficiency": "基本面运营效率包",
    "fund_expectation": "基本面预期包",
    "fund_hf_fusion": "基本面-高频融合包",
    "mined_price_volume": "挖掘量价包",
    "mined_fundamental": "挖掘基本面包",
    "mined_text": "挖掘文本包",
    "mined_fusion": "挖掘融合包",
    "mined_other": "挖掘其他包",
    "mined_custom": "挖掘自定义包",
    "catalog_custom": "Catalog自定义包",
}

_PACKAGE_CN.update(
    {
        "text_sentiment": "金融文本情绪包",
        "text_attention": "金融文本关注度包",
        "text_event": "金融文本事件包",
        "text_topic": "金融文本主题包",
        "text_fusion": "金融文本融合包",
    }
)

_FUND_CAT_CN: Dict[str, str] = {
    "growth": "成长",
    "valuation": "估值",
    "profitability": "盈利能力",
    "quality": "质量",
    "leverage": "杠杆偿债",
    "cashflow": "现金流",
    "efficiency": "运营效率",
    "expectation": "预期",
    "fusion": "融合",
}

_AGG_CN: Dict[str, str] = {
    "mean": "均值",
    "std": "标准差",
    "last": "末值",
    "min": "最小值",
    "max": "最大值",
    "sum": "求和",
}

_TOKEN_CN: Dict[str, str] = {
    "ret": "收益",
    "ma": "均线",
    "gap": "偏离",
    "rv": "实现波动率",
    "range": "振幅",
    "norm": "标准化",
    "amount": "成交额",
    "vol": "成交量",
    "chg": "变化",
    "crowding": "拥挤",
    "proxy": "代理",
    "raw": "原始",
    "score": "评分",
    "trend": "趋势",
    "disp": "离散度",
    "flow": "资金流",
    "jump": "跳跃",
    "intraday": "日内",
    "vwap": "VWAP",
    "sentiment": "情绪",
    "close": "收盘",
    "open": "开盘",
    "high": "最高",
    "low": "最低",
    "noise": "噪声",
    "signal": "信号",
    "liq": "流动性",
    "liquidity": "流动性",
    "micro": "微观",
    "macro": "宏观",
    "period": "周期",
    "momentum": "动量",
    "quality": "质量",
    "breakout": "突破",
    "pressure": "压力",
    "sync": "同步",
    "stress": "压力",
    "carry": "延续",
    "compression": "压缩",
    "dispersion": "离散度",
    "deviation": "偏离",
}

_TOKEN_CN.update(
    {
        "txt": "文本",
        "sentiment": "情绪",
        "attention": "关注度",
        "event": "事件",
        "topic": "主题",
        "news": "新闻",
        "notice": "公告",
        "iwencai": "问财",
        "uncertainty": "不确定性",
        "risk": "风险",
        "novelty": "新颖度",
        "burst": "异动",
        "fusion": "融合",
    }
)

_FORMULA_WORD_CN: Dict[str, str] = {
    "minus": "减",
    "over": "除以",
    "negative": "取负",
    "ratio": "比值",
    "diff": "差值",
    "symmetric": "对称",
    "spread": "价差",
    "with": "结合",
    "adjusted": "调整后",
    "by": "按",
    "to": "到",
    "and": "与",
    "or": "或",
    "level": "原值",
    "return": "收益",
    "volatility": "波动率",
    "liquidity": "流动性",
    "trend": "趋势",
    "noise": "噪声",
    "signal": "信号",
    "crowding": "拥挤",
    "unwind": "反转",
    "tanh": "双曲正切",
    "mean": "均值",
    "std": "标准差",
    "last": "末值",
    "close": "收盘",
    "daily": "日频",
    "intraday": "日内",
}


def _category_cn(category: str) -> str:
    c = str(category or "")
    if c in _CATEGORY_CN:
        return _CATEGORY_CN[c]
    if c.startswith("fundamental_"):
        tail = c[len("fundamental_") :]
        return f"基本面-{_FUND_CAT_CN.get(tail, tail)}"
    return c


def _package_cn(prefix: str, category: str) -> str:
    p = str(prefix or "")
    if p in _PACKAGE_CN:
        return _PACKAGE_CN[p]
    if p.startswith("g_"):
        return f"量价{p[2:]}包"
    if p.startswith("fund_"):
        return f"基本面{p[5:]}包"
    cat_cn = _category_cn(category)
    return f"{cat_cn}包" if cat_cn else p


def _fallback_cn_text(token: str) -> str:
    t = str(token or "")
    parts = [x for x in t.split("_") if x]
    if not parts:
        return t
    mapped = [_TOKEN_CN.get(p, p) for p in parts]
    return "".join(mapped)


def _metric_cn(metric: str) -> str:
    m = str(metric or "").strip()
    if not m:
        return m

    m_fd_raw = re.match(r"^fd_([a-z]+)_raw_(\d+)$", m)
    if m_fd_raw:
        cat = _FUND_CAT_CN.get(m_fd_raw.group(1), m_fd_raw.group(1))
        return f"{cat}原始指标{int(m_fd_raw.group(2)):02d}"

    m_fd_stat = re.match(r"^fd_([a-z]+)_(score|trend|disp)$", m)
    if m_fd_stat:
        cat = _FUND_CAT_CN.get(m_fd_stat.group(1), m_fd_stat.group(1))
        stat_map = {"score": "综合评分", "trend": "趋势变化", "disp": "离散度"}
        return f"{cat}{stat_map.get(m_fd_stat.group(2), m_fd_stat.group(2))}"

    m_fd_hf = re.match(r"^fd_hf_([a-z]+)_([a-z0-9]+)$", m)
    if m_fd_hf:
        cat = _FUND_CAT_CN.get(m_fd_hf.group(1), m_fd_hf.group(1))
        metric_tail = _fallback_cn_text(m_fd_hf.group(2))
        return f"{cat}高频{metric_tail}"

    m_hf = re.match(
        r"^hf_(5min|15min|30min|60min|120min|D|W|M)_to_(5min|15min|30min|60min|120min|D|W|M)_([a-z]+)_(.+)$",
        m,
    )
    if m_hf:
        src, tgt, agg, base = m_hf.groups()
        return f"{src}→{tgt} { _AGG_CN.get(agg, agg) }({ _metric_cn(base) })"

    m_ret = re.match(r"^ret_(\d+)$", m)
    if m_ret:
        return f"{int(m_ret.group(1))}期收益率"

    m_ma_gap = re.match(r"^ma_gap_(\d+)$", m)
    if m_ma_gap:
        return f"均线偏离{int(m_ma_gap.group(1))}"

    m_rv = re.match(r"^rv_(\d+)$", m)
    if m_rv:
        return f"{int(m_rv.group(1))}期实现波动率"

    m_amount_ratio = re.match(r"^amount_ratio_(\d+)$", m)
    if m_amount_ratio:
        return f"成交额比率{int(m_amount_ratio.group(1))}"

    m_vol_chg = re.match(r"^vol_chg_(\d+)$", m)
    if m_vol_chg:
        return f"成交量变化{int(m_vol_chg.group(1))}"

    exact = {
        "range_norm": "振幅标准化",
        "crowding_proxy_raw": "拥挤度原始代理",
        "context_trend_20d": "20日趋势上下文",
        "context_quality_20d": "20日质量上下文",
        "context_liquidity_20d": "20日流动性上下文",
        "context_intraday_mood": "日内情绪上下文",
        "hf_fast_slow_trend_diff": "快慢频趋势差",
        "hf_fast_slow_liquidity_diff": "快慢频流动性差",
        "hf_fast_slow_noise_diff": "快慢频噪声差",
    }
    if m in exact:
        return exact[m]

    return _fallback_cn_text(m)


def _translate_formula_text(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return s
    # Replace phrase-like words first.
    for en, cn in sorted(_FORMULA_WORD_CN.items(), key=lambda x: len(x[0]), reverse=True):
        s = re.sub(rf"\b{re.escape(en)}\b", cn, s, flags=re.IGNORECASE)
    # Replace known metric-like tokens.
    def _replace_token(match: re.Match[str]) -> str:
        tok = match.group(0)
        if tok.isupper() and len(tok) <= 4:
            return tok
        if tok.lower() in {"eps", "abs", "tanh", "mean", "std", "last"}:
            return tok
        return _metric_cn(tok)

    s = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", _replace_token, s)
    s = s.replace("->", "→")
    return s


def _explain_template_factor(name: str, category: str) -> Dict[str, str] | None:
    # Parse only when name starts with known template package prefix,
    # to avoid confusing metric substrings (e.g. *_ratio_12, *_diff) as operators.
    prefixes = sorted(_PACKAGE_CN.keys(), key=len, reverse=True)
    for prefix in prefixes:
        for op in ["_lvl_", "_neg_", "_tanh_"]:
            token = f"{prefix}{op}"
            if not name.startswith(token):
                continue
            col = name[len(token) :]
            p_cn = _package_cn(prefix, category)
            c_cn = _metric_cn(col)
            if op == "_lvl_":
                return {
                    "name_cn": f"{p_cn}·{c_cn}原值",
                    "meaning_cn": f"直接使用 {c_cn} 作为 {p_cn} 信号，反映该基础指标的绝对水平。",
                    "formula_cn": f"{name} = 列({c_cn})",
                }
            if op == "_neg_":
                return {
                    "name_cn": f"{p_cn}·{c_cn}反向",
                    "meaning_cn": f"对 {c_cn} 取负，构造方向相反的因子暴露。",
                    "formula_cn": f"{name} = -列({c_cn})",
                }
            return {
                "name_cn": f"{p_cn}·{c_cn}压缩",
                "meaning_cn": f"对 {c_cn} 做 tanh 压缩，降低极值影响，提升稳健性。",
                "formula_cn": f"{name} = tanh(列({c_cn}))",
            }

        for op, op_cn in [("_diff_", "差值"), ("_ratio_", "比值"), ("_sym_", "对称价差")]:
            token = f"{prefix}{op}"
            if not name.startswith(token):
                continue
            tail = name[len(token) :]
            if "__" not in tail:
                return None
            a, b = tail.split("__", 1)
            p_cn = _package_cn(prefix, category)
            a_cn = _metric_cn(a)
            b_cn = _metric_cn(b)
            if op == "_diff_":
                return {
                    "name_cn": f"{p_cn}·{a_cn}-{b_cn}差值",
                    "meaning_cn": f"比较 {a_cn} 与 {b_cn} 的相对强弱，突出两者差异变化。",
                    "formula_cn": f"{name} = 列({a_cn}) - 列({b_cn})",
                }
            if op == "_ratio_":
                return {
                    "name_cn": f"{p_cn}·{a_cn}/{b_cn}比值",
                    "meaning_cn": f"衡量 {a_cn} 相对 {b_cn} 的比例关系，体现结构性偏离。",
                    "formula_cn": f"{name} = 列({a_cn}) / (|列({b_cn})| + EPS)",
                }
            return {
                "name_cn": f"{p_cn}·{a_cn}-{b_cn}{op_cn}",
                "meaning_cn": f"构造 {a_cn} 与 {b_cn} 的对称归一化价差，减少量纲影响。",
                "formula_cn": (
                    f"{name} = (列({a_cn}) - 列({b_cn})) / "
                    f"(|列({a_cn})| + |列({b_cn})| + EPS)"
                ),
            }

        token_rel = f"{prefix}_rel_"
        if name.startswith(token_rel) and "_to_" in name:
            tail = name[len(token_rel) :]
            c, anc = tail.split("_to_", 1)
            p_cn = _package_cn(prefix, category)
            c_cn = _metric_cn(c)
            anc_cn = _metric_cn(anc)
            return {
                "name_cn": f"{p_cn}·{c_cn}相对{anc_cn}",
                "meaning_cn": f"衡量 {c_cn} 相对锚点 {anc_cn} 的偏离程度，反映相对强弱。",
                "formula_cn": f"{name} = (列({c_cn}) - 列({anc_cn})) / (|列({anc_cn})| + EPS)",
            }
    return None


def _explain_bridge_factor(name: str) -> Dict[str, str] | None:
    m = re.match(r"^bridge_(5min|15min|30min|60min|120min|D|W|M)_(.+)$", name)
    if not m:
        return None
    src = m.group(1)
    suffix = m.group(2)
    meaning_map = {
        "noise": "来源频率聚合后的收益噪声信号比，刻画短周期噪声占比。",
        "trend_carry": "来源频率聚合后的趋势延续度，衡量趋势是否延续。",
        "liq_pulse": "来源频率聚合后的流动性脉冲，衡量成交突发程度。",
        "vol_comp": "来源频率聚合后的波动压缩，刻画波动收敛状态。",
        "close_disp": "来源频率聚合后的收盘离散度，衡量收盘分布分散程度。",
        "close_dev": "来源频率聚合后的收盘偏离，衡量偏离均值程度。",
        "hilo_spread": "来源频率聚合后的高低价差，反映区间波动强度。",
        "crowd_unwind": "来源频率聚合后的拥挤反转，刻画拥挤交易回撤风险。",
        "liq_trend_sync": "来源频率聚合后的流动性与趋势同步性。",
        "micro_shift": "来源频率聚合后的微观状态切换强度。",
        "ret3_std": "来源频率聚合后的中短收益波动强度。",
        "liq_pressure": "来源频率聚合后的流动性压力指标。",
    }
    return {
        "name_cn": f"跨频桥接·{src}→目标频率·{_fallback_cn_text(suffix)}",
        "meaning_cn": meaning_map.get(suffix, "将来源频率信息桥接到目标频率，构造跨频状态特征。"),
        "formula_cn": f"基于 {src}→目标频率 的聚合列计算 {suffix} 特征（见桥接定义）。",
    }


def _explain_factor_row(name: str, category: str, description: str) -> Dict[str, str]:
    tpl = _explain_template_factor(name=name, category=category)
    if tpl is not None:
        return tpl

    bridge = _explain_bridge_factor(name=name)
    if bridge is not None:
        return bridge

    cat_cn = _category_cn(category)
    desc_cn = _translate_formula_text(description)
    return {
        "name_cn": f"{cat_cn}·{_metric_cn(name)}",
        "meaning_cn": f"{cat_cn}因子，刻画：{desc_cn or '该信号的相对变化与强弱。'}",
        "formula_cn": f"按定义计算：{desc_cn or _metric_cn(name)}",
    }


def enrich_factor_metadata_for_display(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Add Chinese-readable factor metadata columns for list display."""
    if meta_df.empty:
        out = meta_df.copy()
        for c in ["name_cn", "meaning_cn", "formula_cn"]:
            out[c] = []
        return out

    required = {"factor", "category", "description"}
    if not required.issubset(set(meta_df.columns)):
        return meta_df.copy()

    out = meta_df.copy().reset_index(drop=True)
    extra_rows: List[Dict[str, str]] = []
    for row in out[["factor", "category", "description"]].itertuples(index=False):
        extra_rows.append(_explain_factor_row(str(row.factor), str(row.category), str(row.description)))
    extra_df = pd.DataFrame(extra_rows)
    out = pd.concat([out, extra_df], axis=1)

    preferred = ["factor", "freq", "category", "name_cn", "meaning_cn", "formula_cn", "description"]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]


def load_custom_factor_module(library: FactorLibrary, module_path: str) -> None:
    mod = import_module_from_file(module_path, module_name="strategy7_custom_factor_module")
    if not hasattr(mod, "register_factors"):
        raise RuntimeError("custom factor module must provide register_factors(library)")
    mod.register_factors(library)


def resolve_selected_factors(library: FactorLibrary, freq: str, factor_list_arg: str, default_set: List[str]) -> List[str]:
    if factor_list_arg.strip():
        selected = [x.strip() for x in factor_list_arg.split(",") if x.strip()]
    else:
        selected = default_set.copy()
    available = set(library.names(freq))
    missing = [f for f in selected if f not in available]
    if missing:
        raise ValueError(f"missing factors for freq={freq}: {missing}")
    return selected


def compute_factor_panel(base_df: pd.DataFrame, library: FactorLibrary, freq: str, selected_factors: List[str]) -> pd.DataFrame:
    panel = base_df.copy()
    computed: Dict[str, pd.Series] = {}
    for fac in selected_factors:
        try:
            values = library.get(freq, fac).func(panel)
        except Exception as exc:
            raise RuntimeError(f"factor computation failed: {fac} @ {freq}") from exc
        if isinstance(values, pd.Series):
            aligned = values.reindex(panel.index)
        else:
            aligned = pd.Series(values)
            if len(aligned) != len(panel):
                raise ValueError(
                    f"factor result length mismatch for {fac} @ {freq}: "
                    f"expected={len(panel)}, got={len(aligned)}"
                )
            aligned.index = panel.index
        computed[fac] = pd.to_numeric(aligned, errors="coerce")
    if computed:
        factor_frame = pd.DataFrame(computed, index=panel.index)
        overlap = [c for c in factor_frame.columns if c in panel.columns]
        if overlap:
            panel = panel.drop(columns=overlap, errors="ignore")
        panel = pd.concat([panel, factor_frame], axis=1)
    return panel


def register_passthrough_panel_factors(
    library: FactorLibrary,
    base_df: pd.DataFrame,
    freq: str,
    *,
    category: str = "auto_panel",
    description_prefix: str = "auto passthrough",
) -> List[str]:
    """Register numeric panel columns as direct-usage factors for the target frequency."""
    if base_df.empty:
        return []

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
    registered: List[str] = []
    for c in base_df.columns:
        if c in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(base_df[c]):
            continue
        if library.has(freq, c):
            continue
        library.register(
            name=str(c),
            category=category,
            description=f"{description_prefix}: {c}",
            func=lambda d, col=c: pd.to_numeric(d[col], errors="coerce"),
            freq=freq,
        )
        registered.append(str(c))
    return sorted(registered)
