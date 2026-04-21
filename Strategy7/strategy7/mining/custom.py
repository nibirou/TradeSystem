"""User-defined factor expressions based on Strategy7 panel columns."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ..core.constants import EPS


@dataclass
class CustomFactorSpec:
    name: str
    expression: str
    freq: str = "D"
    category: str = "custom_factor"
    description: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def load_custom_specs(path: str | Path) -> List[CustomFactorSpec]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        raw = p.read_text(encoding="utf-8-sig")
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            records = parsed
        elif isinstance(parsed, dict):
            records = parsed.get("items", [])
        else:
            records = []
    except Exception:
        return []
    specs: List[CustomFactorSpec] = []
    for rec in records:
        name = str(rec.get("name", "")).strip()
        expr = str(rec.get("expression", "")).strip()
        if not name or not expr:
            continue
        specs.append(
            CustomFactorSpec(
                name=name,
                expression=expr,
                freq=str(rec.get("freq", "D")),
                category=str(rec.get("category", "custom_factor")),
                description=str(rec.get("description", "")),
            )
        )
    return specs


def _safe_custom_name(source_factor: str, *, prefix: str = "custom_eval") -> str:
    token = re.sub(r"[^0-9a-zA-Z_]+", "_", str(source_factor).strip()).strip("_").lower()
    if not token:
        token = "factor"
    if token[0].isdigit():
        token = f"f_{token}"
    base = f"{prefix}_{token}"
    if len(base) <= 96:
        return base
    digest = hashlib.sha1(str(source_factor).encode("utf-8")).hexdigest()[:8]
    keep = max(8, 96 - len(prefix) - len(digest) - 2)
    return f"{prefix}_{token[:keep]}_{digest}"


def build_custom_specs_from_factor_names(
    factor_names: Sequence[str],
    *,
    freq: str = "D",
    category: str = "mined_custom",
    name_prefix: str = "custom_eval",
) -> List[CustomFactorSpec]:
    """Build passthrough custom specs that evaluate existing panel factor columns."""
    out: List[CustomFactorSpec] = []
    used_names: set[str] = set()
    for fac in factor_names:
        source = str(fac).strip()
        if not source:
            continue
        name = _safe_custom_name(source, prefix=name_prefix)
        idx = 2
        while name in used_names:
            name = f"{_safe_custom_name(source, prefix=name_prefix)}_{idx:02d}"
            idx += 1
        used_names.add(name)
        out.append(
            CustomFactorSpec(
                name=name,
                expression=f'col("{source}")',
                freq=str(freq),
                category=str(category),
                description=f"custom passthrough evaluate: {source}",
            )
        )
    return out


class _ExprEvaluator:
    def __init__(self, panel: pd.DataFrame, time_col: str = "date", code_col: str = "code") -> None:
        self.df = panel.copy()
        self.time_col = time_col
        self.code_col = code_col
        self.df[code_col] = self.df[code_col].astype(str)
        self.df[time_col] = pd.to_datetime(self.df[time_col], errors="coerce")
        self.df = self.df.sort_values([code_col, time_col]).copy()

    def eval(self, expression: str) -> pd.Series:
        tree = ast.parse(expression, mode="eval")
        out = self._eval_node(tree.body)
        if isinstance(out, (int, float, np.number)):
            s = pd.Series(float(out), index=self.df.index)
        elif isinstance(out, pd.Series):
            s = out.reindex(self.df.index)
        else:
            raise ValueError("expression did not evaluate to Series")
        return pd.to_numeric(s, errors="coerce").reindex(self.df.index)

    def _series(self, x) -> pd.Series:
        if isinstance(x, pd.Series):
            return pd.to_numeric(x, errors="coerce")
        if isinstance(x, (int, float, np.number)):
            return pd.Series(float(x), index=self.df.index, dtype=float)
        raise ValueError(f"unsupported value type: {type(x)}")

    def _binary(self, left, right, op):
        a = self._series(left)
        b = self._series(right)
        if isinstance(op, ast.Add):
            return a + b
        if isinstance(op, ast.Sub):
            return a - b
        if isinstance(op, ast.Mult):
            return a * b
        if isinstance(op, ast.Div):
            return a / (b + EPS)
        if isinstance(op, ast.Pow):
            return np.sign(a) * (np.abs(a) ** b)
        raise ValueError("unsupported binary op")

    def _compare(self, left, comparators, ops):
        a = self._series(left)
        cur = pd.Series(True, index=self.df.index)
        left_s = a
        for op, comp in zip(ops, comparators):
            b = self._series(comp)
            if isinstance(op, ast.Gt):
                cur = cur & (left_s > b)
            elif isinstance(op, ast.GtE):
                cur = cur & (left_s >= b)
            elif isinstance(op, ast.Lt):
                cur = cur & (left_s < b)
            elif isinstance(op, ast.LtE):
                cur = cur & (left_s <= b)
            elif isinstance(op, ast.Eq):
                cur = cur & (left_s == b)
            elif isinstance(op, ast.NotEq):
                cur = cur & (left_s != b)
            else:
                raise ValueError("unsupported compare op")
            left_s = b
        return cur

    def _call(self, name: str, args: List):
        fn = str(name).lower()

        if fn == "col":
            if not args:
                raise ValueError("col(name) requires one column name argument")
            col_name = str(args[0])
            if col_name not in self.df.columns:
                raise ValueError(f"column not found in panel: {col_name}")
            return pd.to_numeric(self.df[col_name], errors="coerce")

        if fn == "abs":
            return self._series(args[0]).abs()
        if fn == "log":
            x = self._series(args[0])
            return np.sign(x) * np.log1p(np.abs(x))
        if fn == "sqrt":
            x = self._series(args[0])
            return np.sqrt(np.abs(x))
        if fn == "sign":
            return np.sign(self._series(args[0]))
        if fn == "clip":
            x = self._series(args[0])
            lo = float(args[1]) if len(args) > 1 else -3.0
            hi = float(args[2]) if len(args) > 2 else 3.0
            return x.clip(lower=lo, upper=hi)

        if fn == "delay":
            x = self._series(args[0])
            n = int(float(args[1]))
            return x.groupby(self.df[self.code_col]).shift(n)
        if fn == "delta":
            x = self._series(args[0])
            n = int(float(args[1]))
            return x - x.groupby(self.df[self.code_col]).shift(n)
        if fn == "pct":
            x = self._series(args[0])
            n = int(float(args[1]))
            prev = x.groupby(self.df[self.code_col]).shift(n)
            return x / (prev + EPS) - 1.0
        if fn == "ts_mean":
            x = self._series(args[0])
            n = max(1, int(float(args[1])))
            return x.groupby(self.df[self.code_col]).transform(lambda s: s.rolling(n, min_periods=max(2, n // 3)).mean())
        if fn == "ts_std":
            x = self._series(args[0])
            n = max(2, int(float(args[1])))
            return x.groupby(self.df[self.code_col]).transform(lambda s: s.rolling(n, min_periods=max(2, n // 3)).std())
        if fn == "ts_z":
            x = self._series(args[0])
            n = max(2, int(float(args[1])))
            m = x.groupby(self.df[self.code_col]).transform(lambda s: s.rolling(n, min_periods=max(2, n // 3)).mean())
            st = x.groupby(self.df[self.code_col]).transform(lambda s: s.rolling(n, min_periods=max(2, n // 3)).std())
            return (x - m) / (st + EPS)

        if fn == "cs_rank":
            x = self._series(args[0])
            return x.groupby(self.df[self.time_col]).transform(lambda s: s.rank(pct=True, method="average"))
        if fn == "cs_z":
            x = self._series(args[0])

            def _z(v: pd.Series) -> pd.Series:
                std = float(v.std(ddof=0))
                if std <= EPS:
                    return pd.Series(np.zeros(len(v), dtype=float), index=v.index)
                return (v - float(v.mean())) / (std + EPS)

            return x.groupby(self.df[self.time_col]).transform(_z)

        if fn == "where":
            cond = args[0]
            a = self._series(args[1])
            b = self._series(args[2])
            c = cond if isinstance(cond, pd.Series) else pd.Series(bool(cond), index=self.df.index)
            return pd.Series(np.where(c.fillna(False), a, b), index=self.df.index, dtype=float)

        raise ValueError(f"unsupported function: {name}")

    def _eval_node(self, node):
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            name = str(node.id)
            if name not in self.df.columns:
                raise ValueError(f"column not found in panel: {name}")
            return pd.to_numeric(self.df[name], errors="coerce")

        if isinstance(node, ast.UnaryOp):
            v = self._series(self._eval_node(node.operand))
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return v
            raise ValueError("unsupported unary op")

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self._binary(left, right, node.op)

        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left)
            comparators = [self._eval_node(c) for c in node.comparators]
            return self._compare(left, comparators, node.ops)

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("only simple function names are allowed")
            name = node.func.id
            args = [self._eval_node(a) for a in node.args]
            return self._call(name, args)

        if isinstance(node, ast.BoolOp):
            vals = [self._eval_node(v) for v in node.values]
            out = vals[0].copy()
            for x in vals[1:]:
                if isinstance(node.op, ast.And):
                    out = out & x
                elif isinstance(node.op, ast.Or):
                    out = out | x
                else:
                    raise ValueError("unsupported bool op")
            return out

        raise ValueError(f"unsupported syntax node: {type(node).__name__}")


def evaluate_custom_factor_expression(
    panel: pd.DataFrame,
    expression: str,
    time_col: str = "date",
    code_col: str = "code",
) -> pd.Series:
    ev = _ExprEvaluator(panel=panel, time_col=time_col, code_col=code_col)
    return ev.eval(expression)


def evaluate_custom_specs(
    panel: pd.DataFrame,
    specs: List[CustomFactorSpec],
    time_col: str = "date",
    code_col: str = "code",
) -> pd.DataFrame:
    out = panel[[time_col, code_col]].copy()
    for spec in specs:
        out[spec.name] = evaluate_custom_factor_expression(
            panel=panel,
            expression=spec.expression,
            time_col=time_col,
            code_col=code_col,
        )
    return out
