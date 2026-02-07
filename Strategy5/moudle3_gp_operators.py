# gp_operators.py
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F

from moudle1_gp_config import Config


def _safe_div(a, b, eps=1e-6):
    return a / (b.abs() + eps) * b.sign()


def _safe_log(x, eps=1e-6):
    return torch.log(x.abs() + eps) * x.sign()


def _clip(x, lo=-10.0, hi=10.0):
    return torch.clamp(x, lo, hi)


def cs_rank(x: torch.Tensor) -> torch.Tensor:
    """
    x: [T, N] 按每个t对N做rank并缩放到[-0.5,0.5]
    """
    # argsort twice -> rank
    # rank: 0..N-1
    s = torch.argsort(x, dim=1)
    r = torch.argsort(s, dim=1).float()
    n = x.size(1)
    return (r / (n - 1 + 1e-12)) - 0.5


def ts_mean(x: torch.Tensor, win: int) -> torch.Tensor:
    # x [T,N]
    if win <= 1:
        return x
    # 用1d conv实现滑动平均：对时间维做
    # reshape -> [N,1,T]
    xn = x.transpose(0, 1).unsqueeze(1)
    w = torch.ones(1, 1, win, device=x.device) / win
    y = F.conv1d(xn, w, padding=win - 1)  # left padding
    y = y[:, :, : x.size(0)]
    return y.squeeze(1).transpose(0, 1)


def ts_std(x: torch.Tensor, win: int) -> torch.Tensor:
    m = ts_mean(x, win)
    m2 = ts_mean(x * x, win)
    v = torch.clamp(m2 - m * m, min=1e-12)
    return torch.sqrt(v)


def ts_zscore(x: torch.Tensor, win: int) -> torch.Tensor:
    m = ts_mean(x, win)
    s = ts_std(x, win)
    return (x - m) / (s + 1e-6)


def ts_corr(a: torch.Tensor, b: torch.Tensor, win: int) -> torch.Tensor:
    # corr = cov / (std_a*std_b)
    am = ts_mean(a, win)
    bm = ts_mean(b, win)
    cov = ts_mean((a - am) * (b - bm), win)
    sa = ts_std(a, win)
    sb = ts_std(b, win)
    return cov / (sa * sb + 1e-6)


@dataclass(frozen=True)
class Op:
    name: str
    arity: int
    fn: Callable


class OperatorSet:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # binary ops
        self.binary_ops: List[Op] = [
            Op("add", 2, lambda a, b: a + b),
            Op("sub", 2, lambda a, b: a - b),
            Op("mul", 2, lambda a, b: a * b),
            Op("div", 2, lambda a, b: _safe_div(a, b)),
            # corr/组合一般强，给它固定窗口也行（你也可以做成可变窗口的“常数节点”）
            Op("ts_corr_10", 2, lambda a, b: ts_corr(a, b, 10)),
            Op("ts_corr_20", 2, lambda a, b: ts_corr(a, b, 20)),
        ]

        # unary ops
        self.unary_ops: List[Op] = [
            Op("neg", 1, lambda x: -x),
            Op("abs", 1, lambda x: x.abs()),
            Op("log", 1, lambda x: _safe_log(x)),
            Op("clip", 1, lambda x: _clip(x)),
            Op("cs_rank", 1, lambda x: cs_rank(x)),
            Op("ts_mean_5", 1, lambda x: ts_mean(x, 5)),
            Op("ts_mean_10", 1, lambda x: ts_mean(x, 10)),
            Op("ts_std_10", 1, lambda x: ts_std(x, 10)),
            Op("ts_z_10", 1, lambda x: ts_zscore(x, 10)),
            Op("ts_z_20", 1, lambda x: ts_zscore(x, 20)),
        ]

        self.all_ops = self.unary_ops + self.binary_ops

    def sample_op(self, arity: Optional[int] = None) -> Op:
        import random
        if arity is None:
            return random.choice(self.all_ops)
        pool = self.unary_ops if arity == 1 else self.binary_ops
        return random.choice(pool)
