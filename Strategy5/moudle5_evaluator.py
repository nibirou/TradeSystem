# evaluator.py
import math
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, Optional, List

from moudle1_gp_config import Config
from moudle4_gp_tree import GPTree


class FactorEvaluator:
    def __init__(self, cfg: Config, panel: pd.DataFrame):
        self.cfg = cfg
        self.panel = panel

        self.dates = panel.index.get_level_values(0).unique().sort_values()
        self.codes = panel.index.get_level_values(1).unique().sort_values()

        # 转为 [T,N] 张量
        self._X, self._y, self._mask = self._build_tensors(panel)

    def _build_tensors(self, panel: pd.DataFrame):
        # pivot -> [T,N]
        dates = self.dates
        codes = self.codes

        # 特征字典
        X = {}
        for col in (self.cfg.price_features + self.cfg.fundamental_features):
            if col not in panel.columns:
                continue
            pv = panel[col].unstack("code").reindex(index=dates, columns=codes)
            x = torch.tensor(pv.values, dtype=torch.float32, device=self.cfg.device)
            X[col] = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        yname = f"fwd_ret_{self.cfg.forward_ret_days}d"
        yv = panel[yname].unstack("code").reindex(index=dates, columns=codes)
        y = torch.tensor(yv.values, dtype=torch.float32, device=self.cfg.device)

        mask = torch.isfinite(y)  # label存在
        return X, y, mask

    @torch.no_grad()
    def _rankic_series(self, factor: torch.Tensor) -> torch.Tensor:
        """
        factor, y: [T,N]
        return ic[t]，t=0..T-1
        """
        y = self._y
        mask = self._mask

        # 对每个t做截面 rankIC( spearman )：等价于 rank 后 pearson
        # 简化：rank -> demean -> corr
        def cs_rank(x):
            s = torch.argsort(x, dim=1)
            r = torch.argsort(s, dim=1).float()
            n = x.size(1)
            return r / (n - 1 + 1e-12)

        fx = factor.clone()
        fy = y.clone()

        # mask掉无效
        fx[~mask] = float("nan")
        fy[~mask] = float("nan")

        # 对每个t做rank（nan会影响argsort；这里用nan_to_num置极小，随后再用mask约束）
        fx2 = torch.nan_to_num(fx, nan=-1e9)
        fy2 = torch.nan_to_num(fy, nan=-1e9)

        rx = cs_rank(fx2)
        ry = cs_rank(fy2)

        # 使用mask过滤不足样本的截面
        valid_n = mask.sum(dim=1).float()
        ok = valid_n >= self.cfg.ic_min_obs

        # demean
        rx = rx - rx.mean(dim=1, keepdim=True)
        ry = ry - ry.mean(dim=1, keepdim=True)

        num = (rx * ry).sum(dim=1)
        den = torch.sqrt((rx * rx).sum(dim=1) * (ry * ry).sum(dim=1) + 1e-12)
        ic = num / (den + 1e-12)

        ic[~ok] = float("nan")
        return ic

    @torch.no_grad()
    def fitness(self, tree: GPTree) -> float:
        factor = tree.eval(self._X)  # [T,N]
        ic = self._rankic_series(factor)  # [T]
        ic_np = ic.detach().cpu().numpy()
        ic_np = ic_np[np.isfinite(ic_np)]
        if ic_np.size < 10:
            return -1e9

        ic_mean = float(np.mean(ic_np))
        ic_std = float(np.std(ic_np) + 1e-12)
        icir = ic_mean / ic_std

        # 换手：按 topK 选股集合变化率（越小越好）
        turnover = self._turnover_topk(factor, topk=self.cfg.topk)

        # 复杂度惩罚
        complexity = tree.size()

        metric = icir if self.cfg.fitness_metric == "icir" else ic_mean
        score = metric - self.cfg.turnover_penalty * turnover - self.cfg.complexity_penalty * complexity
        return float(score)

    @torch.no_grad()
    def _turnover_topk(self, factor: torch.Tensor, topk: int) -> float:
        # 每期 topk 选股，计算集合变化率
        # factor: [T,N]
        T, N = factor.shape
        # 排名高的买入：取 topk
        idx = torch.argsort(factor, dim=1, descending=True)[:, :topk]  # [T,topk]
        idx_np = idx.detach().cpu().numpy()

        changes = []
        prev = None
        for t in range(T):
            cur = set(idx_np[t].tolist())
            if prev is not None:
                inter = len(cur & prev)
                changes.append(1.0 - inter / topk)
            prev = cur
        if not changes:
            return 1.0
        return float(np.mean(changes))

    @torch.no_grad()
    def full_report(self, tree: GPTree) -> Dict:
        factor = tree.eval(self._X)
        ic = self._rankic_series(factor).detach().cpu().numpy()
        ic = ic[np.isfinite(ic)]
        ic_mean = float(np.mean(ic)) if ic.size else float("nan")
        ic_std = float(np.std(ic) + 1e-12) if ic.size else float("nan")
        icir = ic_mean / ic_std if np.isfinite(ic_mean) else float("nan")

        turnover = self._turnover_topk(factor, topk=self.cfg.topk)
        bt = self._simple_ls_backtest(factor)

        return {
            "expr": tree.to_string(),
            "size": tree.size(),
            "ic_mean": ic_mean,
            "icir": icir,
            "turnover_topk": turnover,
            **bt
        }

    @torch.no_grad()
    def _simple_ls_backtest(self, factor: torch.Tensor) -> Dict:
        """
        简单双边：每个调仓日做 topK - bottomK 的等权组合，收益用 forward_ret 近似（简化）。
        """
        y = self._y  # fwd_ret_Nd
        dates = self.dates.to_pydatetime()

        # 只在 rebalance_freq 采样调仓日
        dser = pd.Series(range(len(dates)), index=pd.to_datetime(dates))
        rebal_idx = dser.resample(self.cfg.rebalance_freq).last().dropna().astype(int).values
        rebal_idx = [int(i) for i in rebal_idx if i < factor.shape[0]]

        topk = self.cfg.topk
        pnl = []

        for t in rebal_idx:
            ft = factor[t]
            yt = y[t]
            # 过滤nan label
            m = torch.isfinite(yt)
            if m.sum().item() < max(self.cfg.ic_min_obs, topk * 2):
                continue

            ft2 = ft.clone()
            ft2[~m] = -1e9
            rank = torch.argsort(ft2, descending=True)

            long_idx = rank[:topk]
            short_idx = rank[-topk:] if self.cfg.long_short else None

            r_long = yt[long_idx].mean().item()
            if self.cfg.long_short:
                r_short = yt[short_idx].mean().item()
                r = r_long - r_short
            else:
                r = r_long
            pnl.append(r)

        if len(pnl) < 5:
            return {"bt_mean": float("nan"), "bt_sharpe": float("nan"), "bt_obs": len(pnl)}

        pnl = np.array(pnl, dtype=float)
        mean = float(pnl.mean())
        std = float(pnl.std() + 1e-12)
        sharpe = mean / std * math.sqrt(52)  # 周频近似年化
        return {"bt_mean": mean, "bt_sharpe": float(sharpe), "bt_obs": len(pnl)}
