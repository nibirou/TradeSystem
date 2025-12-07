# reporting.py
import numpy as np
import pandas as pd


def compute_performance_stats(nav_df: pd.DataFrame, trading_days_per_year: int = 252) -> dict:
    """
    基于净值时间序列计算年度化收益、波动率、夏普、最大回撤等。
    nav_df: [date, nav]
    """
    nav_df = nav_df.sort_values("date").reset_index(drop=True)
    nav = nav_df["nav"].values
    ret = nav[1:] / nav[:-1] - 1.0
    if len(ret) == 0:
        return {}

    # 年化收益
    total_return = nav[-1] / nav[0] - 1.0
    years = len(ret) / trading_days_per_year
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan

    # 年化波动率
    vol_daily = np.std(ret, ddof=1)
    annual_vol = vol_daily * np.sqrt(trading_days_per_year)

    # 夏普比率（无风险利率近似0）
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan

    # 最大回撤
    cum_max = np.maximum.accumulate(nav)
    dd = nav / cum_max - 1.0
    max_drawdown = dd.min()

    stats = {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }
    return stats
