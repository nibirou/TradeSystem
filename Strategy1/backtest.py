# backtest.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from config import (
    INIT_CAPITAL,
    REB_FREQ_DAYS,
    MAX_STOCKS,
    MAX_WEIGHT,
    SLIPPAGE,
    COMMISSION,
    STAMP_DUTY,
    MIN_DAILY_AMOUNT,
    MAX_VOL_RATIO,
)


class BacktestResult:
    def __init__(self, nav_df: pd.DataFrame, trades: pd.DataFrame):
        self.nav_df = nav_df.sort_values("date").reset_index(drop=True)
        self.trades = trades.sort_values(["date", "code"]).reset_index(drop=True)


def _compute_portfolio_value(
    holdings: Dict[str, int],
    cash: float,
    price_map: Dict[str, float],
) -> float:
    value = cash
    for code, shares in holdings.items():
        if code in price_map:
            value += shares * price_map[code]
    return value


def _select_rebalance_dates(dates: List[pd.Timestamp]) -> List[pd.Timestamp]:
    """
    简单按“时间间隔”来确定调仓日（基于日历时间，不是精确的交易日计数），
    更严谨也可以直接按索引编号每隔N个交易日调仓。
    这里采用索引方式，更合理。
    """
    reb_dates = []
    for i, d in enumerate(dates):
        if i % REB_FREQ_DAYS == 0:
            reb_dates.append(d)
    return reb_dates


def backtest_long_only(
    df: pd.DataFrame,
    factor_cols: List[str],
) -> BacktestResult:
    """
    多头仅做多策略回测：
      - df 必须包含：date, code, close, open, volume, amount, score, 以及风控/过滤字段
      - 不使用未来数据：使用 date_t 的 score，在 date_{t+1} 的 open 成交
    """
    data = df.copy()
    data = data.sort_values(["date", "code"]).reset_index(drop=True)

    # 所有交易日
    dates = sorted(data["date"].unique())
    if len(dates) < 2:
        raise ValueError("Not enough trading days in data.")

    # 确定调仓日
    reb_dates = set(_select_rebalance_dates(dates))

    # 状态变量
    cash = INIT_CAPITAL
    holdings: Dict[str, int] = {}  # code -> shares
    nav_records = []
    trade_records = []

    # 主循环（到倒数第二天，因为用 t 的信号在 t+1 交易）
    for i in range(len(dates) - 1):
        date = dates[i]
        next_date = dates[i + 1]

        df_today = data[data["date"] == date]
        df_next = data[data["date"] == next_date]

        # 当前价格 map（用今日收盘价估值）
        price_today = dict(
            zip(df_today["code"], df_today["close"].astype(float))
        )

        # 组合市值（按今日收盘）
        portfolio_value = _compute_portfolio_value(
            holdings, cash, price_today
        )

        nav_records.append({"date": date, "nav": portfolio_value})

        # 若今天不是调仓日，跳到下一天
        if date not in reb_dates:
            continue

        # ========== 选股与目标权重（基于今日因子与过滤） ==========

        # 选股过滤：成交额/停牌/ST 已经在前面预处理，这里再加一层流动性过滤
        df_sel = df_today.copy()
        df_sel["amount"] = pd.to_numeric(df_sel["amount"], errors="coerce")
        df_sel = df_sel[df_sel["amount"] >= MIN_DAILY_AMOUNT]

        # 排除score缺失
        df_sel = df_sel.dropna(subset=["score"])

        # 按score降序，取前 MAX_STOCKS
        df_sel = df_sel.sort_values("score", ascending=False).head(MAX_STOCKS)

        if df_sel.empty:
            # 没有标的可选，直接下一个交易日
            continue

        # 等权 + 单票权重约束
        n = len(df_sel)
        base_w = 1.0 / n
        target_weights = {
            code: min(base_w, MAX_WEIGHT) for code in df_sel["code"]
        }
        # 归一化权重
        w_sum = sum(target_weights.values())
        target_weights = {k: v / w_sum for k, v in target_weights.items()}

        # 准备下一日的 open, volume 信息（用于模拟成交）
        open_next = dict(
            zip(df_next["code"], df_next["open"].astype(float))
        )
        volume_next = dict(
            zip(df_next["code"], df_next["volume"].astype(float))
        )

        # 再次用今日收盘市值作为基准总资金（理论总资产）
        portfolio_value = _compute_portfolio_value(
            holdings, cash, price_today
        )

        # 目标股数（按明日开盘价估算）
        target_shares: Dict[str, int] = {}
        for code, w in target_weights.items():
            if code not in open_next:
                continue
            est_price = open_next[code] * (1 + SLIPPAGE)  # 买入预估价
            tgt_value = portfolio_value * w
            shares = int(tgt_value // est_price)
            if shares <= 0:
                continue

            # 流动性约束：不能超过当日成交量的一定比例
            max_shares = int(volume_next.get(code, 0) * MAX_VOL_RATIO)
            shares = min(shares, max_shares)
            if shares <= 0:
                continue
            target_shares[code] = shares

        # ========== 在下一交易日开盘进行调仓：先卖后买 ==========

        # 1) 卖出所有不在目标中的持仓
        for code, cur_shares in list(holdings.items()):
            if code not in target_shares:
                if code not in open_next:
                    continue
                sell_price = open_next[code] * (1 - SLIPPAGE)
                value = cur_shares * sell_price
                # 成本 = 佣金 + 印花税（卖出才收）
                cost = value * (COMMISSION + STAMP_DUTY)
                cash += (value - cost)

                trade_records.append(
                    {
                        "date": next_date,
                        "code": code,
                        "side": "SELL",
                        "shares": cur_shares,
                        "price": sell_price,
                        "value": value,
                        "cost": cost,
                    }
                )
                del holdings[code]

        # 2) 对在目标中的股票进行调整（加/减仓）
        for code, tgt in target_shares.items():
            cur = holdings.get(code, 0)
            diff = tgt - cur
            if diff == 0:
                continue

            if diff > 0:
                # 买入 diff 股
                if code not in open_next:
                    continue
                buy_price = open_next[code] * (1 + SLIPPAGE)
                value = diff * buy_price
                cost = value * COMMISSION
                total_cost = value + cost
                if total_cost <= cash:
                    cash -= total_cost
                    holdings[code] = holdings.get(code, 0) + diff

                    trade_records.append(
                        {
                            "date": next_date,
                            "code": code,
                            "side": "BUY",
                            "shares": diff,
                            "price": buy_price,
                            "value": value,
                            "cost": cost,
                        }
                    )
                else:
                    # 资金不足则买不到这么多，可以根据需要部分买入，这里简单跳过
                    continue
            else:
                # 卖出 -diff 股
                if code not in open_next:
                    continue
                sell_shares = -diff
                sell_price = open_next[code] * (1 - SLIPPAGE)
                value = sell_shares * sell_price
                cost = value * (COMMISSION + STAMP_DUTY)
                cash += (value - cost)
                holdings[code] = holdings.get(code, 0) - sell_shares
                if holdings[code] <= 0:
                    del holdings[code]

                trade_records.append(
                    {
                        "date": next_date,
                        "code": code,
                        "side": "SELL",
                        "shares": sell_shares,
                        "price": sell_price,
                        "value": value,
                        "cost": cost,
                    }
                )

    # 最后一天组合估值
    last_date = dates[-1]
    df_last = data[data["date"] == last_date]
    price_last = dict(
        zip(df_last["code"], df_last["close"].astype(float))
    )
    final_value = _compute_portfolio_value(holdings, cash, price_last)
    nav_records.append({"date": last_date, "nav": final_value})

    nav_df = pd.DataFrame(nav_records).sort_values("date").reset_index(drop=True)
    trades_df = pd.DataFrame(trade_records)

    return BacktestResult(nav_df=nav_df, trades=trades_df)
