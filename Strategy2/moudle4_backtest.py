import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Analyzer:
    def __init__(self, nav_df: pd.DataFrame, trades_df: pd.DataFrame):
        self.nav_df = nav_df.sort_values("date").reset_index(drop=True)
        self.trades_df = trades_df.sort_values(["entry_date", "code"]).reset_index(drop=True)

    # ==============================================================
    # 1. 计算核心指标
    # ==============================================================

    def compute_performance(self):
        df = self.nav_df.copy()

        # 日收益序列
        df["ret"] = df["nav"].pct_change().fillna(0)

        # ------------------ 年化收益 ------------------
        total_return = df["nav"].iloc[-1] / df["nav"].iloc[0] - 1
        num_days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
        annualized_return = (1 + total_return) ** (365 / num_days) - 1

        # ------------------ 年化波动 ------------------
        annualized_vol = df["ret"].std() * np.sqrt(252)

        # ------------------ 夏普比率 ------------------
        sharpe = annualized_return / (annualized_vol + 1e-9)

        # ------------------ 最大回撤 ------------------
        df["cummax"] = df["nav"].cummax()
        df["drawdown"] = df["nav"] / df["cummax"] - 1
        max_drawdown = df["drawdown"].min()

        metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_vol": annualized_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "num_days": num_days,
        }
        return metrics

    # ==============================================================
    # 2. 单笔交易分析指标
    # ==============================================================

    def compute_trade_stats(self):
        df = self.trades_df.copy()
        if df.empty:
            return {}

        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] <= 0]

        win_rate = len(wins) / len(df)
        avg_win = wins["ret"].mean() if len(wins) > 0 else 0
        avg_loss = losses["ret"].mean() if len(losses) > 0 else 0

        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan

        avg_holding_days = (pd.to_datetime(df["exit_date"]) - pd.to_datetime(df["entry_date"])).dt.days.mean()

        return {
            "num_trades": len(df),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_loss_ratio": profit_loss_ratio,
            "avg_holding_days": avg_holding_days,
        }

    # ==============================================================
    # 3. 可视化：净值曲线
    # ==============================================================

    def plot_nav(self):
        df = self.nav_df

        plt.figure(figsize=(12, 5))
        plt.plot(df["date"], df["nav"], label="Strategy NAV", lw=2)
        plt.title("Strategy Net Asset Value")
        plt.xlabel("Date")
        plt.ylabel("NAV")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ==============================================================
    # 4. 可视化：回撤曲线
    # ==============================================================

    def plot_drawdown(self):
        df = self.nav_df.copy()
        df["cummax"] = df["nav"].cummax()
        df["drawdown"] = df["nav"] / df["cummax"] - 1

        plt.figure(figsize=(12, 3))
        plt.plot(df["date"], df["drawdown"], color="red", lw=1.5)
        plt.title("Drawdown Curve")
        plt.ylabel("Drawdown")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ==============================================================
    # 5. 可视化：单笔收益分布（Histogram）
    # ==============================================================

    def plot_trade_distribution(self):
        df = self.trades_df.copy()
        if df.empty:
            print("No trades available.")
            return

        plt.figure(figsize=(8, 4))
        plt.hist(df["ret"], bins=40, alpha=0.7, color="steelblue")
        plt.title("Distribution of Trade Returns")
        plt.xlabel("Return per Trade")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ==============================================================
    # 6. 可视化：未来 15 日最高收益分布
    # ==============================================================

    def plot_future_window_distribution(self):
        df = self.trades_df.copy()
        if "future_max_ret_15d" not in df.columns:
            print("future_max_ret_15d not found. Please run evaluate_future_window().")
            return

        plt.figure(figsize=(8, 4))
        plt.hist(df["future_max_ret_15d"].dropna(), bins=40, alpha=0.7, color="green")
        plt.title("Future 15-Day Max Return Distribution")
        plt.xlabel("Future Max Return (15d)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ==============================================================
    # 7. 集成：输出全部结果
    # ==============================================================

    def run_all(self):
        print("========== PERFORMANCE SUMMARY ==========")
        perf = self.compute_performance()
        for k, v in perf.items():
            print(f"{k}: {v}")

        print("\n========== TRADE STATS ==========")
        trade_stats = self.compute_trade_stats()
        for k, v in trade_stats.items():
            print(f"{k}: {v}")

        print("\n========== PLOTS ==========")
        self.plot_nav()
        self.plot_drawdown()
        self.plot_trade_distribution()
        self.plot_future_window_distribution()