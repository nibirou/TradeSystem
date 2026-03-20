# quant_preprocess/labels.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


# -----------------------------
# 配置：卖出价统计的可选项
# -----------------------------
PriceSource = Literal["daily", "minute"]

# 日频卖出价规则（基于未来窗口内的“日收盘价序列”）
DailySellMethod = Literal["last_close", "max_close", "topk_mean_close"]

# 分钟卖出价规则（基于未来窗口内所有分钟bar的 close）
MinuteSellMethod = Literal[
    "last_day_intraday_max",      # 窗口最后一天(t+H) 日内最高 close
    "window_intraday_max",        # 整个窗口(t+1..t+H) 所有分钟bar 最高 close
    "window_intraday_topk_mean",  # 整个窗口(t+1..t+H) 所有分钟bar close TopK均值
]


@dataclass
class LabelConfig:
    # 未来窗口长度（交易日步数）：日/周/月/季度本质只是 H 不同
    horizon_n: int = 20

    # 卖出价来源与统计规则
    price_source: PriceSource = "daily"
    daily_sell_method: DailySellMethod = "last_close"
    minute_sell_method: MinuteSellMethod = "window_intraday_max"

    # TopK（用于 topk_mean）
    topk: int = 5

    # 波动率计算方式：默认用“未来窗口内日收益率std”
    # 未来扩展分钟级波动率：可加 "minute_return"
    vol_method: Literal["daily_return"] = "daily_return"

    # 年化系数：日收益率默认 sqrt(252)
    vol_annualize: float = math.sqrt(252)

    # close<=0 / inf 的清洗
    invalid_price_to_nan: bool = True


class LabelEngine:
    """
    输出“日频级别”的标签面板：
      [date, code, buy_px, sell_px, ret, vol, meta...]
    注意：
      - 你传入的 label_buffer_dates 已经包含向后 buffer（<= period.load_end）
      - 因此本模块不再丢弃末尾H天；不完整窗口会自然产生 NaN，同时给 is_complete 标记
    """

    def __init__(self, label_buffer_dates: pd.DatetimeIndex):
        self.label_buffer_dates = pd.DatetimeIndex(pd.to_datetime(label_buffer_dates)).sort_values()

    # -----------------------------
    # 工具：将 label_buffer_dates 映射到未来窗口日期
    # -----------------------------
    def _future_window_dates(self, t: pd.Timestamp, H: int) -> pd.DatetimeIndex:
        t = pd.to_datetime(t)
        idx = self.label_buffer_dates.get_loc(t)
        return self.label_buffer_dates[idx + 1: idx + 1 + int(H)]

    # -----------------------------
    # 工具：安全清洗价格列
    # -----------------------------
    @staticmethod
    def _clean_price(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan)
        s = s.where(s > 0, np.nan)
        return s

    # -----------------------------
    # 1) 构造“日频 close 网格”(date x code)，并按 code ffill
    #    先确保 (date,code) 唯一，避免 non-unique multi-index 报错
    # -----------------------------
    def _build_daily_close_grid(
        self,
        daily: pd.DataFrame,
        code_col: str = "code",
        date_col: str = "date",
        close_col: str = "close",
        ffill_limit: Optional[int] = None,
        restrict_dates: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        df = daily[[date_col, code_col, close_col]].copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df[code_col] = df[code_col].astype(str)

        td = self.label_buffer_dates if restrict_dates is None else pd.DatetimeIndex(pd.to_datetime(restrict_dates)).sort_values()
        df = df[df[date_col].isin(td)]

        # ✅关键：确保 (date,code) 唯一
        df = df.sort_values([date_col, code_col]).drop_duplicates([date_col, code_col], keep="last")

        if close_col in df.columns:
            df[close_col] = self._clean_price(df[close_col])

        codes = df[code_col].unique()
        full_index = pd.MultiIndex.from_product([td, codes], names=[date_col, code_col])
        out = df.set_index([date_col, code_col]).reindex(full_index).reset_index()

        out[close_col] = out.groupby(code_col)[close_col].ffill(limit=ffill_limit)
        return out

    # -----------------------------
    # 2) 分钟数据 -> 日级摘要：
    #    - day_last_close: 当日最后一根bar close
    #    - day_max_close:  当日分钟close最大值
    #    - day_topk_closes: 当日分钟close topK 列表（np.ndarray）
    #
    # 关键性质：窗口TopK 一定来自各日TopK并集 => 不需要保留全量分钟bar
    # -----------------------------
    def _build_minute_daily_summary(
        self,
        minute: pd.DataFrame,
        cfg: LabelConfig,
        code_col: str = "code",
        date_col: str = "date",
        dt_col: str = "datetime",
        close_col: str = "close",
    ) -> pd.DataFrame:
        m = minute[[code_col, date_col, dt_col, close_col]].copy()
        m[code_col] = m[code_col].astype(str)
        m[date_col] = pd.to_datetime(m[date_col])
        m[dt_col] = pd.to_datetime(m[dt_col], errors="coerce")

        m[close_col] = self._clean_price(m[close_col])
        m = m.dropna(subset=[code_col, date_col, dt_col, close_col])

        if m.empty:
            return pd.DataFrame(columns=[date_col, code_col, "day_last_close", "day_max_close", "day_topk_closes"])

        m = m.sort_values([code_col, date_col, dt_col])

        K = max(1, int(cfg.topk))

        def topk_list(x: pd.Series) -> np.ndarray:
            arr = x.to_numpy()
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return np.array([], dtype=float)
            if arr.size <= K:
                return np.sort(arr)[::-1]
            idx = np.argpartition(arr, -K)[-K:]
            return np.sort(arr[idx])[::-1]

        day_last = (
            m.groupby([date_col, code_col], as_index=False)
             .agg(day_last_close=(close_col, "last"))
        )
        day_max = (
            m.groupby([date_col, code_col], as_index=False)
             .agg(day_max_close=(close_col, "max"))
        )
        day_topk = (
            m.groupby([date_col, code_col])[close_col]
             .apply(topk_list)
             .reset_index(name="day_topk_closes")
        )

        out = (
            day_last.merge(day_max, on=[date_col, code_col], how="outer")
                    .merge(day_topk, on=[date_col, code_col], how="outer")
                    .sort_values([date_col, code_col])
                    .reset_index(drop=True)
        )
        return out

    # -----------------------------
    # 3A) 日频卖出价：基于未来窗口的日收盘价序列
    # -----------------------------
    def _compute_sell_price_daily(
        self,
        daily_close_grid: pd.DataFrame,
        cfg: LabelConfig,
        code_col: str = "code",
        date_col: str = "date",
        close_col: str = "close",
    ) -> pd.Series:
        df = daily_close_grid.sort_values([code_col, date_col]).copy()
        H = int(cfg.horizon_n)

        fut_close = df.groupby(code_col)[close_col].shift(-1)

        if cfg.daily_sell_method == "last_close":
            sell = df.groupby(code_col)[close_col].shift(-H)

        elif cfg.daily_sell_method == "max_close":
            sell = (
                fut_close.groupby(df[code_col])
                         .rolling(H, min_periods=H)
                         .max()
                         .shift(-(H - 1))
                         .reset_index(level=0, drop=True)
            )

        else:  # topk_mean_close
            K = max(1, int(cfg.topk))

            def topk_mean(arr: np.ndarray) -> float:
                arr = arr[np.isfinite(arr)]
                if arr.size < K:
                    return np.nan
                idx = np.argpartition(arr, -K)[-K:]
                return float(np.mean(arr[idx]))

            sell = (
                fut_close.groupby(df[code_col])
                         .rolling(H, min_periods=H)
                         .apply(lambda x: topk_mean(x.to_numpy()), raw=False)
                         .shift(-(H - 1))
                         .reset_index(level=0, drop=True)
            )

        return sell

    # -----------------------------
    # 3B) 分钟卖出价：基于未来窗口内分钟close（通过日摘要实现 max/topk）
    # -----------------------------
    def _compute_sell_price_minute(
        self,
        minute_day_summary_aligned: pd.DataFrame,
        cfg: LabelConfig,
        code_col: str = "code",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """
        输入：已对齐到 (date,code) 网格的分钟日摘要（可能含缺失）
            至少含 [date, code, day_max_close, day_topk_closes]
        输出：增加 sell_px_minute 列

        说明：window_intraday_topk_mean 不能用 pandas rolling，因为是 object dtype（ndarray）
            必须按 code 循环计算窗口 topK。
        """
        df = minute_day_summary_aligned.copy()
        if df.empty:
            df["sell_px_minute"] = np.nan
            return df

        df[date_col] = pd.to_datetime(df[date_col])
        df[code_col] = df[code_col].astype(str)
        df = df.sort_values([code_col, date_col]).reset_index(drop=True)

        H = int(cfg.horizon_n)
        K = max(1, int(cfg.topk))

        g = df.groupby(code_col, group_keys=False)

        # 规则1：窗口最后一天(t+H)日内最高 close：等价于 day_max_close.shift(-H)
        if cfg.minute_sell_method == "last_day_intraday_max":
            df["sell_px_minute"] = g["day_max_close"].shift(-H)
            return df

        # 规则2：窗口内所有分钟bar最高 close：等价于 max(day_max_close over t+1..t+H)
        if cfg.minute_sell_method == "window_intraday_max":
            fut_day_max = g["day_max_close"].shift(-1)
            df["sell_px_minute"] = (
                fut_day_max.groupby(df[code_col])
                        .rolling(H, min_periods=H)
                        .max()
                        .shift(-(H - 1))
                        .reset_index(level=0, drop=True)
            )
            return df

        # 规则3：窗口内所有分钟bar close 的 TopK 均值（你的定义）
        # ⚠️不能 rolling(object)，改为按 code 循环（窗口最多 H*K 个数）
        sell_vals = np.full(len(df), np.nan, dtype=float)

        for code, sub_idx in g.indices.items():
            idxs = np.asarray(sub_idx, dtype=int)  # 该code在df中的行号（已按date排序）
            arrs = df.loc[idxs, "day_topk_closes"].to_list()

            # 将每个元素规范为 np.ndarray（可能是 NaN/None/float）
            norm = []
            for a in arrs:
                if isinstance(a, np.ndarray):
                    aa = a.astype(float, copy=False)
                elif a is None or (isinstance(a, float) and np.isnan(a)):
                    aa = np.array([], dtype=float)
                else:
                    # 极少数情况下可能被读成 list/tuple
                    try:
                        aa = np.asarray(a, dtype=float)
                    except Exception:
                        aa = np.array([], dtype=float)
                # 清掉非有限数
                if aa.size > 0:
                    aa = aa[np.isfinite(aa)]
                norm.append(aa)

            n = len(norm)

            # 对 sub 的每个位置 i，窗口取 i+1 .. i+H
            # 只有当 i+H < n 才完整
            for j in range(n):
                end = j + H
                if end >= n:
                    # 不完整窗口：保持 NaN（但你 loader 已 buffer，正常不会太多）
                    continue

                # 拼接未来窗口内各日 topK close
                window_arrays = norm[j + 1: end + 1]
                if not window_arrays:
                    continue

                # print(window_arrays)
                # print(len(window_arrays), j, code)

                cat = np.concatenate([a for a in window_arrays if a.size > 0], axis=0)
                if cat.size < K:
                    continue

                # 取全窗口 topK
                top_idx = np.argpartition(cat, -K)[-K:]
                sell_vals[idxs[j]] = float(np.mean(cat[top_idx]))

        df["sell_px_minute"] = sell_vals
        return df

    # -----------------------------
    # 4) 波动率：未来窗口内“日收益率std”
    # -----------------------------
    def _compute_vol_daily_return(
        self,
        daily_close_grid: pd.DataFrame,
        cfg: LabelConfig,
        code_col: str = "code",
        date_col: str = "date",
        close_col: str = "close",
    ) -> pd.Series:
        df = daily_close_grid.sort_values([code_col, date_col]).copy()

        df["log_close"] = np.log(df[close_col])
        df["ret1"] = df.groupby(code_col)["log_close"].diff()
        df["ret1"] = df["ret1"].replace([np.inf, -np.inf], np.nan)

        H = int(cfg.horizon_n)
        fut_ret = df.groupby(code_col)["ret1"].shift(-1)

        vol = (
            fut_ret.groupby(df[code_col])
                   .rolling(H, min_periods=H)
                   .std(ddof=0)
                   .shift(-(H - 1))
                   .reset_index(level=0, drop=True)
                   * float(cfg.vol_annualize)
        )
        return vol

    # -----------------------------
    # 5) 主函数：构造完整标签面板
    # -----------------------------
    def build_labels_panel(
        self,
        daily: pd.DataFrame,
        minute: Optional[pd.DataFrame],
        cfg: LabelConfig,
        code_col: str = "code",
        date_col: str = "date",
        close_col: str = "close",
        dt_col: str = "datetime",
        ffill_limit: Optional[int] = None,
        # 你通常只想输出 factor 区间内的标签（不含向后buffer段），这里留一个参数
        output_dates: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """
        输出：
          date, code, buy_px, sell_px, ret, vol,
          is_complete, price_source, sell_method, horizon_n, topk

        说明：
          - label_buffer_dates 用于构造未来窗口
          - output_dates 用于裁剪最终输出日期（常用：period.factor_start..period.factor_end）
        """
        td = self.label_buffer_dates
        H = int(cfg.horizon_n)

        # ---- 日频close网格（用于 buy_px, vol, 以及 daily sell）----
        daily_grid = self._build_daily_close_grid(
            daily=daily,
            code_col=code_col,
            date_col=date_col,
            close_col=close_col,
            ffill_limit=ffill_limit,
            restrict_dates=td,
        )

        # buy_px：锚点日收盘价（统一语义）
        buy_px = daily_grid[close_col].copy()

        # vol
        if cfg.vol_method == "daily_return":
            vol = self._compute_vol_daily_return(daily_grid, cfg, code_col, date_col, close_col)
        else:
            raise NotImplementedError("vol_method only supports 'daily_return' for now.")

        # sell_px
        if cfg.price_source == "daily":
            sell_px = self._compute_sell_price_daily(daily_grid, cfg, code_col, date_col, close_col)
            sell_method = cfg.daily_sell_method

        else:  # minute
            sell_method = cfg.minute_sell_method

            if minute is None or minute.empty:
                sell_px = pd.Series(np.nan, index=daily_grid.index)
            else:
                minute_sum = self._build_minute_daily_summary(
                    minute=minute,
                    cfg=cfg,
                    code_col=code_col,
                    date_col=date_col,
                    dt_col=dt_col,
                    close_col=close_col,
                )

                # 对齐到 daily_grid 的 date/code 网格（缺失填 NaN）
                minute_sum = minute_sum.sort_values([date_col, code_col]).drop_duplicates([date_col, code_col], keep="last")
                aligned = daily_grid[[date_col, code_col]].merge(minute_sum, on=[date_col, code_col], how="left")

                aligned2 = self._compute_sell_price_minute(aligned, cfg, code_col, date_col)
                sell_px = aligned2["sell_px_minute"]

        ret = sell_px / buy_px - 1

        # is_complete：未来窗口是否完整（在 label_buffer_dates 范围内）
        pos_map = pd.Series(np.arange(len(td)), index=td)
        pos = daily_grid[date_col].map(pos_map).astype("float64")
        is_complete = np.isfinite(pos) & ((pos + H) < len(td))

        out = pd.DataFrame({
            date_col: daily_grid[date_col],
            code_col: daily_grid[code_col],
            "buy_px": buy_px,
            "sell_px": sell_px,
            "ret": ret,
            "vol": vol,
            "is_complete": is_complete.astype(bool),
            "price_source": cfg.price_source,
            "sell_method": sell_method,
            "horizon_n": H,
            "topk": int(cfg.topk),
        })

        # 最终输出日期裁剪：默认不裁剪（返回包含向后buffer段的全部日期）
        if output_dates is not None:
            out_dates = pd.DatetimeIndex(pd.to_datetime(output_dates)).sort_values()
            out = out[out[date_col].isin(out_dates)].reset_index(drop=True)

        return out
