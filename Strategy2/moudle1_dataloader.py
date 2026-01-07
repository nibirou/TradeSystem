import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict
import pandas as pd
import numpy as np


# ===============================
# 1. 数据配置（根据你的目录结构）
# ===============================
@dataclass
class DataConfig:
    base_dir: str = "data_baostock"
    hist_dir: str = "stock_hist"
    meta_dir: str = "metadata"

    # 股票池列表文件存放路径
    hs300_list: str = "stock_list_hs300.csv"
    zz500_list: str = "stock_list_zz500.csv"
    all_list: str = "stock_list_all.csv"

    # 子目录命名（按你的代码）
    daily_freq: str = "d"
    minute5_freq: str = "5"

    # 交易日历路径
    trade_calendar_dir: str = base_dir + "/metadata/trade_datas.csv"
    all_stock_list_dir: str = base_dir + "/metadata/stock_list_all.csv"

    # 字段标准化
    date_col: str = "date"
    time_col: str = "time"
    code_col: str = "code"

    # 统一输出字段
    unified_datetime_col: str = "datetime"
    unified_date_col: str = "date"



# ===============================
# 2. 用户自定义时间段
# ===============================
@dataclass
class PeriodConfig:
    factor_start: str = "2019-01-01"
    factor_end: str = "2024-12-31"

    backtest_start: str = "2020-01-01"
    backtest_end: str = "2024-12-31"

    factor_buffer_n: int = 30

    def validate(self):
        fs = pd.to_datetime(self.factor_start)
        fe = pd.to_datetime(self.factor_end)
        bs = pd.to_datetime(self.backtest_start)
        be = pd.to_datetime(self.backtest_end)

#         if not (fs < bs < be <= fe):
#             raise ValueError(
#                 f"""
# ❌ 回测区间必须完全包含在因子区间内部。

# 因子区间:     {fs.date()} ~ {fe.date()}
# 回测区间:     {bs.date()} ~ {be.date()}

# 正确关系必须满足:
# factor_start < backtest_start < backtest_end <= factor_end
# """
#             )

def get_all_trade_dates_from_csv(csv_path: str) -> pd.DatetimeIndex:
    """
    从 CSV 交易日历中读取所有交易日，自动兼容：
    1) 有表头：calendar_date,is_trading_day
    2) 无表头：2025-11-10,1
    """
    # 先正常按“有表头”尝试
    df = pd.read_csv(csv_path)

    # ---------- 情况 1：有标准表头 ----------
    if set(["calendar_date", "is_trading_day"]).issubset(df.columns):
        pass

    # ---------- 情况 2：无表头 ----------
    else:
        df = pd.read_csv(
            csv_path,
            header=None,
            names=["calendar_date", "is_trading_day"]
        )

    # 类型清洗
    df["calendar_date"] = pd.to_datetime(
        df["calendar_date"],
        errors="coerce"
    )
    df["is_trading_day"] = pd.to_numeric(
        df["is_trading_day"],
        errors="coerce"
    ).fillna(0).astype(int)

    # 丢掉非法行（比如表头被误读的情况）
    df = df.dropna(subset=["calendar_date"])

    trade_days = (
        df[df["is_trading_day"] == 1]
        .sort_values("calendar_date")["calendar_date"]
        .unique()
    )

    trade_dates = pd.DatetimeIndex(trade_days)

    if trade_dates.empty:
        raise ValueError(f"交易日历为空，请检查文件: {csv_path}")

    return trade_dates


# ===============================
# 3. 加载股票池
# ===============================
def load_stock_pool(cfg: DataConfig, pool: str) -> pd.DataFrame:
    meta_dir = Path(cfg.base_dir) / cfg.meta_dir
    file_map = {
        "hs300": cfg.hs300_list,
        "zz500": cfg.zz500_list,
        "all": cfg.all_list
    }

    file_path = meta_dir / file_map[pool]
    if not file_path.exists():
        raise FileNotFoundError(f"股票池文件不存在: {file_path}")

    df = pd.read_csv(file_path)
    df["code"] = df["code"].astype(str)
    return df


# ===============================
# 4. 加载日频数据（支持时间裁剪）
# ===============================
def load_daily_data(
    cfg: DataConfig,
    pool: str,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None
) -> pd.DataFrame:
    """
    加载 data_baostock/stock_hist/{pool}/d/*.csv
    支持按时间窗口裁剪（start_date ~ end_date）
    """
    root = Path(cfg.base_dir) / cfg.hist_dir / pool / cfg.daily_freq
    if not root.exists():
        raise FileNotFoundError(f"日频目录不存在: {root}")

    all_files = sorted(root.glob("*.csv"))
    dfs = []

    # 预处理时间
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
    if end_date is not None:
        end_date = pd.to_datetime(end_date)

    for f in all_files:
        df = pd.read_csv(f)
        if df.empty:
            continue

        # 基础清洗
        df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
        df[cfg.code_col] = df[cfg.code_col].astype(str)

        # ===============================
        # ★ 时间裁剪（关键优化点）
        # ===============================
        if start_date is not None:
            df = df[df[cfg.date_col] >= start_date]
        if end_date is not None:
            df = df[df[cfg.date_col] <= end_date]

        if df.empty:
            continue

        dfs.append(df)

    if len(dfs) == 0:
        raise RuntimeError(f"未加载到任何日频数据（时间区间内无数据）: {root}")

    daily = pd.concat(dfs, ignore_index=True)

    # 排序、去重（事实表）
    daily = daily.sort_values([cfg.code_col, cfg.date_col])
    daily = daily.drop_duplicates(subset=[cfg.code_col, cfg.date_col], keep="last")
    daily = daily.reset_index(drop=True)

    return daily


# ===============================
# 5. 加载 5 分钟频率数据（支持时间裁剪）
# ===============================
# ===============================
# 5. 加载 5 分钟频率数据（支持时间裁剪，修正 time 解析）
# ===============================
def load_5min_data(
    cfg: DataConfig,
    pool: str,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None
) -> pd.DataFrame:

    root = Path(cfg.base_dir) / cfg.hist_dir / pool / cfg.minute5_freq
    if not root.exists():
        raise FileNotFoundError(f"5分钟目录不存在: {root}")

    all_files = sorted(root.glob("*.csv"))
    dfs = []

    # 预处理时间
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
    if end_date is not None:
        end_date = pd.to_datetime(end_date)

    for f in all_files:
        df = pd.read_csv(f)
        if df.empty:
            continue

        # 基础字段
        df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
        df[cfg.time_col] = df[cfg.time_col].astype(str)
        df[cfg.code_col] = df[cfg.code_col].astype(str)

        # ===============================
        # ★ 先按 date 做“粗裁剪”（性能关键）
        # ===============================
        if start_date is not None:
            df = df[df[cfg.date_col] >= start_date]
        if end_date is not None:
            df = df[df[cfg.date_col] <= end_date]

        if df.empty:
            continue

        # ===============================
        # ★ 正确解析 Baostock 5min time
        # time 格式: YYYYMMDDHHMMSSmmm
        # ===============================
        df[cfg.unified_datetime_col] = pd.to_datetime(
            df[cfg.time_col].str.slice(0, 14),   # 取 YYYYMMDDHHMMSS
            format="%Y%m%d%H%M%S",
            errors="coerce"
        )

        df = df.dropna(subset=[cfg.unified_datetime_col])

        if df.empty:
            continue

        dfs.append(df)

    if len(dfs) == 0:
        raise RuntimeError(f"未加载到任何 5 分钟数据（时间区间内无数据）: {root}")

    minute = pd.concat(dfs, ignore_index=True)

    # 排序 & 事实表整理
    minute = minute.sort_values([cfg.code_col, cfg.unified_datetime_col])
    minute = minute.reset_index(drop=True)

    return minute




# ===============================
# 6. 根据时间段切片
# ===============================
def slice_period_daily(daily: pd.DataFrame, period: PeriodConfig, cfg: DataConfig):
    mask = (daily[cfg.date_col] >= period.factor_start) & (daily[cfg.date_col] <= period.factor_end)
    return daily.loc[mask].copy()


def slice_period_5min(minute: pd.DataFrame, period: PeriodConfig, cfg: DataConfig):
    mask = (minute[cfg.unified_datetime_col].dt.date >= pd.to_datetime(period.factor_start).date()) & \
           (minute[cfg.unified_datetime_col].dt.date <= pd.to_datetime(period.factor_end).date())
    return minute.loc[mask].copy()



# ===============================
# 7. 获取回测交易日序列
# ===============================
def get_backtest_trade_dates(daily: pd.DataFrame, period: PeriodConfig, cfg: DataConfig):
    df = daily[
        (daily[cfg.date_col] >= period.backtest_start) &
        (daily[cfg.date_col] <= period.backtest_end) &
        (daily["tradestatus"] == 1)
    ]
    trade_dates = df[cfg.date_col].drop_duplicates().sort_values()
    return pd.DatetimeIndex(trade_dates)



# ===============================
# 8. 统一入口
# ===============================
def load_data_bundle(cfg: DataConfig, period: PeriodConfig, pools=("hs300", "zz500")):
    # period.validate()

    daily_list = []
    minute_list = []

    for p in pools:
        print(f">>> 加载股票池: {p}")
        stock_pool = load_stock_pool(cfg, p)
        print(f"股票池数量：{len(stock_pool)}")

        print(f">>> 加载 {p} 日频数据")
        daily_p = load_daily_data(cfg, p)
        daily_p["source_pool"] = p
        daily_list.append(daily_p)

        print(f">>> 加载 {p} 5 分钟数据")
        minute_p = load_5min_data(cfg, p)
        minute_p["source_pool"] = p
        minute_list.append(minute_p)

    daily_all = pd.concat(daily_list, ignore_index=True)
    minute_all = pd.concat(minute_list, ignore_index=True)

    # 清洗去重
    # ===============================
    # 日频事实表清洗（核心）
    # ===============================
    before = len(daily_all)

    daily_all = (
        daily_all
        .sort_values(["code", "date"])
        .drop_duplicates(subset=["code", "date"], keep="last")
        .reset_index(drop=True)
    )
    after = len(daily_all)
    print(f">>> 日频去重完成: {before} -> {after} (removed {before - after})")

    # ===============================
    # 5 分钟事实表清洗
    # ===============================
    before = len(minute_all)

    minute_all = (
        minute_all
        .sort_values(["code", "datetime"])
        .drop_duplicates(subset=["code", "datetime"], keep="last")
        .reset_index(drop=True)
    )
    after = len(minute_all)
    print(f">>> 5min 去重完成: {before} -> {after} (removed {before - after})")

    print(">>> 时间切片（因子区间）")
    daily_slice = slice_period_daily(daily_all, period, cfg)
    minute_slice = slice_period_5min(minute_all, period, cfg)

    print(">>> 获取回测交易日序列")
    trade_dates = get_backtest_trade_dates(daily_slice, period, cfg)

    return {
        "daily": daily_slice,
        "minute5": minute_slice,
        "trade_dates": trade_dates,
        "cfg": cfg,
        "period": period
    }





def shift_trade_date(trade_dates: pd.DatetimeIndex,
                     anchor: pd.Timestamp,
                     n_back: int) -> pd.Timestamp:
    """
    从 anchor 向前回退 n_back 个交易日
    """
    anchor = pd.to_datetime(anchor)
    idx = trade_dates.get_loc(anchor)
    start_idx = max(0, idx - n_back)
    return trade_dates[start_idx]

def compute_load_window(period: PeriodConfig,
                        trade_dates: pd.DatetimeIndex,
                        buffer_n: int):
    """
    返回 (load_start, load_end)
    """
    factor_start = pd.to_datetime(period.factor_start)
    factor_end = pd.to_datetime(period.factor_end)

    load_start = shift_trade_date(trade_dates, factor_start, buffer_n)
    load_end = factor_end

    return load_start, load_end


def load_data_bundle_update(cfg: DataConfig,
                     period: PeriodConfig,
                     pools=("hs300", "zz500")):
    daily_list = []
    minute_list = []

    # 准备完整交易日序列（用于 buffer 回退）
    full_trade_dates = get_all_trade_dates_from_csv(cfg.trade_calendar_dir)

    # ===============================
    # 1. 计算加载窗口（buffer）
    # ===============================
    buffer_n = period.factor_buffer_n
    load_start, load_end = compute_load_window(
        period,
        full_trade_dates,
        buffer_n
    )

    print(f">>> 因子区间: {period.factor_start} ~ {period.factor_end}")
    print(f">>> 实际加载数据区间: {load_start.date()} ~ {load_end.date()}")

    for p in pools:
        print(f">>> 加载股票池: {p}")
        stock_pool = load_stock_pool(cfg, p)
        print(f"股票池数量：{len(stock_pool)}")

        print(f">>> 加载 {p} 日频数据（裁剪后）")
        daily_p = load_daily_data(
            cfg, p,
            start_date=load_start,
            end_date=load_end
        )
        daily_p["source_pool"] = p
        daily_list.append(daily_p)

        print(f">>> 加载 {p} 5 分钟数据（裁剪后）")
        minute_p = load_5min_data(
            cfg, p,
            start_date=load_start,
            end_date=load_end
        )
        minute_p["source_pool"] = p
        minute_list.append(minute_p)
    
    # ===============================
    # 2. 合并
    # ===============================
    daily_all = pd.concat(daily_list, ignore_index=True)
    minute_all = pd.concat(minute_list, ignore_index=True)

    # ===============================
    # 3. 日频事实表清洗
    # ===============================
    before = len(daily_all)
    daily_all = (
        daily_all
        .sort_values(["code", "date"])
        .drop_duplicates(subset=["code", "date"], keep="last")
        .reset_index(drop=True)
    )
    after = len(daily_all)
    print(f">>> 日频去重完成: {before} -> {after} (removed {before - after})")

    # ===============================
    # 4. 5 分钟事实表清洗
    # ===============================
    before = len(minute_all)
    minute_all = (
        minute_all
        .sort_values(["code", "datetime"])
        .drop_duplicates(subset=["code", "datetime"], keep="last")
        .reset_index(drop=True)
    )
    after = len(minute_all)
    print(f">>> 5min 去重完成: {before} -> {after} (removed {before - after})")


    print(">>> 获取回测交易日序列")
    trade_dates = trade_dates = full_trade_dates[
                    (full_trade_dates >= period.backtest_start) &
                    (full_trade_dates <= period.backtest_end)
                ]

    return {
        "daily": daily_all,
        "minute5": minute_all,
        "trade_dates": trade_dates,
        "cfg": cfg,
        "period": period,
        "load_start": load_start,
        "load_end": load_end
    }