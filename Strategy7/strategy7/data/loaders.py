"""Market data loader and base feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import DataConfig, DateConfig
from ..core.constants import EPS
from ..core.time_utils import compute_load_window
from ..core.types import FeatureBundle, MarketBundle
from ..core.utils import (
    infer_board_type,
    infer_industry_bucket,
    is_main_board_symbol,
    symbol_key_from_filename,
)
from .base import MarketDataLoader
from .frequency import add_generic_micro_structure_features, build_frequency_views


def list_symbol_keys(daily_dir: Path, file_format: str = "auto") -> List[str]:
    keys: set[str] = set()
    if file_format in {"auto", "parquet"}:
        for p in daily_dir.glob("*_d.parquet"):
            k = symbol_key_from_filename(p.name)
            if k:
                keys.add(k)
    if file_format in {"auto", "csv"}:
        for p in daily_dir.glob("*_d.csv"):
            k = symbol_key_from_filename(p.name)
            if k:
                keys.add(k)
    return sorted(keys)


def pick_existing_file(folder: Path, symbol_key: str, freq_tag: str, file_format: str = "auto") -> Optional[Path]:
    if file_format == "parquet":
        cands = [folder / f"{symbol_key}_{freq_tag}.parquet"]
    elif file_format == "csv":
        cands = [folder / f"{symbol_key}_{freq_tag}.csv"]
    else:
        cands = [folder / f"{symbol_key}_{freq_tag}.parquet", folder / f"{symbol_key}_{freq_tag}.csv"]
    for p in cands:
        if p.exists():
            return p
    return None


def read_data_file(path: Path, usecols: List[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    date_start = start_date.strftime("%Y-%m-%d")
    date_end = end_date.strftime("%Y-%m-%d")
    if path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(path, columns=usecols, filters=[("date", ">=", date_start), ("date", "<=", date_end)])
        except Exception:
            csv_path = path.with_suffix(".csv")
            if not csv_path.exists():
                raise
            df = pd.read_csv(csv_path, usecols=lambda c: c in usecols)
    else:
        df = pd.read_csv(path, usecols=lambda c: c in usecols)
    if "date" not in df.columns:
        return pd.DataFrame(columns=usecols)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    return df


def load_hs300_constituent_keys(hs300_list_path: Path) -> List[str]:
    if not hs300_list_path.exists():
        raise FileNotFoundError(f"hs300 constituent file not found: {hs300_list_path}")
    df = pd.read_csv(hs300_list_path)
    if "code" in df.columns:
        codes = df["code"].astype(str)
    else:
        codes = df.iloc[:, 0].astype(str)
    keys = codes.str.strip().str.lower().str.replace(".", "_", regex=False)
    keys = keys[keys.str.match(r"^[a-z]{2}_\d{6}$", na=False)]
    return sorted(keys.drop_duplicates().tolist())


@dataclass
class HS300MarketDataLoader(MarketDataLoader):
    """Loader for hs300 daily+5min files."""

    data_cfg: DataConfig
    date_cfg: DateConfig
    lookback_days: int
    horizon: int

    def _load_market_frames(
        self,
        data_root: Path,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        file_format: str = "auto",
        max_files: Optional[int] = None,
        hs300_list_path: Optional[Path] = None,
        main_board_only: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        daily_dir = data_root / "d"
        minute_dir = data_root / "5"
        if not daily_dir.exists() or not minute_dir.exists():
            raise FileNotFoundError(f"data directories missing: {daily_dir} or {minute_dir}")

        keys_all = list_symbol_keys(daily_dir, file_format=file_format)
        keys = keys_all
        if hs300_list_path is not None:
            hs300_keys = set(load_hs300_constituent_keys(hs300_list_path))
            keys = [k for k in keys if k in hs300_keys]
        if main_board_only:
            keys = [k for k in keys if is_main_board_symbol(k)]
        if max_files is not None:
            keys = keys[: int(max_files)]

        daily_cols = [
            "date",
            "code",
            "open",
            "high",
            "low",
            "close",
            "preclose",
            "volume",
            "amount",
            "turn",
            "tradestatus",
        ]
        minute_cols = ["date", "time", "code", "open", "high", "low", "close", "volume", "amount"]

        daily_frames: List[pd.DataFrame] = []
        minute_frames: List[pd.DataFrame] = []
        for key in keys:
            daily_path = pick_existing_file(daily_dir, key, "d", file_format=file_format)
            minute_path = pick_existing_file(minute_dir, key, "5", file_format=file_format)
            if daily_path is None or minute_path is None:
                continue

            try:
                ddf = read_data_file(daily_path, daily_cols, start_date, end_date)
                mdf = read_data_file(minute_path, minute_cols, start_date, end_date)
            except Exception:
                # Skip broken files instead of failing the whole run.
                continue
            if ddf.empty or mdf.empty:
                continue

            ddf["code"] = ddf["code"].astype(str).str.strip()
            mdf["code"] = mdf["code"].astype(str).str.strip()
            for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "tradestatus"]:
                if col in ddf.columns:
                    ddf[col] = pd.to_numeric(ddf[col], errors="coerce")
            for col in ["time", "open", "high", "low", "close", "volume", "amount"]:
                if col in mdf.columns:
                    mdf[col] = pd.to_numeric(mdf[col], errors="coerce")

            if "tradestatus" in ddf.columns:
                ddf = ddf[ddf["tradestatus"].fillna(1) == 1].copy()

            # parse datetime from baostock numeric time
            if "time" in mdf.columns:
                tstr = mdf["time"].astype("Int64").astype(str).str.zfill(17).str.slice(0, 14)
                mdf["datetime"] = pd.to_datetime(tstr, format="%Y%m%d%H%M%S", errors="coerce")
                mdf["datetime"] = mdf["datetime"].fillna(pd.to_datetime(mdf["date"], errors="coerce"))
            else:
                mdf["datetime"] = pd.to_datetime(mdf["date"], errors="coerce")
            mdf["date"] = pd.to_datetime(mdf["date"], errors="coerce").dt.normalize()
            mdf = mdf.dropna(subset=["date", "datetime", "code"]).copy()

            daily_frames.append(ddf)
            minute_frames.append(mdf)

        if not daily_frames:
            raise RuntimeError("no daily market data loaded.")
        if not minute_frames:
            raise RuntimeError("no minute market data loaded.")

        daily_df = pd.concat(daily_frames, ignore_index=True).sort_values(["code", "date"]).reset_index(drop=True)
        minute_df = pd.concat(minute_frames, ignore_index=True).sort_values(["code", "datetime"]).reset_index(drop=True)
        daily_df = daily_df.drop_duplicates(["code", "date"], keep="last").reset_index(drop=True)
        minute_df = minute_df.drop_duplicates(["code", "datetime"], keep="last").reset_index(drop=True)
        return daily_df, minute_df

    def load(self) -> MarketBundle:
        load_start, load_end = compute_load_window(
            train_start=self.date_cfg.train_start,
            test_end=self.date_cfg.test_end,
            lookback_days=int(self.lookback_days),
            horizon=int(self.horizon),
        )
        daily_df, minute_df = self._load_market_frames(
            data_root=Path(self.data_cfg.data_root),
            start_date=load_start,
            end_date=load_end,
            file_format=self.data_cfg.file_format,
            max_files=self.data_cfg.max_files,
            hs300_list_path=Path(self.data_cfg.hs300_list_path),
            main_board_only=self.data_cfg.main_board_only,
        )
        codes = sorted(daily_df["code"].astype(str).drop_duplicates().tolist())
        notes = {
            "data_root": self.data_cfg.data_root,
            "hs300_list_path": self.data_cfg.hs300_list_path,
            "main_board_only": str(self.data_cfg.main_board_only),
            "file_format": self.data_cfg.file_format,
        }
        return MarketBundle(
            daily=daily_df,
            minute5=minute_df,
            start_date=load_start,
            end_date=load_end,
            codes=codes,
            source_notes=notes,
        )


def build_minute_daily_features(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate minute bars to daily micro-structure and executable prices."""
    m = minute_df.copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.normalize()
    m["datetime"] = pd.to_datetime(m["datetime"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors="coerce")
    m = m.dropna(subset=["date", "datetime", "code", "open", "close", "volume"]).copy()
    m = m.sort_values(["code", "date", "datetime"])
    grp_cols = ["code", "date"]

    m["bar_idx"] = m.groupby(grp_cols).cumcount()
    m["bar_rev_idx"] = m.groupby(grp_cols).cumcount(ascending=False)
    m["ret_5m"] = m.groupby(grp_cols)["close"].pct_change()
    m["pv"] = m["close"] * m["volume"]
    m["signed_vol"] = np.sign(m["ret_5m"].fillna(0.0)) * m["volume"]
    m["abs_ret_5m"] = m["ret_5m"].abs()

    base = m.groupby(grp_cols, as_index=False).agg(
        first_open_5m=("open", "first"),
        first_close_5m=("close", "first"),
        last_close_5m=("close", "last"),
        day_vol=("volume", "sum"),
        day_pv=("pv", "sum"),
        minute_realized_vol_5m=("ret_5m", "std"),
        minute_up_ratio_5m=("ret_5m", lambda x: float((x > 0).mean()) if x.notna().any() else np.nan),
        minute_ret_skew_5m=("ret_5m", lambda x: float(x.skew()) if x.notna().sum() > 2 else np.nan),
        minute_ret_kurt_5m=("ret_5m", lambda x: float(x.kurt()) if x.notna().sum() > 3 else np.nan),
        signed_vol_sum=("signed_vol", "sum"),
        abs_ret_max=("abs_ret_5m", "max"),
    )
    first30 = m[m["bar_idx"] < 6].groupby(grp_cols, as_index=False).agg(pv_first30=("pv", "sum"), vol_first30=("volume", "sum"), close_30m=("close", "last"))
    first60 = m[m["bar_idx"] < 12].groupby(grp_cols, as_index=False).agg(pv_first60=("pv", "sum"), vol_first60=("volume", "sum"))
    last30 = m[m["bar_rev_idx"] < 6].groupby(grp_cols, as_index=False).agg(twap_last30=("close", "mean"), close_last30_start=("close", "first"), close_last30_end=("close", "last"))

    feat = base.merge(first30, on=grp_cols, how="left").merge(first60, on=grp_cols, how="left").merge(last30, on=grp_cols, how="left")
    feat["vwap_day"] = feat["day_pv"] / (feat["day_vol"] + EPS)
    feat["vwap_first30"] = feat["pv_first30"] / (feat["vol_first30"] + EPS)
    feat["vwap_first60"] = feat["pv_first60"] / (feat["vol_first60"] + EPS)
    feat["signed_vol_imbalance_5m"] = feat["signed_vol_sum"] / (feat["day_vol"] + EPS)
    feat["jump_ratio_5m"] = feat["abs_ret_max"] / (feat["minute_realized_vol_5m"] + EPS)
    feat["open_to_close_intraday"] = feat["last_close_5m"] / (feat["first_open_5m"] + EPS) - 1.0
    feat["morning_momentum_30m"] = feat["close_30m"] / (feat["first_open_5m"] + EPS) - 1.0
    feat["last30_momentum"] = feat["close_last30_end"] / (feat["close_last30_start"] + EPS) - 1.0
    feat["vwap30_vs_day"] = feat["vwap_first30"] / (feat["vwap_day"] + EPS) - 1.0

    feat["px_open5"] = feat["first_open_5m"]
    feat["px_vwap30"] = feat["vwap_first30"]
    feat["px_twap_last30"] = feat["twap_last30"]

    keep_cols = [
        "code",
        "date",
        "vwap_day",
        "vwap_first30",
        "vwap_first60",
        "minute_realized_vol_5m",
        "minute_up_ratio_5m",
        "minute_ret_skew_5m",
        "minute_ret_kurt_5m",
        "signed_vol_imbalance_5m",
        "jump_ratio_5m",
        "morning_momentum_30m",
        "last30_momentum",
        "open_to_close_intraday",
        "vwap30_vs_day",
        "px_open5",
        "px_vwap30",
        "px_twap_last30",
    ]
    return feat[keep_cols]


def _rolling_mean(g: pd.core.groupby.generic.SeriesGroupBy, window: int) -> pd.Series:
    return g.transform(lambda s: s.rolling(window, min_periods=window).mean())


def _rolling_std(g: pd.core.groupby.generic.SeriesGroupBy, window: int) -> pd.Series:
    return g.transform(lambda s: s.rolling(window, min_periods=window).std())


def build_daily_feature_base(daily_df: pd.DataFrame, minute_daily_feat: pd.DataFrame) -> pd.DataFrame:
    """Build robust daily base features for factor library and portfolio models."""
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "code", "open", "high", "low", "close", "volume"]).sort_values(["code", "date"])
    if "preclose" not in df.columns:
        df["preclose"] = np.nan
    df["preclose"] = df["preclose"].where(df["preclose"].notna(), df.groupby("code")["close"].shift(1))
    if "turn" not in df.columns:
        df["turn"] = np.nan

    df = df.merge(minute_daily_feat, on=["code", "date"], how="left")
    g = df.groupby("code")
    df["ret_1d"] = g["close"].pct_change(1)
    df["ret_3d"] = g["close"].pct_change(3)
    df["ret_5d"] = g["close"].pct_change(5)
    df["ret_10d"] = g["close"].pct_change(10)
    df["ret_20d"] = g["close"].pct_change(20)
    df["vol_chg_1d"] = g["volume"].pct_change(1)

    df["ma5"] = _rolling_mean(g["close"], 5)
    df["ma10"] = _rolling_mean(g["close"], 10)
    df["ma20"] = _rolling_mean(g["close"], 20)
    df["roll_high_20"] = g["high"].transform(lambda s: s.rolling(20, min_periods=20).max())
    df["roll_low_20"] = g["low"].transform(lambda s: s.rolling(20, min_periods=20).min())
    df["vol_ma5"] = _rolling_mean(g["volume"], 5)
    df["vol_ma20"] = _rolling_mean(g["volume"], 20)
    df["amount_ma20"] = _rolling_mean(g["amount"], 20)
    df["turn_ma5"] = _rolling_mean(g["turn"], 5)

    df["ma_gap_5"] = df["close"] / (df["ma5"] + EPS) - 1.0
    df["ma_gap_10"] = df["close"] / (df["ma10"] + EPS) - 1.0
    df["ma_gap_20"] = df["close"] / (df["ma20"] + EPS) - 1.0
    df["ma_cross_5_20"] = df["ma5"] / (df["ma20"] + EPS) - 1.0
    df["breakout_20"] = df["close"] / (df["roll_high_20"] + EPS) - 1.0
    df["vol_ratio_5"] = df["volume"] / (df["vol_ma5"] + EPS)
    df["vol_ratio_20"] = df["volume"] / (df["vol_ma20"] + EPS)
    df["amount_ratio_20"] = df["amount"] / (df["amount_ma20"] + EPS)
    df["turn_ratio_5"] = df["turn"] / (df["turn_ma5"] + EPS)
    df["intraday_range"] = (df["high"] - df["low"]) / (df["preclose"] + EPS)

    hl = df["high"] - df["low"]
    df["body_ratio"] = (df["close"] - df["open"]) / (hl + EPS)
    df["close_pos"] = (df["close"] - df["low"]) / (hl + EPS)
    prev_close = g["close"].shift(1)
    tr = np.maximum.reduce([(df["high"] - df["low"]).to_numpy(), (df["high"] - prev_close).abs().to_numpy(), (df["low"] - prev_close).abs().to_numpy()])
    df["tr"] = tr
    df["atr14"] = g["tr"].transform(lambda s: s.rolling(14, min_periods=14).mean())
    df["atr_norm_14"] = df["atr14"] / (df["close"] + EPS)
    df["realized_vol_20"] = _rolling_std(g["ret_1d"], 20)
    df["ret_neg_1d"] = df["ret_1d"].where(df["ret_1d"] < 0.0, 0.0)
    df["downside_vol_20"] = df.groupby("code")["ret_neg_1d"].transform(lambda s: s.rolling(20, min_periods=20).std())
    df["downside_vol_ratio_20"] = df["downside_vol_20"] / (df["realized_vol_20"] + EPS)

    delta = g["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.groupby(df["code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    avg_loss = loss.groupby(df["code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    rs = avg_gain / (avg_loss + EPS)
    df["rsi14"] = 100.0 - (100.0 / (1.0 + rs))

    df["amihud_1d"] = np.abs(df["ret_1d"]) / (df["amount"] + EPS)
    df["amihud_20"] = df.groupby("code")["amihud_1d"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    ret_vol_corr = (
        df.groupby("code", group_keys=False)
        .apply(lambda x: x["ret_1d"].rolling(20, min_periods=20).corr(x["vol_chg_1d"]))
        .reset_index(level=0, drop=True)
    )
    df["ret_vol_corr_20"] = ret_vol_corr
    df["close_to_vwap_day"] = df["close"] / (df["vwap_day"] + EPS) - 1.0
    df["overnight_gap"] = df["open"] / (df["preclose"] + EPS) - 1.0
    df["px_daily_close"] = df["close"]

    # portfolio/timing features
    df["barra_size_proxy"] = np.log(df["amount_ma20"].clip(lower=0.0) + 1.0)
    df["barra_momentum_proxy"] = df["ret_20d"]
    df["barra_volatility_proxy"] = df["realized_vol_20"]
    df["barra_liquidity_proxy"] = -df["amihud_20"]
    df["barra_beta_proxy"] = df["ret_vol_corr_20"]
    df["crowding_proxy_raw"] = 0.45 * df["vol_ratio_20"].abs() + 0.35 * df["turn_ratio_5"].abs() + 0.20 * df["ret_vol_corr_20"].abs()
    df["board_type"] = df["code"].astype(str).map(infer_board_type)
    df["industry_bucket"] = df["code"].astype(str).map(infer_industry_bucket)
    return df


def build_feature_bundle(bundle: MarketBundle) -> FeatureBundle:
    """Generate daily and multi-frequency feature bases."""
    minute_daily_feat = build_minute_daily_features(bundle.minute5)
    daily_base = build_daily_feature_base(bundle.daily, minute_daily_feat)

    views = build_frequency_views(daily_base, bundle.minute5)
    # convert non-daily views to generic features if needed
    for freq in ["5min", "15min", "30min", "60min", "120min"]:
        if freq in views and not views[freq].empty:
            views[freq] = add_generic_micro_structure_features(views[freq], time_col="datetime")
    if "W" in views and not views["W"].empty:
        views["W"] = add_generic_micro_structure_features(views["W"], time_col="date")
    if "M" in views and not views["M"].empty:
        views["M"] = add_generic_micro_structure_features(views["M"], time_col="date")

    price_cols = ["code", "date", "px_open5", "px_vwap30", "px_twap_last30", "px_daily_close"]
    price_table = daily_base[price_cols].copy()
    return FeatureBundle(
        by_freq=views,
        price_table_daily=price_table,
        meta={"daily_rows": len(bundle.daily), "minute_rows": len(bundle.minute5)},
    )


def pick_named_file(folder: Path, stem: str, file_format: str = "auto") -> Optional[Path]:
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


def load_index_benchmark_data(
    index_root: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    file_format: str = "auto",
) -> Dict[str, pd.DataFrame]:
    mapping = {"hs300": "hs300_price", "zz500": "zz500_price", "zz1000": "zz1000_price"}
    idx_data: Dict[str, pd.DataFrame] = {}
    for name, stem in mapping.items():
        fp = pick_named_file(index_root, stem, file_format=file_format)
        if fp is None:
            idx_data[name] = pd.DataFrame(columns=["date", "close"])
            continue
        df = read_data_file(fp, usecols=["date", "close"], start_date=start_date, end_date=end_date)
        if df.empty:
            idx_data[name] = pd.DataFrame(columns=["date", "close"])
            continue
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date")
        idx_data[name] = df[["date", "close"]].copy()
    return idx_data


def lookup_index_period_return(index_df: pd.DataFrame, entry_date: pd.Timestamp, exit_date: pd.Timestamp) -> float:
    if index_df.empty or pd.isna(entry_date) or pd.isna(exit_date):
        return float("nan")
    s = index_df.sort_values("date")
    s_entry = s[s["date"] >= pd.Timestamp(entry_date)]
    s_exit = s[s["date"] >= pd.Timestamp(exit_date)]
    if s_entry.empty or s_exit.empty:
        return float("nan")
    entry_px = float(s_entry.iloc[0]["close"])
    exit_px = float(s_exit.iloc[0]["close"])
    return exit_px / (entry_px + EPS) - 1.0

