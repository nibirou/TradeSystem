# quant_preprocess/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import pandas as pd

from .config import PathsConfig, SchemaConfig, PeriodConfig, LoadRequest
from .market_loader import MarketDataLoader, MarketBundle
# from .factors import FactorEngine
from .factors_update import FactorEngine
from .preprocess import clean_numeric
from .sources_fundamental import AkShareFundamentalSource, BaoStockQFundamentalSource

from .labels import LabelEngine, LabelConfig

from .resample import synthesize_by_freq


@dataclass
class PipelineResult:
    bundle: MarketBundle
    factor_panel: pd.DataFrame
    labels: Dict[str, pd.DataFrame]


class UnifiedPipeline:
    def __init__(self, paths: PathsConfig, schema: SchemaConfig):
        self.paths = paths
        self.schema = schema
        self.market_loader = MarketDataLoader(paths, schema)

    def run(
        self,
        period: PeriodConfig,
        req: LoadRequest,
        factor_selected: Optional[Sequence[str]] = None,
        # factor_start: Optional[str] = None,
        # factor_end: Optional[str] = None,

        # ✅新：标签配置（你可以传两套：日频卖出规则 / 分钟卖出规则）
        label_cfg_daily_sell: Optional[LabelConfig] = None,
        label_cfg_minute_sell: Optional[LabelConfig] = None,

        # ✅控制：最终输出的锚点日期范围（通常是 factor_start..factor_end）
        label_output_start: Optional[str] = None,
        label_output_end: Optional[str] = None,
    ) -> PipelineResult:
        
        # 1) load market bundle（你已在 bundle 里带 label_buffer_dates） 历史行情加载
        bundle = self.market_loader.load(period, req)

        # 未来bundle中可能还有minute15、minute30、minute60、minute120等
        if req.trade_freq == "15min":
            # 把minute5历史行情整合为minute15历史行情
            minute15 = synthesize_by_freq(bundle.daily, bundle.minute5, freq=req.trade_freq)
            minute15 = clean_numeric(minute15, cols=["open", "high", "low", "close", "volume"])
        if req.trade_freq == "30min":
            # 把minute5历史行情整合为minute30历史行情
            minute30 = synthesize_by_freq(bundle.daily, bundle.minute5, freq=req.trade_freq)
            minute30 = clean_numeric(minute30, cols=["open", "high", "low", "close", "volume"])
        if req.trade_freq == "60min":
            # 把minute5历史行情整合为minute60历史行情
            minute60 = synthesize_by_freq(bundle.daily, bundle.minute5, freq=req.trade_freq)
            minute60 = clean_numeric(minute60, cols=["open", "high", "low", "close", "volume"])
        if req.trade_freq == "120min":
            # 把minute5历史行情整合为minute120历史行情
            minute120 = synthesize_by_freq(bundle.daily, bundle.minute5, freq=req.trade_freq)
            minute120 = clean_numeric(minute120, cols=["open", "high", "low", "close", "volume"])
        if req.trade_freq == "W":
            # 把daily历史行情整合为week历史行情
            week = synthesize_by_freq(bundle.daily, bundle.minute5, freq=req.trade_freq)
            week = clean_numeric(week, cols=["open", "high", "low", "close", "volume", "turn", "tradestatus"])
        if req.trade_freq == "M":
            # 把daily历史行情整合为month历史行情
            month = synthesize_by_freq(bundle.daily, bundle.minute5, freq=req.trade_freq)
            month = clean_numeric(month, cols=["open", "high", "low", "close", "volume", "turn", "tradestatus"])

        # 2) clean basics
        daily = bundle.daily.copy()
        minute5 = None if bundle.minute5 is None else bundle.minute5.copy()

        daily = clean_numeric(daily, cols=["open", "high", "low", "close", "volume", "turn", "tradestatus"])
        if minute5 is not None:
            minute5 = clean_numeric(minute5, cols=["open", "high", "low", "close", "volume"])

        # 3) factor engine 量价因子计算与加载
        # fe = FactorEngine(daily=daily, minute5=minute5)
        # 在[load_start, load_end]区间，但是只取trade_dates区间[factor_start, factor_end]
        # factor_panel = fe.compute(
        #     selected=factor_selected,
        #     start=bundle.load_start, 
        #     end=bundle.load_end,
        #     post_winsor_zscore=True,
        #     post_neutralize=False,
        # )
        if req.factor_freq == "5min":
            fe = FactorEngine(daily=daily, freq="5min", minute5=minute5)
        if req.factor_freq == "15min":
            fe = FactorEngine(daily=daily, freq="15min", minute15=minute15)
        if req.factor_freq == "30min":
            fe = FactorEngine(daily=daily, freq="30min", minute30=minute30)
        if req.factor_freq == "60min":
            fe = FactorEngine(daily=daily, freq="60min", minute60=minute60)
        if req.factor_freq == "120min":
            fe = FactorEngine(daily=daily, freq="120min", minute120=minute120)
        if req.factor_freq == "D":
            fe = FactorEngine(daily=daily, freq="D", minute5=minute5)
        if req.factor_freq == "W":
            fe = FactorEngine(daily=daily, freq="W", weekly=week)
        if req.factor_freq == "M":
            fe = FactorEngine(daily=daily, freq="M", monthly=month)

        factor_panel = fe.compute(selected=None, start=bundle.load_start, end=bundle.load_end)
        factor_panel = factor_panel[factor_panel["date"] >= pd.to_datetime(period.factor_start)]
        factor_panel = factor_panel[factor_panel["date"] <= pd.to_datetime(period.factor_end)]


        # 基本面因子加载
        # 4) fundamental merge (可插拔)
        code_universe = sorted(factor_panel["code"].astype(str).unique().tolist())

        if req.use_akshare_fundamental:
            ak_src = AkShareFundamentalSource(base_dir=self.paths.base_dir, pool=req.pools[0])
            ak_daily = ak_src.load_daily(bundle.trade_dates, code_universe)
            if not ak_daily.empty:
                factor_panel = factor_panel.merge(ak_daily, on=["date", "code"], how="left")

        if req.use_baostock_q_fundamental:
            bs_src = BaoStockQFundamentalSource(base_dir=self.paths.base_dir, pool=req.pools[0])
            bs_daily = bs_src.load_daily(bundle.trade_dates, code_universe)
            if not bs_daily.empty:
                factor_panel = factor_panel.merge(bs_daily, on=["date", "code"], how="left")

        # 其他额外因子加载（算法挖掘新因子/文本因子等）



        # 5) labels（✅新 labels.py） 标签（收益率、波动率计算与加载）
        le = LabelEngine(label_buffer_dates=bundle.label_buffer_dates)
        labels: Dict[str, pd.DataFrame] = {}

        # 输出日期：默认=因子区间（不含向后buffer段）
        out_start = pd.to_datetime(label_output_start or period.factor_start)
        out_end = pd.to_datetime(label_output_end or period.factor_end)
        output_dates = bundle.label_buffer_dates[
            (bundle.label_buffer_dates >= out_start) & (bundle.label_buffer_dates <= out_end)
        ]
        # 目前只实现了在日频label_freq尺度上统计收益率、波动率等label
        if req.label_freq == "D":
            # A) 日频close统计卖出价（daily source）
            cfg1 = label_cfg_daily_sell or LabelConfig(
                horizon_n=20,
                price_source="daily",
                daily_sell_method="last_close",
                topk=5,
            )
            labels["daily_label_by_daily_close"] = le.build_labels_panel(
                daily=daily,
                minute=minute5,
                cfg=cfg1,
                output_dates=output_dates,
                code_col=self.schema.code_col,
                date_col=self.schema.date_col,
                close_col="close",
                dt_col=self.schema.datetime_col,
            )

            # B) 5min close统计卖出价（minute source）
            cfg2 = label_cfg_minute_sell or LabelConfig(
                horizon_n=20,
                price_source="minute",
                minute_sell_method="window_intraday_max",
                topk=5,
            )
            
            labels["daily_label_by_minute_close"] = le.build_labels_panel(
                daily=daily,
                minute=minute5,   # 如果 minute5=None，则 sell_px 全 NaN
                cfg=cfg2,
                output_dates=output_dates,
                code_col=self.schema.code_col,
                date_col=self.schema.date_col,
                close_col="close",
                dt_col=self.schema.datetime_col,
            )

        # 行情  因子 和标签（收益率、波动率），都有各自的统计频率和统计来源
        # 比如我要统计日频（D）的行情 因子 和标签，统计来源是日频历史行情、以及5min历史行情（在计算量价因子时，可以用5min数据计算一些日频量价因子；在统计标签时，设置不同的horizon_n长度）
        # 最终把因子、标签都统计到日频上

        # 再比如统计15min 频率的行情 因子 和标签，那么统计来源只能是5min历史行情（或1min）
        # 15min行情由5min（或1min）历史行情合成，15min的一些量价因子可能由5min（或1min）行情数据得到；在统计标签时，在设置5min历史行情（或1min）设置对应horizon_n长度，来得到15min下的标签

        return PipelineResult(bundle=bundle, factor_panel=factor_panel, labels=labels)
