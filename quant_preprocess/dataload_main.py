import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # /workspace/Quant/TradeSystem
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_preprocess.config import PathsConfig, SchemaConfig, PeriodConfig, LoadRequest
from quant_preprocess.pipeline import UnifiedPipeline
from quant_preprocess.labels import LabelConfig


paths = PathsConfig(base_dir="/workspace/Quant/data_baostock")
schema = SchemaConfig()

period = PeriodConfig(
    factor_start="2025-12-02",
    factor_end="2025-12-30",
    train_start="2025-01-02",
    train_end="2025-11-15",
    backtest_start="2025-09-01",
    backtest_end="2025-11-30",
    inference_day="2025-12-30",
    factor_buffer_n=60,
    label_buffer_n=30,   # 你 loader 端会用它来确定 period.load_end
)

req = LoadRequest(
    pools=("hs300", "zz500"),
    load_daily=True,
    load_minute5=True,
    use_akshare_fundamental=True,
    use_baostock_q_fundamental=True,
    minute5_best_price="best_close",  # 这个字段如果现在不用了也无所谓
)

pipe = UnifiedPipeline(paths, schema)

# 1) 日频 close 卖出价：last/max/topk_mean
cfg_daily_sell = LabelConfig(
    horizon_n=20,
    price_source="daily",
    daily_sell_method="topk_mean_close",  # "last_close" / "max_close" / "topk_mean_close"
    topk=5,
)

# 2) 5min close 卖出价：last_day_intraday_max / window_intraday_max / window_intraday_topk_mean
cfg_minute_sell = LabelConfig(
    horizon_n=20,
    price_source="minute",
    minute_sell_method="window_intraday_topk_mean",
    topk=20,
)

res = pipe.run(
    period=period,
    req=req,
    factor_start=period.factor_start,
    factor_end=period.factor_end,
    label_cfg_daily_sell=cfg_daily_sell,
    label_cfg_minute_sell=cfg_minute_sell,
    # 输出锚点日期范围（可不传，默认=因子区间）
    label_output_start=period.factor_start,
    label_output_end=period.factor_end,
)

daily = res.bundle.daily
minute5 = res.bundle.minute5
trade_dates = res.bundle.trade_dates
factor_panel = res.factor_panel

labels_daily = res.labels["daily_label_by_daily_close"]
labels_minute = res.labels["daily_label_by_minute_close"]

print("daily", daily.head())
print("minute5", None if minute5 is None else minute5.head())
print("trade_dates", trade_dates[:5], "...", trade_dates[-5:])

print("factor_panel", factor_panel.head())
print("labels_daily_sell", labels_daily.head())
print("labels_minute_sell", labels_minute.head())

# 你会看到 labels 中带 is_complete 字段
print("labels_daily_sell is_complete ratio:", labels_daily["is_complete"].mean())
