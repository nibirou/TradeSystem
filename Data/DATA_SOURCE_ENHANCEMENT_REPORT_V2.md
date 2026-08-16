# Data Source Enhancement Report V2

日期：2026-05-27

本轮遵守“只新增、不修改现有脚本”的原则，继续补充免费或可免费注册使用的数据渠道，并尽量保持与 Strategy7 当前 `data_baostock` 保存习惯兼容。

## 新增脚本

### Trading

- `Data/Trading/TradingData_TDX_Optional_ED1.py`
  - 可选依赖：`xmtdx` 或 `pytdx` `mootdx` `eltdx` `tdxquant`
  - 数据：通达信 K 线、当前五档盘口快照、历史逐笔成交（取决于公共 TDX 服务器支持）
  - 保存：
    - K 线：`data_baostock/stock_hist/<pool>/<freq>/<code>_<freq>.csv|parquet`
    - 五档快照：`data_baostock/quote_snapshot/xmtdx/<pool>/<date>/quotes_<time>.csv|parquet`
    - 逐笔成交：`data_baostock/tick_trades/xmtdx/<pool>/<date>/<code>_<date>.csv|parquet`

- `Data/Trading/TradingData_EFinance_Optional_ED1.py`
  - 可选依赖：`efinance`
  - 数据：东财公开接口历史 K 线，作为 AkShare/BaoStock 交叉校验源
  - 默认保存：`data_baostock/stock_hist_efinance/<pool>/<freq>/`
  - 可用 `--strategy7-layout` 写入 Strategy7 主行情目录；默认不覆盖主源。

- `Data/Trading/DataQuality_Strategy7_Audit_ED1.py`
  - 数据质量审计：检查缺列、重复键、日期范围、非正价格、读取错误
  - 保存：`data_baostock/quality_reports/strategy7_market_quality_<pool>_<freq>_<timestamp>.csv|parquet`

### Macro / Risk

- `Data/Macro/MacroData_Akshare_ED1.py`
  - 数据：LPR、SHIBOR、CPI、PPI、PMI、GDP、货币供应、外储、贸易、社融、地产、社零、工业增加值等
  - 保存：`data_baostock/macro/akshare/<dataset>.csv|parquet`

- `Data/IndexData/MarketRiskData_Akshare_ED1.py`
  - 数据：两融、北向资金、ETF/指数期权 QVIX
  - 保存：`data_baostock/market_risk/akshare/...`

### Sentiment / Event

- `Data/News/StockHotRank_Eastmoney_Akshare_ED1.py`
  - 数据：东财个股人气排名与关键词
  - 保存：`data_baostock/sentiment/eastmoney_hot_rank/<pool>/<date>/`

- `Data/ResearchReport/Wencai_Query_PyWencai_Optional_ED1.py`
  - 可选依赖：`pywencai`
  - 数据：问财自然语言选股、事件、研报、回购、质押、调研等探索型数据
  - Cookie：通过 `--cookie` 或 `IWENCAI_COOKIE` 传入，不写进代码
  - 保存：`data_baostock/wencai_query/<date>/`

### Fundamental

- `Data/Fundamental/FundamentalData_Tushare_Optional_ED1.py`
  - 可选依赖：`tushare`，需 `TUSHARE_TOKEN`
  - 数据：stock_basic、daily_basic、adj_factor、利润表、资产负债表、现金流量表、财务指标、分红、停复牌、指数日线
  - 保存：`data_baostock/tushare/...`

## 当前免费渠道能力边界

可以稳定补充：

- 日线、周线、月线：BaoStock、AkShare、efinance/Tushare 交叉校验
- 1/5/15/30/60 分钟线：AkShare 东财、BaoStock 部分频率、通达信可选源
- 当日或近端逐笔成交：腾讯/AkShare、xmtdx/TDX，可作为滚动补充
- 当前盘口快照：AkShare `stock_bid_ask_em`、xmtdx 五档
- 公告、研报、资金流、行业/概念、ETF、宏观、两融、北向、QVIX、人气热度

免费公开源通常不能稳定获得：

- 全历史 1 分钟长回溯
- 历史逐笔委托
- 历史 Level2 十档快照
- 订单队列、委托撤单、逐笔成交与订单关联数据

这些通常需要券商 QMT/PTrade/Level2 权限、交易所授权行情，或商业数据商。新增脚本已经把免费公开源可触达的部分尽量落地；真正全量 Level2 只能接入授权源。

## 推荐运行顺序

1. 主行情基准：
   - `TradingData_BaoStock_ED4_Strategy7.py`
2. 近端分钟/逐笔补充：
   - `TradingData_Akshare_Minute_ED4.py`
   - `TradingData_Tick_Akshare_ED1.py`
   - `TradingData_TDX_Optional_ED1.py`
3. 结构化 Alpha 数据：
   - `factor_flow_lhb_Akshare_ED4.py`
   - `StockHotRank_Eastmoney_Akshare_ED1.py`
   - `MarketRiskData_Akshare_ED1.py`
   - `MacroData_Akshare_ED1.py`
4. 基本面和事件补充：
   - `FundamentalData_CorporateActions_Akshare_ED1.py`
   - `FundamentalData_Tushare_Optional_ED1.py`
   - `CommonNotice_Cninfo_ED1.py`
5. 每轮落库后做质量审计：
   - `DataQuality_Strategy7_Audit_ED1.py`

## 调试记录

- 已使用 `env_quant` 对新增脚本执行 `python -m py_compile`
- 已验证所有新增脚本的 `--help`
- `xmtdx`、`pytdx`、`pywencai`、`tushare` 当前未安装，脚本会快速报出可选依赖缺失
- `efinance` 当前能被发现但导入失败，因为其版本尝试在 env 的 `site-packages/efinance/data` 创建缓存目录，当前环境无写权限；脚本已捕获该问题并给出明确提示

