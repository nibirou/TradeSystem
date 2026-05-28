# Data Source Enhancement Report

日期：2026-05-27

## 1. 当前脚本覆盖情况

`Data/Trading`
- 已有：AkShare 日线、BaoStock 日线/5/15/30/60 分钟、XtQuant 本地行情。
- Strategy7 直接依赖：`data_baostock/stock_hist/<universe>/d` 和 `data_baostock/stock_hist/<universe>/5`，文件名为 `sh_600000_d.csv/parquet`、`sh_600000_5.csv/parquet`。
- 主要问题：部分脚本硬编码日期/路径；BaoStock 全市场日期偶发写死；并发使用 BaoStock 容易受全局登录状态影响；AkShare 东财日线容易受反爬影响。

`Data/Fundamental`
- 已有：BaoStock 季频财务表；AkShare 新浪/东财财务指标。
- Strategy7 会自动读取：`ak_fundamental` 与 `baostock_fundamental_q`。
- 主要缺口：公司行为、分红配股、股本变动、ST 状态、股东持仓结构没有统一落盘。

`Data/News`、`Data/CommonNotice`、`Data/ResearchReport`
- 已有：东财新闻/公告/研报，Wencai 研报，若干 NLP 情绪因子。
- Strategy7 会自动读取：`data_em_news`、`data_em_notices`、`data_em_reports`、`data_iwencai_reports`。
- 主要缺口：巨潮公告作为披露主站更适合补齐公告；结构化资金流/LHB 未按股票稳定落盘；研报列表可用 AkShare 轻量补充。

`Data/IndexData`
- 已有：AkShare 下载沪深300、中证500、中证1000指数日线。
- Strategy7 直接读取：`ak_index/hs300_price`、`zz500_price`、`zz1000_price`。
- 主要问题：无失败回退；指数成分股快照未统一保存。

`Data/ETF_Trade_Hist`
- 已有：ETF 日线历史行情，东财/新浪双源。
- 主要缺口：ETF 分钟线、净值/规模/分红等 ETF 轮动常用信息。

`Data/ShenwanIndustry`
- 已有：申万三级成分。
- 主要缺口：东财/同花顺行业、概念板块成员与板块列表。

## 2. 新增脚本

公共工具
- `Data/data_fetch_common.py`
  - 统一 `data_baostock` 路径发现、BaoStock 股票池、代码格式、CSV/Parquet 双写。

行情与微观结构
- `Data/Trading/TradingData_BaoStock_ED4_Strategy7.py`
  - 稳定版 BaoStock 日线/周/月/5/15/30/60 分钟下载。
  - 默认写入 Strategy7 目录：`data_baostock/stock_hist/<pool>/<freq>`。
- `Data/Trading/TradingData_Akshare_Minute_ED4.py`
  - 东财 1/5/15/30/60 分钟滚动补充，适合补最近分钟数据。
- `Data/Trading/TradingData_Tick_Akshare_ED1.py`
  - 腾讯分笔成交，保存到 `tick_trades/tencent/<pool>/<date>`。
- `Data/Trading/TradingData_OrderBook_Snapshot_ED1.py`
  - 当前盘口快照；支持 AkShare `stock_bid_ask_em`，可选 XtQuant `get_full_tick`。
  - 注意：免费公开源通常不是历史 10 档 Level2。

公告、研报、资金面
- `Data/CommonNotice/CommonNotice_Cninfo_ED1.py`
  - 巨潮资讯公告；输出字段兼容 Strategy7 text loader。
- `Data/ResearchReport/ReportsData_Akshare_Research_ED2.py`
  - AkShare 东财个股研报列表轻量版。
- `Data/News/factor_flow_lhb_Akshare_ED4.py`
  - 个股资金流、资金流排名、龙虎榜统计结构化落盘。

基本面与事件
- `Data/Fundamental/FundamentalData_CorporateActions_Akshare_ED1.py`
  - 股本变动、分红配股、ST 列表、十大流通股东持仓。

指数、ETF、行业概念
- `Data/IndexData/IndexData_Akshare_Baostock_ED2.py`
  - AkShare 指数日线 + BaoStock 回退；保存指数成分股快照。
- `Data/ETF_Trade_Hist/ETF_Minute_Info_Akshare_ED1.py`
  - ETF 分钟线、ETF 净值/日表、分红。
- `Data/ShenwanIndustry/ShenwanIndustryData_Akshare_ED2.py`
  - 东财行业/概念板块列表及成分；同花顺板块列表作为补充。

## 3. 免费渠道建议

更稳定的基础盘：
- 日线/周线/月线/5-60 分钟：BaoStock 优先，免费且无需注册，适合作为 Strategy7 主数据源。
- 最近 1 分钟：AkShare 东财分钟线可滚动补，但通常只适合近期数据。
- 本地券商终端：XtQuant/QMT 若有行情权限，是分钟、实时 tick、盘口快照更好的来源。

更准确的披露与事件：
- 公告：巨潮资讯优先，东财公告作为补充。
- 财务：BaoStock 季频表 + AkShare 东财/新浪指标互补。
- 公司行为：巨潮股本变动、AkShare 分红配股、ST 列表、股东持仓。

补充型 alpha 数据：
- 资金流：东财个股资金流、资金流排名。
- 龙虎榜：东财龙虎榜统计。
- 行业/概念：申万三级 + 东财/同花顺行业概念并行。
- ETF：ETF 分钟线、净值、规模、分红。

## 4. Level2/逐笔委托边界

免费公开渠道可以补：
- 最近交易日分笔成交。
- 当前盘口快照。
- 近端 1 分钟/5 分钟 K 线。

免费公开渠道通常不能稳定补：
- 全历史 1 分钟长回溯。
- 历史逐笔成交全量。
- 逐笔委托。
- 历史 Level2 十档快照。

这些数据更现实的来源是券商 QMT/PTrade/Level2 权限、本地行情落盘服务，或交易所/商业数据供应商。

## 5. 运行效率建议

1. 主行情用 BaoStock 做基准，AkShare 只做近端补充与交叉校验。
2. BaoStock 不建议在一个进程内高并发共享登录状态；可按股票池/日期段拆多个独立进程。
3. 所有脚本优先增量读取本地最大日期，再追加新数据。
4. Parquet 优先供 Strategy7 读取，CSV 保留人工排查。
5. 大文本数据保留“列表”和“正文/PDF”两层缓存，避免每次重复抓全文。
6. 对分钟/分笔/盘口类数据限制并发并加随机等待，避免触发反爬。
7. 对 `all` 股票池保存快照，退市/调出股票保留历史文件，不应删除旧数据。
