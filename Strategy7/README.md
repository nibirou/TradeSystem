# Strategy7

模块化量化研究框架，支持：

1. 多频率因子研究（5min/15min/30min/60min/120min/D/W/M）
2. 插件化四类模型（选股、择时、组合权重、执行）
3. 完整回测评估（收益、超额、IC、分层指标、图表）
4. 因子挖掘与 catalog 入库（可自动接入主流程）

## 快速开始

```bash
python Strategy7/run_strategy7.py \
  --train-start 2024-01-01 --train-end 2024-12-31 \
  --test-start 2025-01-01 --test-end 2025-12-31 \
  --factor-freq D \
  --stock-model-type decision_tree \
  --timing-model-type none \
  --portfolio-model-type equal_weight \
  --execution-model-type ideal_fill
```

## 常用命令

列出因子（快速路径，不依赖行情数据）：

```bash
python Strategy7/run_strategy7.py --list-factors --factor-freq D
```

运行 FactorGCL：

```bash
python Strategy7/run_strategy7.py --stock-model-type factor_gcl --label-task return
```

运行 DAFAT（Transformer 自适应方案）：

```bash
python Strategy7/run_strategy7.py --stock-model-type dafat --label-task return
```

运行因子挖掘：

```bash
python Strategy7/run_factor_mining.py --framework fundamental_multiobj
```

列出挖掘可用因子（含 catalog/自定义）：

```bash
python Strategy7/run_factor_mining.py --list-factors --factor-freq 30min
```

股票池切换（主流程与挖掘入口通用）：

```bash
# 上证50
python Strategy7/run_strategy7.py --universe sz50

# 中证500
python Strategy7/run_strategy7.py --universe zz500

# 全市场
python Strategy7/run_strategy7.py --universe all

# 全市场 + 自定义股票列表（例如中证1000/中证2000研究池）
python Strategy7/run_strategy7.py --universe all --stock-list-path D:/PythonProject/Quant/data_baostock/metadata/your_custom_pool.csv
```

基本面数据接入（AK + Baostock）：

```bash
python Strategy7/run_strategy7.py \
  --universe all \
  --stock-list-path D:/PythonProject/Quant/data_baostock/metadata/your_custom_pool.csv \
  --fundamental-root-ak D:/PythonProject/Quant/data_baostock/ak_fundamental \
  --fundamental-root-bsq D:/PythonProject/Quant/data_baostock/baostock_fundamental_q
```

可选基本面因子包（`--factor-packages`）：`fund_growth/fund_valuation/fund_profitability/fund_quality/fund_leverage/fund_cashflow/fund_efficiency/fund_expectation/fund_hf_fusion`。

金融文本数据接入（新闻/公告/研报）：

```bash
python Strategy7/run_strategy7.py \
  --universe all \
  --stock-list-path D:/PythonProject/Quant/data_baostock/metadata/your_custom_pool.csv \
  --text-root-news D:/PythonProject/Quant/data_baostock/data_em_news \
  --text-root-notice D:/PythonProject/Quant/data_baostock/data_em_notices \
  --text-root-report-em D:/PythonProject/Quant/data_baostock/data_em_reports \
  --text-root-report-iwencai D:/PythonProject/Quant/data_baostock/data_iwencai_reports
```

可选文本因子包（`--factor-packages`）：`text_sentiment/text_attention/text_event/text_topic/text_fusion`。

## 文档

1. [完整使用指南](./docs/usage_guide.md)
2. [代码阅读地图](./docs/code_reading_map.md)
3. [FactorGCL 说明](./docs/factor_gcl.md)
4. [DAFAT 复现与工程实现说明](./docs/dafat_transformer.md)
5. [因子挖掘框架说明](./docs/factor_mining_framework.md)

## 脚本

`scripts/` 目录包含常用 `.ps1/.cmd/.sh` 运行脚本，可按场景直接使用或改参数。

