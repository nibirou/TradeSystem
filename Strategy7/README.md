# Strategy7: Modular Quant Research Engine

`Strategy7` 把 `Strategy6` 的单文件回测脚本升级为可插拔工程化框架，覆盖：

- 数据加载与多源融合（量价、基本面、文本/外部因子）
- 多频率因子库（5min/15min/30min/60min/120min/日/周/月）
- 标签工程（方向/收益率/波动率/多任务）
- 四类模型模块化（选股、择时、组合权重、执行引擎）
- 回测评估、IC 统计、曲线绘图、模型与结果落盘

## 目录结构

```text
Strategy7/
  run_strategy7.py
  strategy7/
    config.py
    core/
    data/
    factors/
    models/
      stock_selection/
      timing/
      portfolio/
      execution/
    backtest/
    pipeline/
    plugins/
```

## 快速运行（默认接近 Strategy6）

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

## 开启主板过滤 + 动态组合优化 + 真实成交仿真 + 择时

```bash
python Strategy7/run_strategy7.py \
  --main-board-only \
  --timing-model-type volatility_regime \
  --portfolio-model-type dynamic_opt \
  --execution-model-type realistic_fill \
  --save-models
```

## 插件扩展

- 自定义因子模块：`--custom-factor-py path/to/module.py`（需实现 `register_factors(library)`）
- 自定义外部数据源模块：`--extra-source-module path/to/module.py`（需实现 `register_sources(registry)`）
- 直接追加外部因子表：`--extra-factor-paths file1.csv,file2.parquet`
- 自定义四类模型模块（可选）：
  - `--custom-stock-model-py`
  - `--custom-timing-model-py`
  - `--custom-portfolio-model-py`
  - `--custom-execution-model-py`
  每个模块需提供 `build_model(cfg)` 并返回对应基类实例。

模板文件：

- `strategy7/plugins/custom_factor_template.py`
- `strategy7/plugins/custom_source_template.py`
- `strategy7/plugins/custom_stock_model_template.py`
- `strategy7/plugins/custom_timing_model_template.py`
- `strategy7/plugins/custom_portfolio_model_template.py`
- `strategy7/plugins/custom_execution_model_template.py`

## Factor Mining (New)

`Strategy7` now includes a report-aligned factor mining subsystem:

- `run_factor_mining.py` (CLI entry)
- `strategy7/mining/` (formulas, NSGA-II/III, evaluation, catalog, custom expression)
- `docs/factor_mining_framework.md`

Quick start:

```bash
python Strategy7/run_factor_mining.py \
  --framework fundamental_multiobj \
  --train-start 2021-01-01 --train-end 2023-12-31 \
  --valid-start 2024-01-01 --valid-end 2024-12-31
```

After mining, admitted factors are written to `factor_mining/factor_catalog.json` and can be auto-loaded by `run_strategy7.py` via:

- `--factor-catalog-path auto` (default)
- `--disable-catalog-factors` (opt-out)
