# Strategy7 使用说明（完整指南）

本文档面向日常研究和工程维护，覆盖：

1. 工程结构与运行链路
2. 数据准备规范
3. 回测主流程使用
4. 因子挖掘使用
5. 自定义扩展（因子/数据源/模型）
6. 输出文件说明
7. 常见问题排查

## 1. 工程定位

`Strategy7` 是一个模块化量化研究引擎，核心特点：

1. 数据、因子、模型、回测四层可插拔
2. 同时支持传统树模型与 `FactorGCL/DAFAT` 深度模型
3. 支持因子挖掘、因子 catalog 入库、主流程自动加载
4. 支持自定义因子表达式与自定义插件模型

## 2. 目录结构

核心目录：

1. `run_strategy7.py`：主回测入口
2. `run_factor_mining.py`：因子挖掘入口
3. `strategy7/config.py`：参数定义与校验
4. `strategy7/data/`：数据加载、重采样、预处理
5. `strategy7/factors/`：因子库与标签工程
6. `strategy7/models/`：选股/择时/组合/执行四类模型
7. `strategy7/backtest/`：回测引擎与评价指标
8. `strategy7/mining/`：因子挖掘、评估、NSGA、多因子入库
9. `strategy7/pipeline/runner.py`：端到端主流程编排
10. `scripts/`：常用批处理脚本（`.ps1/.cmd/.sh`）
11. `docs/`：模型与挖掘文档

## 3. 环境依赖

基础依赖：

1. `python>=3.9`
2. `pandas`
3. `numpy`
4. `scikit-learn`
5. `matplotlib`（可选，仅用于画图）

深度模型附加依赖：

1. `torch`（`--stock-model-type factor_gcl` 或 `dafat` 必需）

建议先验证解释器：

```powershell
python -c "import pandas, numpy, sklearn; print('ok')"
python -c "import torch; print(torch.__version__)"
```

## 4. 数据目录规范

默认数据根目录（可改）：

1. `--data-root auto` 时，会按 `--universe` 自动映射到：
2. `D:/PythonProject/Quant/data_baostock/stock_hist/hs300`
3. `D:/PythonProject/Quant/data_baostock/stock_hist/sz50`
4. `D:/PythonProject/Quant/data_baostock/stock_hist/zz500`
5. `D:/PythonProject/Quant/data_baostock/stock_hist/all`

应包含：

1. 日线目录：`d/`
2. 5 分钟目录：`5/`

文件命名：

1. 日线：`sh_600000_d.csv` 或 `.parquet`
2. 分钟：`sh_600000_5.csv` 或 `.parquet`

关键字段（至少）：

1. 日线：`date code open high low close preclose volume amount turn tradestatus`
2. 分钟：`date time code open high low close volume amount`

指数基准目录（默认）：

1. `D:/PythonProject/Quant/data_baostock/ak_index`
2. 文件：`hs300_price/zz500_price/zz1000_price`（csv/parquet）

股票列表文件（可选，用于二次过滤）：

1. `D:/PythonProject/Quant/data_baostock/metadata/stock_list_hs300.csv`
2. `D:/PythonProject/Quant/data_baostock/metadata/stock_list_sz50.csv`
3. `D:/PythonProject/Quant/data_baostock/metadata/stock_list_zz500.csv`
4. `D:/PythonProject/Quant/data_baostock/metadata/stock_list_all.csv`

## 5. 主流程运行

### 5.1 最小可运行示例

```powershell
python Strategy7/run_strategy7.py `
  --train-start 2024-01-01 --train-end 2024-12-31 `
  --test-start 2025-01-01 --test-end 2025-12-31 `
  --factor-freq D `
  --stock-model-type decision_tree `
  --timing-model-type none `
  --portfolio-model-type equal_weight `
  --execution-model-type ideal_fill
```

### 5.2 FactorGCL 示例

```powershell
python Strategy7/run_strategy7.py `
  --label-task return `
  --stock-model-type factor_gcl `
  --fgcl-seq-len 30 `
  --fgcl-future-look 20 `
  --fgcl-hidden-size 128 `
  --fgcl-num-layers 2 `
  --fgcl-num-factor 48 `
  --fgcl-epochs 200 `
  --fgcl-lr 9e-5 `
  --fgcl-device auto
```

### 5.3 DAFAT 示例

```powershell
python Strategy7/run_strategy7.py `
  --label-task return `
  --stock-model-type dafat `
  --dafat-seq-len 40 `
  --dafat-hidden-size 128 `
  --dafat-num-layers 2 `
  --dafat-num-heads 4 `
  --dafat-local-window 20 `
  --dafat-topk-ratio 0.30 `
  --dafat-meso-scale 5 `
  --dafat-macro-scale 20 `
  --dafat-epochs 200 `
  --dafat-lr 1e-4 `
  --dafat-device auto
```

### 5.4 仅列出因子（不加载市场数据）

```powershell
python Strategy7/run_strategy7.py --list-factors --factor-freq D
```

说明：当前版本已支持 `--list-factors` 快速路径，不再依赖本地行情数据存在。

### 5.5 股票池切换与自定义池

```powershell
# 上证50
python Strategy7/run_strategy7.py --universe sz50

# 中证500
python Strategy7/run_strategy7.py --universe zz500

# 全市场
python Strategy7/run_strategy7.py --universe all

# 全市场 + 自定义股票池（中证1000/中证2000等）
python Strategy7/run_strategy7.py `
  --universe all `
  --stock-list-path D:/PythonProject/Quant/data_baostock/metadata/stock_list_custom.csv
```

## 6. 常用参数说明（主流程）

参数闭环审计文档：

1. 详见 `docs/config_usage_audit.md`（记录每组参数在哪个模块生效、哪些参数是条件生效）。

参数分组：

1. 数据：
   `--universe --data-root --stock-list-path(--hs300-list-path 兼容) --index-root --file-format --max-files --main-board-only`
2. 因子：
   `--factor-freq --factor-list --factor-packages --custom-factor-py --label-task --lookback-days`
3. 选股模型：
   `--stock-model-type`（`decision_tree`/`factor_gcl`/`dafat`）
4. 择时模型：
   `--timing-model-type`（`none`/`volatility_regime`）
5. 组合模型：
   `--portfolio-model-type`（`equal_weight`/`dynamic_opt`）
6. 执行模型：
   `--execution-model-type`（`ideal_fill`/`realistic_fill`）
7. 回测：
   `--horizon --top-k --long-threshold --execution-scheme --fee-bps --slippage-bps`
8. 产物：
   `--output-dir --save-models`

### 6.1 分频默认量价因子库（已扩容）

当前默认因子库已按主频 `["5min", "15min", "30min", "60min", "120min", "D", "W", "M"]` 分层设计，核心思路：

1. 5min：
   - 偏微观结构与短时成交行为（冲击、反转、噪声过滤、拥挤压力）
2. 15/30/60/120min：
   - 偏中短周期趋势-波动-流动性联动，并叠加跨频桥接特征
3. D：
   - 偏日频趋势/反转/波动/流动性 + 分钟级桥接（5min/30min/60min/120min -> D）
4. W/M：
   - 偏周期趋势与风险轮动，并融合日频与分钟频向上聚合信息

每个主频的默认因子池都按“包”组织，当前统一支持：

1. `legacy_core`：历史兼容核心因子（原工程默认因子 + 关键扩展）
2. `trend`：趋势/动量包
3. `reversal`：反转/均值回复包
4. `liquidity`：流动性/成交行为包
5. `volatility`：波动率/风险结构包
6. `structure`：K线结构/微观形态包
7. `context`：上下文状态包（日频风格代理、跨阶段背景）
8. `bridge`：跨频桥接包（高频向主频聚合）
9. `multiscale`：多尺度差分包（快慢频对比）

跨频桥接因子命名规则：

1. 基础桥接列：`hf_{source_freq}_to_{target_freq}_{agg}_{base_col}`
2. 默认桥接因子族：`hf_noise_to_signal`、`hf_liquidity_pulse`、`hf_trend_carry`、`hf_fast_slow_*`

`--factor-packages` 用法：

1. 空（默认）：使用该频率全部默认包
2. `--factor-packages trend,reversal`：只启用趋势+反转包
3. `--factor-packages bridge,multiscale`：只做跨频研究
4. `--factor-packages all`：显式启用全部默认包（等价于留空）

示例：

```powershell
python Strategy7/run_strategy7.py `
  --factor-freq 30min `
  --factor-packages trend,liquidity,bridge `
  --stock-model-type factor_gcl
```

建议：

1. 先用 `--list-factors --factor-freq <freq>` 查看当前频率可用因子清单
2. 再通过 `--factor-list` 按研究主题筛选子集（例如偏趋势、偏流动性、偏跨频）

布尔参数注意：

1. `--save-models false` 可正确关闭模型落盘
2. `--save-models`（不带值）等价于 `true`

## 7. 输出目录与文件

每次运行会在 `outputs/<timestamp>/<run_tag>/` 下生成：

1. `summary_*.json`：总汇总（配置、样本规模、模型指标、回测指标）
2. `predictions_*.csv`：样本级预测与标签
3. `trades_*.csv`：调仓周期级收益序列
4. `positions_*.csv`：持仓明细
5. `curve_*.csv`：净值与超额曲线
6. `factor_ic_summary_*.csv`：因子 IC 汇总
7. `model_ic_series_*.csv`：模型分期 IC 序列
8. `backtest_curve_main_*.png / backtest_curve_excess_*.png`
9. `models/`（可选）：四类模型的持久化文件

## 8. 标签任务与评估兼容性

`label_task` 支持：

1. `direction`（二分类）
2. `return`（连续收益标签）
3. `volatility`（连续波动率标签）
4. `multi_task`（当前主训练目标按 `target_up`）

评估输出规则：

1. 二分类任务直接输出 `accuracy/precision/recall/auc`
2. 连续任务会自动派生方向标签（避免分类指标 NaN）
3. 连续任务并行输出回归指标：
   `reg_mae/reg_rmse/reg_r2/reg_pearson_corr/reg_target_std/reg_pred_std`

## 9. 因子挖掘流程

入口：

1. `python Strategy7/run_factor_mining.py ...`
2. 股票池参数同样支持：`--universe` 与 `--stock-list-path`（兼容 `--hs300-list-path`）

框架：

1. `fundamental_multiobj`：基本面参数化 + NSGA-II
2. `minute_parametric`：分钟参数化 + NSGA-III
3. `minute_parametric_plus`：分钟增强参数化 + NSGA-III
4. `ml_ensemble_alpha`：集成学习因子挖掘
5. `gplearn_symbolic_alpha`：基于 `gplearn` 的符号遗传规划挖掘
6. `custom`：用户表达式挖掘

默认素材池（新）：

1. `run_factor_mining.py` 默认会按 `--factor-freq` 注入主因子库默认因子作为挖掘素材
2. 可通过 `--factor-packages` 控制注入包（空=全部）
3. 可通过 `--disable-default-factor-materials` 关闭默认因子素材注入
4. `--index-root` 会加载 HS300 指数并注入市场上下文特征（收益、波动、回撤等）

示例（基本面）：

```powershell
python Strategy7/run_factor_mining.py `
  --framework fundamental_multiobj `
  --train-start 2021-01-01 --train-end 2023-12-31 `
  --valid-start 2024-01-01 --valid-end 2024-12-31 `
  --population-size 128 --generations 20 --top-n 20
```

示例（分钟级）：

```powershell
python Strategy7/run_factor_mining.py `
  --framework minute_parametric `
  --population-size 96 --generations 16 --top-n 20
```

示例（自定义表达式）：

```powershell
python Strategy7/run_factor_mining.py `
  --framework custom `
  --custom-spec-json ./Strategy7/docs/custom_factor_specs.json
```

示例（GP 符号遗传规划）：

```powershell
python Strategy7/run_factor_mining.py `
  --framework gplearn_symbolic_alpha `
  --gp-population-size 400 --gp-generations 12 `
  --gp-num-runs 3 --gp-n-components 24 `
  --gp-prefilter-topk 80
```

## 10. 因子 catalog 与主流程联动

挖掘结果会写入：

1. 因子值表：`data_baostock/factor_mining/.../factor_values_<freq>.parquet/csv`
2. catalog：`data_baostock/factor_mining/factor_catalog.json`

主流程默认自动加载 catalog 因子：

1. `--factor-catalog-path auto`（默认）
2. `--disable-catalog-factors`（关闭）

## 11. 插件扩展

插件模板目录：

1. `strategy7/plugins/custom_factor_template.py`
2. `strategy7/plugins/custom_source_template.py`
3. `strategy7/plugins/custom_stock_model_template.py`
4. `strategy7/plugins/custom_timing_model_template.py`
5. `strategy7/plugins/custom_portfolio_model_template.py`
6. `strategy7/plugins/custom_execution_model_template.py`

接口约定：

1. 自定义因子模块需实现 `register_factors(library)`
2. 自定义数据源模块需实现 `register_sources(registry)`
3. 自定义模型模块需实现 `build_model(cfg)`，并返回对应基类子类实例

## 12. 常见问题排查

1. `ModuleNotFoundError: pandas/torch`
   使用与你训练时一致的解释器，确认环境安装。
2. `FutureWarning: 'M' is deprecated`
   已兼容 `ME`，旧版 pandas 自动回退 `M`。
3. `model_metrics` 出现 NaN
   连续标签下请确认 `target_return/target_volatility` 非空，当前版本已做方向派生与回归并行评估。
4. `--save-models false` 仍保存模型
   当前版本已修复显式布尔解析。
5. 因子挖掘结果空
   放宽 admission 门槛，或增大样本覆盖（`max-files/population/generations`）。

## 13. 性能与复现实务建议

1. 先用 `--max-files` 做小样本烟雾测试，再跑全量
2. 固定 `--random-state` 便于复现
3. 对深度模型先用小 `epochs` 验证流程，再扩到正式训练
4. 多次挖掘后定期审阅 `factor_catalog.json`，下线失效因子

## 14. 推荐阅读

1. [FactorGCL 说明](./factor_gcl.md)
2. [DAFAT 复现与工程实现说明](./dafat_transformer.md)
3. [因子挖掘框架说明](./factor_mining_framework.md)

