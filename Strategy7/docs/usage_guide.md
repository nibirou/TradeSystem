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

基本面/财务数据目录（默认）：

1. `--fundamental-root-ak`：`D:/PythonProject/Quant/data_baostock/ak_fundamental`
2. `--fundamental-root-bsq`：`D:/PythonProject/Quant/data_baostock/baostock_fundamental_q`
3. AK 子源：`financial_indicator_em / financial_indicator_sina / financial_abstract_sina`
4. Baostock 子源：`balance / cash_flow / dupont / forecast / growth / operation / perf_express / profit`
5. 可用 `--fundamental-file-format auto|csv|parquet` 指定格式，`--disable-fundamental-data` 可关闭基本面加载

金融文本数据目录（默认）：

1. `--text-root-news`：`D:/PythonProject/Quant/data_baostock/data_em_news`
2. `--text-root-notice`：`D:/PythonProject/Quant/data_baostock/data_em_notices`
3. `--text-root-report-em`：`D:/PythonProject/Quant/data_baostock/data_em_reports`
4. `--text-root-report-iwencai`：`D:/PythonProject/Quant/data_baostock/data_iwencai_reports`
5. 可用 `--text-file-format auto|csv|parquet` 指定格式，`--disable-text-data` 可关闭文本加载
6. 文本源会自动做日频 NLP 聚合并透传到分钟/周/月频，支持与量价/基本面因子融合

金融文本数据类型与格式（基于 `data_baostock` 目录扫描）：

1. 数据源分类：
   - `data_em_news`：东方财富新闻
   - `data_em_notices`：东方财富公告
   - `data_em_reports`：东方财富研报
   - `data_iwencai_reports`：同花顺问财研报
2. 文件组织：
   - 目录按 `universe` 分层（如 `all/hs300/zz500/sz50`，实际存在的子目录因源而异）
   - 文件按股票命名：`sh_600000.csv` / `sz_000001.csv`（或 parquet）
3. 当前扫描到的 universe 覆盖（示例机数据）：
   - `data_em_news`：`hs300/zz500`
   - `data_em_notices`：`all/hs300/zz500`
   - `data_em_reports`：`hs300`
   - `data_iwencai_reports`：`hs300/zz500`
4. 加载器会按 `--universe` 优先读取，并自动 fallback 到 `all -> hs300 -> zz500 -> sz50`，因此可兼容目录不齐全的情况。
5. 字段映射（核心）：
   - `news`：时间列优先 `发布时间`；代码列优先 `关键词/股票代码`；文本列优先 `新闻标题 + 新闻正文/新闻内容`
   - `notice`：时间列优先 `公告发布时间/公告日期`；代码列优先 `股票代码`；文本列优先 `公告标题 + 公告正文/公告摘要`
   - `em_report`：时间列优先 `研报发布时间/发布日期`；代码列优先 `股票代码`；文本列优先 `研报标题 + 研报正文`；评级变化优先 `评级变动`
   - `iwencai`：时间列优先 `publish_time`；若无代码列则使用文件名反推股票代码；文本列优先 `title + content`
6. 编码与格式：
   - 默认按 UTF-8 读取（`csv/parquet`）；通过 `--text-file-format` 可显式限制格式。
   - 同一路径下支持 csv/parquet 混合，`auto` 会按文件后缀自动识别。

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
清单每行会输出以下字段，便于直接阅读和解释：

1. `factor`：英文因子名（程序主键）
2. `freq`：因子频率
3. `factor_package`：主因子包分类（与 `--factor-packages` 一致）
4. `factor_packages`：该因子所属全部因子包（逗号分隔）
5. `name_cn`：中文名称（按因子结构自动生成）
6. `meaning_cn`：中文含义（该因子想刻画的市场/基本面信息）
7. `formula_cn`：计算公式中文解释（模板因子会给出标准公式）
8. `description`：原始英文说明（便于交叉核对）

同时在清单末尾会输出摘要：

1. `Fundamental Factor Coverage`：对应频率基本面因子的 `expected/listed/missing` 覆盖情况
2. `Factor Package Summary`：按因子包统计量价、基本面、金融文本三大类数量

`--list-factors` 支持导出（可选）：

1. `--export-factor-list`：开启导出
2. `--factor-list-export-format csv|json|markdown`：导出格式
3. `--factor-list-export-path <file>`：自定义导出文件路径（可选）

示例：

```powershell
# 自动写入 output_dir/factor_lists/
python Strategy7/run_strategy7.py `
  --list-factors --factor-freq 30min `
  --export-factor-list --factor-list-export-format csv

# 指定导出路径（Markdown）
python Strategy7/run_strategy7.py `
  --list-factors --factor-freq D `
  --export-factor-list --factor-list-export-format markdown `
  --factor-list-export-path D:/PythonProject/Quant/TradeSystem/Strategy7/outputs/factor_list_D.md
```

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

### 5.6 组合优化 + 现实执行完整示例

```powershell
python Strategy7/run_strategy7.py `
  --train-start 2024-01-01 --train-end 2024-12-31 `
  --test-start 2025-01-01 --test-end 2025-12-31 `
  --factor-freq D `
  --stock-model-type decision_tree `
  --timing-model-type volatility_regime `
  --timing-vol-threshold 0.0 `
  --timing-momentum-threshold 0.0 `
  --portfolio-model-type dynamic_opt `
  --opt-max-weight 0.20 `
  --opt-max-turnover 1.00 `
  --opt-risk-aversion 1.20 `
  --opt-style-penalty 0.80 `
  --execution-model-type realistic_fill `
  --max-participation-rate 0.12 `
  --base-fill-rate 0.92 `
  --latency-bars 1 `
  --execution-scheme vwap30_vwap30 `
  --save-models true
```

### 5.7 主流程“全功能组合”示例（catalog + 因子缓存 + 快照）

```powershell
python Strategy7/run_strategy7.py `
  --train-start 2024-01-01 --train-end 2024-12-31 `
  --test-start 2025-01-01 --test-end 2025-12-31 `
  --factor-freq D `
  --factor-packages trend,liquidity,fund_growth,mined_fusion `
  --factor-catalog-path auto `
  --enable-factor-value-store true `
  --factor-value-store-root auto `
  --factor-value-store-format parquet `
  --auto-export-factor-snapshot true `
  --stock-model-type factor_gcl `
  --label-task return `
  --fgcl-epochs 120 `
  --fgcl-device auto `
  --output-dir auto `
  --save-models true
```

### 5.8 基于已有模型直接回测（不重新训练）

当前主入口已支持双模式：

1. `--model-run-mode train`（默认）：训练新模型后回测
2. `--model-run-mode load`：加载已有模型后直接回测

`load` 模式下的模型路径解析优先级：

1. 显式传入单模型路径（`--stock-model-path/--timing-model-path/...`）
2. `--model-summary-json` 自动读取 `summary_*.json` 里的 `outputs.model_files`
3. `--models-load-dir` + 可选 `--models-load-run-tag` 自动匹配模型文件

`load` 模式参数冲突与约束（新增）：

1. `load + FE` 由 `--load-fe-mode strict|refit|off` 控制，不再是固定阻断：
   - `strict`：按 `--model-summary-json` 中记录的 FE 结果回放（要求 `notes.feature_engineering_summary` 存在且启用；当前不支持 `pca` 回放）。
   - `refit`：按当前样本重新拟合 FE（默认）。
   - `off`：即使 `--enable-factor-engineering true` 也跳过 FE。
2. `strict` 模式必须提供 `--model-summary-json`，否则会显式报错阻断（避免语义错位）。
3. `train` 模式下若传了 `--model-summary-json/--models-load-dir/--*-model-path`，主流程会提示并忽略这些 load 专用参数。
4. 文件后缀建议：
   - `decision_tree`：`.pkl/.pickle`
   - `factor_gcl/dafat`：`.pt/.pth`
   - `volatility_regime`：`.pkl/.json`
   - `dynamic_opt/realistic_fill`：`.pkl/.json`
5. 当对应模型类型为默认无文件实现时，会忽略其路径参数：
   - `timing_model_type=none` 时忽略 `--timing-model-path`
   - `portfolio_model_type=equal_weight` 时忽略 `--portfolio-model-path`
   - `execution_model_type=ideal_fill` 时忽略 `--execution-model-path`

推荐方式（最稳）：直接用历史运行的 `summary_*.json` 回放：

```powershell
python Strategy7/run_strategy7.py `
  --model-run-mode load `
  --load-fe-mode refit `
  --model-summary-json D:/PythonProject/Quant/TradeSystem/Strategy7/outputs/20260404_131548/tr240101241231_te250101251231_D_h5_k10_vwap30vwap30_all_eqw/summary_D_h5_allboards_equal_weight_e89a96a3e4.json `
  --stock-model-type factor_gcl `
  --timing-model-type none `
  --portfolio-model-type equal_weight `
  --execution-model-type ideal_fill `
  --factor-freq D `
  --horizon 5 `
  --top-k 10 `
  --save-models false
```

显式指定四类模型文件（适合做消融实验/控制变量）：

```powershell
python Strategy7/run_strategy7.py `
  --model-run-mode load `
  --load-fe-mode off `
  --stock-model-type decision_tree `
  --timing-model-type volatility_regime `
  --portfolio-model-type dynamic_opt `
  --execution-model-type realistic_fill `
  --stock-model-path D:/.../models/stock_model_tree_XXX.pkl `
  --timing-model-path D:/.../models/timing_vol_regime_XXX.pkl `
  --portfolio-model-path D:/.../models/portfolio_dynamic_XXX.pkl `
  --execution-model-path D:/.../models/execution_realistic_XXX.pkl `
  --factor-freq D --horizon 5 --top-k 10 `
  --save-models false
```

仅提供模型目录自动匹配（run_tag 可选）：

```powershell
python Strategy7/run_strategy7.py `
  --model-run-mode load `
  --load-fe-mode off `
  --stock-model-type decision_tree `
  --timing-model-type none `
  --portfolio-model-type equal_weight `
  --execution-model-type ideal_fill `
  --models-load-dir D:/PythonProject/Quant/TradeSystem/Strategy7/outputs/_tmp_freq_guard/pipeline_W/models `
  --models-load-run-tag W_h3_allboards_equal_weight_edb61102ec `
  --factor-freq W --horizon 3 --top-k 10
```

### 5.9 next bar 快速推理（四类模型联动）

主入口支持在回测同时额外输出“最新信号时点”的组合推理结果（选股分数 + 择时暴露 + 组合权重 + 执行填单）：

1. `--enable-next-bar-inference true`：开启
2. `--inference-top-k`：输出候选数量（按 `pred_score` 排序）

示例：

```powershell
python Strategy7/run_strategy7.py `
  --model-run-mode load `
  --model-summary-json D:/.../summary_XXX.json `
  --enable-next-bar-inference true `
  --inference-top-k 20 `
  --save-models false
```

会新增产物：

1. `next_bar_candidates_*.csv`：最新信号时点候选/入选/目标权重/执行权重/成交率
2. `next_bar_summary_*.json`：最新信号时点的暴露、诊断信息与统计摘要

### 5.10 全功能模板库（scripts/v2）与回归建议

新版模板目录：`Strategy7/scripts/v2`。  
当前模板规模：

1. 主流程 `run_strategy7_v2_*.ps1` 共 `01~19`
2. 挖掘入口 `run_factor_mining_v2_*.ps1` 共 `01~14`

完整模板索引与每个模板覆盖点详见：`Strategy7/scripts/v2/README.md`。

主流程模板（`run_strategy7_v2_*.ps1`）速览：

1. `01~03`：基础/增强/自定义四模型训练回测
2. `04~07`：四种 load 路径（summary refit、summary strict、models_dir off、自定义 load）
3. `08~09`：深度模型冒烟（FactorGCL、DAFAT）
4. `10~11`：`--list-factors` 导出、factor value store build-only
5. `12~19`：扩展链路（W/M/30min、price-only+main_board、custom factor plugin、value store hydrate、显式四路径 load、30min JSON 导出）

挖掘模板（`run_factor_mining_v2_*.ps1`）速览：

1. `01~06`：六类挖掘框架（fundamental/minute/minute+/ml/gp/custom）
2. `07`：`--list-factors` 导出
3. `08~14`：扩展链路（materials+FE+value store、custom-spec-json、30min、price-only+main_board、markdown 导出、disable-default-materials + factor-list、custom factor plugin 列表）

按研究目的选模板（极简索引）：

1. 快速基线：`run_strategy7_v2_01_train_tree_baseline.ps1`
2. 训练全链路（FE+动态组合+realistic+next-bar）：`run_strategy7_v2_02_train_tree_fe_dynamic.ps1`
3. 复现已有模型（summary）：`run_strategy7_v2_04` / `run_strategy7_v2_05`
4. 复现已有模型（models_dir / 显式四路径）：`run_strategy7_v2_06` / `run_strategy7_v2_17`
5. 自定义模型与因子插件：`run_strategy7_v2_03` / `run_strategy7_v2_07` / `run_strategy7_v2_15` / `run_factor_mining_v2_14`
6. 多频与任务扩展（30min/W/M、volatility/multi_task）：`run_strategy7_v2_12` / `13` / `18` / `19` / `run_factor_mining_v2_10`
7. 数据裁剪实验（price-only/main_board）：`run_strategy7_v2_14` / `run_factor_mining_v2_11`
8. 值仓库路径（build-only/hydrate/materials 联动）：`run_strategy7_v2_11` / `16` / `run_factor_mining_v2_08`
9. 挖掘参数扩展（custom spec、禁默认素材+指定因子）：`run_factor_mining_v2_09` / `13`
10. 因子清单导出（json/markdown）：`run_strategy7_v2_10` / `19` / `run_factor_mining_v2_07` / `12`

建议执行方式（`env_quant`）：

```powershell
# 单模板运行（示例）
powershell -ExecutionPolicy Bypass -File Strategy7/scripts/v2/run_strategy7_v2_02_train_tree_fe_dynamic.ps1

# load+models_dir 模板需要显式传模型目录（示例）
powershell -ExecutionPolicy Bypass -File Strategy7/scripts/v2/run_strategy7_v2_06_load_from_models_dir_off.ps1 `
  -ModelsLoadDir D:/PythonProject/Quant/TradeSystem/Strategy7/outputs/smoke_v2/run_strategy7_01_train_tree/models `
  -ModelsLoadRunTag 510c1cd320

# 核心 smoke 套件
powershell -ExecutionPolicy Bypass -File Strategy7/scripts/v2/run_smoke_suite_v2.ps1

# 核心 + 扩展模板全覆盖
powershell -ExecutionPolicy Bypass -File Strategy7/scripts/v2/run_smoke_suite_v2.ps1 -IncludeExtended

# 按需跳过深度模型/挖掘
powershell -ExecutionPolicy Bypass -File Strategy7/scripts/v2/run_smoke_suite_v2.ps1 -IncludeExtended -SkipDeepModels
powershell -ExecutionPolicy Bypass -File Strategy7/scripts/v2/run_smoke_suite_v2.ps1 -IncludeExtended -SkipMining
```

## 6. 常用参数说明（主流程）

参数闭环审计文档：

1. 详见 `docs/config_usage_audit.md`（记录每组参数在哪个模块生效、哪些参数是条件生效）。

参数分组：

1. 数据：
   `--universe --data-root --stock-list-path(--hs300-list-path 兼容) --index-root --file-format --max-files --main-board-only --fundamental-root-ak --fundamental-root-bsq --fundamental-file-format --disable-fundamental-data --text-root-news --text-root-notice --text-root-report-em --text-root-report-iwencai --text-file-format --disable-text-data`
2. 因子：
   `--factor-freq --factor-list --factor-packages --custom-factor-py --list-factors --auto-export-factor-snapshot --export-factor-list --factor-list-export-format --factor-list-export-path --label-task --lookback-days --enable-factor-engineering --fe-min-coverage --fe-min-std --fe-corr-threshold --fe-preselect-top-n --fe-min-factors --fe-max-factors --fe-orth-method --fe-pca-variance-ratio --fe-pca-max-components --enable-factor-value-store --factor-value-store-root --factor-value-store-format --factor-value-store-build-all --factor-value-store-build-only --factor-value-store-chunk-size`
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
8. 模型运行模式与推理：
   `--model-run-mode --load-fe-mode --model-summary-json --models-load-dir --models-load-run-tag --stock-model-path --timing-model-path --portfolio-model-path --execution-model-path --enable-next-bar-inference --inference-top-k`
9. 产物：
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

1. `trend`：趋势/动量包
2. `reversal`：反转/均值回复包
3. `liquidity`：流动性/成交行为包
4. `volatility`：波动率/风险结构包
5. `structure`：K线结构/微观形态包
6. `context`：上下文状态包（日频风格代理、跨阶段背景）
7. `flow`：资金流/量价协同类
8. `crowding`：拥挤度类
9. `price_action`：价格行为/K线形态类
10. `intraday_signature`：日内签名类
11. `intraday_micro`：日内微观结构类
12. `period_signature`：周月周期签名类
13. `oscillator`：摆动指标类
14. `overnight`：隔夜效应类
15. `multi_freq`：跨频主桥接族（`hf_*`）类
16. `bridge`：跨频桥接包（高频向主频聚合）
17. `multiscale`：多尺度差分包（快慢频对比）
18. `text_sentiment`：金融文本情绪类（>=30 初始因子）
19. `text_attention`：金融文本关注度类（>=30 初始因子）
20. `text_event`：金融文本事件风险类（>=30 初始因子）
21. `text_topic`：金融文本主题类（>=30 初始因子）
22. `text_fusion`：金融文本融合类（文本×量价×基本面，>=30 初始因子）
23. `fund_growth`：基本面成长类（>=100 初始因子）
24. `fund_valuation`：基本面估值类（>=100 初始因子）
25. `fund_profitability`：基本面盈利能力类（>=100 初始因子）
26. `fund_quality`：基本面质量类（>=100 初始因子）
27. `fund_leverage`：基本面杠杆/偿债类（>=100 初始因子）
28. `fund_cashflow`：基本面现金流类（>=100 初始因子）
29. `fund_efficiency`：基本面运营效率类（>=100 初始因子）
30. `fund_expectation`：预期/业绩预告类（>=100 初始因子）
31. `fund_hf_fusion`：基本面-高频量价融合类（>=100 初始因子）
32. `mined_price_volume`：挖掘因子-量价类
33. `mined_fundamental`：挖掘因子-基本面类
34. `mined_text`：挖掘因子-金融文本类
35. `mined_fusion`：挖掘因子-融合类（量价/基本面/文本混合）
36. `mined_other`：挖掘因子-其他类
37. `mined_custom`：挖掘因子-自定义类

说明：部分包只在特定频率有非空因子，例如 `period_signature` 仅在 `W/M` 生效，`intraday_signature` 仅在分钟频生效。
挖掘因子还支持维度标签包：`mined_fw_*`（挖掘框架）、`mined_universe_*`（股票池）、`mined_freq_*`（频率）、`mined_materialpkg_*`（素材包）。
历史 catalog 中的 `catalog_custom` 已统一并入 `mined_custom`（兼容读取时自动映射）。

补充说明：当前工程已统一为“仅使用 factor package 分类”，因子清单不再单独展示 `category` 列。
你在清单中看到的分类与 `--factor-packages` 参数使用的是同一套标准。

跨频桥接因子命名规则：

1. 基础桥接列：`hf_{source_freq}_to_{target_freq}_{agg}_{base_col}`
2. 默认桥接因子族：`hf_noise_to_signal`、`hf_liquidity_pulse`、`hf_trend_carry`、`hf_fast_slow_*`

`--factor-packages` 用法：

1. 空（默认）：使用该频率全部默认包
2. `--factor-packages trend,reversal`：只启用趋势+反转包
3. `--factor-packages bridge,multiscale`：只做跨频研究
4. `--factor-packages all`：显式启用全部默认包（等价于留空）
5. `--factor-packages fund_growth,fund_quality`：只做成长+质量基本面研究
6. `--factor-packages text_sentiment,text_event`：只做金融文本情绪+事件研究
7. `--factor-packages text_fusion,fund_hf_fusion,bridge`：做文本/基本面与高频桥接融合研究
8. `--factor-packages mined_fusion`：只启用 catalog 中“挖掘融合类”因子
9. `--factor-packages mined_fw_gpl,mined_universe_hs300`：只启用特定挖掘框架/股票池标签下的挖掘因子

示例：

```powershell
python Strategy7/run_strategy7.py `
  --factor-freq 30min `
  --factor-packages trend,liquidity,bridge `
  --stock-model-type factor_gcl
```

建议：

1. 先用 `--list-factors --factor-freq <freq>` 查看当前频率可用因子清单
2. 再通过 `--factor-list` 按研究主题筛选子集（例如偏趋势、偏流动性、偏跨频、偏文本情绪）

### 6.2 训练前因子特征工程（选股/择时/组合/执行共享）

当因子数量很大（上千）时，建议在训练和回测前开启特征工程：

1. `--enable-factor-engineering true`：启用训练期拟合、测试期仅变换的特征工程流程
2. `--fe-min-coverage`：先按训练期覆盖率过滤低可用因子
3. `--fe-min-std`：再过滤近常数因子
4. `--fe-corr-threshold`：对高相关因子做贪心去冗余（Spearman）
5. `--fe-preselect-top-n`：去冗余前先按质量分预筛，降低大规模相关矩阵开销
6. `--fe-orth-method pca`：可选 PCA 正交降维；`none` 时仅做筛选不过投影
7. `--fe-pca-variance-ratio` / `--fe-pca-max-components`：控制 PCA 主成分数量

示例：

```powershell
python Strategy7/run_strategy7.py `
  --factor-freq D `
  --enable-factor-engineering true `
  --fe-min-coverage 0.75 `
  --fe-corr-threshold 0.90 `
  --fe-max-factors 500 `
  --fe-orth-method none
```

自动因子快照导出（新增，默认关闭）：

1. 两个入口 `run_strategy7.py` / `run_factor_mining.py` 通过 `--auto-export-factor-snapshot` 显式开启导出
2. 导出根目录：`Strategy7/outputs/factor_snapshots/<entrypoint>/<timestamp_freq_tag>/`
3. 目录结构按因子类型与包分类：`by_group/<price_volume|fundamental|text|mined>/<factor_package>/`
4. 每级都会包含：
   - `all.csv`（该分类全部因子）
   - `used.csv`（该分类本次使用因子）
5. 根目录额外包含：
   - `all_factors.csv`
   - `used_factors.csv`
   - `snapshot_summary.json`（统计信息）
   - `snapshot_overview.md`（直观对比概览）
6. 之前的 `--export-factor-list` / `--factor-list-export-format` 功能继续保留（手动导出清单）

因子值缓存仓库（新增，默认关闭）：

1. 主入口可通过 `--enable-factor-value-store` 开启“优先读缓存，缺失再计算并回写”
2. 默认仓库根目录（`--factor-value-store-root auto`）：`<data_baostock>/factor_value_store`
3. 目录结构：`<factor_freq>/by_group/<price_volume|fundamental|text|mined>/<factor_package>/<code>.parquet|csv`
4. 可按当前频率完整因子清单批量构建：`--factor-value-store-build-all true`
5. 仅构建缓存后退出（不训练/不回测）：`--factor-value-store-build-only true`
6. 分块大小：`--factor-value-store-chunk-size`（默认 `64`）
7. 每次写入会自动更新跨度汇总：`<factor_freq>/factor_span_summary.csv`
8. 若 `data_baostock` 目录不可写，请显式指定 `--factor-value-store-root` 到可写路径

推荐两步法（主入口）：

1. 先离线构建当前频率完整缓存：

```powershell
python Strategy7/run_strategy7.py `
  --factor-freq D `
  --enable-factor-value-store true `
  --factor-value-store-build-all true `
  --factor-value-store-build-only true
```

2. 再在研究运行中复用缓存（只算缺失增量）：

```powershell
python Strategy7/run_strategy7.py `
  --factor-freq D `
  --enable-factor-value-store true `
  --factor-packages trend,liquidity,fund_growth
```

`--factor-list` 构建示例（以 30min 为例）：

1. 先查看清单并导出：

```powershell
python Strategy7/run_strategy7.py `
  --list-factors --factor-freq 30min `
  --export-factor-list --factor-list-export-format csv
```

2. 在导出的 `factor_list_30min_*.csv` 里按 `factor_package` 选取目标主题因子（例如 `trend` + `liquidity` + `multi_freq`），拼成逗号分隔字符串：

```text
ret_1,ret_3,ma_gap_12,amount_ratio_12,liquidity_stress,hf_noise_to_signal,hf_fast_slow_trend_diff
```

3. 回填到 `--factor-list` 运行：

```powershell
python Strategy7/run_strategy7.py `
  --factor-freq 30min `
  --factor-list ret_1,ret_3,ma_gap_12,amount_ratio_12,liquidity_stress,hf_noise_to_signal,hf_fast_slow_trend_diff `
  --stock-model-type factor_gcl
```

布尔参数注意：

1. `--save-models false` 可正确关闭模型落盘
2. `--save-models`（不带值）等价于 `true`
3. `--enable-next-bar-inference`（不带值）等价于 `true`

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
10. `next_bar_candidates_*.csv`（可选）：最新信号时点快速推理候选结果
11. `next_bar_summary_*.json`（可选）：最新信号时点推理摘要

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
3. 新增与主流程对齐参数：
   - `--factor-list`：显式指定挖掘素材因子名列表
   - `--custom-factor-py`：加载自定义因子插件作为可选素材
   - `--list-factors` / `--export-factor-list`：列出并导出挖掘可用因子清单
   - `--auto-export-factor-snapshot`：可选导出“全部因子 vs 本次使用因子”快照（默认关闭）
   - `--factor-catalog-path` / `--disable-catalog-factors`：控制 catalog 因子加载与过滤
   - `--enable-factor-value-store --factor-value-store-root --factor-value-store-format`：启用素材因子值缓存复用（命中则直读，缺失增量计算）

框架：

1. `fundamental_multiobj`：基本面参数化 + NSGA-II
2. `minute_parametric`：分钟参数化 + NSGA-III
3. `minute_parametric_plus`：分钟增强参数化 + NSGA-III
4. `ml_ensemble_alpha`：集成学习因子挖掘
5. `gplearn_symbolic_alpha`：基于 `gplearn` 的符号遗传规划挖掘
6. `custom`：自定义评估模式（优先按 `--factor-list` 逐因子评估；兼容 `--custom-spec-json` 表达式）

默认素材池（新）：

1. `run_factor_mining.py` 默认会按 `--factor-freq` 注入主因子库默认因子作为挖掘素材
2. 可通过 `--factor-packages` 控制注入包（空=全部）
3. 可通过 `--disable-default-factor-materials` 关闭默认因子素材注入
4. `--index-root` 会加载 HS300 指数并注入市场上下文特征（收益、波动、回撤等）
5. 基本面素材默认参与面板构建（AK+Baostock）；可通过 `--disable-fundamental-data` 关闭
6. 金融文本素材默认参与面板构建（news/notice/report）；可通过 `--disable-text-data` 关闭
7. catalog 因子素材支持按 `--factor-packages` 过滤（如 `mined_fusion`）或按 `--factor-list` 精确点名
8. 当 `--factor-list` 非空时，挖掘素材将按显式名单解析（默认包仅作为补充可选来源）
9. 可选素材特征工程（默认关闭）：`--enable-material-feature-engineering`
10. 素材特征工程参数：`--material-fe-min-coverage --material-fe-min-std --material-fe-corr-threshold --material-fe-preselect-top-n --material-fe-min-factors --material-fe-max-factors`
11. 素材特征工程只在训练期拟合筛选规则，并在挖掘前裁剪素材列，避免信息泄漏

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

示例（分钟增强参数化）：

```powershell
python Strategy7/run_factor_mining.py `
  --framework minute_parametric_plus `
  --factor-freq 30min `
  --train-start 2021-01-01 --train-end 2023-12-31 `
  --valid-start 2024-01-01 --valid-end 2024-12-31 `
  --population-size 96 --generations 16 --top-n 20
```

示例（ML 集成挖掘）：

```powershell
python Strategy7/run_factor_mining.py `
  --framework ml_ensemble_alpha `
  --factor-freq D `
  --factor-packages trend,reversal,liquidity,flow `
  --train-start 2021-01-01 --train-end 2023-12-31 `
  --valid-start 2024-01-01 --valid-end 2024-12-31 `
  --ml-population-size 48 --ml-generations 10 `
  --ml-model-pool rf,et,hgbt `
  --ml-prefilter-topk 80 `
  --ml-feature-min 10 --ml-feature-max 36 `
  --ml-train-sample-frac 0.40 `
  --ml-max-train-rows 220000 `
  --ml-num-jobs -1
```

示例（custom 逐因子评估，推荐）：

```powershell
python Strategy7/run_factor_mining.py `
  --framework custom `
  --factor-freq 30min `
  --custom-factor-py ./Strategy7/strategy7/plugins/custom_factor_template.py `
  --factor-list custom_ret_accel_5_20,custom_intraday_mr
```

示例（自定义表达式 JSON，兼容模式）：

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

示例（仅查看挖掘可用因子清单并导出）：

```powershell
python Strategy7/run_factor_mining.py `
  --list-factors --factor-freq 30min `
  --factor-catalog-path D:/PythonProject/Quant/TradeSystem/Strategy7/outputs/mining_test/factor_mining/factor_catalog.json `
  --factor-packages mined_fusion `
  --export-factor-list --factor-list-export-format csv
```

说明：

1. 若不显式传 `--factor-list-export-path`，导出默认目录为：`<factor-store-root>/factor_mining/factor_lists/`。
2. `--execution-scheme` 与主流程保持一致，可选值：`open5_open5`、`vwap30_vwap30`、`open5_twap_last30`、`daily_close_daily_close`。

示例（显式指定挖掘素材因子）：

```powershell
python Strategy7/run_factor_mining.py `
  --framework fundamental_multiobj `
  --factor-freq 30min `
  --factor-list amount_ratio_12,ret_12,rv_12 `
  --factor-store-root D:/PythonProject/Quant/TradeSystem/Strategy7/outputs/mining_test
```

示例（挖掘入口复用因子值缓存）：

```powershell
python Strategy7/run_factor_mining.py `
  --framework fundamental_multiobj `
  --factor-freq D `
  --enable-factor-value-store `
  --factor-value-store-root auto `
  --factor-packages fund_growth,fund_quality
```

示例（挖掘前素材去冗余，推荐在素材很多时开启）：

```powershell
python Strategy7/run_factor_mining.py `
  --framework ml_ensemble_alpha `
  --factor-freq D `
  --enable-material-feature-engineering `
  --material-fe-min-coverage 0.75 `
  --material-fe-corr-threshold 0.95 `
  --material-fe-max-factors 600
```

## 10. 因子 catalog 与主流程联动

挖掘结果会写入：

1. 因子值表：`data_baostock/factor_mining/.../factor_values_<freq>.parquet/csv`
2. catalog：`data_baostock/factor_mining/factor_catalog.json`

主流程默认自动加载 catalog 因子：

1. `--factor-catalog-path auto`（默认）
2. `--disable-catalog-factors`（关闭）

主流程验证示例：

```powershell
# 使用 catalog 因子（默认）
python Strategy7/run_strategy7.py `
  --factor-freq D `
  --factor-packages mined_fusion `
  --factor-catalog-path auto

# 显式关闭 catalog 因子
python Strategy7/run_strategy7.py `
  --factor-freq D `
  --disable-catalog-factors
```

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
2. 自定义数据源模块需实现 `register_sources(registry)`（兼容模式，建议优先迁移到自定义因子插件）
3. 自定义模型模块需实现 `build_model(cfg)`，并返回对应基类子类实例
4. 若要支持 `--model-run-mode load`，自定义模型插件还需实现 `load_model(cfg, model_path)`（返回对应模型实例）

自定义因子插件新增推荐能力：

1. 可在插件中直接读取外部 CSV/Parquet 并注册为因子
2. 参考工具函数：`strategy7.factors.base.register_external_factor_table`

参数迁移建议：

1. `--extra-factor-paths`、`--extra-source-module` 仍可用，但已标记为 deprecated
2. 新增外部因子建议统一在 `--custom-factor-py` 中注册（可直接从外部 CSV/Parquet 注册）

插件调用示例：

```powershell
# 自定义因子插件
python Strategy7/run_strategy7.py `
  --custom-factor-py Strategy7/strategy7/plugins/custom_factor_template.py

# 自定义选股模型
python Strategy7/run_strategy7.py `
  --custom-stock-model-py Strategy7/strategy7/plugins/custom_stock_model_template.py

# 自定义择时模型
python Strategy7/run_strategy7.py `
  --custom-timing-model-py Strategy7/strategy7/plugins/custom_timing_model_template.py

# 自定义组合模型
python Strategy7/run_strategy7.py `
  --custom-portfolio-model-py Strategy7/strategy7/plugins/custom_portfolio_model_template.py

# 自定义执行模型
python Strategy7/run_strategy7.py `
  --custom-execution-model-py Strategy7/strategy7/plugins/custom_execution_model_template.py
```

自定义模型在 `load` 模式下的调用示例（四类模型可独立混搭）：

```powershell
python Strategy7/run_strategy7.py `
  --model-run-mode load `
  --custom-stock-model-py Strategy7/strategy7/plugins/custom_stock_model_template.py `
  --custom-timing-model-py Strategy7/strategy7/plugins/custom_timing_model_template.py `
  --custom-portfolio-model-py Strategy7/strategy7/plugins/custom_portfolio_model_template.py `
  --custom-execution-model-py Strategy7/strategy7/plugins/custom_execution_model_template.py `
  --stock-model-path D:/.../custom_stock_constant_xxx.json `
  --timing-model-path D:/.../custom_timing_fixed_xxx.json `
  --portfolio-model-path D:/.../custom_portfolio_top1_xxx.json `
  --execution-model-path D:/.../custom_execution_half_fill_xxx.json `
  --factor-freq D --horizon 5 --top-k 10
```

说明：

1. 工程内四个自定义模型模板文件都已提供 `build_model(cfg)` 和 `load_model(cfg, model_path)` 双接口示例。
2. 若你只想替换其中一个模块做消融，其余三个模块可继续使用内置模型类型。

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
6. `model_run_mode=load` 报“missing factor cols”
   说明当前运行可用因子与模型训练时不一致；优先使用原始 `summary_*.json` 回放，或对齐 `--factor-freq/--factor-packages/--factor-list`。
7. next bar 推理文件未生成
   检查 `--enable-next-bar-inference true`，并确认最新信号时点存在可用样本（`code/time/factor` 不为空）。
8. `model_run_mode=load` 且 `enable_factor_engineering=true` 的 FE 行为不符合预期
   显式设置 `--load-fe-mode`：`strict`（按 summary 回放）、`refit`（当前样本重拟合）、`off`（跳过 FE）。若用 `strict`，必须传 `--model-summary-json` 且源实验 FE 非 PCA。

## 13. 性能与复现实务建议

1. 先用 `--max-files` 做小样本烟雾测试，再跑全量
2. 固定 `--random-state` 便于复现
3. 对深度模型先用小 `epochs` 验证流程，再扩到正式训练
4. 多次挖掘后定期审阅 `factor_catalog.json`，下线失效因子

## 14. 推荐阅读

1. [FactorGCL 说明](./factor_gcl.md)
2. [DAFAT 复现与工程实现说明](./dafat_transformer.md)
3. [因子挖掘框架说明](./factor_mining_framework.md)

