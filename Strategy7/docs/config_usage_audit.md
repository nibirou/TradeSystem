# Strategy7 配置项使用审计（run_strategy7 / run_factor_mining）

更新时间：2026-04-23  
审计目标：确认配置项“可配置即可生效”，排查空转参数并修复。

## 1. 审计结论

1. `run_strategy7.py`（通过 `strategy7/config.py`）的参数链路已闭环：  
   - CLI 参数 -> `build_run_config` -> `RunConfig` -> `pipeline/模型/回测` 全链路生效。  
2. `run_factor_mining.py` 已完成闭环修复：  
   - 发现并修复了 `--index-root` 先前未实际参与挖掘流程的问题。  
   - 现在 `--index-root` 会加载 HS300 指数并注入市场上下文素材列。  
3. 条件生效参数说明：  
   - 某些参数仅在对应模型/框架下生效（如 `launch_boost_*`、`fgcl_*`、`dafat_*`、`gp_*`、`ml_*`），这是设计行为，不是空转。

## 2. 本次修复项（重点）

## 2.1 `run_factor_mining.py` 的 `--index-root` 空转修复

修复前：
1. `--index-root` 仅进入 `DataConfig`，但挖掘核心未消费该信息。  

修复后：
1. 挖掘流程会读取 `index_root` 下 HS300 指数数据。  
2. 自动生成并注入市场上下文列（如 `mkt_hs300_ret_1d/5d/20d`、`mkt_hs300_vol_20d` 等）。  
3. 注入后的列进入挖掘素材池，可参与表达式/模型组合。  
4. `mining_summary.json` 增加 `index_context_cols`，用于可追溯审计。

实现位置：
1. `run_factor_mining.py`：指数加载与上下文 merge。  
2. `strategy7/mining/runner.py`：新增 `index_context_cols` 到配置与摘要。  

## 2.2 额外稳健性修复

1. `run_factor_mining.py` 增加 `max_files` 正数校验，避免非法负值导致非预期截断行为。  

## 2.3 2026-04-14 续做复核

1. 自动扫描 `run_factor_mining.py` 的 CLI 参数定义与 `args.xxx` 消费路径：64/64 全量消费。  
2. 自动扫描 `strategy7/config.py`：  
   - `--log-level/--quiet/--verbose` 属于入口运行时日志控制，在 `run_strategy7.py` 中生效（不进入 `RunConfig`）。  
   - `--output-dir` 通过 `resolve_output_dir(args)` 消费（函数级消费，不是 `args.output_dir` 直连赋值）。  
3. 结论：本次复核未发现新的“定义了但完全未使用”的配置项。

## 2.4 2026-04-22 增量优化

1. `run_factor_mining.py` 新增通用边界校验（非框架专属）：  
   - `horizon/population-size/generations/elite-size/top-n` 正数约束  
   - `elite-size <= population-size`  
   - `mutation-rate/crossover-rate` 范围 `[0,1]`  
   - `corr-threshold/material-fe-corr-threshold` 范围 `[0,1)`  
   - `top-frac` 范围 `(0,1]`  
   - `min-cross-section > 1`
2. `run_factor_mining.py` 将 `ml_*` 与 `gp_*` 校验改为“按框架条件触发”：  
   - `framework=ml_ensemble_alpha` 时校验 `ml_*`  
   - `framework=gplearn_symbolic_alpha` 时校验 `gp_*`  
   - 避免非对应框架运行时被无关参数误拦截
3. `run_factor_mining.py` 的 `--execution-scheme` 接入枚举校验（与主流程一致）：  
   - `open5_open5 / vwap30_vwap30 / open5_twap_last30 / daily_close_daily_close`
4. `run_factor_mining.py --list-factors --export-factor-list` 默认导出目录修正为：  
   - `<factor-store-root>/factor_mining/factor_lists/`  
   - 与参数帮助文案和使用手册保持一致

## 3. run_strategy7 参数生效映射（分组）

1. 数据参数（`universe/data-root/stock-list-path(兼容hs300-list-path)/index-root/file-format/max-files/main-board-only`）  
生效模块：`data/loaders.py`、`pipeline/runner.py`、`load_index_benchmark_data`。

2. 因子参数（`factor-freq/factor-list/factor-packages/custom-factor-py/list-factors/lookback-days/min-ic-cross-section/label-task`）  
生效模块：`pipeline/runner.py`、`factors/defaults.py`、`factors/labeling.py`、`backtest` 评估。

3. 选股模型参数（`stock-model-type` + `launch_boost_*` + `fgcl_*` + `dafat_*` + 决策树参数）  
生效模块：`models/stock_selection/factory.py` -> 各模型实现。

4. 择时参数（`timing-model-type/timing-*`）  
生效模块：`models/timing/factory.py`、`models/timing/models.py`。

5. 组合参数（`portfolio-model-type/opt-*`）  
生效模块：`models/portfolio/factory.py`、`models/portfolio/weighting.py`。

6. 执行参数（`execution-model-type/max-participation-rate/base-fill-rate/latency-bars`）  
生效模块：`models/execution/factory.py`、`models/execution/engines.py`。

7. 回测参数（`horizon/top-k/long-threshold/execution-scheme/fee-bps/slippage-bps/rebalance-stride/ic-eval-mode`）  
生效模块：`pipeline/runner.py`、`backtest/engine.py`、`backtest/metrics.py`。

8. 模型运行与推理参数（`model-run-mode/load-fe-mode/model-summary-json/models-load-dir/models-load-run-tag/*-model-path/enable-next-bar-inference/inference-top-k`）  
生效模块：`strategy7/config.py`、`pipeline/runner.py`、`models/loading.py`。

9. 输出参数（`output-dir/save-models`）  
生效模块：`config.resolve_output_dir`、`pipeline/runner.py` 产物落盘。

10. 日志参数（`log-level/quiet/verbose`）  
生效模块：`run_strategy7.py` 入口日志控制（运行时行为参数，不进入 RunConfig）。

## 4. run_factor_mining 参数生效映射（分组）

1. 数据与频率参数（`universe/data-root/stock-list-path(兼容hs300-list-path)/index-root/file-format/max-files/main-board-only/factor-freq/horizon`）  
生效模块：`run_factor_mining.py` 数据加载、标签生成、指数上下文注入。

2. 默认因子素材参数（`factor-packages/disable-default-factor-materials`）  
生效模块：`run_factor_mining.py` 默认因子计算并注入面板。

3. 进化参数（`population-size/generations/elite-size/mutation-rate/crossover-rate/top-n/...`）  
生效模块：`mining/runner.py`、`mining/models/*.py`。

4. 框架专用参数（`ml_*`、`gp_*`）  
生效模块：对应 `ml_ensemble_alpha` 与 `gplearn_symbolic_alpha` 模型文件。  
注意：仅在对应 framework 下生效。

5. custom 参数（`custom-spec-json`）  
生效模块：`framework=custom` 分支加载表达式规格。

6. 入库参数（`factor-store-root/catalog-path/save-format`）  
生效模块：`mining/runner.py` 因子落盘与 catalog 更新。

7. 准入阈值覆盖（`min-*`）  
生效模块：`mining/runner.py` 覆盖 admission profile 阈值后执行筛选。

8. 日志参数（`log-level/quiet/verbose`）  
生效模块：`run_factor_mining.py` 入口日志控制（运行时行为参数）。

## 5. 关键说明

1. 条件生效不等于未使用：  
`launch_boost_*` 在非 `launch_boost` 下不会生效，`fgcl_*` 在非 `factor_gcl` 下不会生效，`gp_*` 在非 `gplearn_symbolic_alpha` 下不会生效，这是预期设计。  

2. 运行时参数（如日志级别）不会进入 `RunConfig/FactorMiningConfig`，但已在入口处实际生效。  

3. 若后续新增模型或框架，建议同步更新本审计文档的“生效映射”章节，避免配置漂移。

## 6. 快速回归命令（建议保留）

1. 主流程清单模式（验证快速路径）：  
`python Strategy7/run_strategy7.py --list-factors --factor-freq D --log-level quiet`

2. 挖掘清单模式（验证导出目录）：  
`python Strategy7/run_factor_mining.py --list-factors --factor-freq D --factor-store-root D:/PythonProject/Quant/data_baostock --export-factor-list --factor-list-export-format csv --log-level quiet`

3. 验证框架条件校验（应通过，不应被 `gp_*` 误拦截）：  
`python Strategy7/run_factor_mining.py --framework fundamental_multiobj --list-factors --factor-freq D --gp-num-jobs 0`

4. 验证执行方案枚举校验（应在 CLI 直接报错）：  
`python Strategy7/run_factor_mining.py --list-factors --execution-scheme bad_scheme`

## 7. 2026-04-22~2026-04-23 主流程模型加载回测与快速推理参数闭环

新增参数（`strategy7/config.py`）：

1. `--model-run-mode train|load`
2. `--load-fe-mode strict|refit|off`
3. `--model-summary-json`
4. `--models-load-dir`
5. `--models-load-run-tag`
6. `--stock-model-path --timing-model-path --portfolio-model-path --execution-model-path`
7. `--enable-next-bar-inference`
8. `--inference-top-k`

生效链路：

1. CLI -> `build_run_config` -> `RunConfig.model_run`
2. `pipeline/runner.py` 在步骤 9 根据 `model_run.mode` 分支：
   - `train`：保持原逻辑（训练后回测）
   - `load`：调用 `models/loading.py` 解析并加载四类模型
3. `pipeline/runner.py` 在步骤 8.5 根据 `load_fe_mode` 分支：
   - `strict`：按 `summary` 中 `notes.feature_engineering_summary` 回放 FE（当前不支持 PCA 回放）
   - `refit`：按当前样本重拟合 FE（默认）
   - `off`：跳过 FE（即使 `enable_factor_engineering=true`）
4. `models/loading.py` 负责：
   - 从 `summary_*.json` / 模型目录 / 显式路径解析模型文件
   - 加载内置四类模型（含 tree/launch_boost/factor_gcl/dafat + timing/portfolio/execution）
   - `peek_stock_model_factor_cols` 在 `custom_stock_model_py` 场景优先走插件探测，不再错误按内置 `pickle` 反序列化
5. `pipeline/runner.py` 在回测后按 `enable_next_bar_inference` 生成：
   - `next_bar_candidates_*.csv`
   - `next_bar_summary_*.json`

审计结论：

1. 新增参数已完成“可配置即可生效”闭环接入。
2. 默认参数不变时，仍保持原有 `train` 模式行为。
3. `load` 模式下若因子列与模型文件不一致，会显式报错阻断（避免静默错误回测）。
4. `timing=none / portfolio=equal_weight / execution=ideal_fill` 时，对应 `--*-model-path` 将被忽略，不再触发“路径不存在”误报。
5. `load + custom_stock_model` 已修复 JSON/非 pickle 产物误按 `pickle.load` 解析导致的潜在崩溃问题。

## 8. 2026-04-23 模板覆盖与回归闭环（scripts/v2）

新增目录：`Strategy7/scripts/v2/`，目标是“覆盖完整功能链路 + 可复用配置模板”。

模板范围：

1. `run_strategy7_v2_01..11.ps1`：覆盖 train/load、FE 三模式、custom 四模型、FactorGCL、DAFAT、list-factors、factor-value-store。
2. `run_factor_mining_v2_01..07.ps1`：覆盖 `fundamental_multiobj/minute_parametric/minute_parametric_plus/ml_ensemble_alpha/gplearn/custom/list-factors`。
3. `run_smoke_suite_v2.ps1`：按“先 train 再 load”顺序串联执行模板，支持 `-SkipDeepModels/-SkipMining`。

模板修复与规范化：

1. 统一 `repoRoot` 到 `..\..\..`，避免路径定位到 `Strategy7/Strategy7`。
2. 统一 `--factor-packages` 为单字符串参数（PowerShell 下避免逗号拆分）。
3. 每个模板追加 `$LASTEXITCODE` 检查，非零退出立即抛错，避免假通过。
4. `run_strategy7_v2_06_load_from_models_dir_off.ps1` 设计为强制显式传入 `-ModelsLoadDir`，确保 load 回测来源可追踪。

2026-04-23 实跑结果（`env_quant`）：

1. `run_strategy7_v2_01..11` 全部可跑通（`06` 需按设计传 `-ModelsLoadDir`）。
2. `run_factor_mining_v2_01..07` 全部可跑通。
3. 产物统一落盘到 `Strategy7/outputs/smoke_v2/...`，可直接复核 summary/metrics/模型文件/因子导出。

