# Strategy7 Templates V2

本目录提供覆盖 Strategy7 主要功能链路的新版脚本模板（`env_quant`），并可由 `run_smoke_suite_v2.ps1` / `run_smoke_suite_v2.sh` 统一回归。

## 0) 平台版本

- 每个 v2 模板均提供同名双版本：`*.ps1`（Windows PowerShell）与 `*.sh`（Linux Ubuntu Bash）。
- 推荐约定：
  - Windows：`powershell -ExecutionPolicy Bypass -File ...ps1`
  - Linux：`bash ...sh`
- Linux 首次使用可先执行：`chmod +x Strategy7/scripts/v2/*.sh`
- Linux 默认路径通过 `QUANT_ROOT` 推断（默认取 `repo_root/..`）；可显式设置：
  - `export QUANT_ROOT=/workspace/Quant`
  - `export CONDA_ENV=env_quant`

## 1) 主入口模板（run_strategy7.py）

| 模板 | 作用 | 覆盖点 |
|---|---|---|
| `run_strategy7_v2_01_train_tree_baseline.ps1` | 决策树基础训练回测 | train、内置四模型默认链路 |
| `run_strategy7_v2_02_train_tree_fe_dynamic.ps1` | 决策树+FE+择时+动态组合+真实执行 | FE(refit 基础)、volatility_regime、dynamic_opt、realistic_fill、next-bar |
| `run_strategy7_v2_03_train_custom_all_models.ps1` | 自定义四模型训练回测 | custom stock/timing/portfolio/execution |
| `run_strategy7_v2_04_load_from_summary_refit.ps1` | 基于 summary 的 load 回测（refit FE） | model_run_mode=load、load_fe_mode=refit |
| `run_strategy7_v2_05_load_from_summary_strict.ps1` | 基于 summary 的 load 回测（strict FE） | model_run_mode=load、load_fe_mode=strict |
| `run_strategy7_v2_06_load_from_models_dir_off.ps1` | 基于 models 目录的 load 回测（off FE） | models_load_dir、load_fe_mode=off |
| `run_strategy7_v2_07_load_custom_from_summary.ps1` | 自定义四模型的 load 回测 | custom load_model 链路 |
| `run_strategy7_v2_08_train_factor_gcl_smoke.ps1` | FactorGCL 轻量训练回测 | factor_gcl 分支 |
| `run_strategy7_v2_09_train_dafat_smoke.ps1` | DAFAT 轻量训练回测 | dafat 分支 |
| `run_strategy7_v2_10_list_factors_export.ps1` | 因子清单列出并导出 | list-factors、factor-list-export、snapshot |
| `run_strategy7_v2_11_factor_value_store_build_only.ps1` | 因子值仓库构建（仅构建） | factor value store build-all/build-only |
| `run_strategy7_v2_12_train_weekly_volatility.ps1` | 周频训练回测 | `factor_freq=W`、`label_task=volatility` |
| `run_strategy7_v2_13_train_30min_intraday_realistic.ps1` | 30min 日内链路训练回测 | `factor_freq=30min`、realistic execution、next-bar |
| `run_strategy7_v2_14_train_price_only_mainboard.ps1` | 价格因子主板子集训练回测 | `universe=all` + `main_board_only` + 关闭基本面/文本 |
| `run_strategy7_v2_15_train_custom_factor_plugin.ps1` | 自定义因子插件训练回测 | `--custom-factor-py` |
| `run_strategy7_v2_16_train_factor_value_store_hydrate.ps1` | 因子值仓库“水合”回测 | `enable-factor-value-store=true` + 非 build-only |
| `run_strategy7_v2_17_load_explicit_paths_off.ps1` | 显式四路径 load 回测 | `--stock/timing/portfolio/execution-model-path` |
| `run_strategy7_v2_18_train_monthly_multitask_catalog_off.ps1` | 月频多任务训练回测 | `factor_freq=M`、`label_task=multi_task`、`enable_factor_catalog=false` |
| `run_strategy7_v2_19_list_factors_30min_json_export.ps1` | 30min 因子清单 JSON 导出 | `--list-factors` + `factor_freq=30min` + JSON export |
| `run_strategy7_v2_20_train_allmarket_bottom_launch_10d.ps1` | 全市场低位启动10日训练回测 | `factor_freq=D`、`horizon=10`、`stock_model_type=launch_boost`、`factor_packages` 含 `bottom_launch` |
| `run_strategy7_v2_21_load_allmarket_bottom_launch_10d.ps1` | 全市场低位启动10日 load 回测 | `model_run_mode=load`、`model_summary_json`、`launch_boost`、next-bar |

## 2) 挖掘入口模板（run_factor_mining.py）

| 模板 | 作用 | 覆盖点 |
|---|---|---|
| `run_factor_mining_v2_01_fundamental_smoke.ps1` | 基本面多目标挖掘 | framework=fundamental_multiobj |
| `run_factor_mining_v2_02_minute_parametric_smoke.ps1` | 分钟参数化挖掘 | framework=minute_parametric |
| `run_factor_mining_v2_03_minute_parametric_plus_smoke.ps1` | 分钟增强参数化挖掘 | framework=minute_parametric_plus |
| `run_factor_mining_v2_04_ml_ensemble_smoke.ps1` | ML 集成挖掘 | framework=ml_ensemble_alpha |
| `run_factor_mining_v2_05_gplearn_smoke.ps1` | GP-Library 符号挖掘 | framework=gplearn_symbolic_alpha |
| `run_factor_mining_v2_06_custom_smoke.ps1` | custom 因子表达式评估 | framework=custom、factor-list passthrough |
| `run_factor_mining_v2_07_list_factors_export.ps1` | 挖掘入口 list-factors 导出 | list-factors、snapshot、export |
| `run_factor_mining_v2_08_material_fe_value_store.ps1` | 素材+FE+值仓库联动挖掘 | factor materials、factor engineering、value store |
| `run_factor_mining_v2_09_custom_spec_json_smoke.ps1` | 自定义规格 JSON 挖掘 | `--custom-spec-json` |
| `run_factor_mining_v2_10_minute_parametric_30min_smoke.ps1` | 30min 参数化挖掘 | `framework=minute_parametric` + `factor_freq=30min` |
| `run_factor_mining_v2_11_price_only_mainboard_all.ps1` | 价格因子主板子集挖掘 | `universe=all` + `main_board_only` + 关闭基本面/文本 |
| `run_factor_mining_v2_12_list_factors_markdown_export.ps1` | 因子清单 Markdown 导出 | `--list-factors` + markdown export |
| `run_factor_mining_v2_13_disable_default_materials_with_factor_list.ps1` | 指定因子列表挖掘 | `--disable-default-factor-materials` + `--factor-list` |
| `run_factor_mining_v2_14_list_factors_with_custom_plugin.ps1` | 自定义因子插件列表模式 | `--list-factors` + `--custom-factor-py` |

## 3) 按研究目的选模板（极简索引）

| 研究目的 | 推荐模板 |
|---|---|
| 快速基线训练回测（最小可用） | `run_strategy7_v2_01_train_tree_baseline.ps1` |
| FE + 动态组合 + realistic 执行 + next-bar 全链路 | `run_strategy7_v2_02_train_tree_fe_dynamic.ps1` |
| 自定义四模型训练 / load 验证 | `run_strategy7_v2_03_train_custom_all_models.ps1`、`run_strategy7_v2_07_load_custom_from_summary.ps1` |
| 复现已有模型（summary 引导） | `run_strategy7_v2_04_load_from_summary_refit.ps1`、`run_strategy7_v2_05_load_from_summary_strict.ps1` |
| 复现已有模型（models 目录） | `run_strategy7_v2_06_load_from_models_dir_off.ps1` |
| 复现已有模型（显式四路径） | `run_strategy7_v2_17_load_explicit_paths_off.ps1` |
| 深度模型轻量冒烟（选股） | `run_strategy7_v2_08_train_factor_gcl_smoke.ps1`、`run_strategy7_v2_09_train_dafat_smoke.ps1` |
| 频率扩展验证（30min/W/M） | `run_strategy7_v2_13_train_30min_intraday_realistic.ps1`、`run_strategy7_v2_12_train_weekly_volatility.ps1`、`run_strategy7_v2_18_train_monthly_multitask_catalog_off.ps1` |
| 数据裁剪实验（price-only / 主板） | `run_strategy7_v2_14_train_price_only_mainboard.ps1`、`run_factor_mining_v2_11_price_only_mainboard_all.ps1` |
| 自定义因子插件接入 | `run_strategy7_v2_15_train_custom_factor_plugin.ps1`、`run_factor_mining_v2_14_list_factors_with_custom_plugin.ps1` |
| 因子值仓库路径（build-only / hydrate） | `run_strategy7_v2_11_factor_value_store_build_only.ps1`、`run_strategy7_v2_16_train_factor_value_store_hydrate.ps1` |
| 因子清单导出（json/markdown） | `run_strategy7_v2_10_list_factors_export.ps1`、`run_strategy7_v2_19_list_factors_30min_json_export.ps1`、`run_factor_mining_v2_07_list_factors_export.ps1`、`run_factor_mining_v2_12_list_factors_markdown_export.ps1` |
| 挖掘框架覆盖（fundamental/minute/ml/gp/custom） | `run_factor_mining_v2_01..06` |
| 挖掘参数扩展（custom spec / 30min / 关闭默认素材） | `run_factor_mining_v2_09_custom_spec_json_smoke.ps1`、`run_factor_mining_v2_10_minute_parametric_30min_smoke.ps1`、`run_factor_mining_v2_13_disable_default_materials_with_factor_list.ps1` |
| 低位启动10日趋势选股（全市场） | `run_strategy7_v2_20_train_allmarket_bottom_launch_10d.ps1`、`run_strategy7_v2_21_load_allmarket_bottom_launch_10d.ps1` |

## 4) 一键回归

- `run_smoke_suite_v2.ps1` / `run_smoke_suite_v2.sh`：按“先 train 再 load”顺序逐条执行模板并输出通过/失败清单。
- 默认执行核心模板（`run_strategy7:01~11`、`run_factor_mining:01~07`）；如需覆盖新增模板，附加 `-IncludeExtended`。
- `run_strategy7_v2_20/21` 是研究型全市场模板（执行时间通常较长），默认不加入 `run_smoke_suite_v2.ps1`。
- 可按需提速：
  - `-SkipDeepModels`
  - `-SkipMining`

示例：

```powershell
# 核心模板回归
powershell -ExecutionPolicy Bypass -File Strategy7/scripts/v2/run_smoke_suite_v2.ps1

# 核心 + 扩展模板全覆盖回归
powershell -ExecutionPolicy Bypass -File Strategy7/scripts/v2/run_smoke_suite_v2.ps1 -IncludeExtended

# 全覆盖但跳过深度模型
powershell -ExecutionPolicy Bypass -File Strategy7/scripts/v2/run_smoke_suite_v2.ps1 -IncludeExtended -SkipDeepModels

# 全覆盖但跳过挖掘入口
powershell -ExecutionPolicy Bypass -File Strategy7/scripts/v2/run_smoke_suite_v2.ps1 -IncludeExtended -SkipMining
```

```bash
# 核心模板回归
bash Strategy7/scripts/v2/run_smoke_suite_v2.sh

# 核心 + 扩展模板全覆盖回归
bash Strategy7/scripts/v2/run_smoke_suite_v2.sh --include-extended

# 全覆盖但跳过深度模型
bash Strategy7/scripts/v2/run_smoke_suite_v2.sh --include-extended --skip-deep-models

# 全覆盖但跳过挖掘入口
bash Strategy7/scripts/v2/run_smoke_suite_v2.sh --include-extended --skip-mining
```

## 5) 约定

- 默认使用 `conda run -n env_quant --no-capture-output python`。
- 输出根目录统一到 `Strategy7/outputs/smoke_v2/...`。
- 所有模板均可单独执行，`load` 模板通过参数接收上一步产出的 summary/models 路径。
- `run_strategy7_v2_06_load_from_models_dir_off.ps1` 需要显式传入 `-ModelsLoadDir`（可选 `-ModelsLoadRunTag`），示例：
  - `powershell -ExecutionPolicy Bypass -File Strategy7/scripts/v2/run_strategy7_v2_06_load_from_models_dir_off.ps1 -ModelsLoadDir D:/PythonProject/Quant/TradeSystem/Strategy7/outputs/smoke_v2/run_strategy7_01_train_tree/models -ModelsLoadRunTag 510c1cd320`
- `run_strategy7_v2_06_load_from_models_dir_off.sh` 参数等价为 `--models-load-dir`（可选 `--models-load-run-tag`），示例：
  - `bash Strategy7/scripts/v2/run_strategy7_v2_06_load_from_models_dir_off.sh --models-load-dir /workspace/Quant/TradeSystem/Strategy7/outputs/smoke_v2/run_strategy7_01_train_tree/models --models-load-run-tag 510c1cd320`
- `run_strategy7_v2_17_load_explicit_paths_off.ps1` 需要显式传四类模型路径：
  - `-StockModelPath -TimingModelPath -PortfolioModelPath -ExecutionModelPath`
- `run_strategy7_v2_17_load_explicit_paths_off.sh` 参数等价为：
  - `--stock-model-path --timing-model-path --portfolio-model-path --execution-model-path`
- `run_strategy7_v2_20_train_allmarket_bottom_launch_10d.ps1` / `run_strategy7_v2_21_load_allmarket_bottom_launch_10d.ps1` 支持：
  - `-DataRoot`（默认 `auto`，按主入口自动解析全市场数据目录）
  - `-IndexRoot`（默认空，留空则走主入口默认）
  - `-TrainStart/-TrainEnd/-TestStart/-TestEnd`（默认长期研究窗口，可按需改成短窗口调试）
  - `-MaxFiles`（可选小样本调试）
- `run_strategy7_v2_20_train_allmarket_bottom_launch_10d.sh` / `run_strategy7_v2_21_load_allmarket_bottom_launch_10d.sh` 参数等价为：
  - `--data-root --index-root --train-start --train-end --test-start --test-end --max-files`
  - `run_strategy7_v2_21` 额外需要 `--model-summary-json`
