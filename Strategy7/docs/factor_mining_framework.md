# Strategy7 因子挖掘框架（报告复现版）

本文档说明 `Strategy7` 新增的因子挖掘子系统，覆盖：

1. 基本面参数化多目标挖掘（对应 2025-08-11 华泰报告）
2. 分钟级特征参数化多目标挖掘（对应 2026-03-31 华泰报告）
3. 分钟级增强参数化挖掘（`minute_parametric_plus`）
4. 集成学习因子挖掘（`ml_ensemble_alpha`，基于 `scikit-learn`）
5. 符号遗传规划挖掘（`gplearn_symbolic_alpha`，基于 `gplearn`）
6. 用户自定义因子表达式挖掘、评估、入库
7. 因子 catalog 入库与主回测框架自动适配

## 1. 模块结构

- `strategy7/mining/formulas.py`
  - `FundamentalFormulaSpec`（11 参数）
  - `MinuteFormulaSpec`（10 参数）
  - 基本面/分钟公式计算、后处理（去极值、中性化、标准化）
- `strategy7/mining/evaluation.py`
  - 多目标评估指标（|IC|、IC 胜率、NDCG、多头收益/夏普/胜率）
  - 频率分层入库标准与准入检查
- `strategy7/mining/nsga.py`
  - NSGA-II（基本面三目标）
  - NSGA-III（分钟级五目标）
  - 动态短板惩罚机制
- `strategy7/mining/custom.py`
  - 用户自定义表达式解析与计算（ts/cs 运算支持）
- `strategy7/mining/catalog.py`
  - 因子入库目录（catalog）读写
  - 因子表 merge + 因子库自动注册
- `strategy7/mining/runner.py`
  - 挖掘全流程（进化、评估、筛选、入库）
  - `MLEnsembleFormulaSpec`（模型结构 + 特征子集参数化）
- `strategy7/mining/models/gplearn_symbolic_alpha_model.py`
  - `GPLearnProgramSpec`
  - `SymbolicTransformer` 多轮符号进化 + 稳定性/简洁性约束
- `run_factor_mining.py`
  - 挖掘 CLI 入口

## 2. 报告方法到代码映射

### 2.1 基本面多目标框架（NSGA-II）

- 参数化表达式（11 参数）
  - `y_field/x_field`、`y_log/x_log`
  - `y_tr/x_tr`、`y_tr_period/x_tr_period`（qoq/yoy）
  - `y_tr_form/x_tr_form`（diff/pct/std）
  - `mode`（ratio/diff/sum/prod/mean/corr20/beta20）
- 目标函数（3 维）
  - `abs_ic_mean`
  - `ic_win_rate`
  - `ndcg_k`
- 排序选择
  - `NSGA-II` 非支配排序 + 拥挤距离

### 2.2 分钟级参数化框架（NSGA-III）

- 四步流程（输入切片 -> 时序掩码 -> 算子降维 -> 因子后处理）
- 参数化表达式（10 参数）
  - `a_field/b_field/window/slice_pos/mask_field/mask_rule/mode/op_name/cross_op_name/b_shift_lag`
- 目标函数（5 维）
  - `abs_ic_mean`
  - `ic_win_rate`
  - `long_ret_annualized`
  - `long_sharpe`
  - `long_win_rate`
- 排序选择
  - `NSGA-III` 参考点机制
  - `动态短板惩罚`（弱维度惩罚）

### 2.3 分钟增强参数化框架（`minute_parametric_plus`）

- 在 `minute_parametric` 基础上扩展：
  - 窗口池（新增更短/更长窗口）
  - 算子池（新增 `median/mad/iqr/entropy/autocorr1/spearman_corr/cosine_sim/downside_beta` 等）
  - 稳定性目标（训练/验证 `abs_ic_mean` 差异惩罚）
- 适用场景：
  - 想提升分钟级挖掘的表达能力
  - 同时控制过拟合（通过稳定性目标约束）

### 2.4 ML 集成框架（`ml_ensemble_alpha`）

- 候选体包含两部分：
  - 模型参数：`rf/et/hgbt` 超参数
  - 特征子集：从预筛后的日频特征池中抽样
- 训练流程：
  - 训练期 IC 预筛特征
  - 进化搜索候选（模型 + 特征子集）
  - 对预测分数做去极值/中性化/标准化后评估
- 目标函数（5 维）：
  - `abs_ic_mean`
  - `ic_win_rate`
  - `ndcg_k`
  - `long_excess_annualized`
  - `long_sharpe`

### 2.5 符号遗传规划框架（`gplearn_symbolic_alpha`）

- 底层库：
  - `gplearn.genetic.SymbolicTransformer`
- 核心流程：
  - 训练期特征预筛（IC Top-K）
  - 多随机种子进化搜索（`gp_num_runs`）
  - 每次输出多个表达式组件（`gp_n_components`）
  - 表达式输出统一走去极值/中性化/标准化
  - 训练/验证双样本评估后再入候选池
- 目标函数（7 维）：
  - 与 `ml_ensemble_alpha` 相同的 5 维效果目标
  - 稳定性目标：`-|abs_ic_train - abs_ic_valid|`
  - 简洁性目标：`-log(1+program_length)`（抑制表达式膨胀）

### 2.6 默认因子库素材注入（新）

- 挖掘入口 `run_factor_mining.py` 默认会按 `--factor-freq` 调用主因子库（`strategy7/factors/defaults.py`）。
- 注入后，默认因子会并入挖掘面板，与原始特征一起成为挖掘素材。
- 素材包可用参数：
  - `--factor-packages`：逗号分隔，控制注入哪些默认包（空=全部）
  - `--disable-default-factor-materials`：关闭默认因子素材注入
- 对分钟框架：
  - 当 `factor_freq` 为分钟频时，注入因子直接进入分钟面板素材池
  - 当 `factor_freq=D` 且使用分钟框架时，日频素材会作为“日内常量字段”参与分钟表达式计算

### 2.7 指数上下文素材（`--index-root` 生效）

- 挖掘入口会从 `--index-root` 读取 HS300 指数数据，并自动构建市场上下文特征：
  - `mkt_hs300_ret_1d/5d/20d`
  - `mkt_hs300_vol_20d`
  - `mkt_hs300_ma_gap_20d`
  - `mkt_hs300_drawdown_60d`
- 这些特征会 merge 到挖掘面板，并与默认因子素材一起参与挖掘模型。
- `mining_summary.json` 会记录 `index_context_cols` 以便追溯本次挖掘是否成功注入指数上下文。

### 2.8 股票池与自定义池加载（`--universe/--stock-list-path`）

- 挖掘入口与主回测入口都支持：
  - `--universe hs300/sz50/zz500/all`
  - `--stock-list-path`（旧参数 `--hs300-list-path` 仍兼容）
- 当 `--universe all` 且提供 `--stock-list-path` 时，可在全市场历史行情上加载任意自定义股票池（例如中证1000/中证2000）。
- `--data-root auto` 会按 `--universe` 自动映射到 `data_baostock/stock_hist/<universe>`。

## 3. 入库标准（频率分层）

`strategy7/mining/evaluation.py` 中内置频率+框架标准：

- 基本面（`*_fundamental_nsga2_v1`）
  - `abs_ic_mean >= 0.02`
  - `ic_win_rate >= 0.55`
  - `ic_ir >= 0.20`
  - `long_excess_annualized >= 0.02`
  - `long_sharpe >= 0.60`
  - `coverage >= 0.90`
- 分钟级（`*_minute_nsga3_v1`）
  - `abs_ic_mean >= 0.015`
  - `ic_win_rate >= 0.53`
  - `ic_ir >= 0.15`
  - `long_excess_annualized >= 0.02`
  - `long_sharpe >= 0.80`
  - `coverage >= 0.85`
- 分钟增强（`*_minute_nsga3_plus_v1`）
  - `abs_ic_mean >= 0.018`
  - `ic_win_rate >= 0.54`
  - `ic_ir >= 0.18`
  - `long_excess_annualized >= 0.03`
  - `long_sharpe >= 0.90`
  - `coverage >= 0.88`
- ML 集成（`*_ml_ensemble_v1`）
  - `abs_ic_mean >= 0.018`
  - `ic_win_rate >= 0.54`
  - `ic_ir >= 0.16`
  - `long_excess_annualized >= 0.02`
  - `long_sharpe >= 0.70`
  - `coverage >= 0.90`
- GP 符号挖掘（`*_gplearn_symbolic_v1`）
  - `abs_ic_mean >= 0.020`
  - `ic_win_rate >= 0.54`
  - `ic_ir >= 0.18`
  - `long_excess_annualized >= 0.02`
  - `long_sharpe >= 0.80`
  - `coverage >= 0.90`

仅通过准入标准的因子才会以 `status=active` 入 catalog。

## 4. 因子入库与主框架适配

挖掘完成后会输出：

- 因子值表：`factor_values_<freq>.parquet/csv`
- catalog：`factor_mining/factor_catalog.json`
- summary：`mining_summary.json`

主回测 `run_strategy7.py` 已支持自动加载 catalog 因子：

- 新参数：
  - `--factor-catalog-path`（默认 `auto`）
  - `--disable-catalog-factors`（关闭自动加载）
- `run_pipeline` 会：
  - 自动 merge catalog 因子值
  - 自动注册到 `FactorLibrary`
  - `--factor-list` 可直接引用 catalog 因子名

## 5. 用户自定义因子

### 5.1 自定义规格文件

`--custom-spec-json` 使用 JSON 列表：

```json
[
  {
    "name": "custom_alpha_01",
    "expression": "cs_z(0.6*mom_20 - 0.4*amihud_20)",
    "freq": "D",
    "category": "custom_factor",
    "description": "示例：动量-流动性"
  }
]
```

### 5.2 支持算子

- 基础：`abs/log/sqrt/sign/clip/where`
- 时序：`delay/delta/pct/ts_mean/ts_std/ts_z`
- 截面：`cs_rank/cs_z`

表达式会先计算，再走统一评估与入库标准。

## 6. 运行示例

### 6.1 基本面挖掘

```bash
python Strategy7/run_factor_mining.py \
  --framework fundamental_multiobj \
  --train-start 2021-01-01 --train-end 2023-12-31 \
  --valid-start 2024-01-01 --valid-end 2024-12-31 \
  --population-size 128 --generations 20 --top-n 20
```

### 6.2 分钟级挖掘

```bash
python Strategy7/run_factor_mining.py \
  --framework minute_parametric \
  --factor-freq 15min \
  --factor-packages trend,liquidity,bridge \
  --train-start 2021-01-01 --train-end 2023-12-31 \
  --valid-start 2024-01-01 --valid-end 2024-12-31 \
  --population-size 96 --generations 16 --top-n 20
```

### 6.3 自定义因子挖掘

```bash
python Strategy7/run_factor_mining.py \
  --framework custom \
  --custom-spec-json ./Strategy7/docs/custom_factor_specs.json \
  --train-start 2021-01-01 --train-end 2023-12-31 \
  --valid-start 2024-01-01 --valid-end 2024-12-31
```

### 6.4 分钟增强参数化挖掘

```bash
python Strategy7/run_factor_mining.py \
  --framework minute_parametric_plus \
  --train-start 2021-01-01 --train-end 2023-12-31 \
  --valid-start 2024-01-01 --valid-end 2024-12-31 \
  --population-size 96 --generations 16 --top-n 20
```

### 6.5 ML 集成因子挖掘

```bash
python Strategy7/run_factor_mining.py \
  --framework ml_ensemble_alpha \
  --factor-freq D \
  --factor-packages legacy_core,trend,reversal,liquidity \
  --train-start 2021-01-01 --train-end 2023-12-31 \
  --valid-start 2024-01-01 --valid-end 2024-12-31 \
  --ml-population-size 48 --ml-generations 10 \
  --ml-model-pool rf,et,hgbt \
  --ml-prefilter-topk 80 --ml-feature-min 10 --ml-feature-max 36
```

### 6.6 GP 符号遗传规划因子挖掘

```bash
python Strategy7/run_factor_mining.py \
  --framework gplearn_symbolic_alpha \
  --train-start 2021-01-01 --train-end 2023-12-31 \
  --valid-start 2024-01-01 --valid-end 2024-12-31 \
  --gp-population-size 400 --gp-generations 12 \
  --gp-num-runs 3 --gp-n-components 24 \
  --gp-function-set add,sub,mul,div,sqrt,log,abs,neg,max,min \
  --gp-prefilter-topk 80
```

依赖说明：

- 该框架需要 `gplearn`，请先在运行环境安装：
  - `pip install gplearn`

## 7. 说明

- 当前工程使用通用可解释参数化结构复现研报核心流程（参数化表达式 + 多目标进化 + 分层准入 + 因子入库）。
- 若你的数据源已补充更丰富的基本面字段/分钟字段，框架会自动扩大可选搜索空间。
- 若需进一步“逐表逐字段”严格对齐研报原始 71 指标/26 指标定义，可在数据源层增补字段后直接复用本框架挖掘引擎。

### Admission Threshold Overrides

You can override built-in admission thresholds from CLI, for example:

```bash
python Strategy7/run_factor_mining.py \
  --framework fundamental_multiobj \
  --factor-packages trend,context \
  --min-abs-ic-mean 0.015 \
  --min-ic-win-rate 0.53 \
  --min-long-excess-annualized 0.015
```
