# Strategy7 因子挖掘框架（报告复现版）

本文档说明 `Strategy7` 新增的因子挖掘子系统，覆盖：

1. 基本面参数化多目标挖掘（对应 2025-08-11 华泰报告）
2. 分钟级特征参数化多目标挖掘（对应 2026-03-31 华泰报告）
3. 用户自定义因子表达式挖掘、评估、入库
4. 因子 catalog 入库与主回测框架自动适配

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

## 7. 说明

- 当前工程使用通用可解释参数化结构复现研报核心流程（参数化表达式 + 多目标进化 + 分层准入 + 因子入库）。
- 若你的数据源已补充更丰富的基本面字段/分钟字段，框架会自动扩大可选搜索空间。
- 若需进一步“逐表逐字段”严格对齐研报原始 71 指标/26 指标定义，可在数据源层增补字段后直接复用本框架挖掘引擎。

### Admission Threshold Overrides

You can override built-in admission thresholds from CLI, for example:

```bash
python Strategy7/run_factor_mining.py \
  --framework fundamental_multiobj \
  --min-abs-ic-mean 0.015 \
  --min-ic-win-rate 0.53 \
  --min-long-excess-annualized 0.015
```
