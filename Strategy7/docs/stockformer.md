# StockFormer in Strategy7

本文档说明民生证券研报《StockFormer：基于 Transformer 的强化学习模型探究》在 `Strategy7` 中的工程化复现方式。

## 1. 研报结构映射

研报中的 StockFormer 分为两个阶段：

1. 三个改造 Transformer 先做 predictive coding：
   - 关系状态 `S_relat`：原论文用 252 日价格协方差矩阵和技术因子；本工程保留 252 日关系分支，并用当前可用因子、行业/板块分组、市场截面偏离构造动态关系输入，以避免固定 88 只股票的限制。
   - 短期收益状态 `S_short`：预测 1 bar/1 日收益状态。
   - 长期收益状态 `S_long`：预测当前框架 `horizon` 对应的未来收益状态。
2. SAC 强化学习再优化交易动作：
   - 先用多头注意力融合 `S_long` 与 `S_short` 得到 `S_future`。
   - 再用多头注意力融合 `S_future` 与 `S_relat` 得到 SAC 状态 `S_t`。
   - Actor 输出每个截面股票的连续权重动作；双 Q Critic、目标 Q 网络软更新、熵正则和自适应熵系数按 SAC 训练。

代码入口：

- `strategy7/models/stock_selection/stockformer_model.py`
- `strategy7/models/stock_selection/factory.py`
- `strategy7/models/loading.py`
- `strategy7/config.py`

## 2. 与框架的适配

`StockFormerStockModel` 继承 `StockSelectionModel`，因此完整支持现有主流程：

- `fit(train_df, factor_cols, target_col)`
- `predict_score(df, factor_cols)`
- `save(folder, run_tag)`
- `model_run_mode=load` 加载 `.pt` checkpoint

研报模型直接输出持仓动作；Strategy7 回测引擎先用 `pred_score` 选股，再进入择时/组合/执行层。因此实现中使用 Actor 的确定性权重作为原始动作，并在每个 `signal_ts` 截面转成 rank percentile `pred_score`，保持与 `top_k`、`long_threshold`、IC 诊断、next-bar 推理兼容。

## 3. 频率与因子支持

模型不硬编码日频或固定因子列，`factor_cols` 完全来自主流程配置：

- 频率：`5min/15min/30min/60min/120min/D/W/M`
- 因子来源：默认量价因子、跨频桥接因子、基本面因子、文本因子、catalog 因子、自定义因子插件、FE 派生因子
- 时间列自动识别：`signal_ts`、`datetime`、`date`

推荐日频研究时使用：

- `--factor-freq D`
- `--label-task return`
- `--horizon 5`
- `--stockformer-seq-len 60`
- `--stockformer-rel-seq-len 252`
- `--stockformer-label-transform csrank`

周频/月频也可运行，但研报指出低频强化学习样本期数不足，建议降低模型规模、增加滚动训练窗口，并把 `--stockformer-learning-starts` 调小。

## 4. 奖励函数

研报实证将奖励函数改为“超额收益 - 跟踪误差 - 交易费用”。工程实现对应为：

```text
reward = portfolio_return
       - equal_weight_benchmark_return
       - turnover_penalty * reward_cost_bps * turnover / 10000
       - tracking_penalty * tracking_error_proxy
```

其中 `reward_cost_bps` 默认 `30.0`，对应研报中双边千分之三的交易成本假设。若希望与当前回测默认 `fee_bps/slippage_bps` 更接近，可把它调低到 `3.0`。

## 5. 关键参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--stockformer-seq-len` | `60` | 短/长收益 Transformer 输入长度 |
| `--stockformer-rel-seq-len` | `252` | 关系状态 Transformer 输入长度 |
| `--stockformer-hidden-size` | `64` | 隐层维度 |
| `--stockformer-num-layers` | `2` | Transformer 层数 |
| `--stockformer-num-heads` | `10` | 多头注意力头数；实现允许 hidden 不被 heads 整除 |
| `--stockformer-pretrain-epochs` | `50` | 三路 Transformer predictive coding 预训练轮数 |
| `--stockformer-sac-episodes` | `50` | SAC episode 轮数 |
| `--stockformer-sac-lr` | `3e-4` | SAC 学习率 |
| `--stockformer-gamma` | `0.999` | 折现因子 |
| `--stockformer-init-alpha` | `0.5` | 初始熵正则权重 |
| `--stockformer-learning-starts` | `100` | 第多少个交易日后开始 SAC 更新 |
| `--stockformer-reward-cost-bps` | `30.0` | 奖励函数换手成本 |
| `--stockformer-min-cross-section` | `8` | 单个截面最小股票数 |

## 6. 运行示例

轻量冒烟：

```powershell
powershell -ExecutionPolicy Bypass -File Strategy7/scripts/v2/run_strategy7_v2_26_train_stockformer_smoke.ps1
```

Linux：

```bash
bash Strategy7/scripts/v2/run_strategy7_v2_26_train_stockformer_smoke.sh
```

研究型训练：

```powershell
powershell -ExecutionPolicy Bypass -File Strategy7/scripts/v2/run_strategy7_v2_27_train_stockformer.ps1
```

## 7. 实现边界

原论文的关系状态分支依赖固定股票池协方差矩阵，预测时也要求同一批股票存在。Strategy7 面向动态股票池和多频因子，因此使用行业/板块/市场截面偏离构造可变股票池关系状态，这是为了兼容全市场、成分股滚动、主板过滤、分钟频率和自定义因子插件所做的必要修改。
