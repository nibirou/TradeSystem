# DFQ-TimesNet in Strategy7

本文档说明研报《DFQ-TimesNet：捕捉量价特征周期规律，提升股票收益预测效果》在 `Strategy7` 中的工程化复现方式。

## 1. 研报结构映射

DFQ-TimesNet 的核心是把一维量价序列折叠成二维周期结构，用二维卷积同时捕捉“周期内变化”和“跨周期关联”：

1. `TokenEmbedding`：用 kernel size = 3 的 1D 卷积把 `[batch, seq_len, C_in]` 映射到 `[batch, seq_len, hidden_size]`。
2. `TimesBlock`：按固定周期折叠序列。默认周期为 `5,60`，对应周度与季度交易节律。
3. `Inception`：每个周期分支使用两层 Inception 卷积，结构为 `Conv -> GELU -> Conv`，默认卷积核数量为 3，即 `1x1/3x3/5x5`。
4. 周期融合：不同周期输出直接平均，不使用 FFT 振幅加权或线性学习权重。
5. 残差连接：融合输出与 TimesBlock 输入相加。
6. 预测头：取最后一个时间步，通过 `Linear(hidden_size, 1)` 输出原始预测值；回测前按横截面转为 `pred_score` 分位数。

实现文件：

- `strategy7/models/stock_selection/dfq_timesnet_model.py`
- `strategy7/models/stock_selection/factory.py`
- `strategy7/models/loading.py`
- `strategy7/config.py`

## 2. 与框架的适配

模型实现继承 `StockSelectionModel`，因此与现有四模型链路一致：

- `fit(train_df, factor_cols, target_col)`
- `predict_score(df, factor_cols)`
- `save(folder, run_tag)`
- `load` 模式下通过 `models/loading.py` 恢复 `.pt` checkpoint

频率支持：

- 支持 `5min/15min/30min/60min/120min/D/W/M`。
- 时序窗口按 `signal_ts`、`datetime`、`date` 自动识别。
- 日/周/月按日期截面分组；日内频率按具体 bar 时间截面分组。

因子类型支持：

- 可使用默认量价因子、跨频桥接因子、基本面因子、文本因子、挖掘 catalog 因子、自定义因子插件输出。
- 研报最优设定偏向基础量价特征；工程中不硬编码 60 个输入特征，实际 `factor_cols` 由 `--factor-list`、`--factor-packages`、FE、catalog 等配置共同决定。

数据处理：

- 主流程已在模型前执行截面 winsorize + zscore。
- TimesNet 内部额外提供 `--timesnet-input-clip`，默认 `3.0`，用于贴近研报的 `Z-score + clip(-3,3)`。
- 标签默认 `--timesnet-label-transform cszscore`，贴近研报“未来收益截面 Z-score”的训练目标。

## 3. 默认超参

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--timesnet-seq-len` | `60` | 历史序列长度 |
| `--timesnet-hidden-size` | `128` | TokenEmbedding 输出维度 |
| `--timesnet-e-layers` | `1` | TimesBlock 层数 |
| `--timesnet-hidden-size2` | `128` | Inception 中间通道数 |
| `--timesnet-periods` | `5,60` | 固定双周期 |
| `--timesnet-num-kernels` | `3` | Inception 卷积核数量 |
| `--timesnet-dropout` | `0.0` | dropout |
| `--timesnet-epochs` | `200` | 最大训练轮数 |
| `--timesnet-lr` | `9e-5` | 学习率 |
| `--timesnet-early-stop` | `20` | 验证集 RankIC 早停 |
| `--timesnet-smooth-steps` | `5` | 最优权重平滑窗口 |
| `--timesnet-per-epoch-batch` | `100` | 每轮抽取截面数 |
| `--timesnet-batch-size` | `-1` | 单截面全量股票 |
| `--timesnet-label-transform` | `cszscore` | 标签截面标准化 |

## 4. 运行示例

正式研报风格日频训练：

```powershell
python Strategy7/run_strategy7.py `
  --factor-freq D `
  --label-task return `
  --horizon 20 `
  --stock-model-type dfq_timesnet `
  --timesnet-seq-len 60 `
  --timesnet-periods "5,60" `
  --timesnet-hidden-size 128 `
  --timesnet-hidden-size2 128 `
  --timesnet-num-kernels 3 `
  --timesnet-label-transform cszscore `
  --timesnet-input-clip 3.0 `
  --timesnet-epochs 200 `
  --timesnet-lr 9e-5
```

轻量冒烟：

```powershell
powershell -ExecutionPolicy Bypass -File Strategy7/scripts/v2/run_strategy7_v2_24_train_dfq_timesnet_smoke.ps1
```

Linux：

```bash
bash Strategy7/scripts/v2/run_strategy7_v2_24_train_dfq_timesnet_smoke.sh
```
