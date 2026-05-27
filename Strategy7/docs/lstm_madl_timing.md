# LSTM+MADL 择时模型

本文记录 `--timing-model-type lstm_madl` 的研报复现口径和 Strategy7 适配方式。

## 研报映射

参考研报：华福证券《金融工程专题：基于 LSTM 神经网络的择时融合多因子选股策略》。

核心结构：

1. 输入层：历史序列特征。
2. 隐层：`[LSTM + BatchNorm] x N`，默认 `512,256,128`，并使用 Dropout。
3. 输出层：全连接 + `Tanh`，输出区间为 `[-1, 1]`。
4. 损失：MADL，方向正确且绝对收益更大时奖励更高；框架默认使用 `madl_mse`，在 MADL 上加入很小的 MSE 稳定项。

研报日频版本使用过去 20 个交易日的 7 个基础量价特征：`open/high/low/close/volume/amount/trade_count`。研报分钟版本使用一日 240 根分钟线，基础量价特征加 44 个技术指标。框架中的 `auto` 模式会按研究频率自动选择：

1. `D/W/M` 等非日内频率：7 个基础量价特征。
2. `5min/15min/30min/60min/120min` 等日内频率：7 个基础量价特征 + 44 个技术指标。

## Strategy7 适配

现有回测链路已经按四类模型解耦：

1. 选股模型生成 `pred_score` 和股票池。
2. 择时模型输出整体仓位 `exposure`。
3. 组合优化模型在股票池内生成个股权重。
4. 执行模型模拟成交约束。

`lstm_madl` 只控制第二步。最终权重仍由 `portfolio_model` 决定，并在回测引擎中乘以择时仓位，因此它可以自然叠加 `decision_tree/launch_boost/factor_gcl/dafat/dfq_timesnet/dtlc_rl/stockformer` 等选股模型，也可以叠加 `equal_weight/dynamic_opt` 和 `ideal_fill/realistic_fill`。

由于 Strategy7 的训练数据是股票面板，不是单独指数面板，模型会先按 `signal_ts/datetime/date` 聚合成市场代理序列。默认 `amount_weighted` 更接近指数或宽基资金权重，也支持 `mean/median`。

## 频率与因子

常用输入模式：

1. `auto`：默认推荐。非日内用研报日频 7 特征，日内用研报技术指标组。
2. `daily_bar`：强制只用 7 个基础量价特征。
3. `technical`：强制使用基础量价 + 44 技术指标。
4. `hybrid`：在 `technical` 基础上额外选择基本面、文本、catalog 或自定义因子列。
5. `all_numeric`：在所有数值列中择优选择，适合做探索实验。

模型支持框架已有的 `factor-freq`：`5min/15min/30min/60min/120min/D/W/M`。当前框架的最低内置研究频率是 `5min`，一个交易日约 48 根，默认 `--timing-lstm-intraday-seq-len 48` 更合适；若未来接入 1 分钟频率，可把序列长度设为 `240` 以贴近研报分钟版。

## 仓位映射

研报可以做多做空，但 A 股多头框架中负信号会被映射为空仓或低仓位。可选方式：

1. `long_only_bands`：默认分段仓位，贴近研报分钟多头分段：`<-0.1` 空仓，`[-0.1,0.1)` 30%，`[0.1,0.6)` 50%，`[0.6,1)` 80%，更高满仓。
2. `report_daily_long`/`long_only_threshold`：信号大于 `--timing-lstm-long-threshold` 才满仓，默认阈值 `-0.3`。
3. `raw_clip`：`max(signal, 0)` 作为仓位。

如果手动传入以负数开头的分段阈值，请使用等号形式，例如 `--timing-lstm-band-thresholds=-0.1,0.1,0.6,0.999999`。

## 推荐命令

```powershell
python Strategy7/run_strategy7.py `
  --train-start 2020-01-01 --train-end 2024-12-31 `
  --test-start 2025-01-01 --test-end 2025-12-31 `
  --universe hs300 `
  --factor-freq D `
  --factor-packages trend,reversal,liquidity,volatility,flow,price_action `
  --label-task return `
  --horizon 1 `
  --stock-model-type decision_tree `
  --timing-model-type lstm_madl `
  --timing-lstm-seq-len 20 `
  --timing-lstm-hidden-sizes 512,256,128 `
  --timing-lstm-feature-mode auto `
  --timing-lstm-loss-mode madl_mse `
  --timing-lstm-exposure-mode long_only_bands `
  --portfolio-model-type dynamic_opt `
  --execution-model-type realistic_fill `
  --execution-scheme open5_open5 `
  --save-models true
```

## 产物

保存模型时会生成：

1. `models/timing_lstm_madl_<run_tag>.pt`：PyTorch checkpoint。
2. `models/timing_lstm_madl_<run_tag>.json`：元信息，包含特征数、训练摘要和 checkpoint 路径。

`load` 模式推荐通过 `--timing-model-summary-json` 或 `--timing-models-load-dir` 重新加载该模型；兼容旧的 `--model-summary-json`、`--models-load-dir`，以及不推荐的 `--timing-model-path`。加载时会读取 checkpoint 中保存的 `extra_cols`、特征标准化参数和历史市场状态，并把所需额外因子加入当期推理面板构建。
