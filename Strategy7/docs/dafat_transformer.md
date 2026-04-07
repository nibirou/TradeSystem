# DAFAT 在 Strategy7 中的复现与实现说明

本文档说明研报《DAFAT：基于 Transformer 模型的自适应解决方案》在 `Strategy7` 的工程化复现方式、代码映射关系和运行方法。

## 1. 接入位置

DAFAT 作为 `stock_selection` 模块中的新模型接入：

1. 新增模型实现：`strategy7/models/stock_selection/dafat_transformer_model.py`
2. 模型工厂注册：`strategy7/models/stock_selection/factory.py`
3. CLI 参数与配置：`strategy7/config.py`
4. 运行脚本：`scripts/run_strategy7_12_dafat_daily.*`

## 2. 研报结构到代码的映射

### 2.1 动态位置编码（DPE）

研报要点：

1. 时间周期编码（周/月/季度）
2. 市场状态编码（市场波动率、行业轮动、市场流动性）
3. 双通道门控融合

代码映射：

1. 时间周期编码：`_calendar_features_from_times` 生成 `sin/cos` 周期特征
2. 市场状态编码：`_build_market_state_frame` + `state_gru`
3. 门控融合：`DynamicPositionalEncoding.forward`

### 2.2 稀疏注意力（SA）

研报要点：

1. 波动率门控
2. 局部注意力窗口
3. Top-k 稀疏选择

代码映射：

1. 波动率门控：`SparseSelfAttention.forward` 中基于 `vol_quantile` 的 key 掩码
2. 局部窗口：`local_window` 带状掩码
3. Top-k：每行保留 `topk_ratio` 的注意力连接

### 2.3 多尺度融合（MF）

研报要点：

1. 微观/中观/宏观三尺度
2. 跨尺度注意力
3. 门控融合 + 残差

代码映射：

1. 三尺度构造：`MultiScaleFusion._pool_and_upsample`（默认尺度 1/5/20）
2. 跨尺度注意力：`attn_micro / attn_meso / attn_macro`
3. 门控融合：`gate_net` + 残差回注入主干

## 3. 训练目标与流程

研报基线采用 `1 - IC` 作为核心训练目标。工程实现中使用：

1. 主损失：`L_ic = 1 - corr(pred, target)`（可微 IC 损失）
2. 稳定项：`L_mse = mse(pred, target)`
3. 总损失：`L = L_ic + dafat_mse_weight * L_mse`

训练组织方式：

1. 按交易日做截面 batch（与 IC 目标对齐）
2. 训练/验证日期切分 80%/20%
3. 早停 + 最优状态平滑

## 4. 标签与输入规范

1. 推荐 `--label-task return`
2. DAFAT 输入仍走 `Strategy7` 的统一 `factor_cols`，并复用主流程的数据/因子/预处理链路
3. 训练后 `predict_score` 输出按日截面分位数，兼容现有回测引擎

## 5. 关键参数（默认值）

1. `dafat_seq_len=40`
2. `dafat_hidden_size=128`
3. `dafat_num_layers=2`
4. `dafat_num_heads=4`
5. `dafat_local_window=20`
6. `dafat_topk_ratio=0.30`
7. `dafat_meso_scale=5`
8. `dafat_macro_scale=20`
9. `dafat_epochs=200`
10. `dafat_lr=1e-4`
11. `dafat_use_dpe=true`
12. `dafat_use_sparse_attn=true`
13. `dafat_use_multiscale=true`

## 6. 一键运行

Windows PowerShell：

```powershell
.\Strategy7\scripts\run_strategy7_12_dafat_daily.ps1
```

Linux/macOS：

```bash
bash Strategy7/scripts/run_strategy7_12_dafat_daily.sh
```

## 7. 说明与边界

1. 研报中的部分专有数据处理细节未公开，工程实现采用了可复现的等价实现（特别是市场状态与多尺度构造细节）。
2. 参数、损失函数和模块结构已经在 `stock_selection` 接口内完整打通，并与 Strategy7 回测链路保持一致。
3. 可通过 `--dafat-use-dpe/--dafat-use-sparse-attn/--dafat-use-multiscale` 做消融测试。
