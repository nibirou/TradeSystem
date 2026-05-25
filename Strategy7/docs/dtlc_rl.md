# DTLC_RL in Strategy7

本文档说明研报《强化学习驱动下的解耦时序对比选股模型》在 `Strategy7` 中的工程化复现方式。模型入口为 `--stock-model-type dtlc_rl`。

## 1. 研报结构映射

研报模型为 Decoupled Temporal Contrastive Learning with Reinforcement Learning，核心包含四层：

1. Beta 空间：面向市场系统风险，使用 TCN 编码器。工程实现为线性投影后接两个因果扩张 TCN 块，扩张率为 1 和 2，并用全局平均池化输出 32 维默认编码。
2. Alpha 空间：面向个股特异量价信号，使用 20/40/60 三尺度 Transformer。工程实现分别截取多尺度窗口，Transformer 编码后上采样到统一 `seq_len`，再用可学习门控加权融合。
3. Theta 空间：面向基本面/安全边际，使用门控残差 MLP。工程实现为输入投影、若干 GRN 块和 LayerNorm 输出。
4. 强化学习融合：将三空间编码与市场环境状态输入 PPO Actor，输出三维空间权重，对三空间编码加权后由预测头输出未来收益预测。

监督训练阶段同时优化各空间预测头、线性融合头和 PPO Actor 的确定性融合头。损失函数包含 `1-IC`、MSE 辅助项、InfoNCE 对比损失和三空间正交约束。PPO 阶段固定编码器和预测头，用 RankIC、权重稳定性和权重分散度构造奖励。

## 2. 与框架适配

研报固定使用日频特征：Beta 5 个、Alpha 13 个、Theta 8 个。Strategy7 的因子来自配置和插件，可能是日频、分钟频、周频、月频、基本面、文本、catalog 或 FE 派生因子，因此实现中按因子名自动映射：

- Beta：`beta/mkt/market/context/vol/rv/liq/turn/amount/size/sent/crowding`
- Theta：`fund/pe/pb/roe/roic/eps/dividend/profit/cashflow/leverage/valuation/growth/quality`
- Alpha：其余因子

如果某个空间没有匹配因子，会回退到全量 `factor_cols`，保证所有频率和因子类型都能运行。模型仍使用统一接口：

- `fit(train_df, factor_cols, target_col)`
- `predict_score(df, factor_cols)`
- `save(folder, run_tag)`
- `model_run_mode=load` 加载 `.pt` checkpoint

## 3. 推荐设置

贴近研报的日频研究配置：

```bash
--stock-model-type dtlc_rl \
--factor-freq D \
--label-task return \
--horizon 20 \
--rebalance-stride 20 \
--dtlc-seq-len 60 \
--dtlc-alpha-scales 20,40,60 \
--dtlc-hidden-size 64 \
--dtlc-latent-size 32 \
--dtlc-lr 1e-4 \
--dtlc-label-transform cszscore
```

如果使用分钟频、周频或月频，建议先降低 `--dtlc-seq-len`、`--dtlc-pretrain-epochs`、`--dtlc-ppo-epochs` 做稳定性验证，再放大窗口。

## 4. 关键参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--dtlc-seq-len` | `60` | 三空间时序窗口长度 |
| `--dtlc-alpha-scales` | `20,40,60` | Alpha 多尺度 Transformer 窗口 |
| `--dtlc-hidden-size` | `64` | 编码器隐层维度 |
| `--dtlc-latent-size` | `32` | 每个空间输出编码维度 |
| `--dtlc-pretrain-epochs` | `80` | 监督预训练轮数 |
| `--dtlc-ppo-epochs` | `30` | PPO 融合控制器训练轮数 |
| `--dtlc-contrastive-weight` | `0.05` | InfoNCE 权重 |
| `--dtlc-orthogonal-weight` | `0.05` | 三空间正交约束权重 |
| `--dtlc-stable-weight` | `0.05` | PPO 奖励中的权重稳定项 |
| `--dtlc-diversity-weight` | `0.02` | PPO 奖励中的权重分散项 |

## 5. 运行示例

轻量冒烟：

```powershell
powershell -ExecutionPolicy Bypass -File Strategy7/scripts/v2/run_strategy7_v2_28_train_dtlc_rl_smoke.ps1
```

研究型训练：

```powershell
powershell -ExecutionPolicy Bypass -File Strategy7/scripts/v2/run_strategy7_v2_29_train_dtlc_rl.ps1
```

Linux：

```bash
bash Strategy7/scripts/v2/run_strategy7_v2_28_train_dtlc_rl_smoke.sh
bash Strategy7/scripts/v2/run_strategy7_v2_29_train_dtlc_rl.sh
```

## 6. 实现边界

研报的对比学习正样本基于未来 20 日收益率序列相关系数大于 80%，负样本基于相关系数小于 0%。Strategy7 当前训练接口只向选股模型暴露单一未来收益标签，因此实现采用收益排名邻域作为 InfoNCE 正样本近似；若后续主流程提供未来收益路径标签，可在 `_contrastive_loss` 中替换为原始相关系数构造方式。
