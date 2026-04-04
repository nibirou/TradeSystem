# Strategy7 代码阅读地图

本文档给出建议阅读顺序，帮助快速理解 `Strategy7`。

## 1. 推荐阅读顺序（主流程）

1. `run_strategy7.py`
2. `strategy7/config.py`
3. `strategy7/pipeline/runner.py`
4. `strategy7/data/loaders.py`
5. `strategy7/factors/defaults.py` + `strategy7/factors/labeling.py`
6. `strategy7/models/*`
7. `strategy7/backtest/engine.py` + `strategy7/backtest/metrics.py`
8. `strategy7/pipeline/artifacts.py`

## 2. 关键对象关系

1. `RunConfig`：整次实验的统一配置
2. `MarketBundle`：原始日线/分钟数据容器
3. `FeatureBundle`：按频率组织的特征容器
4. `FactorLibrary`：因子注册中心（默认 + catalog + 自定义）
5. `StockSelectionModel` / `TimingModel` / `PortfolioModel` / `ExecutionModel`：四类可插拔模型接口

## 3. 一次完整运行发生了什么

1. `config` 解析参数并做合法性校验
2. `loaders` 读取行情并构建基础特征
3. `pipeline` 合并 catalog 因子与外部数据源
4. `factors` 计算因子面板并做截面预处理
5. `labeling` 构造未来收益/方向/波动率标签并时间切分
6. `models` 训练选股/择时/组合/执行模型
7. `backtest` 计算收益、净值、超额、统计指标
8. `artifacts` 保存结果、图表、模型与 summary

## 4. 因子挖掘阅读顺序

1. `run_factor_mining.py`
2. `strategy7/mining/runner.py`
3. `strategy7/mining/formulas.py`
4. `strategy7/mining/evaluation.py`
5. `strategy7/mining/nsga.py`
6. `strategy7/mining/catalog.py`

## 5. 易混点提示

1. `factor_freq` 决定模型训练/回测使用哪个频率视图
2. `label_task` 决定训练目标列；评估会兼容连续与二分类
3. `--list-factors` 当前是“快速路径”，不会依赖行情文件
4. catalog 因子是“值表 + 注册定义”两步生效

