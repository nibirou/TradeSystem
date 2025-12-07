<!-- 基于baostock日频及5分钟频的多因子策略 -->

# 数据→因子→多因子模型→组合→回测→风控→绩效评估

1. config.py — 全局配置
2. data_loader.py — 数据加载模块
3. features.py — 特征工程与因子构建
4. factor_model.py — 因子IC、滚动权重、多因子打分
5. backtest.py — 回测与交易执行引擎
6. reporting.py — 绩效评估与指标
7. main.py — 整体流程串联

读数据 → 特征工程 → 因子标准化 → 计算前瞻收益 → 滚动IC权重 → 打分 → 回测 → 输出结果。