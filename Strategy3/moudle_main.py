# 建仓后5个交易日内卖出，每只股票取这期间的最大收益，不设止盈，期间止损设10%，统计一段时间内，股票池中每只股票的收益率

# 每隔N日，整合所有股票的收益率和因子值，把因子值作为因子暴露，进行一次截面回归，得到每个因子的因子收益率，然后取平均作为最终的因子收益率

# 当日的建仓价格依据5分钟行情的数据情况来决定

# 模型构建完成后，通过一定统计检验方法检验因子收益率（卡方统计量或者t统计量）

# 模型推理：T日输入因子计算结果，输出T+1日所有股票的预测收益率，进行预测收益率排名

# 模型回测：1、策略历史胜率统计（选股统计胜率，第一步）   2、策略实盘仿真回测（构建投资组合/投资组合动态调整，第二步）（未来要把这两部分回测完善封装）

from moudle1_dataloader import DataConfig, PeriodConfig, load_data_bundle_update
from moudle2_factorengine import FactorEngine
from module3_labelbuilder import LabelBuilder
from module5_stat_backtest import StatBacktester

import pandas as pd

if __name__ == "__main__":
    # 1) 数据加载（模块 1）
    data_cfg = DataConfig(
        # base_dir="./data_baostock",
        base_dir="/workspace/Quant/data_baostock",
        trade_calendar_dir="/workspace/Quant/data_baostock/metadata/trade_datas.csv"
    )
    period_cfg = PeriodConfig(
        factor_start="2025-10-15",   # 计算某些因子需要过去一段时间窗口的行情数据，设置了一个自定义提前量；进行截面回归，需要确定两次截面回归之间的时间间隔，设置了一个自定义提前量
        factor_end="2025-12-30",     # 计算因子的时间段拆分成训练 和 回测两部分
        train_start="2025-10-15",    # 训练
        train_end="2025-11-15",
        backtest_start="2025-11-16", # 回测
        backtest_end="2025-11-30",
        inference_day="2025-12-30"   # 预测
    )
    # bundle = load_data_bundle(data_cfg, period_cfg, pools=("hs300", "zz500"))
    bundle = load_data_bundle_update(data_cfg, period_cfg, pools=("hs300", "zz500"))
    daily = bundle["daily"]
    minute5 = bundle["minute5"]
    trade_dates = bundle["trade_dates"] # factor_start至factor_end之间的交易日列表
    
    # print(trade_dates)
    # print(minute5)

    # 2) 因子计算（模块 2）
    fe = FactorEngine(daily, minute5, trade_dates)
    factor_df = fe.compute_all_factors()
    
    # === 第一类回测，统计策略胜率 ===
    label_builder = LabelBuilder(minute5, trade_dates)

    factor_cols = ["L1","L2","S1","S2","S3","S4","F1","F2","R1"]

    bt = StatBacktester(
        factor_df=factor_df,
        label_builder=label_builder,
        trade_dates=trade_dates,
        factor_cols=factor_cols,
        date_stride=10,          # <<< 每隔N日取样
        fit_mode="fmb_mean",     # 推荐：先每日回归再均值  pooled/fmb_mean/per_stock_ts
        topk=30
    )

    # 训练
    train_panel = bt.build_dataset(
        start_date=period_cfg.train_start,
        end_date=period_cfg.train_end,
        horizon_n=10,
        stop_loss=-0.10,
        label="train"
    )
    
    # test_mode = "backtest"  
    test_mode = "inference"  

    if test_mode == "backtest":
        # 一定时间段内回测
        test_panel = bt.build_dataset(
            start_date=period_cfg.backtest_start,
            end_date=period_cfg.backtest_end,
            horizon_n=10,
            stop_loss=-0.10,
            label="backtest" 
        )
        print("backtest_panel", test_panel)

    if test_mode == "inference":
        # 某日预测（比如最新交易日预测）
        test_panel = bt.build_dataset(
            start_date=period_cfg.inference_day,
            end_date=period_cfg.inference_day,
            label="inference" 
        )
    
    print("test_panel", test_panel, "\n", "test_mode", test_mode)
    
    if bt.fit_mode == "per_stock_ts": 
        beta_global, beta_by_stock, residuals_by_stock = bt.fit_global_beta(train_panel)
    else:
        beta_global = bt.fit_global_beta(train_panel)

    if test_mode == "backtest":
        daily_df = bt.evaluate_in_sample(test_panel, beta_global)
        print(daily_df.describe())
    
    if test_mode == "inference":
        pred_df = bt.inference_in_sample(test_panel, beta_global)

        # 剔除 300 30 688开头股票
        # 提取小数点后的股票代码前缀
        code_prefix = pred_df['code'].str.split('.').str[1]
        # 创建过滤条件：保留不以 '30'、'300'、'688' 开头的行
        mask = ~(
            code_prefix.str.startswith('30') |
            code_prefix.str.startswith('300') |
            code_prefix.str.startswith('688')
        )
        # 应用过滤
        df_filtered = pred_df[mask]
        print(df_filtered)

        all_stock = pd.read_csv("/workspace/Quant/data_baostock/metadata/stock_list_all.csv")
        print(all_stock)
        # selected_stock = all_stock.loc[all_stock["code"].isin(df_filtered["code"])]
        selected_stock = pd.merge(df_filtered, all_stock, on='code', how='inner')
        print(selected_stock)

        print(f"{period_cfg.factor_end}日因子得分策略选股（前20名，剔除创业板和北证）：", selected_stock.head(20))

