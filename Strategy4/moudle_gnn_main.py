# gnn_module_main.py
from moudle1_dataloader import DataConfig, PeriodConfig, load_data_bundle_update
from module3_labelbuilder import LabelBuilder

from module2_gnn_factorengine import GNNFactorEngine
from module4_gnn_dataset import GNNDataConfig, make_gnn_samples
from module5_gnn_trainer import train_gnn, predict_gnn
from module6_gnn_backtest import backtest_score_long_only, stat_winrate_with_labelbuilder

import pandas as pd

if __name__ == "__main__":
    # 1) 数据加载（模块 1）
    data_cfg = DataConfig(
        base_dir="/workspace/Quant/data_baostock",
        trade_calendar_dir="/workspace/Quant/data_baostock/metadata/trade_datas.csv",
        all_stock_list_dir="/workspace/Quant/data_baostock/metadata/stock_list_all.csv",
    )
    # data_cfg = DataConfig(
    #     base_dir="E:/pythonProject/data_baostock",
    #     trade_calendar_dir="E:/pythonProject/data_baostock/metadata/trade_datas.csv",
    #     all_stock_list_dir="E:/pythonProject/data_baostock/metadata/stock_list_all.csv",
    # )

    period_cfg = PeriodConfig(
        factor_start="2025-01-02",   # 因子统计时间
        factor_end="2025-12-30",     # 因子统计时间
        train_start="2025-01-02",    # 训练
        train_end="2025-11-15",
        backtest_start="2025-11-16", # 回测
        backtest_end="2025-11-30",
        inference_day="2025-12-30",   # 预测
        factor_buffer_n=60
    )

    bundle = load_data_bundle_update(data_cfg, period_cfg, pools=("hs300",))  # 先单池跑通
    daily = bundle["daily"]
    minute5 = bundle["minute5"]
    trade_dates = bundle["trade_dates"]

    # 1) 因子（保持一致：date/code + 因子列）
    fe = GNNFactorEngine(daily, minute5, trade_dates, base_dir=data_cfg.base_dir, pool="hs300")
    factor_df = fe.compute_all_factors_with_fundamental(use_akshare=False, use_baostock_q=True)

    # 2) 构建图样本
    gcfg = GNNDataConfig(lookback_corr=60, horizon=20, topk_graph=10, min_nodes=80)
    samples = make_gnn_samples(trade_dates, factor_df, daily, gcfg)
    print("samples:", len(samples), "feat_dim:", samples[0]["X"].shape[1])

    print(samples)

    # 3) 训练（按你 PeriodConfig）
    model, scaler = train_gnn(
        samples=samples,
        train_start=period_cfg.train_start,
        train_end=period_cfg.train_end,
        val_end=period_cfg.backtest_start,   # 简单用 backtest_start 作为 val_end，你也可另设
        hidden=128, layers=2, dropout=0.1,
        lr=1e-3, weight_decay=1e-4, epochs=30,
        device="cuda"
    )

    # 4) 推理：得到 pred_ret/pred_vol
    pred = predict_gnn(model, samples, scaler, device="cuda")
    print(pred.head())

    # 5) 回测A：标准持有期收益回测（用 close）
    bt = backtest_score_long_only(
        daily=daily,
        pred=pred,
        trade_dates=trade_dates,
        start_date=period_cfg.backtest_start,
        end_date=period_cfg.backtest_end,
        rebalance="M",
        topN=50,
        lam_vol=0.3,
        vol_filter_q=0.9
    )
    print(bt.tail())

    # 6) 回测B：复用你 LabelBuilder 的“胜率统计”（可选）
    label_builder = LabelBuilder(minute5, trade_dates)
    stat = stat_winrate_with_labelbuilder(
        pred=pred,
        label_builder=label_builder,
        trade_dates=trade_dates,
        start_date=period_cfg.backtest_start,
        end_date=period_cfg.backtest_end,
        topk=30,
        horizon_n=10,
        stop_loss=-0.10,
        lam_vol=0.3
    )
    print(stat.describe())

    # 保存
    out_dir = f"{data_cfg.base_dir}/gnn_outputs"
    import os, torch
    os.makedirs(out_dir, exist_ok=True)
    pred.to_parquet(f"{out_dir}/pred_hs300.parquet", index=False)
    bt.to_csv(f"{out_dir}/bt_hs300.csv", index=False, encoding="utf-8")
    stat.to_csv(f"{out_dir}/stat_hs300.csv", encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "scaler": scaler}, f"{out_dir}/model_hs300.pt")
    print("saved:", out_dir)
