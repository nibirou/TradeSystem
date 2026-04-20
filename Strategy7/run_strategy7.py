"""CLI entry for Strategy7 modular quant framework."""

from __future__ import annotations

import json

from strategy7.config import build_run_config, parse_args
from strategy7.core.utils import log_progress, set_log_level
from strategy7.pipeline.runner import run_pipeline


def main() -> None:
    # 第一步：解析并标准化日志级别（quiet/normal/verbose）。
    args = parse_args()
    effective_log_level = set_log_level(
        level=str(args.log_level),
        quiet=bool(args.quiet),
        verbose=bool(args.verbose),
    )
    log_progress(f"log level: {effective_log_level}", module="run_strategy7")
    log_progress("parsing and validating CLI arguments", module="run_strategy7")

    # 第二步：将 CLI 参数转换为强类型配置对象，并执行参数合法性校验。
    cfg = build_run_config(args)
    log_progress("configuration ready, start pipeline", module="run_strategy7")

    # 第三步：打印运行参数快照，便于复现实验。
    print("=== Strategy7 Parameters ===")
    print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2, default=str))
    print()

    # 第四步：执行端到端主流程（数据->因子->模型->回测->产物落盘）。
    summary = run_pipeline(cfg)
    summary_path = summary.get("outputs", {}).get("summary_json", "")
    if summary_path:
        log_progress(f"pipeline finished, summary: {summary_path}", module="run_strategy7")
    else:
        log_progress("pipeline finished", module="run_strategy7")

    # 第五步：输出关键评估结果，便于命令行快速观察。
    print("=== Model Metrics ===")
    print(json.dumps(summary.get("model_metrics", {}), ensure_ascii=False, indent=2))
    print("=== Backtest Metrics (Strategy) ===")
    print(json.dumps(summary.get("backtest_metrics", {}).get("strategy", {}), ensure_ascii=False, indent=2))
    print("=== Model IC Summary ===")
    print(json.dumps(summary.get("model_ic_summary", {}), ensure_ascii=False, indent=2))
    print("=== Score Spread (Q5-Q1) ===")
    print(json.dumps(summary.get("model_score_spread", {}), ensure_ascii=False, indent=2))
    print()

    outputs = summary.get("outputs", {})
    # 第六步：打印核心产物路径（预测、持仓、交易、曲线、IC等）。
    for k in [
        "predictions_csv",
        "trades_csv",
        "positions_csv",
        "curve_csv",
        "factor_ic_summary_csv",
        "model_ic_series_csv",
        "backtest_main_plot_png",
        "backtest_excess_plot_png",
    ]:
        if k in outputs:
            print(f"{k:24s}: {outputs[k]}")

    if bool(summary.get("factor_list_exported", False)):
        print(f"{'factor_list_export':24s}: {summary.get('factor_list_export_path', '')}")

    save_models_enabled = bool(outputs.get("save_models_enabled", False))
    model_files = outputs.get("model_files", {}) or {}
    # 第七步：给出模型落盘状态（是否保存成功）。
    if save_models_enabled:
        print(f"{'model_files':24s}: {len(model_files)} models persisted")
    else:
        print(f"{'model_files':24s}: not saved (enable --save-models)")


if __name__ == "__main__":
    main()
