"""CLI entry for Strategy7 modular quant framework."""

from __future__ import annotations

import json

from strategy7.config import build_run_config, parse_args
from strategy7.pipeline.runner import run_pipeline


def main() -> None:
    args = parse_args()
    cfg = build_run_config(args)

    print("=== Strategy7 Parameters ===")
    print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2, default=str))
    print()

    summary = run_pipeline(cfg)

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
    save_models_enabled = bool(outputs.get("save_models_enabled", False))
    model_files = outputs.get("model_files", {}) or {}
    if save_models_enabled:
        print(f"{'model_files':24s}: {len(model_files)} models persisted")
    else:
        print(f"{'model_files':24s}: not saved (enable --save-models)")


if __name__ == "__main__":
    main()
