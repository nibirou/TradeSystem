#!/usr/bin/env bash
# bash /workspace/Quant/TradeSystem/Strategy7/scripts/run_strategy7_01_compat_daily.sh
set -euo pipefail

cd /workspace/Quant/TradeSystem

python3 ./Strategy7/run_strategy7.py \
  --data-root /workspace/Quant/data_baostock/stock_hist/hs300 \
  --hs300-list-path /workspace/Quant/data_baostock/metadata/stock_list_hs300.csv \
  --index-root /workspace/Quant/data_baostock/ak_index \
  --train-start 2024-01-01 \
  --train-end 2024-12-31 \
  --test-start 2025-01-01 \
  --test-end 2025-12-31 \
  --factor-freq D \
  --label-task direction \
  --stock-model-type decision_tree \
  --timing-model-type none \
  --portfolio-model-type equal_weight \
  --execution-model-type ideal_fill \
  --execution-scheme vwap30_vwap30 \
  --horizon 5 \
  --top-k 10 \
  --long-threshold 0.50 \
  --fee-bps 1.5 \
  --slippage-bps 1.5 \
  --output-dir auto

