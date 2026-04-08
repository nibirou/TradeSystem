#!/usr/bin/env bash
# bash /workspace/Quant/TradeSystem/Strategy7/scripts/run_strategy7_05_mine_fundamental_daily.sh
set -euo pipefail

cd /workspace/Quant/TradeSystem

python3 ./Strategy7/run_factor_mining.py \
  --framework fundamental_multiobj \
  --data-root /workspace/Quant/data_baostock/stock_hist/hs300 \
  --hs300-list-path /workspace/Quant/data_baostock/metadata/stock_list_hs300.csv \
  --index-root /workspace/Quant/data_baostock/ak_index \
  --train-start 2025-11-01 \
  --train-end 2025-11-30 \
  --valid-start 2025-12-01 \
  --valid-end 2025-12-31 \
  --factor-freq 5min \
  --horizon 15 \
  --population-size 128 \
  --generations 5 \
  --elite-size 12 \
  --mutation-rate 0.25 \
  --crossover-rate 0.70 \
  --top-n 20 \
  --corr-threshold 0.60 \
  --min-cross-section 30 \
  --top-frac 0.10 \
  --random-state 42 \
  --factor-store-root /workspace/Quant/data_baostock \
  --catalog-path auto \
  --save-format csv
