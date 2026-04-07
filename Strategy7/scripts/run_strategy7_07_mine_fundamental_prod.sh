#!/usr/bin/env bash
# bash /workspace/Quant/TradeSystem/Strategy7/scripts/run_strategy7_07_mine_fundamental_prod.sh
set -euo pipefail

cd /workspace/Quant/TradeSystem

python3 ./Strategy7/run_factor_mining.py \
  --framework fundamental_multiobj \
  --data-root /workspace/Quant/data_baostock/stock_hist/hs300 \
  --hs300-list-path /workspace/Quant/data_baostock/metadata/stock_list_hs300.csv \
  --index-root /workspace/Quant/data_baostock/ak_index \
  --train-start 2021-01-01 \
  --train-end 2023-12-31 \
  --valid-start 2024-01-01 \
  --valid-end 2024-12-31 \
  --factor-freq D \
  --horizon 5 \
  --population-size 128 \
  --generations 20 \
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
  --save-format parquet \
  --min-abs-ic-mean 0.010 \
  --min-ic-win-rate 0.52 \
  --min-ic-ir 0.10 \
  --min-long-excess-annualized 0.01 \
  --min-long-sharpe 0.30 \
  --min-long-win-rate 0.50 \
  --min-coverage 0.80
