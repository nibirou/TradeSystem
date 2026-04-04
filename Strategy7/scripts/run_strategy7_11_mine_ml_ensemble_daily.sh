#!/usr/bin/env bash
# bash /workspace/Quant/TradeSystem/Strategy7/scripts/run_strategy7_11_mine_ml_ensemble_daily.sh
set -euo pipefail

cd /workspace/Quant/TradeSystem

python3 ./Strategy7/run_factor_mining.py \
  --framework ml_ensemble_alpha \
  --data-root /workspace/Quant/data_baostock/stock_hist/hs300 \
  --hs300-list-path /workspace/Quant/data_baostock/metadata/stock_list_hs300.csv \
  --index-root /workspace/Quant/data_baostock/ak_index \
  --train-start 2021-01-01 \
  --train-end 2023-12-31 \
  --valid-start 2024-01-01 \
  --valid-end 2024-12-31 \
  --factor-freq D \
  --horizon 5 \
  --elite-size 8 \
  --mutation-rate 0.30 \
  --crossover-rate 0.70 \
  --top-n 20 \
  --corr-threshold 0.60 \
  --min-cross-section 30 \
  --top-frac 0.10 \
  --random-state 42 \
  --ml-population-size 48 \
  --ml-generations 10 \
  --ml-model-pool rf,et,hgbt \
  --ml-prefilter-topk 80 \
  --ml-feature-min 10 \
  --ml-feature-max 36 \
  --ml-train-sample-frac 0.40 \
  --ml-max-train-rows 220000 \
  --ml-num-jobs -1 \
  --factor-store-root /workspace/Quant/data_baostock \
  --catalog-path auto \
  --save-format parquet
