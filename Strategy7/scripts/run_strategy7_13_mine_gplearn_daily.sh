#!/usr/bin/env bash
# bash /workspace/Quant/TradeSystem/Strategy7/scripts/run_strategy7_13_mine_gplearn_daily.sh
set -euo pipefail

cd /workspace/Quant/TradeSystem

python3 ./Strategy7/run_factor_mining.py \
  --framework gplearn_symbolic_alpha \
  --data-root /workspace/Quant/data_baostock/stock_hist/hs300 \
  --hs300-list-path /workspace/Quant/data_baostock/metadata/stock_list_hs300.csv \
  --index-root /workspace/Quant/data_baostock/ak_index \
  --train-start 2021-01-01 \
  --train-end 2023-12-31 \
  --valid-start 2024-01-01 \
  --valid-end 2024-12-31 \
  --factor-freq D \
  --horizon 5 \
  --top-n 20 \
  --corr-threshold 0.60 \
  --min-cross-section 30 \
  --top-frac 0.10 \
  --random-state 42 \
  --gp-population-size 400 \
  --gp-generations 12 \
  --gp-num-runs 3 \
  --gp-n-components 24 \
  --gp-hall-of-fame 64 \
  --gp-tournament-size 20 \
  --gp-parsimony 0.001 \
  --gp-metric spearman \
  --gp-function-set add,sub,mul,div,sqrt,log,abs,neg,max,min \
  --gp-prefilter-topk 80 \
  --gp-train-sample-frac 0.40 \
  --gp-max-train-rows 220000 \
  --gp-max-depth 5 \
  --gp-max-samples 0.90 \
  --gp-num-jobs -1 \
  --factor-store-root /workspace/Quant/data_baostock \
  --catalog-path auto \
  --save-format parquet

