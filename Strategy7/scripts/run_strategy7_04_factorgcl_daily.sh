#!/usr/bin/env bash
# bash /workspace/Quant/TradeSystem/Strategy7/scripts/run_strategy7_04_factorgcl_daily.sh
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
  --label-task return \
  --stock-model-type factor_gcl \
  --fgcl-seq-len 30 \
  --fgcl-future-look 20 \
  --fgcl-hidden-size 128 \
  --fgcl-num-layers 2 \
  --fgcl-num-factor 48 \
  --fgcl-gamma 1.0 \
  --fgcl-tau 0.25 \
  --fgcl-epochs 200 \
  --fgcl-lr 9e-5 \
  --fgcl-early-stop 20 \
  --fgcl-smooth-steps 5 \
  --fgcl-per-epoch-batch 100 \
  --fgcl-batch-size -1 \
  --fgcl-label-transform csranknorm \
  --fgcl-weight-decay 1e-4 \
  --fgcl-dropout 0.0 \
  --fgcl-device auto \
  --timing-model-type none \
  --portfolio-model-type equal_weight \
  --execution-model-type ideal_fill \
  --execution-scheme vwap30_vwap30 \
  --horizon 20 \
  --top-k 10 \
  --long-threshold 0.50 \
  --fee-bps 1.5 \
  --slippage-bps 1.5 \
  --save-models True \
  --output-dir auto
