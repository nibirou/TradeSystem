#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
quant_root="${QUANT_ROOT:-$(cd "${repo_root}/.." && pwd)}"
cd "${repo_root}"

output_dir="${repo_root}/Strategy7/outputs/smoke_v2/run_strategy7_28_train_dtlc_rl"
mkdir -p "${output_dir}"

conda run -n "${CONDA_ENV:-env_quant}" --no-capture-output python \
  ./Strategy7/run_strategy7.py \
  --data-root ${quant_root}/data_baostock/stock_hist/hs300 \
  --hs300-list-path ${quant_root}/data_baostock/metadata/stock_list_hs300.csv \
  --index-root ${quant_root}/data_baostock/ak_index \
  --train-start 2023-07-01 \
  --train-end 2024-03-31 \
  --test-start 2024-04-01 \
  --test-end 2024-06-30 \
  --factor-freq D \
  --factor-packages "trend,reversal,liquidity,volatility" \
  --label-task return \
  --max-files 15 \
  --data-load-workers "${STRATEGY7_DATA_LOAD_WORKERS:-0}" \
  --stock-model-type dtlc_rl \
  --dtlc-seq-len 12 \
  --dtlc-alpha-scales 4,8,12 \
  --dtlc-hidden-size 32 \
  --dtlc-latent-size 16 \
  --dtlc-num-heads 4 \
  --dtlc-encoder-layers 1 \
  --dtlc-grn-layers 1 \
  --dtlc-ffn-mult 2 \
  --dtlc-pretrain-epochs 1 \
  --dtlc-ppo-epochs 1 \
  --dtlc-ppo-update-epochs 1 \
  --dtlc-per-epoch-batch 8 \
  --dtlc-batch-size 64 \
  --dtlc-ppo-batch-size 4 \
  --dtlc-early-stop 1 \
  --dtlc-min-cross-section 4 \
  --dtlc-device cpu \
  --timing-model-type none \
  --portfolio-model-type equal_weight \
  --execution-model-type ideal_fill \
  --model-run-mode train \
  --horizon 5 \
  --top-k 10 \
  --long-threshold 0.5 \
  --save-models true \
  --output-dir "${output_dir}"
