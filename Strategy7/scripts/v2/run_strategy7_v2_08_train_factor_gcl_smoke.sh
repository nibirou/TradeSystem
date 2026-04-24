#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
quant_root="${QUANT_ROOT:-$(cd "${repo_root}/.." && pwd)}"
cd "${repo_root}"

output_dir="${repo_root}/Strategy7/outputs/smoke_v2/run_strategy7_08_train_factor_gcl"

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
  --stock-model-type factor_gcl \
  --fgcl-seq-len 12 \
  --fgcl-future-look 5 \
  --fgcl-hidden-size 48 \
  --fgcl-num-layers 1 \
  --fgcl-num-factor 16 \
  --fgcl-epochs 1 \
  --fgcl-lr 2e-4 \
  --fgcl-early-stop 1 \
  --fgcl-smooth-steps 2 \
  --fgcl-per-epoch-batch 8 \
  --fgcl-batch-size 64 \
  --fgcl-label-transform csranknorm \
  --fgcl-device cpu \
  --timing-model-type none \
  --portfolio-model-type equal_weight \
  --execution-model-type ideal_fill \
  --model-run-mode train \
  --horizon 5 \
  --top-k 10 \
  --long-threshold 0.5 \
  --save-models true \
  --output-dir "${output_dir}"

