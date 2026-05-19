#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
quant_root="${QUANT_ROOT:-$(cd "${repo_root}/.." && pwd)}"
cd "${repo_root}"

output_dir="${repo_root}/Strategy7/outputs/smoke_v2/run_strategy7_24_train_dfq_timesnet"

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
  --stock-model-type dfq_timesnet \
  --timesnet-seq-len 12 \
  --timesnet-hidden-size 48 \
  --timesnet-e-layers 1 \
  --timesnet-hidden-size2 48 \
  --timesnet-periods "3,12" \
  --timesnet-num-kernels 2 \
  --timesnet-dropout 0.0 \
  --timesnet-epochs 1 \
  --timesnet-lr 2e-4 \
  --timesnet-weight-decay 0.0 \
  --timesnet-early-stop 1 \
  --timesnet-smooth-steps 1 \
  --timesnet-per-epoch-batch 8 \
  --timesnet-batch-size 64 \
  --timesnet-label-transform cszscore \
  --timesnet-input-clip 3.0 \
  --timesnet-device cpu \
  --timing-model-type none \
  --portfolio-model-type equal_weight \
  --execution-model-type ideal_fill \
  --model-run-mode train \
  --horizon 5 \
  --top-k 10 \
  --long-threshold 0.5 \
  --save-models true \
  --output-dir "${output_dir}"
