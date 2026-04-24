#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
quant_root="${QUANT_ROOT:-$(cd "${repo_root}/.." && pwd)}"
cd "${repo_root}"

output_dir="${repo_root}/Strategy7/outputs/smoke_v2/run_strategy7_12_train_weekly_volatility"

mkdir -p "${output_dir}"

conda run -n "${CONDA_ENV:-env_quant}" --no-capture-output python \
  ./Strategy7/run_strategy7.py \
  --data-root ${quant_root}/data_baostock/stock_hist/hs300 \
  --hs300-list-path ${quant_root}/data_baostock/metadata/stock_list_hs300.csv \
  --index-root ${quant_root}/data_baostock/ak_index \
  --train-start 2021-01-01 \
  --train-end 2023-12-31 \
  --test-start 2024-01-01 \
  --test-end 2024-12-31 \
  --factor-freq W \
  --factor-packages "trend,reversal,liquidity,volatility,context,multi_freq" \
  --label-task volatility \
  --max-files 20 \
  --enable-factor-engineering true \
  --fe-orth-method none \
  --fe-corr-threshold 0.9 \
  --fe-preselect-top-n 120 \
  --stock-model-type decision_tree \
  --timing-model-type none \
  --portfolio-model-type equal_weight \
  --execution-model-type ideal_fill \
  --model-run-mode train \
  --horizon 3 \
  --top-k 10 \
  --long-threshold 0.5 \
  --execution-scheme daily_close_daily_close \
  --fee-bps 1.5 \
  --slippage-bps 1.5 \
  --save-models true \
  --output-dir "${output_dir}"

