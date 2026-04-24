#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
quant_root="${QUANT_ROOT:-$(cd "${repo_root}/.." && pwd)}"
cd "${repo_root}"

output_dir="${repo_root}/Strategy7/outputs/smoke_v2/run_strategy7_16_train_factor_value_store_hydrate"
store_root="${repo_root}/Strategy7/outputs/smoke_v2/factor_value_store_hydrate"

mkdir -p "${output_dir}"
mkdir -p "${store_root}"

conda run -n "${CONDA_ENV:-env_quant}" --no-capture-output python \
  ./Strategy7/run_strategy7.py \
  --data-root ${quant_root}/data_baostock/stock_hist/hs300 \
  --hs300-list-path ${quant_root}/data_baostock/metadata/stock_list_hs300.csv \
  --index-root ${quant_root}/data_baostock/ak_index \
  --train-start 2024-01-01 \
  --train-end 2024-06-30 \
  --test-start 2024-07-01 \
  --test-end 2024-09-30 \
  --factor-freq D \
  --factor-packages "trend,reversal,liquidity,volatility" \
  --label-task direction \
  --max-files 20 \
  --enable-factor-value-store true \
  --factor-value-store-root "${store_root}" \
  --factor-value-store-format parquet \
  --factor-value-store-build-all false \
  --factor-value-store-build-only false \
  --auto-export-factor-snapshot true \
  --stock-model-type decision_tree \
  --timing-model-type none \
  --portfolio-model-type equal_weight \
  --execution-model-type ideal_fill \
  --model-run-mode train \
  --horizon 5 \
  --top-k 10 \
  --long-threshold 0.5 \
  --execution-scheme vwap30_vwap30 \
  --fee-bps 1.5 \
  --slippage-bps 1.5 \
  --save-models false \
  --output-dir "${output_dir}"

