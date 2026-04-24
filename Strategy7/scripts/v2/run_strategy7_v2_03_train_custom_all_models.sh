#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
quant_root="${QUANT_ROOT:-$(cd "${repo_root}/.." && pwd)}"
cd "${repo_root}"

output_dir="${repo_root}/Strategy7/outputs/smoke_v2/run_strategy7_03_train_custom_all"

mkdir -p "${output_dir}"

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
  --max-files 20 \
  --stock-model-type decision_tree \
  --custom-stock-model-py ./Strategy7/strategy7/plugins/custom_stock_model_template.py \
  --timing-model-type none \
  --custom-timing-model-py ./Strategy7/strategy7/plugins/custom_timing_model_template.py \
  --portfolio-model-type equal_weight \
  --custom-portfolio-model-py ./Strategy7/strategy7/plugins/custom_portfolio_model_template.py \
  --execution-model-type ideal_fill \
  --custom-execution-model-py ./Strategy7/strategy7/plugins/custom_execution_model_template.py \
  --model-run-mode train \
  --horizon 5 \
  --top-k 10 \
  --long-threshold 0.5 \
  --save-models true \
  --output-dir "${output_dir}"

