#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
quant_root="${QUANT_ROOT:-$(cd "${repo_root}/.." && pwd)}"
cd "${repo_root}"

output_dir="${repo_root}/Strategy7/outputs/smoke_v2/run_strategy7_13_train_30min_intraday_realistic"

mkdir -p "${output_dir}"

conda run -n "${CONDA_ENV:-env_quant}" --no-capture-output python \
  ./Strategy7/run_strategy7.py \
  --data-root ${quant_root}/data_baostock/stock_hist/hs300 \
  --hs300-list-path ${quant_root}/data_baostock/metadata/stock_list_hs300.csv \
  --index-root ${quant_root}/data_baostock/ak_index \
  --train-start 2024-01-01 \
  --train-end 2024-02-29 \
  --test-start 2024-03-01 \
  --test-end 2024-03-31 \
  --factor-freq 30min \
  --factor-packages "trend,reversal,liquidity,volatility,intraday_signature,intraday_micro,price_action,bridge,multi_freq" \
  --label-task direction \
  --max-files 12 \
  --enable-factor-engineering false \
  --stock-model-type decision_tree \
  --timing-model-type volatility_regime \
  --portfolio-model-type dynamic_opt \
  --execution-model-type realistic_fill \
  --max-participation-rate 0.2 \
  --base-fill-rate 0.9 \
  --latency-bars 1 \
  --model-run-mode train \
  --enable-next-bar-inference true \
  --inference-top-k 10 \
  --horizon 8 \
  --top-k 8 \
  --long-threshold 0.55 \
  --execution-scheme open5_twap_last30 \
  --fee-bps 2.0 \
  --slippage-bps 2.0 \
  --save-models true \
  --output-dir "${output_dir}"

