#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
quant_root="${QUANT_ROOT:-$(cd "${repo_root}/.." && pwd)}"
cd "${repo_root}"

store_root="${repo_root}/Strategy7/outputs/smoke_v2/factor_mining_store"

mkdir -p "${store_root}"

conda run -n "${CONDA_ENV:-env_quant}" --no-capture-output python \
  ./Strategy7/run_factor_mining.py \
  --framework minute_parametric \
  --data-root ${quant_root}/data_baostock/stock_hist/hs300 \
  --hs300-list-path ${quant_root}/data_baostock/metadata/stock_list_hs300.csv \
  --index-root ${quant_root}/data_baostock/ak_index \
  --train-start 2024-01-01 \
  --train-end 2024-02-29 \
  --valid-start 2024-03-01 \
  --valid-end 2024-03-31 \
  --factor-freq 30min \
  --factor-packages "trend,reversal,liquidity,volatility,intraday_signature,intraday_micro,price_action,bridge,multi_freq" \
  --max-files 12 \
  --horizon 8 \
  --execution-scheme open5_twap_last30 \
  --population-size 8 \
  --generations 1 \
  --elite-size 2 \
  --mutation-rate 0.25 \
  --crossover-rate 0.7 \
  --top-n 5 \
  --corr-threshold 0.65 \
  --min-cross-section 12 \
  --top-frac 0.1 \
  --factor-store-root "${store_root}" \
  --save-format parquet

