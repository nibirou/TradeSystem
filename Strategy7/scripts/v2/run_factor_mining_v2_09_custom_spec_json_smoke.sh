#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
quant_root="${QUANT_ROOT:-$(cd "${repo_root}/.." && pwd)}"
cd "${repo_root}"

store_root="${repo_root}/Strategy7/outputs/smoke_v2/factor_mining_store"
spec_path="${repo_root}/Strategy7/scripts/v2/configs/custom_factor_specs_example.json"

mkdir -p "${store_root}"

conda run -n "${CONDA_ENV:-env_quant}" --no-capture-output python \
  ./Strategy7/run_factor_mining.py \
  --framework custom \
  --data-root ${quant_root}/data_baostock/stock_hist/hs300 \
  --hs300-list-path ${quant_root}/data_baostock/metadata/stock_list_hs300.csv \
  --index-root ${quant_root}/data_baostock/ak_index \
  --disable-fundamental-data \
  --disable-text-data \
  --train-start 2022-01-01 \
  --train-end 2023-06-30 \
  --valid-start 2023-07-01 \
  --valid-end 2023-12-31 \
  --factor-freq D \
  --factor-packages "trend,reversal,liquidity,volatility" \
  --custom-spec-json "${spec_path}" \
  --max-files 20 \
  --horizon 5 \
  --population-size 8 \
  --generations 1 \
  --elite-size 2 \
  --mutation-rate 0.2 \
  --crossover-rate 0.7 \
  --top-n 4 \
  --corr-threshold 0.7 \
  --min-cross-section 20 \
  --top-frac 0.1 \
  --factor-store-root "${store_root}" \
  --save-format parquet

