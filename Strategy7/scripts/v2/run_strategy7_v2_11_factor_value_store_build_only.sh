#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
quant_root="${QUANT_ROOT:-$(cd "${repo_root}/.." && pwd)}"
cd "${repo_root}"

output_dir="${repo_root}/Strategy7/outputs/smoke_v2/run_strategy7_11_factor_value_store"
store_root="${repo_root}/Strategy7/outputs/smoke_v2/factor_value_store"

mkdir -p "${output_dir}"
mkdir -p "${store_root}"

conda run -n "${CONDA_ENV:-env_quant}" --no-capture-output python \
  ./Strategy7/run_strategy7.py \
  --data-root ${quant_root}/data_baostock/stock_hist/hs300 \
  --hs300-list-path ${quant_root}/data_baostock/metadata/stock_list_hs300.csv \
  --index-root ${quant_root}/data_baostock/ak_index \
  --factor-freq D \
  --factor-packages "trend,reversal" \
  --max-files 15 \
  --enable-factor-value-store true \
  --factor-value-store-root "${store_root}" \
  --factor-value-store-format parquet \
  --factor-value-store-build-all true \
  --factor-value-store-build-only true \
  --output-dir "${output_dir}" \
  --log-level normal

