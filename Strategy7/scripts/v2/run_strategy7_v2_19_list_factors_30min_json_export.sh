#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
quant_root="${QUANT_ROOT:-$(cd "${repo_root}/.." && pwd)}"
cd "${repo_root}"

output_dir="${repo_root}/Strategy7/outputs/smoke_v2/run_strategy7_19_list_factors_30min_json"

mkdir -p "${output_dir}"
mkdir -p "${output_dir}"

conda run -n "${CONDA_ENV:-env_quant}" --no-capture-output python \
  ./Strategy7/run_strategy7.py \
  --list-factors \
  --factor-freq 30min \
  --factor-packages "trend,reversal,liquidity,volatility,intraday_signature,intraday_micro,price_action,bridge,multi_freq" \
  --auto-export-factor-snapshot true \
  --export-factor-list \
  --factor-list-export-format json \
  --factor-list-export-path $exportPath \
  --output-dir "${output_dir}"

