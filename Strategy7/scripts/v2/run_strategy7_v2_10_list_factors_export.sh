#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
quant_root="${QUANT_ROOT:-$(cd "${repo_root}/.." && pwd)}"
cd "${repo_root}"

output_dir="${repo_root}/Strategy7/outputs/smoke_v2/run_strategy7_10_list_factors"

mkdir -p "${output_dir}"

conda run -n "${CONDA_ENV:-env_quant}" --no-capture-output python \
  ./Strategy7/run_strategy7.py \
  --list-factors \
  --factor-freq D \
  --factor-packages "trend,reversal,liquidity,volatility,flow,crowding" \
  --auto-export-factor-snapshot true \
  --export-factor-list \
  --factor-list-export-format markdown \
  --output-dir "${output_dir}" \
  --log-level quiet

