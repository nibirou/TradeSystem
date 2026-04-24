#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
quant_root="${QUANT_ROOT:-$(cd "${repo_root}/.." && pwd)}"
cd "${repo_root}"

store_root="${repo_root}/Strategy7/outputs/smoke_v2/factor_mining_store"

mkdir -p "${store_root}"
mkdir -p "${store_root}"

conda run -n "${CONDA_ENV:-env_quant}" --no-capture-output python \
  ./Strategy7/run_factor_mining.py \
  --framework fundamental_multiobj \
  --list-factors \
  --factor-freq D \
  --factor-packages "trend,reversal,liquidity,volatility,flow,crowding" \
  --auto-export-factor-snapshot \
  --export-factor-list \
  --factor-list-export-format markdown \
  --factor-list-export-path $exportPath \
  --factor-store-root "${store_root}"

