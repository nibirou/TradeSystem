#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
quant_root="${QUANT_ROOT:-$(cd "${repo_root}/.." && pwd)}"
cd "${repo_root}"

models_load_dir=""
models_load_run_tag=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-load-dir)
      models_load_dir="$2"
      shift 2
      ;;
    --models-load-run-tag)
      models_load_run_tag="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${models_load_dir}" ]]; then
  echo "Usage: bash scripts/v2/run_strategy7_v2_06_load_from_models_dir_off.sh --models-load-dir <path> [--models-load-run-tag <tag>]" >&2
  exit 2
fi

output_dir="${repo_root}/Strategy7/outputs/smoke_v2/run_strategy7_06_load_models_dir_off"
mkdir -p "${output_dir}"

cmd=(
  conda run -n "${CONDA_ENV:-env_quant}" --no-capture-output python ./Strategy7/run_strategy7.py
  --data-root "${quant_root}/data_baostock/stock_hist/hs300"
  --hs300-list-path "${quant_root}/data_baostock/metadata/stock_list_hs300.csv"
  --index-root "${quant_root}/data_baostock/ak_index"
  --train-start 2024-01-01
  --train-end 2024-06-30
  --test-start 2024-07-01
  --test-end 2024-09-30
  --factor-freq D
  --factor-packages "trend,reversal,liquidity,volatility"
  --max-files 20
  --enable-factor-engineering true
  --stock-model-type decision_tree
  --timing-model-type none
  --portfolio-model-type equal_weight
  --execution-model-type ideal_fill
  --model-run-mode load
  --load-fe-mode off
  --models-load-dir "${models_load_dir}"
  --horizon 5
  --top-k 10
  --long-threshold 0.5
  --save-models false
  --output-dir "${output_dir}"
)

if [[ -n "${models_load_run_tag}" ]]; then
  cmd+=(--models-load-run-tag "${models_load_run_tag}")
fi

"${cmd[@]}"
