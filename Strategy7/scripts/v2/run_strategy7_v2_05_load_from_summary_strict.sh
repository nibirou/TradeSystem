#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
quant_root="${QUANT_ROOT:-$(cd "${repo_root}/.." && pwd)}"
cd "${repo_root}"

model_summary_json=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-summary-json)
      model_summary_json="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${model_summary_json}" ]]; then
  echo "Usage: bash scripts/v2/run_strategy7_v2_05_load_from_summary_strict.sh --model-summary-json <path>" >&2
  exit 2
fi

output_dir="${repo_root}/Strategy7/outputs/smoke_v2/run_strategy7_05_load_strict"
mkdir -p "${output_dir}"

conda run -n "${CONDA_ENV:-env_quant}" --no-capture-output python \
  ./Strategy7/run_strategy7.py \
  --data-root "${quant_root}/data_baostock/stock_hist/hs300" \
  --hs300-list-path "${quant_root}/data_baostock/metadata/stock_list_hs300.csv" \
  --index-root "${quant_root}/data_baostock/ak_index" \
  --train-start 2024-01-01 \
  --train-end 2024-06-30 \
  --test-start 2024-07-01 \
  --test-end 2024-09-30 \
  --factor-freq D \
  --factor-packages "trend,reversal,liquidity,volatility,structure,context" \
  --max-files 20 \
  --enable-factor-engineering true \
  --stock-model-type decision_tree \
  --timing-model-type volatility_regime \
  --portfolio-model-type dynamic_opt \
  --execution-model-type realistic_fill \
  --model-run-mode load \
  --load-fe-mode strict \
  --model-summary-json "${model_summary_json}" \
  --enable-next-bar-inference true \
  --inference-top-k 15 \
  --horizon 5 \
  --top-k 12 \
  --long-threshold 0.55 \
  --execution-scheme vwap30_vwap30 \
  --fee-bps 1.5 \
  --slippage-bps 1.5 \
  --save-models false \
  --output-dir "${output_dir}"
