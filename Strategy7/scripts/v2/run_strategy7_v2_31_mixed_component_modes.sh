#!/usr/bin/env bash
set -euo pipefail

timing_models_load_dir=""
execution_models_load_dir=""
timing_models_load_run_tag=""
execution_models_load_run_tag=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --timing-models-load-dir)
      timing_models_load_dir="$2"; shift 2 ;;
    --execution-models-load-dir)
      execution_models_load_dir="$2"; shift 2 ;;
    --timing-models-load-run-tag)
      timing_models_load_run_tag="$2"; shift 2 ;;
    --execution-models-load-run-tag)
      execution_models_load_run_tag="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${timing_models_load_dir}" || -z "${execution_models_load_dir}" ]]; then
  echo "Usage: bash scripts/v2/run_strategy7_v2_31_mixed_component_modes.sh --timing-models-load-dir <models_dir> --execution-models-load-dir <models_dir>" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
quant_root="${QUANT_ROOT:-$(cd "${repo_root}/.." && pwd)}"
cd "${repo_root}"

output_dir="${repo_root}/Strategy7/outputs/smoke_v2/run_strategy7_31_mixed_component_modes"
mkdir -p "${output_dir}"

cmd=(
  conda run -n "${CONDA_ENV:-env_quant}" --no-capture-output python
  ./Strategy7/run_strategy7.py \
  --data-root "${quant_root}/data_baostock/stock_hist/hs300" \
  --hs300-list-path "${quant_root}/data_baostock/metadata/stock_list_hs300.csv" \
  --index-root "${quant_root}/data_baostock/ak_index" \
  --train-start "${STRATEGY7_TRAIN_START:-2024-01-01}" \
  --train-end "${STRATEGY7_TRAIN_END:-2024-06-30}" \
  --test-start "${STRATEGY7_TEST_START:-2024-07-01}" \
  --test-end "${STRATEGY7_TEST_END:-2024-09-30}" \
  --factor-freq "${STRATEGY7_FACTOR_FREQ:-D}" \
  --factor-packages "trend,reversal,liquidity,volatility,flow,price_action" \
  --label-task return \
  --max-files "${STRATEGY7_MAX_FILES:-40}" \
  --disable-text-data \
  --enable-factor-engineering true \
  --fe-corr-threshold 0.90 \
  --fe-preselect-top-n 160 \
  --stock-model-type decision_tree \
  --timing-model-type lstm_madl \
  --portfolio-model-type dynamic_opt \
  --execution-model-type realistic_fill \
  --model-run-mode train \
  --stock-model-run-mode train \
  --timing-model-run-mode load \
  --portfolio-model-run-mode train \
  --execution-model-run-mode load \
  --timing-models-load-dir "${timing_models_load_dir}" \
  --execution-models-load-dir "${execution_models_load_dir}" \
  --horizon "${STRATEGY7_HORIZON:-1}" \
  --top-k "${STRATEGY7_TOP_K:-20}" \
  --long-threshold 0.5 \
  --execution-scheme open5_open5 \
  --fee-bps 1.5 \
  --slippage-bps 1.5 \
  --save-models true \
  --output-dir "${output_dir}"
)

if [[ -n "${timing_models_load_run_tag}" ]]; then
  cmd+=(--timing-models-load-run-tag "${timing_models_load_run_tag}")
fi
if [[ -n "${execution_models_load_run_tag}" ]]; then
  cmd+=(--execution-models-load-run-tag "${execution_models_load_run_tag}")
fi

"${cmd[@]}"
