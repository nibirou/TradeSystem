#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
quant_root="${QUANT_ROOT:-$(cd "${repo_root}/.." && pwd)}"
cd "${repo_root}"

infer_date=""
stock_model_path=""
timing_model_path=""
stock_models_dir="${repo_root}/Strategy7/outputs/run_strategy7_27_train_stockformer/models"
timing_models_dir="${repo_root}/Strategy7/outputs/run_strategy7_30_train_lstm_madl_timing/models"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --infer-date)
      infer_date="$2"; shift 2 ;;
    --stock-model-path)
      stock_model_path="$2"; shift 2 ;;
    --timing-model-path)
      timing_model_path="$2"; shift 2 ;;
    --stock-models-dir)
      stock_models_dir="$2"; shift 2 ;;
    --timing-models-dir)
      timing_models_dir="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${infer_date}" ]]; then
  echo "Usage: bash Strategy7/scripts/v2/run_strategy7_v2_32_load_stockformer_lstm_infer_day.sh --infer-date YYYY-MM-DD [--stock-model-path <pt>] [--timing-model-path <pt|json>]" >&2
  exit 2
fi

latest_match() {
  local dir="$1"
  local pattern="$2"
  if [[ ! -d "${dir}" ]]; then
    echo "Model directory not found: ${dir}" >&2
    exit 2
  fi
  shopt -s nullglob
  local files=("${dir}"/${pattern})
  shopt -u nullglob
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "No model file matched ${dir}/${pattern}" >&2
    exit 2
  fi
  ls -t "${files[@]}" | head -n 1
}

if [[ -z "${stock_model_path}" ]]; then
  stock_model_path="$(latest_match "${stock_models_dir}" "stock_model_stockformer_*.pt")"
fi
if [[ -z "${timing_model_path}" ]]; then
  timing_model_path="$(latest_match "${timing_models_dir}" "timing_lstm_madl_*.pt")"
fi

output_dir="${repo_root}/Strategy7/outputs/run_strategy7_32_load_stockformer_lstm_infer_${infer_date//[^0-9]/}"
mkdir -p "${output_dir}"

test_end="${STRATEGY7_TEST_END:-2025-12-31}"
train_start="${STRATEGY7_TRAIN_START:-2024-01-01}"
train_end="${STRATEGY7_TRAIN_END:-2024-12-31}"

conda run -n "${CONDA_ENV:-env_quant}" --no-capture-output python \
  ./Strategy7/run_strategy7.py \
  --train-start "${train_start}" \
  --train-end "${train_end}" \
  --test-start "${infer_date}" \
  --test-end "${test_end}" \
  --universe "${STRATEGY7_UNIVERSE:-zz500}" \
  --data-root auto \
  --index-root "${quant_root}/data_baostock/ak_index" \
  --factor-freq "${STRATEGY7_FACTOR_FREQ:-D}" \
  --disable-text-data \
  --factor-packages "trend,reversal,liquidity,volatility,flow,price_action" \
  --label-task return \
  --data-load-workers "${STRATEGY7_DATA_LOAD_WORKERS:-8}" \
  --enable-factor-value-store true \
  --factor-value-store-root auto \
  --factor-value-store-format csv \
  --factor-value-store-workers "${STRATEGY7_FACTOR_STORE_WORKERS:-8}" \
  --enable-factor-engineering false \
  --stock-model-type stockformer \
  --timing-model-type lstm_madl \
  --portfolio-model-type equal_weight \
  --execution-model-type ideal_fill \
  --model-run-mode train \
  --stock-model-run-mode load \
  --timing-model-run-mode load \
  --portfolio-model-run-mode train \
  --execution-model-run-mode train \
  --stock-model-path "${stock_model_path}" \
  --timing-model-path "${timing_model_path}" \
  --enable-next-bar-inference true \
  --inference-signal-ts "${infer_date}" \
  --inference-top-k "${STRATEGY7_INFERENCE_TOP_K:-50}" \
  --horizon "${STRATEGY7_HORIZON:-5}" \
  --top-k "${STRATEGY7_TOP_K:-50}" \
  --long-threshold "${STRATEGY7_LONG_THRESHOLD:-0.5}" \
  --execution-scheme "${STRATEGY7_EXECUTION_SCHEME:-vwap30_vwap30}" \
  --fee-bps "${STRATEGY7_FEE_BPS:-1.5}" \
  --slippage-bps "${STRATEGY7_SLIPPAGE_BPS:-1.5}" \
  --save-models false \
  --output-dir "${output_dir}"

echo
echo "StockFormer model : ${stock_model_path}"
echo "Timing model      : ${timing_model_path}"
echo "Output directory  : ${output_dir}"

latest_output() {
  local pattern="$1"
  shopt -s nullglob
  local files=("${output_dir}"/${pattern})
  shopt -u nullglob
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "<not generated>"
    return 0
  fi
  ls -t "${files[@]}" | head -n 1
}

echo "Candidate file    : $(latest_output "next_bar_candidates_*.csv")"
echo "Summary file      : $(latest_output "next_bar_summary_*.json")"
