#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
quant_root="${QUANT_ROOT:-$(cd "${repo_root}/.." && pwd)}"
cd "${repo_root}"

infer_date="2025-01-16"
stock_model_summary_json=""
stock_models_dir="${repo_root}/Strategy7/outputs/run_strategy7_27_train_stockformer/models"
stock_models_run_tag=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --infer-date)
      infer_date="$2"; shift 2 ;;
    --stock-model-summary-json)
      stock_model_summary_json="$2"; shift 2 ;;
    --stock-models-load-dir)
      stock_models_dir="$2"; shift 2 ;;
    --stock-models-load-run-tag)
      stock_models_run_tag="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${infer_date}" ]]; then
  echo "Usage: bash Strategy7/scripts/v2/run_strategy7_v2_33_load_stockformer_infer_day.sh --infer-date YYYY-MM-DD [--stock-models-load-dir <models_dir>]" >&2
  exit 2
fi

if [[ -z "${stock_model_summary_json}" && ! -d "${stock_models_dir}" ]]; then
  echo "Stock model directory not found: ${stock_models_dir}" >&2
  exit 2
fi

output_dir="${repo_root}/Strategy7/outputs/run_strategy7_v2_33_load_stockformer_infer_${infer_date//[^0-9]/}"
mkdir -p "${output_dir}"

date_shift() {
  python3 -c "import datetime,sys; print((datetime.date.fromisoformat(sys.argv[1]) + datetime.timedelta(days=int(sys.argv[2]))).isoformat())" "$1" "$2"
}

# In load/inference mode these dates define the current data window, not the
# source model's training window. Keep enough history for rolling factors and
# StockFormer sequence bootstrap, and enough future span for the framework's
# label-based test split when running historical day inference.
history_days="${STRATEGY7_INFERENCE_HISTORY_DAYS:-540}"
forward_days="${STRATEGY7_INFERENCE_FORWARD_DAYS:-90}"
test_end="${STRATEGY7_TEST_END:-$(date_shift "${infer_date}" "${forward_days}")}"
train_start="${STRATEGY7_TRAIN_START:-$(date_shift "${infer_date}" "-${history_days}")}"
train_end="${STRATEGY7_TRAIN_END:-$(date_shift "${infer_date}" "-1")}"
lookback_days="${STRATEGY7_LOOKBACK_DAYS:-252}"

cmd=(
python3
  ./Strategy7/run_strategy7.py \
  --train-start "${train_start}" \
  --train-end "${train_end}" \
  --test-start "${infer_date}" \
  --test-end "${test_end}" \
  --universe "${STRATEGY7_UNIVERSE:-zz500}" \
  --data-root auto \
  --index-root "${quant_root}/data_baostock/ak_index" \
  --factor-freq "${STRATEGY7_FACTOR_FREQ:-D}" \
  --lookback-days "${lookback_days}" \
  --disable-text-data \
  --label-task return \
  --data-load-workers "${STRATEGY7_DATA_LOAD_WORKERS:-8}" \
  --enable-factor-value-store true \
  --factor-value-store-root auto \
  --factor-value-store-format csv \
  --factor-value-store-workers "${STRATEGY7_FACTOR_STORE_WORKERS:-8}" \
  --enable-factor-engineering false \
  --stock-model-run-mode load \
  --stock-model-type stockformer \
  --timing-model-type none \
  --portfolio-model-type equal_weight \
  --execution-model-type ideal_fill \
  --load-fe-mode off \
  --enable-next-bar-inference true \
  --inference-signal-ts "${infer_date}" \
  --inference-top-k "${STRATEGY7_INFERENCE_TOP_K:-5}" \
  --horizon "${STRATEGY7_HORIZON:-5}" \
  --top-k "${STRATEGY7_TOP_K:-5}" \
  --output-dir "${output_dir}"
)

if [[ -n "${stock_model_summary_json}" ]]; then
  cmd+=(--stock-model-summary-json "${stock_model_summary_json}")
else
  cmd+=(--stock-models-load-dir "${stock_models_dir}")
fi
"${cmd[@]}"

echo
echo "Stock source      : ${stock_model_summary_json:-${stock_models_dir}}"
echo "Data window       : train=${train_start}~${train_end}, test=${infer_date}~${test_end}, lookback_days=${lookback_days}"
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
