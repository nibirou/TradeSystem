#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
quant_root="${QUANT_ROOT:-$(cd "${repo_root}/.." && pwd)}"
cd "${repo_root}"

infer_dates=()
default_infer_dates="${STRATEGY7_INFER_DATES:-2025-01-16, 2025-01-17}"
stock_model_summary_json=""
stock_models_dir="${repo_root}/Strategy7/outputs/run_strategy7_27_train_stockformer/models"
stock_models_run_tag=""

append_infer_dates() {
  local expr="$1"
  expr="${expr//;/,}"
  expr="${expr#\[}"
  expr="${expr%\]}"
  local parts=()
  IFS=',' read -r -a parts <<< "${expr}"
  local raw item
  for raw in "${parts[@]}"; do
    item="${raw#"${raw%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    item="${item#\"}"; item="${item%\"}"
    item="${item#\'}"; item="${item%\'}"
    if [[ -n "${item}" ]]; then
      infer_dates+=("${item}")
    fi
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --infer-date)
      append_infer_dates "$2"; shift 2 ;;
    --infer-dates)
      append_infer_dates "$2"; shift 2 ;;
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

if [[ ${#infer_dates[@]} -eq 0 ]]; then
  append_infer_dates "${default_infer_dates}"
fi
if [[ ${#infer_dates[@]} -eq 0 ]]; then
  echo "Usage: bash Strategy7/scripts/v2/run_strategy7_v2_33_load_stockformer_infer_day.sh --infer-date YYYY-MM-DD [--infer-dates YYYY-MM-DD,YYYY-MM-DD] [--stock-models-load-dir <models_dir>]" >&2
  exit 2
fi
mapfile -t infer_dates < <(printf '%s\n' "${infer_dates[@]}" | awk 'NF && !seen[$0]++')
mapfile -t sorted_infer_dates < <(printf '%s\n' "${infer_dates[@]}" | sort)
first_infer_date="${sorted_infer_dates[0]}"
last_idx=$((${#sorted_infer_dates[@]} - 1))
last_infer_date="${sorted_infer_dates[$last_idx]}"
infer_dates_csv="$(IFS=,; echo "${infer_dates[*]}")"
if [[ ${#infer_dates[@]} -eq 1 ]]; then
  infer_tag="${first_infer_date//[^0-9A-Za-z]/}"
else
  infer_tag="${first_infer_date//[^0-9A-Za-z]/}_${last_infer_date//[^0-9A-Za-z]/}_n${#infer_dates[@]}"
fi

if [[ -z "${stock_model_summary_json}" && ! -d "${stock_models_dir}" ]]; then
  echo "Stock model directory not found: ${stock_models_dir}" >&2
  exit 2
fi

output_dir="${repo_root}/Strategy7/outputs/run_strategy7_v2_33_load_stockformer_infer_${infer_tag}"
mkdir -p "${output_dir}"

test_start="${STRATEGY7_TEST_START:-${first_infer_date}}"
test_end="${STRATEGY7_TEST_END:-${last_infer_date}}"
train_start="${STRATEGY7_TRAIN_START:-2025-01-01}"
train_end="${STRATEGY7_TRAIN_END:-2025-01-15}"

cmd=(
conda run -n "${CONDA_ENV:-quant}" --no-capture-output python
  ./Strategy7/run_strategy7.py \
  --train-start "${train_start}" \
  --train-end "${train_end}" \
  --test-start "${test_start}" \
  --test-end "${test_end}" \
  --universe "${STRATEGY7_UNIVERSE:-zz500}" \
  --data-root auto \
  --index-root "${quant_root}/data_baostock/ak_index" \
  --factor-freq "${STRATEGY7_FACTOR_FREQ:-D}" \
  --lookback-days "${STRATEGY7_LOOKBACK_DAYS:-10}" \
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
  --inference-signal-ts "${infer_dates_csv}" \
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
if [[ -n "${stock_models_run_tag}" ]]; then
  cmd+=(--stock-models-load-run-tag "${stock_models_run_tag}")
fi
"${cmd[@]}"

echo
echo "Stock source      : ${stock_model_summary_json:-${stock_models_dir}}"
echo "Infer dates       : ${infer_dates_csv}"
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
