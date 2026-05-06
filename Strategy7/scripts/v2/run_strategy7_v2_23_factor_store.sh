#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "${repo_root}"

data_root="auto"
index_root=""
train_start="2020-01-01"
train_end="2024-12-31"
test_start="2025-01-01"
test_end="2025-12-31"
max_files="0"
file_format="auto"
fundamental_file_format="auto"
text_file_format="auto"
store_format="csv"
chunk_size="16"
log_level="normal"
diagnose_lite="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      data_root="$2"
      shift 2
      ;;
    --index-root)
      index_root="$2"
      shift 2
      ;;
    --train-start)
      train_start="$2"
      shift 2
      ;;
    --train-end)
      train_end="$2"
      shift 2
      ;;
    --test-start)
      test_start="$2"
      shift 2
      ;;
    --test-end)
      test_end="$2"
      shift 2
      ;;
    --max-files)
      max_files="$2"
      shift 2
      ;;
    --file-format)
      file_format="$2"
      shift 2
      ;;
    --fundamental-file-format)
      fundamental_file_format="$2"
      shift 2
      ;;
    --text-file-format)
      text_file_format="$2"
      shift 2
      ;;
    --factor-value-store-format)
      store_format="$2"
      shift 2
      ;;
    --factor-value-store-chunk-size)
      chunk_size="$2"
      shift 2
      ;;
    --log-level)
      log_level="$2"
      shift 2
      ;;
    --diagnose-lite)
      diagnose_lite="1"
      shift 1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "${diagnose_lite}" == "1" ]]; then
  # For crash localization: reduce scale + avoid parquet native engine path.
  if [[ "${max_files}" == "0" ]]; then
    max_files="200"
  fi
  file_format="csv"
  fundamental_file_format="csv"
  text_file_format="csv"
  store_format="csv"
  chunk_size="8"
  log_level="verbose"
fi

output_dir="${repo_root}/Strategy7/outputs/smoke_v2/run_strategy7_23_factor_store"
mkdir -p "${output_dir}"

export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

args=(
  ./Strategy7/run_strategy7.py
  --train-start "${train_start}"
  --train-end "${train_end}"
  --test-start "${test_start}"
  --test-end "${test_end}"
  --universe all
  --data-root "${data_root}"
  --file-format "${file_format}"
  --fundamental-file-format "${fundamental_file_format}"
  --text-file-format "${text_file_format}"
  --disable-catalog-factors
  --factor-freq D
  --enable-factor-value-store true
  --factor-value-store-format "${store_format}"
  --factor-value-store-build-all true
  --factor-value-store-build-only true
  --factor-value-store-chunk-size "${chunk_size}"
  --factor-packages "bottom_launch,trend,reversal,liquidity,volatility,price_action,crowding,oscillator,overnight,multi_freq,context,fund_quality,fund_cashflow"
  --log-level "${log_level}"
  --output-dir "${output_dir}"
)

if [[ -n "${index_root}" ]]; then
  args+=(--index-root "${index_root}")
fi

if [[ "${max_files}" != "0" ]]; then
  args+=(--max-files "${max_files}")
fi

conda run -n "${CONDA_ENV:-env_quant}" --no-capture-output python "${args[@]}"
