#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "${repo_root}"

data_root="auto"
index_root=""
train_start="2023-01-01"
train_end="2024-12-31"
test_start="2025-01-01"
test_end="2025-12-31"
max_files="0"
file_format="auto"
fundamental_file_format="auto"
text_file_format="auto"
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
  log_level="verbose"
fi

output_dir="${repo_root}/Strategy7/outputs/smoke_v2/run_strategy7_22_train_allmarket_bottom_launch_10d"
mkdir -p "${output_dir}"

export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

args=(
  ./Strategy7/run_strategy7.py
  --universe all
  --data-root "${data_root}"
  --file-format "${file_format}"
  --fundamental-file-format "${fundamental_file_format}"
  --text-file-format "${text_file_format}"
  --disable-catalog-factors
  --train-start "${train_start}"
  --train-end "${train_end}"
  --test-start "${test_start}"
  --test-end "${test_end}"
  --factor-freq D
  --factor-packages "bottom_launch,trend,reversal,liquidity,volatility,price_action,crowding,oscillator,overnight,multi_freq,context,fund_quality,fund_cashflow"
  --label-task return
  --enable-factor-value-store true
  --factor-value-store-root auto
  --factor-value-store-format csv
  --enable-factor-engineering true
  --fe-corr-threshold 0.90
  --fe-preselect-top-n 300
  --fe-min-factors 40
  --stock-model-type launch_boost
  --launch-boost-max-depth 5
  --launch-boost-learning-rate 0.01
  --launch-boost-max-iter 2000
  --launch-boost-l2 1.5
  --launch-boost-return-head-weight 0.40
  --timing-model-type none
  --portfolio-model-type equal_weight
  --execution-model-type ideal_fill
  --model-run-mode train
  --enable-next-bar-inference true
  --inference-top-k 20
  --horizon 10
  --top-k 10
  --long-threshold 0.60
  --execution-scheme daily_close_daily_close
  --fee-bps 2.0
  --slippage-bps 2.0
  --save-models true
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

# #!/usr/bin/env bash
# # bash /workspace/Quant/TradeSystem/Strategy7/scripts/run_strategy7_12_dafat_daily.sh
# set -euo pipefail

# cd /workspace/Quant/TradeSystem

# python3 ./Strategy7/run_strategy7.py \
#   --universe all \
#   --data-root auto \
#   --disable-catalog-factors \
#   --train-start 2023-01-01 \
#   --train-end 2024-12-31 \
#   --test-start 2025-01-01 \
#   --test-end 2025-12-31 \
#   --factor-freq D \
#   --factor-packages bottom_launch,trend,reversal,liquidity,volatility,price_action,crowding,oscillator,overnight,multi_freq,context,fund_quality,fund_cashflow \
#   --label-task return \
#   --enable-factor-value-store true \
#   --factor-value-store-root auto \
#   --factor-value-store-format csv \
#   --enable-factor-engineering true \
#   --fe-corr-threshold 0.90 \
#   --fe-preselect-top-n 300 \
#   --fe-min-factors 40 \
#   --stock-model-type launch_boost \
#   --launch-boost-max-depth 5 \
#   --launch-boost-learning-rate 0.01 \
#   --launch-boost-max-iter 2000 \
#   --launch-boost-l2 1.5 \
#   --launch-boost-return-head-weight 0.40 \
#   --timing-model-type none \
#   --portfolio-model-type equal_weight \
#   --execution-model-type ideal_fill \
#   --model-run-mode train \
#   --enable-next-bar-inference true \
#   --inference-top-k 20 \
#   --horizon 10 \
#   --top-k 10 \
#   --long-threshold 0.60 \
#   --execution-scheme daily_close_daily_close \
#   --fee-bps 2.0 \
#   --slippage-bps 2.0 \
#   --save-models true \