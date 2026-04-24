#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "${repo_root}"

data_root="auto"
index_root=""
train_start="2020-01-01"
train_end="2023-12-31"
test_start="2024-01-01"
test_end="2025-12-31"
max_files="0"

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
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

output_dir="${repo_root}/Strategy7/outputs/smoke_v2/run_strategy7_20_train_allmarket_bottom_launch_10d"
mkdir -p "${output_dir}"

args=(
  ./Strategy7/run_strategy7.py
  --universe all
  --data-root "${data_root}"
  --train-start "${train_start}"
  --train-end "${train_end}"
  --test-start "${test_start}"
  --test-end "${test_end}"
  --factor-freq D
  --factor-packages "bottom_launch,trend,reversal,liquidity,volatility,price_action,crowding,oscillator,overnight,multi_freq,context,fund_quality,fund_cashflow,text_event,text_sentiment"
  --label-task return
  --enable-factor-engineering true
  --fe-corr-threshold 0.90
  --fe-preselect-top-n 300
  --fe-min-factors 40
  --stock-model-type launch_boost
  --launch-boost-max-depth 5
  --launch-boost-learning-rate 0.04
  --launch-boost-max-iter 400
  --launch-boost-l2 1.5
  --launch-boost-return-head-weight 0.40
  --timing-model-type volatility_regime
  --portfolio-model-type dynamic_opt
  --execution-model-type realistic_fill
  --max-participation-rate 0.15
  --base-fill-rate 0.92
  --latency-bars 1
  --model-run-mode train
  --enable-next-bar-inference true
  --inference-top-k 30
  --horizon 10
  --top-k 25
  --long-threshold 0.60
  --execution-scheme daily_close_daily_close
  --fee-bps 2.0
  --slippage-bps 2.0
  --save-models true
  --output-dir "${output_dir}"
)

if [[ -n "${index_root}" ]]; then
  args+=(--index-root "${index_root}")
fi

if [[ "${max_files}" != "0" ]]; then
  args+=(--max-files "${max_files}")
fi

conda run -n "${CONDA_ENV:-env_quant}" --no-capture-output python "${args[@]}"
