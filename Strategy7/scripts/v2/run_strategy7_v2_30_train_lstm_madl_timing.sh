#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
quant_root="${QUANT_ROOT:-$(cd "${repo_root}/.." && pwd)}"
cd "${repo_root}"

output_dir="${repo_root}/Strategy7/outputs/run_strategy7_30_train_lstm_madl_timing"
mkdir -p "${output_dir}"

conda run -n "${CONDA_ENV:-env_quant}" --no-capture-output python \
  ./Strategy7/run_strategy7.py \
  --train-start "${STRATEGY7_TRAIN_START:-2020-01-01}" \
  --train-end "${STRATEGY7_TRAIN_END:-2024-12-31}" \
  --test-start "${STRATEGY7_TEST_START:-2025-01-01}" \
  --test-end "${STRATEGY7_TEST_END:-2025-12-31}" \
  --universe "${STRATEGY7_UNIVERSE:-hs300}" \
  --data-root auto \
  --index-root "${quant_root}/data_baostock/ak_index" \
  --factor-freq "${STRATEGY7_FACTOR_FREQ:-D}" \
  --disable-text-data \
  --factor-packages "trend,reversal,liquidity,volatility,flow,price_action" \
  --label-task return \
  --horizon "${STRATEGY7_HORIZON:-1}" \
  --data-load-workers "${STRATEGY7_DATA_LOAD_WORKERS:-8}" \
  --enable-factor-value-store true \
  --factor-value-store-root auto \
  --factor-value-store-format csv \
  --factor-value-store-workers "${STRATEGY7_FACTOR_STORE_WORKERS:-8}" \
  --enable-factor-engineering true \
  --fe-corr-threshold 0.90 \
  --fe-preselect-top-n 300 \
  --fe-min-factors 40 \
  --stock-model-type decision_tree \
  --timing-model-type lstm_madl \
  --timing-lstm-seq-len 20 \
  --timing-lstm-intraday-seq-len "${STRATEGY7_TIMING_LSTM_INTRADAY_SEQ_LEN:-48}" \
  --timing-lstm-hidden-sizes "${STRATEGY7_TIMING_LSTM_HIDDEN_SIZES:-512,256,128}" \
  --timing-lstm-dropout 0.20 \
  --timing-lstm-epochs "${STRATEGY7_TIMING_LSTM_EPOCHS:-120}" \
  --timing-lstm-lr 1e-3 \
  --timing-lstm-weight-decay 1e-4 \
  --timing-lstm-early-stop 15 \
  --timing-lstm-batch-size 128 \
  --timing-lstm-min-train-samples 80 \
  --timing-lstm-feature-mode "${STRATEGY7_TIMING_LSTM_FEATURE_MODE:-auto}" \
  --timing-lstm-loss-mode "${STRATEGY7_TIMING_LSTM_LOSS_MODE:-madl_mse}" \
  --timing-lstm-exposure-mode "${STRATEGY7_TIMING_LSTM_EXPOSURE_MODE:-long_only_bands}" \
  --timing-lstm-market-agg amount_weighted \
  --timing-lstm-device "${STRATEGY7_TIMING_LSTM_DEVICE:-auto}" \
  --portfolio-model-type dynamic_opt \
  --opt-max-weight 0.12 \
  --opt-max-turnover 1.00 \
  --opt-risk-aversion 1.20 \
  --execution-model-type realistic_fill \
  --max-participation-rate 0.12 \
  --base-fill-rate 0.92 \
  --latency-bars 1 \
  --model-run-mode train \
  --execution-scheme open5_open5 \
  --fee-bps 1.5 \
  --slippage-bps 1.5 \
  --top-k 50 \
  --long-threshold 0.5 \
  --inference-top-k 20 \
  --save-models true \
  --output-dir "${output_dir}"
