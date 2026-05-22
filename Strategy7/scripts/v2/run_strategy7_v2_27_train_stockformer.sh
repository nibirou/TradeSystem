#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
quant_root="${QUANT_ROOT:-$(cd "${repo_root}/.." && pwd)}"
cd "${repo_root}"

output_dir="${repo_root}/Strategy7/outputs/run_strategy7_27_train_stockformer"
mkdir -p "${output_dir}"

conda run -n "${CONDA_ENV:-env_quant}" --no-capture-output python \
  ./Strategy7/run_strategy7.py \
  --train-start 2019-01-01 \
  --train-end 2024-12-31 \
  --test-start 2025-01-01 \
  --test-end 2025-12-31 \
  --universe zz500 \
  --data-root auto \
  --index-root ${quant_root}/data_baostock/ak_index \
  --factor-freq D \
  --disable-text-data \
  --factor-packages "trend,reversal,liquidity,volatility,flow,price_action" \
  --label-task return \
  --data-load-workers "${STRATEGY7_DATA_LOAD_WORKERS:-8}" \
  --enable-factor-value-store true \
  --factor-value-store-root auto \
  --factor-value-store-format csv \
  --factor-value-store-workers "${STRATEGY7_FACTOR_STORE_WORKERS:-8}" \
  --enable-factor-engineering true \
  --fe-corr-threshold 0.90 \
  --fe-preselect-top-n 300 \
  --fe-min-factors 40 \
  --stock-model-type stockformer \
  --stockformer-seq-len 60 \
  --stockformer-rel-seq-len 252 \
  --stockformer-hidden-size 64 \
  --stockformer-num-layers 2 \
  --stockformer-num-heads 10 \
  --stockformer-ffn-mult 4 \
  --stockformer-dropout 0.10 \
  --stockformer-pretrain-epochs 50 \
  --stockformer-sac-episodes 50 \
  --stockformer-lr 1e-3 \
  --stockformer-sac-lr 3e-4 \
  --stockformer-gamma 0.999 \
  --stockformer-init-alpha 0.5 \
  --stockformer-early-stop 20 \
  --stockformer-learning-starts 100 \
  --stockformer-batch-transitions 16 \
  --stockformer-updates-per-step 1 \
  --stockformer-per-epoch-batch 100 \
  --stockformer-batch-size -1 \
  --stockformer-label-transform csrank \
  --stockformer-reward-cost-bps 30.0 \
  --stockformer-tracking-penalty 0.05 \
  --stockformer-min-cross-section 8 \
  --stockformer-device auto \
  --timing-model-type none \
  --portfolio-model-type equal_weight \
  --execution-model-type ideal_fill \
  --model-run-mode train \
  --horizon 5 \
  --top-k 50 \
  --long-threshold 0.5 \
  --inference-top-k 20 \
  --save-models true \
  --output-dir "${output_dir}"
