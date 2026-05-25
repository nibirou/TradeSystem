$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..")
$QuantRoot = if ($env:QUANT_ROOT) { $env:QUANT_ROOT } else { Resolve-Path (Join-Path $RepoRoot "..") }
Set-Location $RepoRoot

$OutputDir = Join-Path $RepoRoot "Strategy7\outputs\run_strategy7_29_train_dtlc_rl"
$IndexRoot = Join-Path $QuantRoot "data_baostock\ak_index"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$CondaEnv = if ($env:CONDA_ENV) { $env:CONDA_ENV } else { "env_quant" }
$DataWorkers = if ($env:STRATEGY7_DATA_LOAD_WORKERS) { $env:STRATEGY7_DATA_LOAD_WORKERS } else { "8" }
$StoreWorkers = if ($env:STRATEGY7_FACTOR_STORE_WORKERS) { $env:STRATEGY7_FACTOR_STORE_WORKERS } else { "8" }

conda run -n $CondaEnv --no-capture-output python `
  .\Strategy7\run_strategy7.py `
  --train-start 2019-01-01 `
  --train-end 2024-12-31 `
  --test-start 2025-01-01 `
  --test-end 2025-12-31 `
  --universe all `
  --data-root auto `
  --index-root $IndexRoot `
  --factor-freq D `
  --disable-text-data `
  --factor-packages "trend,reversal,liquidity,volatility,flow,price_action,fund_valuation,fund_quality,fund_growth,fund_profitability,fund_leverage,fund_cashflow" `
  --label-task return `
  --data-load-workers $DataWorkers `
  --enable-factor-value-store true `
  --factor-value-store-root auto `
  --factor-value-store-format csv `
  --factor-value-store-workers $StoreWorkers `
  --enable-factor-engineering true `
  --fe-corr-threshold 0.90 `
  --fe-preselect-top-n 300 `
  --fe-min-factors 40 `
  --stock-model-type dtlc_rl `
  --dtlc-seq-len 60 `
  --dtlc-alpha-scales 20,40,60 `
  --dtlc-hidden-size 64 `
  --dtlc-latent-size 32 `
  --dtlc-num-heads 4 `
  --dtlc-encoder-layers 2 `
  --dtlc-grn-layers 2 `
  --dtlc-ffn-mult 4 `
  --dtlc-dropout 0.10 `
  --dtlc-pretrain-epochs 80 `
  --dtlc-ppo-epochs 30 `
  --dtlc-lr 1e-4 `
  --dtlc-ppo-lr 3e-4 `
  --dtlc-weight-decay 1e-4 `
  --dtlc-early-stop 20 `
  --dtlc-per-epoch-batch 100 `
  --dtlc-batch-size -1 `
  --dtlc-label-transform cszscore `
  --dtlc-mse-weight 0.05 `
  --dtlc-ic-loss-weight 1.0 `
  --dtlc-contrastive-weight 0.05 `
  --dtlc-orthogonal-weight 0.05 `
  --dtlc-stable-weight 0.05 `
  --dtlc-diversity-weight 0.02 `
  --dtlc-min-cross-section 8 `
  --dtlc-device auto `
  --timing-model-type none `
  --portfolio-model-type equal_weight `
  --execution-model-type ideal_fill `
  --model-run-mode train `
  --horizon 20 `
  --rebalance-stride 20 `
  --top-k 300 `
  --long-threshold 0.5 `
  --inference-top-k 50 `
  --save-models true `
  --output-dir $OutputDir
