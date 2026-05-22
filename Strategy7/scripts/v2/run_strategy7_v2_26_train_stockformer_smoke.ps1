$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..")
$QuantRoot = if ($env:QUANT_ROOT) { $env:QUANT_ROOT } else { Resolve-Path (Join-Path $RepoRoot "..") }
Set-Location $RepoRoot

$OutputDir = Join-Path $RepoRoot "Strategy7\outputs\smoke_v2\run_strategy7_26_train_stockformer"
$DataRoot = Join-Path $QuantRoot "data_baostock\stock_hist\hs300"
$StockList = Join-Path $QuantRoot "data_baostock\metadata\stock_list_hs300.csv"
$IndexRoot = Join-Path $QuantRoot "data_baostock\ak_index"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$CondaEnv = if ($env:CONDA_ENV) { $env:CONDA_ENV } else { "env_quant" }
$DataWorkers = if ($env:STRATEGY7_DATA_LOAD_WORKERS) { $env:STRATEGY7_DATA_LOAD_WORKERS } else { "0" }

conda run -n $CondaEnv --no-capture-output python `
  .\Strategy7\run_strategy7.py `
  --data-root $DataRoot `
  --hs300-list-path $StockList `
  --index-root $IndexRoot `
  --train-start 2023-07-01 `
  --train-end 2024-03-31 `
  --test-start 2024-04-01 `
  --test-end 2024-06-30 `
  --factor-freq D `
  --factor-packages "trend,reversal,liquidity,volatility" `
  --label-task return `
  --max-files 15 `
  --data-load-workers $DataWorkers `
  --stock-model-type stockformer `
  --stockformer-seq-len 12 `
  --stockformer-rel-seq-len 20 `
  --stockformer-hidden-size 32 `
  --stockformer-num-layers 1 `
  --stockformer-num-heads 4 `
  --stockformer-ffn-mult 2 `
  --stockformer-dropout 0.10 `
  --stockformer-pretrain-epochs 1 `
  --stockformer-sac-episodes 1 `
  --stockformer-lr 1e-3 `
  --stockformer-sac-lr 3e-4 `
  --stockformer-early-stop 1 `
  --stockformer-learning-starts 4 `
  --stockformer-batch-transitions 2 `
  --stockformer-updates-per-step 1 `
  --stockformer-per-epoch-batch 8 `
  --stockformer-batch-size 64 `
  --stockformer-label-transform csrank `
  --stockformer-min-cross-section 4 `
  --stockformer-device cpu `
  --timing-model-type none `
  --portfolio-model-type equal_weight `
  --execution-model-type ideal_fill `
  --model-run-mode train `
  --horizon 5 `
  --top-k 10 `
  --long-threshold 0.5 `
  --save-models true `
  --output-dir $OutputDir
