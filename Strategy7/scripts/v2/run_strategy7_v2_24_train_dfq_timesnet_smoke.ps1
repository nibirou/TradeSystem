Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $repoRoot

$outputDir = Join-Path $repoRoot "Strategy7\outputs\smoke_v2\run_strategy7_24_train_dfq_timesnet"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

conda run -n env_quant --no-capture-output python .\Strategy7\run_strategy7.py `
  --data-root D:/PythonProject/Quant/data_baostock/stock_hist/hs300 `
  --hs300-list-path D:/PythonProject/Quant/data_baostock/metadata/stock_list_hs300.csv `
  --index-root D:/PythonProject/Quant/data_baostock/ak_index `
  --train-start 2023-07-01 `
  --train-end 2024-03-31 `
  --test-start 2024-04-01 `
  --test-end 2024-06-30 `
  --factor-freq D `
  --factor-packages "trend,reversal,liquidity,volatility" `
  --label-task return `
  --max-files 15 `
  --stock-model-type dfq_timesnet `
  --timesnet-seq-len 12 `
  --timesnet-hidden-size 48 `
  --timesnet-e-layers 1 `
  --timesnet-hidden-size2 48 `
  --timesnet-periods "3,12" `
  --timesnet-num-kernels 2 `
  --timesnet-dropout 0.0 `
  --timesnet-epochs 1 `
  --timesnet-lr 2e-4 `
  --timesnet-weight-decay 0.0 `
  --timesnet-early-stop 1 `
  --timesnet-smooth-steps 1 `
  --timesnet-per-epoch-batch 8 `
  --timesnet-batch-size 64 `
  --timesnet-label-transform cszscore `
  --timesnet-input-clip 3.0 `
  --timesnet-device cpu `
  --timing-model-type none `
  --portfolio-model-type equal_weight `
  --execution-model-type ideal_fill `
  --model-run-mode train `
  --horizon 5 `
  --top-k 10 `
  --long-threshold 0.5 `
  --save-models true `
  --output-dir $outputDir

if ($LASTEXITCODE -ne 0) {
  throw "Template execution failed with exit code $LASTEXITCODE"
}
