Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $repoRoot

$outputDir = Join-Path $repoRoot "Strategy7\outputs\smoke_v2\run_strategy7_18_train_monthly_multitask_catalog_off"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

conda run -n env_quant --no-capture-output python .\Strategy7\run_strategy7.py `
  --data-root D:/PythonProject/Quant/data_baostock/stock_hist/hs300 `
  --hs300-list-path D:/PythonProject/Quant/data_baostock/metadata/stock_list_hs300.csv `
  --index-root D:/PythonProject/Quant/data_baostock/ak_index `
  --disable-catalog-factors `
  --train-start 2021-01-01 `
  --train-end 2023-12-31 `
  --test-start 2024-01-01 `
  --test-end 2024-12-31 `
  --factor-freq M `
  --factor-packages "trend,reversal,liquidity,volatility,period_signature,multi_freq" `
  --label-task multi_task `
  --max-files 20 `
  --stock-model-type decision_tree `
  --timing-model-type none `
  --portfolio-model-type equal_weight `
  --execution-model-type ideal_fill `
  --model-run-mode train `
  --horizon 2 `
  --top-k 8 `
  --long-threshold 0.5 `
  --execution-scheme daily_close_daily_close `
  --fee-bps 1.5 `
  --slippage-bps 1.5 `
  --save-models true `
  --output-dir $outputDir

if ($LASTEXITCODE -ne 0) {
  throw "Template execution failed with exit code $LASTEXITCODE"
}
