Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $repoRoot

$outputDir = Join-Path $repoRoot "Strategy7\outputs\smoke_v2\run_strategy7_14_train_price_only_mainboard"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

conda run -n env_quant --no-capture-output python .\Strategy7\run_strategy7.py `
  --universe all `
  --data-root auto `
  --index-root D:/PythonProject/Quant/data_baostock/ak_index `
  --main-board-only `
  --disable-fundamental-data `
  --disable-text-data `
  --train-start 2024-01-01 `
  --train-end 2024-06-30 `
  --test-start 2024-07-01 `
  --test-end 2024-09-30 `
  --factor-freq D `
  --factor-packages "trend,reversal,liquidity,volatility,structure,context" `
  --label-task direction `
  --max-files 20 `
  --stock-model-type decision_tree `
  --timing-model-type none `
  --portfolio-model-type equal_weight `
  --execution-model-type ideal_fill `
  --model-run-mode train `
  --horizon 5 `
  --top-k 10 `
  --long-threshold 0.5 `
  --execution-scheme vwap30_vwap30 `
  --fee-bps 1.5 `
  --slippage-bps 1.5 `
  --save-models true `
  --output-dir $outputDir

if ($LASTEXITCODE -ne 0) {
  throw "Template execution failed with exit code $LASTEXITCODE"
}
