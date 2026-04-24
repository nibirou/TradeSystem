Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $repoRoot

$outputDir = Join-Path $repoRoot "Strategy7\outputs\smoke_v2\run_strategy7_02_train_tree_fe_dynamic"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

conda run -n env_quant --no-capture-output python .\Strategy7\run_strategy7.py `
  --data-root D:/PythonProject/Quant/data_baostock/stock_hist/hs300 `
  --hs300-list-path D:/PythonProject/Quant/data_baostock/metadata/stock_list_hs300.csv `
  --index-root D:/PythonProject/Quant/data_baostock/ak_index `
  --train-start 2024-01-01 `
  --train-end 2024-06-30 `
  --test-start 2024-07-01 `
  --test-end 2024-09-30 `
  --factor-freq D `
  --factor-packages "trend,reversal,liquidity,volatility,structure,context" `
  --label-task return `
  --max-files 20 `
  --enable-factor-engineering true `
  --fe-orth-method none `
  --fe-corr-threshold 0.90 `
  --fe-preselect-top-n 120 `
  --stock-model-type decision_tree `
  --timing-model-type volatility_regime `
  --portfolio-model-type dynamic_opt `
  --execution-model-type realistic_fill `
  --max-participation-rate 0.20 `
  --base-fill-rate 0.90 `
  --latency-bars 1 `
  --model-run-mode train `
  --enable-next-bar-inference true `
  --inference-top-k 15 `
  --horizon 5 `
  --top-k 12 `
  --long-threshold 0.55 `
  --execution-scheme vwap30_vwap30 `
  --fee-bps 1.5 `
  --slippage-bps 1.5 `
  --save-models true `
  --output-dir $outputDir

if ($LASTEXITCODE -ne 0) {
  throw "Template execution failed with exit code $LASTEXITCODE"
}
