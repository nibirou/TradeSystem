param(
  [Parameter(Mandatory = $true)]
  [string]$ModelSummaryJson
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..\\..")).Path
Set-Location $repoRoot

$outputDir = Join-Path $repoRoot "Strategy7\outputs\smoke_v2\run_strategy7_07_load_custom_summary"
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
  --factor-packages "trend,reversal,liquidity,volatility" `
  --max-files 20 `
  --stock-model-type decision_tree `
  --custom-stock-model-py .\Strategy7\strategy7\plugins\custom_stock_model_template.py `
  --timing-model-type none `
  --custom-timing-model-py .\Strategy7\strategy7\plugins\custom_timing_model_template.py `
  --portfolio-model-type equal_weight `
  --custom-portfolio-model-py .\Strategy7\strategy7\plugins\custom_portfolio_model_template.py `
  --execution-model-type ideal_fill `
  --custom-execution-model-py .\Strategy7\strategy7\plugins\custom_execution_model_template.py `
  --model-run-mode load `
  --load-fe-mode off `
  --model-summary-json $ModelSummaryJson `
  --horizon 5 `
  --top-k 10 `
  --long-threshold 0.5 `
  --save-models false `
  --output-dir $outputDir

if ($LASTEXITCODE -ne 0) {
  throw "Template execution failed with exit code $LASTEXITCODE"
}



