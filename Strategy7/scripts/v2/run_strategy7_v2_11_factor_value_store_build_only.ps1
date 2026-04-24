Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..\\..")).Path
Set-Location $repoRoot

$outputDir = Join-Path $repoRoot "Strategy7\outputs\smoke_v2\run_strategy7_11_factor_value_store"
$storeRoot = Join-Path $repoRoot "Strategy7\outputs\smoke_v2\factor_value_store"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
New-Item -ItemType Directory -Force -Path $storeRoot | Out-Null

conda run -n env_quant --no-capture-output python .\Strategy7\run_strategy7.py `
  --data-root D:/PythonProject/Quant/data_baostock/stock_hist/hs300 `
  --hs300-list-path D:/PythonProject/Quant/data_baostock/metadata/stock_list_hs300.csv `
  --index-root D:/PythonProject/Quant/data_baostock/ak_index `
  --factor-freq D `
  --factor-packages "trend,reversal" `
  --max-files 15 `
  --enable-factor-value-store true `
  --factor-value-store-root $storeRoot `
  --factor-value-store-format parquet `
  --factor-value-store-build-all true `
  --factor-value-store-build-only true `
  --output-dir $outputDir `
  --log-level normal

if ($LASTEXITCODE -ne 0) {
  throw "Template execution failed with exit code $LASTEXITCODE"
}



