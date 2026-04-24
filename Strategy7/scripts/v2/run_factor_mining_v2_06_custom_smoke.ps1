Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..\\..")).Path
Set-Location $repoRoot

$storeRoot = Join-Path $repoRoot "Strategy7\outputs\smoke_v2\factor_mining_store"
New-Item -ItemType Directory -Force -Path $storeRoot | Out-Null

conda run -n env_quant --no-capture-output python .\Strategy7\run_factor_mining.py `
  --framework custom `
  --data-root D:/PythonProject/Quant/data_baostock/stock_hist/hs300 `
  --hs300-list-path D:/PythonProject/Quant/data_baostock/metadata/stock_list_hs300.csv `
  --index-root D:/PythonProject/Quant/data_baostock/ak_index `
  --disable-fundamental-data `
  --disable-text-data `
  --train-start 2022-01-01 `
  --train-end 2023-06-30 `
  --valid-start 2023-07-01 `
  --valid-end 2023-12-31 `
  --factor-freq D `
  --factor-list "breakout_20,amihud_20,atr_norm_14" `
  --max-files 20 `
  --horizon 5 `
  --population-size 8 `
  --generations 1 `
  --elite-size 2 `
  --mutation-rate 0.2 `
  --crossover-rate 0.7 `
  --top-n 4 `
  --corr-threshold 0.7 `
  --min-cross-section 20 `
  --top-frac 0.1 `
  --factor-store-root $storeRoot `
  --save-format parquet

if ($LASTEXITCODE -ne 0) {
  throw "Template execution failed with exit code $LASTEXITCODE"
}



