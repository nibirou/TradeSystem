Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $repoRoot

$storeRoot = Join-Path $repoRoot "Strategy7\outputs\smoke_v2\factor_mining_store"
New-Item -ItemType Directory -Force -Path $storeRoot | Out-Null

conda run -n env_quant --no-capture-output python .\Strategy7\run_factor_mining.py `
  --framework minute_parametric `
  --data-root D:/PythonProject/Quant/data_baostock/stock_hist/hs300 `
  --hs300-list-path D:/PythonProject/Quant/data_baostock/metadata/stock_list_hs300.csv `
  --index-root D:/PythonProject/Quant/data_baostock/ak_index `
  --train-start 2024-01-01 `
  --train-end 2024-02-29 `
  --valid-start 2024-03-01 `
  --valid-end 2024-03-31 `
  --factor-freq 30min `
  --factor-packages "trend,reversal,liquidity,volatility,intraday_signature,intraday_micro,price_action,bridge,multi_freq" `
  --max-files 12 `
  --horizon 8 `
  --execution-scheme open5_twap_last30 `
  --population-size 8 `
  --generations 1 `
  --elite-size 2 `
  --mutation-rate 0.25 `
  --crossover-rate 0.7 `
  --top-n 5 `
  --corr-threshold 0.65 `
  --min-cross-section 12 `
  --top-frac 0.1 `
  --factor-store-root $storeRoot `
  --save-format parquet

if ($LASTEXITCODE -ne 0) {
  throw "Template execution failed with exit code $LASTEXITCODE"
}
