Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $repoRoot

$outputDir = Join-Path $repoRoot "Strategy7\outputs\smoke_v2\run_strategy7_19_list_factors_30min_json"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$exportPath = Join-Path $outputDir "factor_list_30min.json"

conda run -n env_quant --no-capture-output python .\Strategy7\run_strategy7.py `
  --list-factors `
  --factor-freq 30min `
  --factor-packages "trend,reversal,liquidity,volatility,intraday_signature,intraday_micro,price_action,bridge,multi_freq" `
  --auto-export-factor-snapshot true `
  --export-factor-list `
  --factor-list-export-format json `
  --factor-list-export-path $exportPath `
  --output-dir $outputDir

if ($LASTEXITCODE -ne 0) {
  throw "Template execution failed with exit code $LASTEXITCODE"
}
