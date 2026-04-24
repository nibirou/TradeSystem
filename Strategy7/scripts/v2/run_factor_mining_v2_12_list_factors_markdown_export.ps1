Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $repoRoot

$storeRoot = Join-Path $repoRoot "Strategy7\outputs\smoke_v2\factor_mining_store"
$exportPath = Join-Path $storeRoot "factor_mining\factor_lists\factor_list_mining_D_markdown.md"
New-Item -ItemType Directory -Force -Path $storeRoot | Out-Null

conda run -n env_quant --no-capture-output python .\Strategy7\run_factor_mining.py `
  --framework fundamental_multiobj `
  --list-factors `
  --factor-freq D `
  --factor-packages "trend,reversal,liquidity,volatility,flow,crowding" `
  --auto-export-factor-snapshot `
  --export-factor-list `
  --factor-list-export-format markdown `
  --factor-list-export-path $exportPath `
  --factor-store-root $storeRoot

if ($LASTEXITCODE -ne 0) {
  throw "Template execution failed with exit code $LASTEXITCODE"
}
