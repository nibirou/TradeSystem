Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $repoRoot

$storeRoot = Join-Path $repoRoot "Strategy7\outputs\smoke_v2\factor_mining_store"
New-Item -ItemType Directory -Force -Path $storeRoot | Out-Null

conda run -n env_quant --no-capture-output python .\Strategy7\run_factor_mining.py `
  --framework fundamental_multiobj `
  --list-factors `
  --factor-freq D `
  --custom-factor-py Strategy7/strategy7/plugins/custom_factor_template.py `
  --factor-packages "trend,reversal,liquidity,volatility" `
  --auto-export-factor-snapshot `
  --export-factor-list `
  --factor-list-export-format csv `
  --factor-store-root $storeRoot

if ($LASTEXITCODE -ne 0) {
  throw "Template execution failed with exit code $LASTEXITCODE"
}
