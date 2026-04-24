Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..\\..")).Path
Set-Location $repoRoot

$outputDir = Join-Path $repoRoot "Strategy7\outputs\smoke_v2\run_strategy7_10_list_factors"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

conda run -n env_quant --no-capture-output python .\Strategy7\run_strategy7.py `
  --list-factors `
  --factor-freq D `
  --factor-packages "trend,reversal,liquidity,volatility,flow,crowding" `
  --auto-export-factor-snapshot true `
  --export-factor-list `
  --factor-list-export-format markdown `
  --output-dir $outputDir `
  --log-level quiet

if ($LASTEXITCODE -ne 0) {
  throw "Template execution failed with exit code $LASTEXITCODE"
}


