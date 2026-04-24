param(
  [Parameter(Mandatory = $true)]
  [string]$ModelSummaryJson,
  [string]$DataRoot = "auto",
  [string]$IndexRoot = "",
  [string]$TrainStart = "2020-01-01",
  [string]$TrainEnd = "2023-12-31",
  [string]$TestStart = "2024-01-01",
  [string]$TestEnd = "2025-12-31",
  [int]$MaxFiles = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $repoRoot

$outputDir = Join-Path $repoRoot "Strategy7\outputs\smoke_v2\run_strategy7_21_load_allmarket_bottom_launch_10d"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$args = @(
  ".\\Strategy7\\run_strategy7.py",
  "--universe", "all",
  "--data-root", $DataRoot,
  "--train-start", $TrainStart,
  "--train-end", $TrainEnd,
  "--test-start", $TestStart,
  "--test-end", $TestEnd,
  "--factor-freq", "D",
  "--factor-packages", "bottom_launch,trend,reversal,liquidity,volatility,price_action,crowding,oscillator,overnight,multi_freq,context,fund_quality,fund_cashflow,text_event,text_sentiment",
  "--label-task", "return",
  "--enable-factor-engineering", "true",
  "--stock-model-type", "launch_boost",
  "--launch-boost-max-depth", "5",
  "--launch-boost-learning-rate", "0.04",
  "--launch-boost-max-iter", "400",
  "--launch-boost-l2", "1.5",
  "--launch-boost-return-head-weight", "0.40",
  "--timing-model-type", "volatility_regime",
  "--portfolio-model-type", "dynamic_opt",
  "--execution-model-type", "realistic_fill",
  "--model-run-mode", "load",
  "--load-fe-mode", "refit",
  "--model-summary-json", $ModelSummaryJson,
  "--enable-next-bar-inference", "true",
  "--inference-top-k", "30",
  "--horizon", "10",
  "--top-k", "25",
  "--long-threshold", "0.60",
  "--execution-scheme", "daily_close_daily_close",
  "--fee-bps", "2.0",
  "--slippage-bps", "2.0",
  "--save-models", "false",
  "--output-dir", $outputDir
)

if (-not [string]::IsNullOrWhiteSpace($IndexRoot)) {
  $args += @("--index-root", $IndexRoot)
}

if ($MaxFiles -gt 0) {
  $args += @("--max-files", "$MaxFiles")
}

& conda run -n env_quant --no-capture-output python @args

if ($LASTEXITCODE -ne 0) {
  throw "Template execution failed with exit code $LASTEXITCODE"
}
