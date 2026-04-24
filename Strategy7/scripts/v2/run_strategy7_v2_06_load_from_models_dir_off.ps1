param(
  [Parameter(Mandatory = $true)]
  [string]$ModelsLoadDir,
  [string]$ModelsLoadRunTag = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $repoRoot

$outputDir = Join-Path $repoRoot "Strategy7\outputs\smoke_v2\run_strategy7_06_load_models_dir_off"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$cmd = @(
  "run", "-n", "env_quant", "--no-capture-output", "python", ".\\Strategy7\\run_strategy7.py",
  "--data-root", "D:/PythonProject/Quant/data_baostock/stock_hist/hs300",
  "--hs300-list-path", "D:/PythonProject/Quant/data_baostock/metadata/stock_list_hs300.csv",
  "--index-root", "D:/PythonProject/Quant/data_baostock/ak_index",
  "--train-start", "2024-01-01",
  "--train-end", "2024-06-30",
  "--test-start", "2024-07-01",
  "--test-end", "2024-09-30",
  "--factor-freq", "D",
  "--factor-packages", "trend,reversal,liquidity,volatility",
  "--max-files", "20",
  "--enable-factor-engineering", "true",
  "--stock-model-type", "decision_tree",
  "--timing-model-type", "none",
  "--portfolio-model-type", "equal_weight",
  "--execution-model-type", "ideal_fill",
  "--model-run-mode", "load",
  "--load-fe-mode", "off",
  "--models-load-dir", $ModelsLoadDir,
  "--horizon", "5",
  "--top-k", "10",
  "--long-threshold", "0.5",
  "--save-models", "false",
  "--output-dir", $outputDir
)
if ($ModelsLoadRunTag) {
  $cmd += @("--models-load-run-tag", $ModelsLoadRunTag)
}

& conda @cmd

if ($LASTEXITCODE -ne 0) {
  throw "Template execution failed with exit code $LASTEXITCODE"
}
