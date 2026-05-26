param(
  [Parameter(Mandatory = $true)]
  [string]$TimingModelPath,
  [Parameter(Mandatory = $true)]
  [string]$ExecutionModelPath
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..")
$QuantRoot = if ($env:QUANT_ROOT) { $env:QUANT_ROOT } else { Resolve-Path (Join-Path $RepoRoot "..") }
Set-Location $RepoRoot

$OutputDir = Join-Path $RepoRoot "Strategy7\outputs\smoke_v2\run_strategy7_31_mixed_component_modes"
$IndexRoot = Join-Path $QuantRoot "data_baostock\ak_index"
$DataRoot = Join-Path $QuantRoot "data_baostock\stock_hist\hs300"
$Hs300List = Join-Path $QuantRoot "data_baostock\metadata\stock_list_hs300.csv"
$CondaEnv = if ($env:CONDA_ENV) { $env:CONDA_ENV } else { "env_quant" }
$TrainStart = if ($env:STRATEGY7_TRAIN_START) { $env:STRATEGY7_TRAIN_START } else { "2024-01-01" }
$TrainEnd = if ($env:STRATEGY7_TRAIN_END) { $env:STRATEGY7_TRAIN_END } else { "2024-06-30" }
$TestStart = if ($env:STRATEGY7_TEST_START) { $env:STRATEGY7_TEST_START } else { "2024-07-01" }
$TestEnd = if ($env:STRATEGY7_TEST_END) { $env:STRATEGY7_TEST_END } else { "2024-09-30" }
$FactorFreq = if ($env:STRATEGY7_FACTOR_FREQ) { $env:STRATEGY7_FACTOR_FREQ } else { "D" }
$Horizon = if ($env:STRATEGY7_HORIZON) { $env:STRATEGY7_HORIZON } else { "1" }
$TopK = if ($env:STRATEGY7_TOP_K) { $env:STRATEGY7_TOP_K } else { "20" }
$MaxFiles = if ($env:STRATEGY7_MAX_FILES) { $env:STRATEGY7_MAX_FILES } else { "40" }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

conda run -n $CondaEnv --no-capture-output python `
  .\Strategy7\run_strategy7.py `
  --data-root $DataRoot `
  --hs300-list-path $Hs300List `
  --index-root $IndexRoot `
  --train-start $TrainStart `
  --train-end $TrainEnd `
  --test-start $TestStart `
  --test-end $TestEnd `
  --factor-freq $FactorFreq `
  --factor-packages "trend,reversal,liquidity,volatility,flow,price_action" `
  --label-task return `
  --max-files $MaxFiles `
  --disable-text-data `
  --enable-factor-engineering true `
  --fe-corr-threshold 0.90 `
  --fe-preselect-top-n 160 `
  --stock-model-type decision_tree `
  --timing-model-type lstm_madl `
  --portfolio-model-type dynamic_opt `
  --execution-model-type realistic_fill `
  --model-run-mode train `
  --stock-model-run-mode train `
  --timing-model-run-mode load `
  --portfolio-model-run-mode train `
  --execution-model-run-mode load `
  --timing-model-path $TimingModelPath `
  --execution-model-path $ExecutionModelPath `
  --horizon $Horizon `
  --top-k $TopK `
  --long-threshold 0.5 `
  --execution-scheme open5_open5 `
  --fee-bps 1.5 `
  --slippage-bps 1.5 `
  --save-models true `
  --output-dir $OutputDir
