$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..")
$QuantRoot = if ($env:QUANT_ROOT) { $env:QUANT_ROOT } else { Resolve-Path (Join-Path $RepoRoot "..") }
Set-Location $RepoRoot

$OutputDir = Join-Path $RepoRoot "Strategy7\outputs\run_strategy7_30_train_lstm_madl_timing"
$IndexRoot = Join-Path $QuantRoot "data_baostock\ak_index"
$CondaEnv = if ($env:CONDA_ENV) { $env:CONDA_ENV } else { "env_quant" }
$DataWorkers = if ($env:STRATEGY7_DATA_LOAD_WORKERS) { $env:STRATEGY7_DATA_LOAD_WORKERS } else { "8" }
$StoreWorkers = if ($env:STRATEGY7_FACTOR_STORE_WORKERS) { $env:STRATEGY7_FACTOR_STORE_WORKERS } else { "8" }
$TrainStart = if ($env:STRATEGY7_TRAIN_START) { $env:STRATEGY7_TRAIN_START } else { "2020-01-01" }
$TrainEnd = if ($env:STRATEGY7_TRAIN_END) { $env:STRATEGY7_TRAIN_END } else { "2024-12-31" }
$TestStart = if ($env:STRATEGY7_TEST_START) { $env:STRATEGY7_TEST_START } else { "2025-01-01" }
$TestEnd = if ($env:STRATEGY7_TEST_END) { $env:STRATEGY7_TEST_END } else { "2025-12-31" }
$Universe = if ($env:STRATEGY7_UNIVERSE) { $env:STRATEGY7_UNIVERSE } else { "hs300" }
$FactorFreq = if ($env:STRATEGY7_FACTOR_FREQ) { $env:STRATEGY7_FACTOR_FREQ } else { "D" }
$Horizon = if ($env:STRATEGY7_HORIZON) { $env:STRATEGY7_HORIZON } else { "1" }
$LstmEpochs = if ($env:STRATEGY7_TIMING_LSTM_EPOCHS) { $env:STRATEGY7_TIMING_LSTM_EPOCHS } else { "120" }
$LstmHidden = if ($env:STRATEGY7_TIMING_LSTM_HIDDEN_SIZES) { $env:STRATEGY7_TIMING_LSTM_HIDDEN_SIZES } else { "512,256,128" }
$LstmIntradaySeq = if ($env:STRATEGY7_TIMING_LSTM_INTRADAY_SEQ_LEN) { $env:STRATEGY7_TIMING_LSTM_INTRADAY_SEQ_LEN } else { "48" }
$LstmFeatureMode = if ($env:STRATEGY7_TIMING_LSTM_FEATURE_MODE) { $env:STRATEGY7_TIMING_LSTM_FEATURE_MODE } else { "auto" }
$LstmLossMode = if ($env:STRATEGY7_TIMING_LSTM_LOSS_MODE) { $env:STRATEGY7_TIMING_LSTM_LOSS_MODE } else { "madl_mse" }
$LstmExposureMode = if ($env:STRATEGY7_TIMING_LSTM_EXPOSURE_MODE) { $env:STRATEGY7_TIMING_LSTM_EXPOSURE_MODE } else { "long_only_bands" }
$LstmDevice = if ($env:STRATEGY7_TIMING_LSTM_DEVICE) { $env:STRATEGY7_TIMING_LSTM_DEVICE } else { "auto" }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

conda run -n $CondaEnv --no-capture-output python `
  .\Strategy7\run_strategy7.py `
  --train-start $TrainStart `
  --train-end $TrainEnd `
  --test-start $TestStart `
  --test-end $TestEnd `
  --universe $Universe `
  --data-root auto `
  --index-root $IndexRoot `
  --factor-freq $FactorFreq `
  --disable-text-data `
  --factor-packages "trend,reversal,liquidity,volatility,flow,price_action" `
  --label-task return `
  --horizon $Horizon `
  --data-load-workers $DataWorkers `
  --enable-factor-value-store true `
  --factor-value-store-root auto `
  --factor-value-store-format csv `
  --factor-value-store-workers $StoreWorkers `
  --enable-factor-engineering true `
  --fe-corr-threshold 0.90 `
  --fe-preselect-top-n 300 `
  --fe-min-factors 40 `
  --stock-model-type decision_tree `
  --timing-model-type lstm_madl `
  --timing-lstm-seq-len 20 `
  --timing-lstm-intraday-seq-len $LstmIntradaySeq `
  --timing-lstm-hidden-sizes $LstmHidden `
  --timing-lstm-dropout 0.20 `
  --timing-lstm-epochs $LstmEpochs `
  --timing-lstm-lr 1e-3 `
  --timing-lstm-weight-decay 1e-4 `
  --timing-lstm-early-stop 15 `
  --timing-lstm-batch-size 128 `
  --timing-lstm-min-train-samples 80 `
  --timing-lstm-feature-mode $LstmFeatureMode `
  --timing-lstm-loss-mode $LstmLossMode `
  --timing-lstm-exposure-mode $LstmExposureMode `
  --timing-lstm-market-agg amount_weighted `
  --timing-lstm-device $LstmDevice `
  --portfolio-model-type dynamic_opt `
  --opt-max-weight 0.12 `
  --opt-max-turnover 1.00 `
  --opt-risk-aversion 1.20 `
  --execution-model-type realistic_fill `
  --max-participation-rate 0.12 `
  --base-fill-rate 0.92 `
  --latency-bars 1 `
  --model-run-mode train `
  --execution-scheme open5_open5 `
  --fee-bps 1.5 `
  --slippage-bps 1.5 `
  --top-k 50 `
  --long-threshold 0.5 `
  --inference-top-k 20 `
  --save-models true `
  --output-dir $OutputDir
