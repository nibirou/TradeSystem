Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..\\..")).Path
Set-Location $repoRoot

$outputDir = Join-Path $repoRoot "Strategy7\outputs\smoke_v2\run_strategy7_09_train_dafat"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

conda run -n env_quant --no-capture-output python .\Strategy7\run_strategy7.py `
  --data-root D:/PythonProject/Quant/data_baostock/stock_hist/hs300 `
  --hs300-list-path D:/PythonProject/Quant/data_baostock/metadata/stock_list_hs300.csv `
  --index-root D:/PythonProject/Quant/data_baostock/ak_index `
  --train-start 2023-07-01 `
  --train-end 2024-03-31 `
  --test-start 2024-04-01 `
  --test-end 2024-06-30 `
  --factor-freq D `
  --factor-packages "trend,reversal,liquidity,volatility" `
  --label-task return `
  --max-files 15 `
  --stock-model-type dafat `
  --dafat-seq-len 16 `
  --dafat-hidden-size 64 `
  --dafat-num-layers 1 `
  --dafat-num-heads 2 `
  --dafat-ffn-mult 2 `
  --dafat-dropout 0.1 `
  --dafat-local-window 10 `
  --dafat-topk-ratio 0.25 `
  --dafat-vol-quantile 0.4 `
  --dafat-meso-scale 3 `
  --dafat-macro-scale 8 `
  --dafat-epochs 1 `
  --dafat-lr 2e-4 `
  --dafat-weight-decay 1e-4 `
  --dafat-early-stop 1 `
  --dafat-per-epoch-batch 8 `
  --dafat-batch-size 64 `
  --dafat-label-transform csranknorm `
  --dafat-mse-weight 0.05 `
  --dafat-use-dpe true `
  --dafat-use-sparse-attn true `
  --dafat-use-multiscale true `
  --dafat-device cpu `
  --timing-model-type none `
  --portfolio-model-type equal_weight `
  --execution-model-type ideal_fill `
  --model-run-mode train `
  --horizon 5 `
  --top-k 10 `
  --long-threshold 0.5 `
  --save-models true `
  --output-dir $outputDir

if ($LASTEXITCODE -ne 0) {
  throw "Template execution failed with exit code $LASTEXITCODE"
}



