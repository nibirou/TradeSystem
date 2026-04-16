Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location "D:\PythonProject\Quant\TradeSystem"

python .\Strategy7\run_factor_mining.py `
  --framework minute_parametric `
  --universe hs300 `
  --data-root D:/PythonProject/Quant/data_baostock/stock_hist/hs300 `
  --stock-list-path D:/PythonProject/Quant/data_baostock/metadata/stock_list_hs300.csv `
  --index-root D:/PythonProject/Quant/data_baostock/ak_index `
  --train-start 2021-01-01 `
  --train-end 2023-12-31 `
  --valid-start 2024-01-01 `
  --valid-end 2024-12-31 `
  --factor-freq D `
  --horizon 5 `
  --population-size 96 `
  --generations 16 `
  --elite-size 10 `
  --mutation-rate 0.30 `
  --crossover-rate 0.70 `
  --top-n 20 `
  --corr-threshold 0.60 `
  --min-cross-section 30 `
  --top-frac 0.10 `
  --random-state 42 `
  --factor-store-root D:/PythonProject/Quant/data_baostock `
  --catalog-path auto `
  --save-format parquet
