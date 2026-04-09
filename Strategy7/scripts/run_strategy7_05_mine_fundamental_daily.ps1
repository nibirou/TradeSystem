Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# & conda activate env_quant
Set-Location "D:\PythonProject\Quant\TradeSystem"

"D:\miniforge3\envs\env_quant\python.exe" .\Strategy7\run_factor_mining.py `
  --framework fundamental_multiobj `
  --data-root D:/PythonProject/Quant/data_baostock/stock_hist/hs300 `
  --hs300-list-path D:/PythonProject/Quant/data_baostock/metadata/stock_list_hs300.csv `
  --index-root D:/PythonProject/Quant/data_baostock/ak_index `
  --train-start 2021-01-01 `
  --train-end 2023-12-31 `
  --valid-start 2024-01-01 `
  --valid-end 2024-12-31 `
  --factor-freq D `
  --horizon 5 `
  --population-size 128 `
  --generations 20 `
  --elite-size 12 `
  --mutation-rate 0.25 `
  --crossover-rate 0.70 `
  --top-n 20 `
  --corr-threshold 0.60 `
  --min-cross-section 30 `
  --top-frac 0.10 `
  --random-state 42 `
  --factor-store-root D:/PythonProject/Quant/data_baostock `
  --catalog-path auto `
  --save-format parquet
