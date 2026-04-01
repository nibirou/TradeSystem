Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location "D:\PythonProject\Quant\TradeSystem"

python .\Strategy7\run_strategy7.py `
  --data-root D:/PythonProject/Quant/data_baostock/stock_hist/hs300 `
  --hs300-list-path D:/PythonProject/Quant/data_baostock/metadata/stock_list_hs300.csv `
  --index-root D:/PythonProject/Quant/data_baostock/ak_index `
  --train-start 2024-01-01 `
  --train-end 2024-12-31 `
  --test-start 2025-01-01 `
  --test-end 2025-12-31 `
  --main-board-only `
  --factor-freq D `
  --label-task direction `
  --stock-model-type decision_tree `
  --timing-model-type volatility_regime `
  --timing-vol-threshold 0.0 `
  --timing-momentum-threshold 0.0 `
  --portfolio-model-type dynamic_opt `
  --opt-max-weight 0.25 `
  --opt-max-turnover 1.20 `
  --opt-liquidity-scale 3.0 `
  --opt-expected-return-weight 1.0 `
  --opt-risk-aversion 1.2 `
  --opt-style-penalty 0.8 `
  --opt-industry-penalty 0.7 `
  --opt-crowding-penalty 0.5 `
  --opt-transaction-cost-penalty 0.6 `
  --opt-max-iter 220 `
  --opt-step-size 0.08 `
  --opt-tolerance 1e-6 `
  --execution-model-type realistic_fill `
  --max-participation-rate 0.15 `
  --base-fill-rate 0.95 `
  --latency-bars 0 `
  --execution-scheme vwap30_vwap30 `
  --horizon 5 `
  --top-k 10 `
  --long-threshold 0.50 `
  --fee-bps 1.5 `
  --slippage-bps 1.5 `
  --save-models `
  --output-dir auto

