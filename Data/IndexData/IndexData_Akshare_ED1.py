# 通过akshare获取沪深300、中证500、中证1000的日频数据
# sh000300 sh000905 sh000852

import akshare as ak

hs300 = ak.stock_zh_index_daily(symbol="sh000300")
hs300.to_csv("/workspace/Quant/data_baostock/ak_index/hs300_price.csv", index=False)
hs300.to_parquet("/workspace/Quant/data_baostock/ak_index/hs300_price.parquet", index=False)
# hs300.to_csv("E:/pythonProject/data_baostock/ak_index/hs300_price.csv", index=False)
# hs300.to_parquet("E:/pythonProject/data_baostock/ak_index/hs300_price.parquet", index=False)

zz500 = ak.stock_zh_index_daily(symbol="sh000905")
zz500.to_csv("/workspace/Quant/data_baostock/ak_index/zz500_price.csv", index=False)
zz500.to_parquet("/workspace/Quant/data_baostock/ak_index/zz500_price.parquet", index=False)
# zz500.to_csv("E:/pythonProject/data_baostock/ak_index/zz500_price.csv", index=False)
# zz500.to_parquet("E:/pythonProject/data_baostock/ak_index/zz500_price.parquet", index=False)


zz1000 = ak.stock_zh_index_daily(symbol="sh000852")
zz1000.to_csv("/workspace/Quant/data_baostock/ak_index/zz1000_price.csv", index=False)
zz1000.to_parquet("/workspace/Quant/data_baostock/ak_index/zz1000_price.parquet", index=False)
# zz1000.to_csv("E:/pythonProject/data_baostock/ak_index/zz1000_price.csv", index=False)
# zz1000.to_parquet("E:/pythonProject/data_baostock/ak_index/zz1000_price.parquet", index=False)