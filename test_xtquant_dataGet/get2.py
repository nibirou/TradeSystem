
from xtquant import xtdata
import pandas as pd
import time
def download_minute_data(stock_code, period):
    xtdata.download_history_data(
        stock_code=stock_code,
        period=period
    )
    print(f"{period}数据下载完成！")
def get_minute_data(stock_code, period):
    data = xtdata.get_market_data(
        field_list=["time", "open", "high", "low", "close", "volume"],
        stock_list=[stock_code],
        period=period,
        fill_data=True  # 填充缺失数据
    )
    return data
if __name__ == "__main__":
    # ===== 配置参数 =====
    stock_code = "000001.SZ"  # 平安银行
    # ===== 获取分钟周期数据 =====
    for period in ["1m"]:
        download_minute_data(stock_code, period)
        df = get_minute_data(stock_code, period)
        print(df)
    
    for period in ["5m"]:
        download_minute_data(stock_code, period)
        df5 = get_minute_data(stock_code, period)
        print(df5)
    df15 = get_minute_data(stock_code, '15m')
    print(df15)
    df60 = get_minute_data(stock_code, '60m')
    print(df60)