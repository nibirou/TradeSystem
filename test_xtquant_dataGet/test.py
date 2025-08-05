
from xtquant import xtdata
import datetime
import pandas as pd
def download_full_market_history():
    """下载全市场近1年日线数据"""
    # 动态计算时间范围
    end_date = datetime.datetime.now().strftime("%Y%m%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y%m%d")

    # 获取沪深A股全部股票
    all_stocks = xtdata.get_stock_list_in_sector("沪深A股")

    # 带进度监控的批量下载
    def on_progress(data):
        print(f"进度: {data['finished']}/{data['total']} - {data['stockcode']}")

   
    xtdata.download_history_data2(
        stock_list=all_stocks,
        period="1d",
        start_time=start_date,
        end_time=end_date,
        callback=on_progress
    )
if __name__ == "__main__":
    # download_full_market_history()

   
    data = xtdata.get_local_data(
        field_list=["open", "high", "low", "close", "volume", ],  # 必须指定字段
        stock_list=['600519.SH'],
        period="1d",
        start_time="", 
        end_time=""
    )

    print(data)