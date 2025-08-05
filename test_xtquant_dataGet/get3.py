from xtquant import xtdata
# xtdata基础版权限
# 5m数据-1年
# 1m数据-1年
# tick数据-1个月
# 1d数据-289天（一年半）

if __name__ == "__main__":
    # ===== 配置参数 =====
    stock_code = "601088.SH"  # 平安银行
    # ===== 获取分钟周期数据 =====
    for period in ["1d"]:
        xtdata.download_history_data(
            stock_code=stock_code,
            period=period,
            start_time="20250730",
            end_time="20250805"
        )
    print(f"{period}数据下载完成！")
    df = xtdata.get_market_data(
        field_list=["time", "open", "high", "low", "close", "volume"],
        stock_list=[stock_code],
        period=period,
        start_time="20250730",
        end_time="20250805",
        fill_data=False  # 填充缺失数据
    )
    print(df)