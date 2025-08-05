from xtquant import xtdata
# xtdata基础版权限
# 5m数据-1年
# 1m数据-1年
# tick数据-1个月
# 1d数据-289天（一年半）
# 财务数据也只能获取到近一年，且存在大量缺失

if __name__ == "__main__":
    # ===== 配置参数 =====
    stock_code = "601088.SH"  # 平安银行
    # ===== 获取财务数据 =====
    xtdata.download_financial_data2([stock_code])
    print(f"财务数据下载完成！")
    df = xtdata.get_financial_data([stock_code], table_list=['Capital'], start_time='', end_time='', report_type='report_time')
    print(df)