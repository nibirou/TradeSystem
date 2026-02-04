# 测试akshre获取个股基本面数据接口（主要关键指标）

# 新浪可用-关键指标（79个）
import akshare as ak
stock_financial_abstract_df = ak.stock_financial_abstract(symbol="600004")
print(stock_financial_abstract_df)

# 同花顺-关键指标（akshare接口有问题，接口不对，无法使用）
import akshare as ak
stock_financial_abstract_new_ths_df = ak.stock_financial_abstract_new_ths(symbol="000063", indicator="按报告期")
print(stock_financial_abstract_new_ths_df)

# 东财-关键指标（140个）
import akshare as ak
stock_financial_analysis_indicator_em_df = ak.stock_financial_analysis_indicator_em(symbol="301389.SZ", indicator="按报告期")
print(stock_financial_analysis_indicator_em_df.iloc[:, :50])

# 新浪-财务指标（87个）
import akshare as ak
stock_financial_analysis_indicator_df = ak.stock_financial_analysis_indicator(symbol="600004", start_year="2020")
print(stock_financial_analysis_indicator_df)