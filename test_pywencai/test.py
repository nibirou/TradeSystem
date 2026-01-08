# 使用pywencai
import pywencai
import pandas as pd

# 列名与数据对其显示
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

pd.set_option('display.max_rows', None)  # 设置显示无限制行
pd.set_option('display.max_columns', None)  # 设置显示无限制列

pd.set_option('display.expand_frame_repr', False) #设置不折叠数据
pd.set_option('display.max_colwidth', 100)


date ="20260108"
param = "{date}涨停，非涉嫌信息披露违规且非立案调查且非ST，非科创板，非北交所"
df = pywencai.get(query= param ,sort_key='成交金额', sort_order='desc')

selected_columns = ['股票代码', '股票简称', '最新价','最新涨跌幅', '首次涨停时间['+date + ']', '连续涨停天数['+date + ']','涨停原因类别['+date + ']','a股市值(不含限售股)['+date + ']','涨停类型['+date + ']']
jj_df = df[selected_columns]
#print(jj_df)
#
# # 按照'连板数'列进行降序排序
sorted_temp_df = jj_df.sort_values(by='连续涨停天数['+date + ']', ascending=False)
# 输出排序后的DataFrame
print(sorted_temp_df)