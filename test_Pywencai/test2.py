# pywencai 获取个股数据

import pywencai
import pandas as pd

# 列名与数据对其显示
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

pd.set_option('display.max_rows', None)  # 设置显示无限制行
pd.set_option('display.max_columns', None)  # 设置显示无限制列

pd.set_option('display.expand_frame_repr', False) #设置不折叠数据
pd.set_option('display.max_colwidth', 100)

code ="600519"
param = f"{code}"
df = pywencai.get(query= param , queryType='report')

print(df)

