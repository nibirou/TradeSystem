# 基于 AkShare 金融数据接口，构建了一个A股市场数据的本地缓存系统，
# 用于批量采集和存储股票列表、历史行情、财务指标和板块概念信息，
# 并支持按需刷新与分频率（日/周/月）处理。适合用于量化研究与策略开发的前置数据准备。

import os
import pandas as pd
import akshare as ak
from tqdm import tqdm
from datetime import datetime
# from sqlalchemy import create_engine

# ========== 配置路径 ==========
BASE_DIR = "data"
HIST_DIR = os.path.join(BASE_DIR, "stock_hist")
FIN_DIR = os.path.join(BASE_DIR, "stock_finance")
META_DIR = os.path.join(BASE_DIR, "metadata")

for d in [BASE_DIR, HIST_DIR, FIN_DIR, META_DIR]:
    os.makedirs(d, exist_ok=True)

# ========== MySQL配置 ==========
# DB_URI = "mysql+pymysql://zongcaicv:zongcaicv-mysql@10.223.48.244:8660/stock_data?charset=utf8mb4"
# engine = create_engine(DB_URI)

# 判断下载的CSV是否存在且有效（非空）
def is_valid_csv(path):
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
        return not df.empty
    except:
        return False
# ========== 保存函数：CSV + Parquet + MySQL ==========
def save_data(df, path_prefix, table_name):
    # 保存 CSV
    df.to_csv(f"{path_prefix}.csv", index=False)
    # 保存 Parquet
    df.to_parquet(f"{path_prefix}.parquet", index=False)
    # 保存到 MySQL
    # df.to_sql(table_name, engine, if_exists='replace', index=False)

# ========== 获取最新的交易日（不晚于今天） ==========
def get_latest_trade_date():
    """
    获取最新的交易日（不晚于今天）
    """
    from datetime import datetime

    today = datetime.today().date()  # 👈 转成 datetime.date 类型
    trade_dates = ak.tool_trade_date_hist_sina()
    trade_dates["trade_date"] = pd.to_datetime(trade_dates["trade_date"]).dt.date  # 👈 确保列为 date 类型
    trade_dates = trade_dates[trade_dates["trade_date"] < today]
    latest_date = trade_dates["trade_date"].max()
    return latest_date.strftime("%Y%m%d")  # 👈 最终返回字符串格式如 '20250729'

# ========== 股票列表 ==========
def get_stock_list(refresh=False):
    path_prefix = os.path.join(META_DIR, "stock_list")
    table_name = "stock_list"

    if os.path.exists(f"{path_prefix}.csv") and not refresh:
        return pd.read_csv(f"{path_prefix}.csv", dtype=str)

    # 1. 初步筛选：实时行情数据（静态信息用）
    df = ak.stock_zh_a_spot_em()
    print(f"[初筛] 股票数量: {len(df)}")

    # 字段转换
    df["总市值"] = pd.to_numeric(df["总市值"], errors="coerce")

    # 初筛条件：非ST + 总市值 > 200亿 + 排除300/688
    df = df[~df["名称"].str.contains("ST", na=False)]
    df = df[df["总市值"] > 200e8]
    df["代码"] = df["代码"].apply(lambda x: x[:6])
    df = df[~df["代码"].str.startswith(("300", "688"))]

    print(f"[初筛] 股票数量: {len(df)}")

    # 2. 获取前一交易日（自动识别）
    end_date = get_latest_trade_date()
    print(f"[使用交易日] {end_date}")

    # 3. 精筛成交量 > 0（逐个获取历史行情）
    filtered = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        code = row["代码"]
        try:
            hist = ak.stock_zh_a_hist(symbol=code, start_date=end_date, end_date=end_date, adjust="qfq")
            if hist.empty:
                continue

            volume = pd.to_numeric(hist.at[0, "成交量"], errors="coerce")
            if volume > 0:
                filtered.append({
                    "代码": code,
                    "名称": row["名称"],
                    "总市值": row["总市值"],
                    "成交量": volume
                })
        except:
            continue

    df_final = pd.DataFrame(filtered)
    save_data(df_final, path_prefix, table_name)
    print(f"[最终筛选] 股票数量: {len(df_final)}")
    return df_final

# ========== 历史行情 ==========
def get_stock_hist(code, start_date="20100101", end_date = "20250730", adjust="qfq", freq="D"):
    symbol = code
    path_prefix = os.path.join(HIST_DIR, f"{symbol}_{freq}")
    table_name = f"stock_hist_{freq}_{symbol}"

    if os.path.exists(f"{path_prefix}.csv"):
        return pd.read_csv(f"{path_prefix}.csv", parse_dates=["日期"])

    raw = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust=adjust)
    if raw.empty:
        return pd.DataFrame()

    raw["日期"] = pd.to_datetime(raw["日期"])
    raw.set_index("日期", inplace=True)

    # 字段重命名
    raw = raw.rename(columns={
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "change_percent",
        "涨跌额": "change",
        "换手率": "turnover_rate"
    })

    # 分频处理
    if freq == "W":
        df = raw.resample("W").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum",
            "change": "sum",  # 总涨跌额（可选）
            "change_percent": "mean",  # 周均涨幅
            "amplitude": "mean",
            "turnover_rate": "mean"
        })
    elif freq == "M":
        df = raw.resample("M").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum",
            "change": "sum",
            "change_percent": "mean",
            "amplitude": "mean",
            "turnover_rate": "mean"
        })
    else:
        df = raw[[
            "open", "high", "low", "close", "volume",
            "amount", "amplitude", "change_percent", "change", "turnover_rate"
        ]]

    df = df.dropna().reset_index()
    save_data(df, path_prefix, table_name)
    return df

# ========== 财务指标 ==========
# def get_finance_data(code):
#     path_prefix = os.path.join(FIN_DIR, code)
#     table_name = f"stock_finance_{code}"

#     if os.path.exists(f"{path_prefix}.csv"):
#         return pd.read_csv(f"{path_prefix}.csv")

#     try:
#         df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2010")
#         save_data(df, path_prefix, table_name)
#         return df
#     except:
#         return pd.DataFrame()
def get_finance_data(code):
    path_prefix = os.path.join(FIN_DIR, code)
    table_name = f"stock_finance_{code}"
    csv_path = f"{path_prefix}.csv"

    if is_valid_csv(csv_path):
        return

    # try:
    #     df = fetch_finance_with_retry(symbol=code)
    #     if df.empty:
    #         return
    #     save_data(df, path_prefix, table_name)
    # except Exception as e:
    #     print(f"[失败] 财务数据获取失败：{code} → {e}")
    df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2010")
    if df.empty:
        return
    save_data(df, path_prefix, table_name)

# ========== 概念板块 ==========
def get_stock_concept():
    path_prefix = os.path.join(META_DIR, "stock_concept")
    table_name = "stock_concept"

    if os.path.exists(f"{path_prefix}.csv"):
        return pd.read_csv(f"{path_prefix}.csv", dtype=str)

    df = ak.stock_board_concept_name_em()
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        concept = row["板块名称"]
        try:
            members = ak.stock_board_concept_cons_em(concept)
            for _, m in members.iterrows():
                records.append({"代码": m["代码"], "名称": m["名称"], "概念": concept})
        except:
            continue

    concept_df = pd.DataFrame(records)
    save_data(concept_df, path_prefix, table_name)
    return concept_df

# ========== 全量初始化 ==========
def init_all_data():
    stocks = get_stock_list()
    for code in tqdm(stocks["代码"].tolist()):
        # get_stock_hist(code)
        get_finance_data(code)
    # get_stock_concept()

# ========== 启动入口 ==========
if __name__ == '__main__':
    init_all_data()
