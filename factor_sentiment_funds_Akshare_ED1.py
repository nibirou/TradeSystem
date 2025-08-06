# 个股消息面 + 资金面数据提取与处理
import os
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from snownlp import SnowNLP
from tqdm import tqdm

META_DIR = "data/metadata"
SENTIMENT_DIR = os.path.join(META_DIR, "sentiment")
FUND_DIR = os.path.join(META_DIR, "fund")
os.makedirs(SENTIMENT_DIR, exist_ok=True)
os.makedirs(FUND_DIR, exist_ok=True)

# ========== 新闻情绪因子模块 ==========
def get_news_sentiment(code, topk=20):
    """
    获取个股最近新闻标题情感平均值（+1 到 -1）
    """
    path = os.path.join(SENTIMENT_DIR, f"{code}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        try:
            df = ak.stock_news_em(symbol=code)
            df.to_csv(path, index=False)
        except:
            return 0.0

    if df is None or df.empty:
        return 0.0

    titles = df["标题"].dropna().astype(str).head(topk).tolist()
    scores = []
    for title in titles:
        try:
            s = SnowNLP(title).sentiments
            scores.append(s)
        except:
            continue

    if not scores:
        return 0.0

    # 转为 [-1,1] 区间，并求平均
    scores = [(s - 0.5) * 2 for s in scores]
    return round(sum(scores) / len(scores), 4)

# ========== 资金流因子模块 ==========
def get_fund_flow(code, days=5):
    """
    获取近days日主力净流入金额合计（正表示流入）
    """
    path = os.path.join(FUND_DIR, f"{code}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        try:
            df = ak.stock_individual_fund_flow(stock=code, market="沪深")
            df.to_csv(path, index=False)
        except:
            return 0.0

    if df is None or df.empty:
        return 0.0

    try:
        df["主力净流入-净额"] = pd.to_numeric(df["主力净流入-净额"], errors="coerce")
        return round(df.tail(days)["主力净流入-净额"].sum() / 1e4, 2)  # 返回万元
    except:
        return 0.0

# ========== 龙虎榜热度模块 ==========
def get_lhb_flag(code, start_date=None, end_date=None):
    """
    判断近一段时间是否多次上龙虎榜（热度标志）
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

    try:
        df = ak.stock_lhb_stock_statistic_em(symbol=code, start_date=start_date, end_date=end_date)
        if df is not None and not df.empty:
            return 1
        else:
            return 0
    except:
        return 0

# ========== 批量提取所有消息面/资金面因子 ==========
def extract_sentiment_fund_features(code_list):
    """
    提取情绪得分、资金流、龙虎榜标志
    """
    results = []
    for code in tqdm(code_list):
        senti = get_news_sentiment(code)
        net_in = get_fund_flow(code)
        lhb = get_lhb_flag(code)
        results.append({
            "代码": code,
            "新闻情绪": senti,
            "主力净流入_5日(万元)": net_in,
            "龙虎榜": lhb
        })
    return pd.DataFrame(results)

if __name__ == '__main__':
    # 示例：跑一遍股票列表
    df_list = pd.read_csv("data/metadata/stock_list.csv", dtype=str)
    df_feat = extract_sentiment_fund_features(df_list["代码"].tolist()[:50])
    df_feat.to_csv(os.path.join(META_DIR, "sentiment_fund_factors.csv"), index=False)
    print(df_feat.head())
