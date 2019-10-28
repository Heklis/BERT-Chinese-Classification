import pandas as pd


df = pd.read_csv("./foodsafe_data/train.csv")
# 保存前8000条样本为训练集，后2000条样本为验证集
# 去掉行索引和表头
df.iloc[:8000].to_csv("./foodsafe_data/train.csv", index=None, header=None)
df.iloc[8000:].to_csv("./foodsafe_data/dev.csv", index=None, header=None)
