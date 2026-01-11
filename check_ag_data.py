import pandas as pd
import numpy as np

# 读取AG的历史数据
ag_data = pd.read_csv('History_Data/hot_daily_market_data/ag.csv', index_col=0, parse_dates=True)

# 计算下一天的收益率和标签
ag_data['next_day_return'] = ag_data['close'].shift(-1) - ag_data['close']
ag_data['label'] = np.sign(ag_data['next_day_return'])

# 查看2025年10月至12月的数据
ag_2025 = ag_data.loc['2025-10-01':'2025-12-31']

print("AG 2025年10月至12月数据:")
print(ag_2025[['close', 'next_day_return', 'label']])

# 统计上涨和下跌的天数
up_days = (ag_2025['label'] == 1).sum()
down_days = (ag_2025['label'] == -1).sum()

print(f"\n上涨天数: {up_days}")
print(f"下跌天数: {down_days}")

# 保存到文件
ag_2025[['close', 'next_day_return', 'label']].to_csv('ag_2025_data.csv', encoding='gbk')
print("\nAG 2025年数据已保存到 ag_2025_data.csv")
