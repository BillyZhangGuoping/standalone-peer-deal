import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 检查AG品种的历史数据和标签生成
print("1. 检查AG品种的历史数据和标签生成...")
ag_data = pd.read_csv('History_Data/hot_daily_market_data/ag.csv', index_col=0, parse_dates=True)

# 计算下一天的收益率和标签
ag_data['next_day_return'] = ag_data['close'].shift(-1) - ag_data['close']
ag_data['label'] = np.sign(ag_data['next_day_return'])

# 查看2025年10月至12月的数据
ag_2025 = ag_data.loc['2025-10-01':'2025-12-31']

print(f"AG 2025年10-12月数据量: {len(ag_2025)} 条")
print(f"\nAG 2025年10-12月收盘价统计:")
print(f"  最小值: {ag_2025['close'].min():.2f}")
print(f"  最大值: {ag_2025['close'].max():.2f}")
print(f"  平均值: {ag_2025['close'].mean():.2f}")
print(f"  总涨幅: {(ag_2025['close'].iloc[-1] - ag_2025['close'].iloc[0])/ag_2025['close'].iloc[0]:.2%}")

print(f"\nAG 2025年10-12月标签分布:")
print(ag_2025['label'].value_counts())

# 2. 检查特征工程
print("\n2. 检查特征工程...")
from random_forest_strategy.random_forest_main import preprocess_data

# 预处理AG数据
ag_processed = preprocess_data(ag_data.copy())
print(f"\nAG处理后数据量: {len(ag_processed)} 条")
print(f"\n处理后特征列表 (共 {len(ag_processed.columns)} 个):")
for col in ag_processed.columns:
    if col not in ['open', 'high', 'low', 'close', 'volume', 'symbol']:
        print(f"  - {col}")

# 3. 检查模型训练数据的特征相关性
print("\n3. 检查关键特征与标签的相关性...")
# 只检查数值特征
numeric_cols = ag_processed.select_dtypes(include=[np.number]).columns

# 计算特征与标签的相关性
corr_with_label = ag_processed[numeric_cols].corrwith(ag_processed['label']).sort_values(ascending=False)
print(f"\n特征与标签的相关性 (前10名):")
print(corr_with_label.head(10))
print(f"\n特征与标签的相关性 (后10名):")
print(corr_with_label.tail(10))

# 4. 检查模型预测逻辑
print("\n4. 检查模型预测逻辑...")
print("查看AG 2025年12月的特征和实际涨跌情况:")
ag_dec = ag_2025.loc['2025-12-01':'2025-12-31']

# 计算一些关键特征
ag_dec['return_10'] = ag_dec['close'].pct_change(10)
ag_dec['volatility_20'] = ag_dec['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
ag_dec['ma_5'] = ag_dec['close'].rolling(window=5).mean()
ag_dec['ma_20'] = ag_dec['close'].rolling(window=20).mean()
ag_dec['ma_60'] = ag_dec['close'].rolling(window=60).mean()

# 显示关键数据
print(ag_dec[['close', 'next_day_return', 'label', 'return_10', 'volatility_20', 'ma_5', 'ma_20', 'ma_60']].dropna())

# 5. 可视化分析
print("\n5. 生成可视化分析...")
plt.figure(figsize=(12, 6))
plt.plot(ag_2025['close'], label='AG收盘价')
plt.plot(ag_2025['ma_5'], label='5日均线')
plt.plot(ag_2025['ma_20'], label='20日均线')
plt.plot(ag_2025['ma_60'], label='60日均线')
plt.title('AG 2025年10-12月收盘价及均线')
plt.legend()
plt.savefig('ag_close_analysis.png')
print("  AG收盘价及均线图已保存到 ag_close_analysis.png")

# 保存诊断报告
with open('model_diagnosis_report.txt', 'w', encoding='gbk') as f:
    f.write("模型诊断报告\n")
    f.write("="*50 + "\n\n")
    f.write("1. 数据基本情况\n")
    f.write(f"   AG 2025年10-12月数据量: {len(ag_2025)} 条\n")
    f.write(f"   收盘价范围: {ag_2025['close'].min():.2f} - {ag_2025['close'].max():.2f}\n")
    f.write(f"   总涨幅: {(ag_2025['close'].iloc[-1] - ag_2025['close'].iloc[0])/ag_2025['close'].iloc[0]:.2%}\n\n")
    
    f.write("2. 标签分布\n")
    f.write(f"   {ag_2025['label'].value_counts().to_string()}\n\n")
    
    f.write("3. 特征与标签相关性\n")
    f.write(f"   {corr_with_label.to_string()}\n\n")
    
    f.write("4. 关键结论\n")
    f.write("   - 请根据上述分析结果，检查模型训练数据、特征工程和模型参数\n")
    f.write("   - 重点关注相关性低的特征和标签生成逻辑\n")
    f.write("   - 考虑增加更多趋势类特征，或调整模型参数以减少过拟合\n")

print("\n模型诊断报告已生成: model_diagnosis_report.txt")
print("\n请查看生成的诊断报告和可视化图表，分析模型预测方向错误的根本原因。")
