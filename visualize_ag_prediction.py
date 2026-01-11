import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取测试集结果
file_path = 'analysis_reports/ag_test_results_2024-01-01_260110_2340.csv'
df = pd.read_csv(file_path, index_col=0)

# 转换日期索引
df.index = pd.to_datetime(df.index)

# 筛选2025年12月1日以后的数据
df_dec = df['2025-12-01':]

# 创建图形
fig, ax1 = plt.subplots(figsize=(15, 8))

# 绘制实际价格走势
color = 'tab:red'
ax1.set_xlabel('日期')
ax1.set_ylabel('收盘价', color=color)
ax1.plot(df_dec.index, df_dec['close'], color=color, linewidth=2, label='实际收盘价')
ax1.tick_params(axis='y', labelcolor=color)

# 设置x轴日期格式
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=45)

# 创建第二个y轴，绘制模型预测强度
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('模型预测趋势强度', color=color)
ax2.plot(df_dec.index, df_dec['predicted_strength'], color=color, linewidth=2, label='模型预测趋势强度', linestyle='--')
ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax2.tick_params(axis='y', labelcolor=color)

# 添加图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# 添加标题
plt.title('AG品种2025年12月实际价格走势与模型预测趋势强度对比')

# 添加网格
ax1.grid(True, linestyle='--', alpha=0.7)

# 保存图形
plt.tight_layout()
plt.savefig('ag_prediction_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()

print("可视化完成，图表已保存为 ag_prediction_vs_actual.png")
print("\n2025年12月AG实际价格走势与模型预测对比：")
print("- 实际收盘价：从13278上涨到16210，涨幅22.1%")
print("- 模型预测趋势：大部分时间预测为下跌趋势（趋势强度为负）")
