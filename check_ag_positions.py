import os
import pandas as pd
import glob

# 获取最新的目标头寸文件夹
target_dir = 'random_forest_strategy/target_position'
subdirs = [d for d in glob.glob(os.path.join(target_dir, '*')) if os.path.isdir(d)]
latest_subdir = max(subdirs, key=os.path.getmtime)

print(f"最新的目标头寸文件夹: {latest_subdir}")

# 读取该文件夹下所有CSV文件
csv_files = glob.glob(os.path.join(latest_subdir, '*.csv'))

# 筛选出包含AG品种的数据
ag_data = []
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file, encoding='gbk')
        # 查找AG相关的行（不区分大小写）
        ag_rows = df[df['symbol'].str.contains('AG|ag', na=False)]
        if not ag_rows.empty:
            # 添加日期列
            date = os.path.basename(csv_file).split('_')[-1].split('.')[0]
            ag_rows['date'] = date
            ag_data.append(ag_rows)
    except Exception as e:
        print(f"读取文件 {csv_file} 时出错: {e}")

if ag_data:
    # 合并所有AG数据
    ag_df = pd.concat(ag_data, ignore_index=True)
    # 按日期排序
    ag_df = ag_df.sort_values('date')
    
    # 显示AG品种的头寸数据
    print("\nAG品种头寸数据:")
    print(ag_df[['date', 'symbol', 'current_price', 'position_size', 'signal']])
    
    # 保存到文件便于查看
    ag_df.to_csv('ag_positions.csv', index=False, encoding='gbk')
    print("\nAG头寸数据已保存到 ag_positions.csv")
else:
    print("未找到AG品种的头寸数据")
