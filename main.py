
import os
import sys
sys.path.append(r"C:\Users\I333248\OneDrive - SAP SE\Desktop\Downloads\trade")
import datetime
import pandas as pd

today = datetime.date(year=2024,month=6,day=28).strftime('%Y%m%d')

# 数据检查
data_path = r"History_Data\\hot_daily_market_data"

for f in os.listdir(data_path):
    variety_name = f.split(".")[0]
    df_bar = pd.read_csv(os.path.join(data_path, f), index_col=[0])
    last_date = df_bar.index[-1]
    if pd.to_datetime(last_date) < pd.to_datetime(today):
        print(f"数据缺失, variety: {variety_name}, today: {today}, last_date: {last_date}!")


from daily.position import run


if __name__ == '__main__':
    run()

