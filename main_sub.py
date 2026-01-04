
import os
import sys
import pathlib
p = str(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(p)
import datetime
import pandas as pd

today = datetime.date.today().strftime('%Y%m%d')

# 数据检查
data_path = r"C:\Users\Administrator\Desktop\hot_daily_market_data"

for f in os.listdir(data_path):
    variety_name = f.split(".")[0]
    df_bar = pd.read_csv(os.path.join(data_path, f), index_col=[0])
    last_date = df_bar.index[-1]
    if pd.to_datetime(last_date) < pd.to_datetime(today):
        print(f"数据缺失, variety: {variety_name}, today: {today}, last_date: {last_date}!")


from daily.position_main import run as run_main
from daily.position_sub import run as run_sub


if __name__ == '__main__':
    date_main, df_main = run_main()
    print("run_main done!")
    date_sub, df_sub = run_sub()
    print("run_sub done!")
    assert date_main == date_sub

    df_pos = pd.concat([df_main, df_sub], axis=0).sort_values(by=['market_value'], ascending=False)

    position_path = r"C:\Users\Administrator\Desktop\position"
    # position_path = r"C:\Users\Administrator\Desktop"
    file = os.path.join(position_path, f"position_{today}.csv")
    df_pos.to_csv(file, index=False)

