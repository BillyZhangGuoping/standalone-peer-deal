
import os
import sys
import pathlib
import time

import numpy as np

# p = str(pathlib.Path(__file__).parent.parent.absolute())
# sys.path.append(p)
sys.path.append(r"C:\trade")

import datetime
import pandas as pd

today = datetime.date.today().strftime('%Y%m%d')
# today = '20250317'

variety_list = ['i', 'rb', 'hc', 'jm', 'j', 'ss', 'ni', 'SF', 'SM', 'lc', 'sn', 'si', 'pb', 'cu', 'al', 'zn',
                'ao', 'SH', 'au', 'ag', 'OI', 'a', 'b', 'm', 'RM', 'p', 'y', 'CF', 'SR', 'c', 'cs', 'AP', 'PK',
                'jd', 'CJ', 'lh', 'sc', 'fu', 'lu', 'pg', 'bu', 'eg', 'TA', 'PF', 'PX', 'l', 'v', 'MA', 'pp',
                'eb', 'ru', 'nr', 'br', 'SA', 'FG', 'UR', 'sp', 'IF', 'IC', 'IH', 'IM', 'ec', 'T', 'TF'
                ]  # 'CY',

# 数据检查
data_path = r"C:\Users\Administrator\Desktop\hot_daily_market_data"

is_data_ready = True
while True:
    for variety_name in variety_list:
        f = f"{variety_name.upper()}.csv"
        df_bar = pd.read_csv(os.path.join(data_path, f), index_col=[0])
        last_date = df_bar.index[-1]
        if pd.to_datetime(last_date) < pd.to_datetime(today):
            print(f"数据缺失, variety: {variety_name}, today: {today}, last_date: {last_date}!!!!")
            is_data_ready = False
            continue

    if is_data_ready:
        break
    else:
        minute = 5
        print(f"wait {min} minutes  ... ")
        time.sleep(60 * minute)


from daily.position_202405 import run as run_202405
from daily.position_202409 import run as run_202409
from daily.position_202501 import run as run_202501
# from daily.position_aw_beta_2503 import run as run_aw_beta_2503
from position_aw_beta_2503 import run as run_aw_beta_2503


if __name__ == '__main__':
    # df_out.columns = ['symbol', 'market_value', 'price', 'target_position']
    today_05, df_out_05 = run_202405()
    today_09, df_out_09 = run_202409()
    today_2501, df_out_2501 = run_202501()
    today_aw_beta_2503, df_out_aw_beta_2503 = run_aw_beta_2503()

    assert today_05 == today_09
    assert today_05 == today_2501
    assert today_05 == today_aw_beta_2503

    # combine
    df_out_05 = df_out_05.set_index('symbol')
    df_out_05.columns = [f"{i}_05" for i in df_out_05.columns]
    df_out_09 = df_out_09.set_index('symbol')
    df_out_09.columns = [f"{i}_09" for i in df_out_09.columns]
    df_out_2501 = df_out_2501.set_index('symbol')
    df_out_2501.columns = [f"{i}_2501" for i in df_out_2501.columns]
    df_out_aw_beta_2503 = df_out_aw_beta_2503.set_index('symbol')
    df_out_aw_beta_2503.columns = [f"{i}_aw_beta_2503" for i in df_out_aw_beta_2503.columns]

    df_out = df_out_05.merge(df_out_09, left_index=True, right_index=True, how='outer')
    df_out = df_out.merge(df_out_2501, left_index=True, right_index=True, how='outer')
    df_out = df_out.merge(df_out_aw_beta_2503, left_index=True, right_index=True, how='outer')

    # ['signal', '合约乘数', 'market_value', 'price', 'target_position']
    # df_out['market_value'] = (df_out['market_value_05'] * weight_05 + df_out['market_value_09'] * weight_09).round(0)
    # df_out['target_position'] = (df_out['target_position_05'] * weight_05 + df_out['target_position_09'] * weight_09).round(0)

    # df_out['price'] = np.maximum(df_out['price_05'].fillna(0), df_out['price_09'].fillna(0),
    #                              df_out['price_2501'].fillna(0), df_out['price_aw_beta_2503'].fillna(0)
    #                              )
    # df_out['合约乘数'] = np.maximum(df_out['合约乘数_05'].fillna(0), df_out['合约乘数_09'].fillna(0),
    #                                 df_out['合约乘数_2501'].fillna(0), df_out['合约乘数_aw_beta_2503'].fillna(0),
    #                                 )

    df_out['price'] = df_out[['price_05', 'price_09', 'price_2501', 'price_aw_beta_2503']].max(axis=1)
    df_out['合约乘数'] = df_out[['合约乘数_05', '合约乘数_09', '合约乘数_2501', '合约乘数_aw_beta_2503']].max(axis=1)
    
    # weight combine
    #total_money = 13000e4  # 20250506
    total_money = 9500e4  # 20250701
    cta = 0.65
    aw = 1 - cta

    weight_05 = 0.6 * cta
    weight_09 = 0.3 * cta
    weight_2501 = 0.1 * cta
    weight_aw_beta_2503 = aw

    print(f"total_money: {total_money}, weight_05: {weight_05}, weight_09: {weight_09}, "
          f"weight_2501: {weight_2501}, weight_aw_beta_2503: {weight_aw_beta_2503}")

    df_out['signal'] = (df_out['signal_05'].fillna(0) * weight_05 +
                        df_out['signal_09'].fillna(0) * weight_09 +
                        df_out['signal_2501'].fillna(0) * weight_2501 +
                        df_out['signal_aw_beta_2503'].fillna(0) * weight_aw_beta_2503
                        )

    multi = min(2.0, 1.0 / df_out['signal'].abs().sum())
    df_out['signal'] = df_out['signal'] * multi

    df_out['market_value'] = (df_out['signal'] * total_money).round(0)
    df_out['target_position'] = (df_out['market_value'] / df_out['price'] / df_out['合约乘数']).round(0)

    # 防止价格出现为0的bug
    df_out.loc[df_out['price'] <= 0, 'target_position'] = 0

    df_output = df_out.reset_index()

    position_path = r"C:\Users\Administrator\Desktop\position"
    # position_path = r"C:\Users\Administrator\Desktop\position_combine"
    save_file = f"{position_path}/position_{today}.csv"

    df_output = df_output[['symbol', 'market_value', 'price', 'target_position']]
    df_output = df_output.sort_values(by=['market_value'], ascending=False)
    df_output.to_csv(save_file, index=False)

    print("run done!")

