"""
    tick数据每日盘后落地
"""
import datetime
import os, sys
import pathlib
from pymongo import MongoClient
import datetime
# sys.path.append(os.path.abspath(__file__ + '/../'))
p = str(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(p)


from gm.api import *

token_id = "d333c8c3938dcaa3f57a6d4377bc08c5cb77f35d"
set_token(token_id)

# 获取各交易所的交易日历
exchange_list = [
    "SHSE",  # 上海证券交易所
    "SZSE",  # 深圳证券交易所
    "CFFEX",  # 中金所
    "SHFE",  # 上期所
    "DCE",  # 大商所
    "CZCE",  # 郑商所
    "INE",  # 上海国际能源交易中心
    "GFEX",  # 广期所
]

# 交易日期
exchange_trading_dates = {}
year = datetime.datetime.now().year
for exchange in exchange_list:
    df_trade_dates = get_trading_dates_by_year(exchange, year, year)
    exchange_trading_dates[exchange] = df_trade_dates.set_index('date')


# 获取当天所有期货合约
def get_all_contract(trade_date):
    df = get_symbols(
        sec_type1=1040,
        trade_date=trade_date,
        df=True
    )
    contract_list = []
    if len(df) <= 0:
        print(f"当天没有合约数据！, trade_date: {trade_date}")
        return contract_list

    for sec_id, ex in zip(df.sec_id, df.exchange):
        d = [i for i in sec_id if not i.isalpha()]
        if len(d) > 0:
            contract_list.append(f"{ex}.{sec_id}")

    return contract_list


# 保存的字段
col_list = [
    'symbol',
    'created_at',
    'open',
    'high',
    'low',
    'price',
    'cum_volume',
    'cum_amount',
    'last_amount',
    'last_volume',
    'cum_position',
    'trade_type',  # 交易类型 1: ‘双开’, 2: ‘双平’, 3: ‘多开’, 4: ‘空开’, 5: ‘空平’, 6: ‘多平’, 7: ‘多换’, 8: ‘空换’
    'quotes',  # 期货提供买卖一档数据; 跌停时无买方报价，涨停时无卖方报价
]


# 获取所有合约
today = datetime.datetime.now().strftime("%Y-%m-%d")
# today = '2024-07-12'
contract_list = get_all_contract(trade_date=today)
print(f"日期: {today}, 合约数量: {len(contract_list)}")

# 连接到本地 MongoDB 数据库 - Billy added
client = MongoClient('localhost', 27017)
db = client['vnpy']  
collection = db['db_parquet_data']

for k, symbol in enumerate(contract_list):
    # symbol = 'DCE.a2405'
    # if k < 467:
    #     continue

    ex, contract = symbol.split(".")
    variety = ''.join([i for i in contract if i.isalpha()])

    df_dates = exchange_trading_dates[ex]

    trade_date = df_dates.loc[today, 'trade_date']
    if trade_date == '':
        continue

    next_trade_date = df_dates.loc[today, 'next_trade_date']
    pre_trade_date = df_dates.loc[today, 'pre_trade_date']

    df_tick = history(symbol=symbol,
                      frequency='tick',
                      start_time=f'{pre_trade_date} 17:00:00',
                      end_time=f'{today} 17:00:00',
                      df=True,
                      fields=','.join(col_list)
                      )

    if len(df_tick) <= 0:
        print(f"数据长度为0, 请检查: {symbol}, {today}")
        continue

    df_tick = df_tick[col_list]
    
    
    # update到本地 MongoDB 数据库 - Billy added
    if len(df_tick[df_tick['cum_volume']>0]) <= 0:
        insrt_data = df_tick.iloc[0].to_dict()
    else:
        insrt_data = df_tick[df_tick['cum_volume']>0].iloc[0].to_dict()
    
    parts = insrt_data['symbol'].split('.')
    insrt_data['symbol'] = f"{parts[1]}.{parts[0]}"
    insrt_data.pop('quotes')
    insrt_data['datetime'] = insrt_data.pop('created_at')
    insrt_data['datetime'] = insrt_data['datetime'].replace(tzinfo=None)
    # print(insrt_data['datetime'])
    print(f"insert to data {insrt_data['symbol']}, {insrt_data['datetime']}, {insrt_data['cum_volume']}")
    collection.update_one({'symbol': insrt_data['symbol'],'datetime': insrt_data['datetime']}, {'$set': insrt_data}, upsert=True)
    # Billy added done
    
    # save_root = r"C:\Users\Administrator\Desktop\gmdata\tick"

    # path = os.path.join(save_root, variety, contract)
    # if not os.path.exists(path):
        # os.makedirs(path)

    # file = os.path.join(path, f"{contract}_{trade_date.replace('-', '')}.parquet")
    # df_tick.to_parquet(file)

    print(f"下载完成, {today}, {symbol}, k: {k} / {len(contract_list) - 1}")


print("all done!")

