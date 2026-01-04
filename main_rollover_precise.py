import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta

# 配置参数
HOT_DATA_DIR = 'History_Data/hot_daily_market_data'
ROLLOVER_DATA_DIR = 'History_Data/main_rollover_daily_market_data'

def ensure_directory_exists(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def load_hot_contract_data():
    """加载主力合约数据"""
    hot_data = {}
    files = glob.glob(os.path.join(HOT_DATA_DIR, '*.csv'))
    
    for file in files:
        try:
            # 提取基础品种代码（文件名，不包含扩展名）
            base_symbol = os.path.basename(file).split('.')[0].upper()
            # 读取数据，设置索引为日期
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            hot_data[base_symbol] = df
            print(f"成功加载 {base_symbol} 的主力合约数据")
        except Exception as e:
            print(f"加载文件 {file} 失败: {e}")
    
    return hot_data

def detect_contract_rollover_events(hot_data, base_symbol):
    """检测特定基础品种的主力合约切换事件"""
    if base_symbol not in hot_data:
        print(f"未找到 {base_symbol} 的数据")
        return []
    
    df = hot_data[base_symbol]
    rollover_events = []
    
    # 按日期排序
    df = df.sort_index()
    
    # 记录前一个合约
    prev_contract = df['symbol'].iloc[0]
    
    for i in range(1, len(df)):
        curr_contract = df['symbol'].iloc[i]
        curr_date = df.index[i]
        
        # 检测合约切换
        if curr_contract != prev_contract:
            # 记录切换事件
            rollover_events.append({
                'date': curr_date,
                'prev_contract': prev_contract,
                'curr_contract': curr_contract,
                'prev_date': df.index[i-1]
            })
            print(f"检测到 {base_symbol} 合约切换: {prev_contract} -> {curr_contract} (日期: {curr_date})")
            prev_contract = curr_contract
    
    return rollover_events

def check_missing_contract_data(hot_data, base_symbol, rollover_events):
    """检查切换日期是否缺少旧合约数据"""
    if base_symbol not in hot_data:
        return []
    
    df = hot_data[base_symbol]
    missing_data = []
    
    for event in rollover_events:
        rollover_date = event['date']
        prev_contract = event['prev_contract']
        
        # 筛选旧合约的数据
        prev_contract_data = df[df['symbol'] == prev_contract]
        
        # 检查切换日期是否有旧合约数据
        if rollover_date not in prev_contract_data.index:
            print(f"{base_symbol} 在 {rollover_date} 缺少旧合约 {prev_contract} 的数据")
            missing_data.append({
                'date': rollover_date,
                'contract': prev_contract
            })
    
    return missing_data

def query_data_with_gm_api(contract, date):
    """使用gm.api查询指定合约和日期的数据"""
    try:
        import gm.api
        # 初始化gm.api（需要用户自行设置token）
        # 注意：在实际使用前，需要将your_token替换为真实的gm.api token
        gm.api.set_token('your_token_here')
        
        print(f"正在使用gm.api查询 {contract} 在 {date.strftime('%Y-%m-%d')} 的数据")
        
        # 转换日期格式为gm.api要求的格式
        start_time = date.strftime('%Y-%m-%d') + ' 09:00:00'
        end_time = date.strftime('%Y-%m-%d') + ' 15:00:00'
        
        # 查询日线数据
        data = gm.api.get_history_n(
            symbol=contract,
            frequency='1d',
            start_time=start_time,
            end_time=end_time,
            fields='symbol,open,high,low,close,volume,open_interest',
            adjust=gm.api.ADJUST_PREV
        )
        
        if data:
            # 转换为DataFrame
            df = pd.DataFrame(data)
            # 转换时间戳为datetime
            df['eob'] = pd.to_datetime(df['eob'])
            # 设置日期为索引
            df.set_index('eob', inplace=True)
            # 只保留需要的列
            df = df[['symbol', 'open', 'high', 'low', 'close', 'volume', 'open_interest']]
            print(f"成功查询到 {contract} 在 {date.strftime('%Y-%m-%d')} 的数据")
            return df
        else:
            print(f"未查询到 {contract} 在 {date.strftime('%Y-%m-%d')} 的数据")
            return None
    
    except ImportError:
        print("错误：未安装gm.api模块，请先安装 'pip install gm-python-sdk'")
        return None
    except Exception as e:
        print(f"gm.api查询失败: {e}")
        return None

def save_rollover_data(base_symbol, data):
    """保存或追加数据到指定文件"""
    # 确保输出目录存在
    ensure_directory_exists(ROLLOVER_DATA_DIR)
    
    # 构造保存路径
    save_path = os.path.join(ROLLOVER_DATA_DIR, f'{base_symbol}.csv')
    
    try:
        # 检查文件是否存在
        if os.path.exists(save_path):
            # 文件存在，读取现有数据
            existing_df = pd.read_csv(save_path, index_col=0, parse_dates=True)
            # 合并数据
            combined_df = pd.concat([existing_df, data])
            # 去重，按日期保留最新数据
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            # 保存数据
            combined_df.to_csv(save_path)
            print(f"数据已追加到 {save_path}")
        else:
            # 文件不存在，直接保存
            data.to_csv(save_path)
            print(f"数据已保存到 {save_path}")
    except Exception as e:
        print(f"保存数据失败: {e}")

def main():
    """主函数"""
    print("开始处理主力合约切换数据...")
    
    # 1. 加载主力合约数据
    hot_data = load_hot_contract_data()
    
    # 2. 遍历所有基础品种，检测主力合约切换
    for base_symbol in hot_data.keys():
        print(f"\n处理品种: {base_symbol}")
        
        # 3. 检测主力合约切换事件
        rollover_events = detect_contract_rollover_events(hot_data, base_symbol)
        
        if not rollover_events:
            print(f"{base_symbol} 未检测到主力合约切换")
            continue
        
        # 4. 检查切换日期是否缺少旧合约数据
        missing_data = check_missing_contract_data(hot_data, base_symbol, rollover_events)
        
        if not missing_data:
            print(f"{base_symbol} 切换日期的旧合约数据完整")
            continue
        
        # 5. 使用gm.api查询缺失的数据
        for item in missing_data:
            contract = item['contract']
            date = item['date']
            
            # 查询数据
            queried_data = query_data_with_gm_api(contract, date)
            
            if queried_data is not None and not queried_data.empty:
                # 6. 保存查询到的数据
                save_rollover_data(base_symbol, queried_data)
    
    print("\n主力合约切换数据处理完成！")

if __name__ == "__main__":
    main()
