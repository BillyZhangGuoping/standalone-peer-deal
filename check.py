"""
check.pyd - 数据检查模块
推测功能：用于检查和验证交易数据的完整性和正确性
"""

import pandas as pd
import numpy as np

def check_data_completeness(data, required_columns):
    """检查数据是否包含所有必需的列"""
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"缺少必需的列: {missing_columns}")
    return True

def check_date_continuity(data):
    """检查日期是否连续"""
    # 假设索引是日期类型
    data_sorted = data.sort_index()
    date_diff = data_sorted.index.to_series().diff().dropna()
    max_gap = date_diff.max()
    
    if max_gap > pd.Timedelta(days=1):
        print(f"警告: 日期存在不连续，最大间隔为 {max_gap}")
    
    return max_gap <= pd.Timedelta(days=1)

def check_price_validity(data):
    """检查价格数据的有效性"""
    # 检查收盘价是否为正数
    if (data['close'] <= 0).any():
        raise ValueError("存在无效的收盘价（<= 0）")
    
    # 检查最高价是否大于等于最低价
    if (data['high'] < data['low']).any():
        raise ValueError("存在最高价小于最低价的情况")
    
    # 检查开盘价和收盘价是否在最高价和最低价之间
    if ((data['open'] > data['high']) | (data['open'] < data['low'])).any():
        raise ValueError("存在开盘价超出最高价/最低价范围的情况")
    
    if ((data['close'] > data['high']) | (data['close'] < data['low'])).any():
        raise ValueError("存在收盘价超出最高价/最低价范围的情况")
    
    return True

def check_volume_validity(data):
    """检查成交量数据的有效性"""
    if (data['volume'] < 0).any():
        raise ValueError("存在负的成交量数据")
    return True

def check_data_range(data, min_date=None, max_date=None):
    """检查数据的日期范围"""
    # 假设索引是日期类型
    data_dates = data.index
    
    if min_date:
        min_date = pd.to_datetime(min_date)
        if (data_dates < min_date).any():
            raise ValueError(f"存在早于 {min_date} 的数据")
    
    if max_date:
        max_date = pd.to_datetime(max_date)
        if (data_dates > max_date).any():
            raise ValueError(f"存在晚于 {max_date} 的数据")
    
    return True

def check_duplicate_data(data):
    """检查是否存在重复的日期数据"""
    # 假设索引是日期类型，检查索引是否有重复
    if data.index.duplicated().any():
        raise ValueError("存在重复的索引数据")
    return True

def validate_data(data, required_columns=None, min_date=None, max_date=None):
    """综合验证数据的完整性和正确性"""
    # 假设索引是日期类型
    if required_columns is None:
        required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # 执行所有检查
    check_data_completeness(data, required_columns)
    check_date_continuity(data)
    check_price_validity(data)
    check_volume_validity(data)
    check_data_range(data, min_date=min_date, max_date=max_date)
    check_duplicate_data(data)
    
    return True

# 添加缺失的et函数
def et():
    """返回一个默认的日期字符串，用于解决AttributeError: module 'check' has no attribute 'et'错误"""
    return '2024-06-28'