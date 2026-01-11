"""
data_process.pyd - 数据处理模块
推测功能：用于数据清洗、转换和预处理
"""

import pandas as pd
import numpy as np

def clean_data(data):
    """清洗数据，处理缺失值和异常值"""
    # 复制数据以避免修改原始数据
    cleaned_data = data.copy()
    
    # 删除完全空的行
    cleaned_data = cleaned_data.dropna(how='all')
    
    # 使用前向填充处理价格数据的缺失值
    price_columns = ['open', 'high', 'low', 'close']
    cleaned_data[price_columns] = cleaned_data[price_columns].ffill()
    
    # 成交量缺失值填充为0
    if 'volume' in cleaned_data.columns:
        cleaned_data['volume'] = cleaned_data['volume'].fillna(0)
    
    return cleaned_data

def convert_tick_to_bar(tick_data, bar_period='1min'):
    """将Tick数据转换为K线数据"""
    # 设置时间索引
    tick_data = tick_data.set_index('datetime')
    
    # 转换为不同周期的K线
    bar_data = tick_data['price'].resample(bar_period).ohlc()
    bar_data['volume'] = tick_data['volume'].resample(bar_period).sum()
    
    # 重置索引
    bar_data = bar_data.reset_index()
    
    return bar_data

def normalize_data(data, columns=None):
    """归一化数据"""
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    normalized_data = data.copy()
    for col in columns:
        if col in normalized_data.columns:
            col_min = normalized_data[col].min()
            col_max = normalized_data[col].max()
            normalized_data[col] = (normalized_data[col] - col_min) / (col_max - col_min) if (col_max - col_min) > 0 else 0
    
    return normalized_data

def standardize_data(data, columns=None):
    """标准化数据"""
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    standardized_data = data.copy()
    for col in columns:
        if col in standardized_data.columns:
            col_mean = standardized_data[col].mean()
            col_std = standardized_data[col].std()
            standardized_data[col] = (standardized_data[col] - col_mean) / col_std if col_std > 0 else 0
    
    return standardized_data

def calculate_returns(data, price_column='close'):
    """计算收益率"""
    returns_data = data.copy()
    returns_data['return'] = returns_data[price_column].pct_change()
    returns_data['log_return'] = np.log(returns_data[price_column] / returns_data[price_column].shift())
    return returns_data

def resample_data(data, new_frequency, date_column='date'):
    """重新采样数据到新的频率"""
    data_resampled = data.set_index(date_column).resample(new_frequency).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    return data_resampled

def merge_data(data_list, on='date', how='inner'):
    """合并多个数据源"""
    if not data_list:
        return pd.DataFrame()
    
    merged_data = data_list[0]
    for data in data_list[1:]:
        merged_data = pd.merge(merged_data, data, on=on, how=how)
    
    return merged_data

def filter_data_by_date(data, start_date, end_date, date_column='date'):
    """按日期范围过滤数据"""
    filtered_data = data.copy()
    filtered_data[date_column] = pd.to_datetime(filtered_data[date_column])
    filtered_data = filtered_data[(filtered_data[date_column] >= start_date) & (filtered_data[date_column] <= end_date)]
    return filtered_data