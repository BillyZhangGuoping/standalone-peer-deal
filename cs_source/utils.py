"""
utils.pyd - 工具函数模块
推测功能：提供各种通用的工具函数
"""

import pandas as pd
import numpy as np
import os
import datetime
from typing import Any, Dict, List, Tuple

def load_data(file_path: str, file_format: str = 'csv') -> pd.DataFrame:
    """加载数据文件"""
    if file_format.lower() == 'csv':
        return pd.read_csv(file_path)
    elif file_format.lower() == 'excel' or file_format.lower() == 'xlsx':
        return pd.read_excel(file_path)
    elif file_format.lower() == 'parquet':
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_format}")

def save_data(data: pd.DataFrame, file_path: str, file_format: str = 'csv') -> None:
    """保存数据到文件"""
    if file_format.lower() == 'csv':
        data.to_csv(file_path, index=False)
    elif file_format.lower() == 'excel' or file_format.lower() == 'xlsx':
        data.to_excel(file_path, index=False)
    elif file_format.lower() == 'parquet':
        data.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"不支持的文件格式: {file_format}")

def format_date(date_str: str, input_format: str = '%Y-%m-%d', output_format: str = '%Y%m%d') -> str:
    """格式化日期字符串"""
    date_obj = datetime.datetime.strptime(date_str, input_format)
    return date_obj.strftime(output_format)

def get_current_date(format: str = '%Y%m%d') -> str:
    """获取当前日期"""
    return datetime.datetime.now().strftime(format)

def calculate_returns(data: pd.DataFrame, price_column: str = 'close') -> pd.DataFrame:
    """计算收益率"""
    df = data.copy()
    df['return'] = df[price_column].pct_change()
    df['log_return'] = np.log(df[price_column] / df[price_column].shift())
    return df

def normalize_data(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """标准化数据"""
    df = data.copy()
    for col in columns:
        df[f'{col}_normalized'] = (df[col] - df[col].mean()) / df[col].std()
    return df

def standardize_data(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """标准化数据（归一化到0-1范围）"""
    df = data.copy()
    for col in columns:
        df[f'{col}_standardized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

def calculate_volatility(data: pd.DataFrame, returns_column: str = 'return', window: int = 20) -> pd.DataFrame:
    """计算波动率"""
    df = data.copy()
    df['volatility'] = df[returns_column].rolling(window=window).std() * np.sqrt(252)
    return df

def get_trading_days(start_date: str, end_date: str) -> List[str]:
    """获取交易日列表（简化处理，实际应该从数据库或API获取）"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    all_days = pd.date_range(start, end)
    
    # 简单地排除周末
    trading_days = [day.strftime('%Y-%m-%d') for day in all_days if day.weekday() < 5]
    
    return trading_days

def merge_dataframes(df_list: List[pd.DataFrame], on: str = 'date', how: str = 'inner') -> pd.DataFrame:
    """合并多个DataFrame"""
    if not df_list:
        return pd.DataFrame()
    
    merged_df = df_list[0]
    for df in df_list[1:]:
        merged_df = pd.merge(merged_df, df, on=on, how=how)
    
    return merged_df

def filter_data_by_date_range(data: pd.DataFrame, start_date: str, end_date: str, date_column: str = 'date') -> pd.DataFrame:
    """按日期范围筛选数据"""
    df = data.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
    return df.loc[mask]

def calculate_moving_averages(data: pd.DataFrame, price_column: str = 'close', windows: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
    """计算多个周期的移动平均线"""
    df = data.copy()
    for window in windows:
        df[f'ma_{window}'] = df[price_column].rolling(window=window).mean()
    return df

def calculate_bollinger_bands(data: pd.DataFrame, price_column: str = 'close', window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """计算布林带"""
    df = data.copy()
    df['ma'] = df[price_column].rolling(window=window).mean()
    df['std'] = df[price_column].rolling(window=window).std()
    df['upper_band'] = df['ma'] + (df['std'] * num_std)
    df['lower_band'] = df['ma'] - (df['std'] * num_std)
    return df