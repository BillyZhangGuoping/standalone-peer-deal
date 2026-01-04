"""
mom.pyd - 动量策略模块
推测功能：实现各种动量交易策略
"""

import pandas as pd
import numpy as np

def calculate_momentum(data, period=12):
    """计算动量指标"""
    data['momentum'] = data['close'] / data['close'].shift(period) - 1
    return data

def generate_momentum_signal(data, period=12, lookback=3):
    """基于动量指标生成信号"""
    # 计算动量
    data = calculate_momentum(data, period)
    
    # 动量排名
    data['momentum_rank'] = data['momentum'].rolling(window=lookback).rank(ascending=False)
    
    # 信号：1=买入，-1=卖出，0=持有
    data['signal'] = 0
    data.loc[data['momentum_rank'] == 1, 'signal'] = 1
    data.loc[data['momentum_rank'] == lookback, 'signal'] = -1
    
    return data

def generate_cross_sectional_momentum_signal(data_list, period=12):
    """生成横截面动量信号"""
    # 计算每个品种的动量
    momentum_dict = {}
    for symbol, data in data_list.items():
        # 检查数据行数是否足够
        if len(data) >= period:
            momentum = data['close'].iloc[-1] / data['close'].iloc[-period] - 1
            momentum_dict[symbol] = momentum
    
    # 排序动量
    sorted_momentum = sorted(momentum_dict.items(), key=lambda x: x[1], reverse=True)
    
    # 生成信号
    signals = {}
    n = len(sorted_momentum)
    
    if n == 0:
        return signals
    
    top_percent = int(n * 0.2)  # 买入前20%
    bottom_percent = int(n * 0.2)  # 卖出后20%
    
    for i, (symbol, momentum) in enumerate(sorted_momentum):
        if i < top_percent:
            signals[symbol] = 1  # 买入
        elif i >= n - bottom_percent:
            signals[symbol] = -1  # 卖出
        else:
            signals[symbol] = 0  # 持有
    
    return signals

def calculate_relative_strength(data, benchmark, period=12):
    """计算相对强弱"""
    data['relative_strength'] = data['close'] / benchmark['close']
    data['relative_strength_change'] = data['relative_strength'] / data['relative_strength'].shift(period) - 1
    return data

def generate_relative_strength_signal(data, benchmark, period=12):
    """基于相对强弱生成信号"""
    # 计算相对强弱
    data = calculate_relative_strength(data, benchmark, period)
    
    # 相对强弱上升：买入信号
    data['signal'] = 0
    data.loc[data['relative_strength_change'] > 0, 'signal'] = 1
    data.loc[data['relative_strength_change'] < 0, 'signal'] = -1
    
    return data

def calculate_dual_momentum(data, price_period=12, trend_period=6):
    """计算双重动量"""
    # 价格动量
    data['price_momentum'] = data['close'] / data['close'].shift(price_period) - 1
    
    # 趋势动量（短期均线）
    data['trend_momentum'] = data['close'].rolling(window=trend_period).mean() / data['close'].shift(trend_period) - 1
    
    return data

def generate_dual_momentum_signal(data, price_period=12, trend_period=6):
    """基于双重动量生成信号"""
    # 计算双重动量
    data = calculate_dual_momentum(data, price_period, trend_period)
    
    # 双重动量为正：买入信号
    data['signal'] = 0
    data.loc[(data['price_momentum'] > 0) & (data['trend_momentum'] > 0), 'signal'] = 1
    data.loc[(data['price_momentum'] < 0) | (data['trend_momentum'] < 0), 'signal'] = -1
    
    return data

def calculate_absolute_momentum(data, period=12):
    """计算绝对动量"""
    data['absolute_momentum'] = data['close'] / data['close'].shift(period) - 1
    return data

def generate_absolute_momentum_signal(data, period=12):
    """基于绝对动量生成信号"""
    # 计算绝对动量
    data = calculate_absolute_momentum(data, period)
    
    # 绝对动量为正：买入信号
    data['signal'] = 0
    data.loc[data['absolute_momentum'] > 0, 'signal'] = 1
    data.loc[data['absolute_momentum'] <= 0, 'signal'] = -1
    
    return data