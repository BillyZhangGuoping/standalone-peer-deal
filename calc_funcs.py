"""
calc_funcs.pyd - 计算函数模块
推测功能：包含各种技术指标和金融计算函数
"""

import pandas as pd
import numpy as np

def calculate_ma(data, period):
    """计算移动平均线"""
    return data['close'].rolling(window=period).mean()

def calculate_ema(data, period):
    """计算指数移动平均线"""
    return data['close'].ewm(span=period, adjust=False).mean()

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """计算MACD指标"""
    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)
    macd = ema_fast - ema_slow
    signal = calculate_ema(pd.DataFrame({'close': macd}), signal_period)
    histogram = macd - signal
    return macd, signal, histogram

def calculate_rsi(data, period=14):
    """计算相对强弱指标"""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data, period=20, num_std=2):
    """计算布林带"""
    sma = calculate_ma(data, period)
    std = data['close'].rolling(window=period).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def calculate_atr(data, period=14):
    """计算平均真实波动幅度"""
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_volume_weighted_average_price(data, period=14):
    """计算成交量加权平均价格"""
    vwap = (data['close'] * data['volume']).rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
    return vwap