"""
long_short_signals.pyd - 多空信号模块
推测功能：生成各种多空交易信号
"""

import pandas as pd
import numpy as np
from calc_funcs import calculate_ma, calculate_macd, calculate_rsi, calculate_bollinger_bands

def generate_ma_crossover_signal(data, short_period=5, long_period=20):
    """生成均线交叉信号"""
    data['ma_short'] = calculate_ma(data, short_period)
    data['ma_long'] = calculate_ma(data, long_period)
    
    # 金叉：短期均线上穿长期均线
    data['golden_cross'] = (data['ma_short'] > data['ma_long']) & (data['ma_short'].shift(1) <= data['ma_long'].shift(1))
    
    # 死叉：短期均线下穿长期均线
    data['death_cross'] = (data['ma_short'] < data['ma_long']) & (data['ma_short'].shift(1) >= data['ma_long'].shift(1))
    
    # 信号：1=买入，-1=卖出，0=持有
    data['signal'] = 0
    data.loc[data['golden_cross'], 'signal'] = 1
    data.loc[data['death_cross'], 'signal'] = -1
    
    return data

def generate_macd_signal(data):
    """基于MACD生成信号"""
    macd, signal, histogram = calculate_macd(data)
    data['macd'] = macd
    data['macd_signal'] = signal
    data['macd_histogram'] = histogram
    
    # MACD金叉：MACD线上穿信号线
    data['macd_golden_cross'] = (data['macd'] > data['macd_signal']) & (data['macd'].shift(1) <= data['macd_signal'].shift(1))
    
    # MACD死叉：MACD线下穿信号线
    data['macd_death_cross'] = (data['macd'] < data['macd_signal']) & (data['macd'].shift(1) >= data['macd_signal'].shift(1))
    
    # 信号：1=买入，-1=卖出，0=持有
    data['signal'] = 0
    data.loc[data['macd_golden_cross'], 'signal'] = 1
    data.loc[data['macd_death_cross'], 'signal'] = -1
    
    return data

def generate_rsi_signal(data, overbought=70, oversold=30):
    """基于RSI生成信号"""
    data['rsi'] = calculate_rsi(data)
    
    # RSI超卖：买入信号
    data['rsi_oversold'] = (data['rsi'] <= oversold) & (data['rsi'].shift(1) > oversold)
    
    # RSI超买：卖出信号
    data['rsi_overbought'] = (data['rsi'] >= overbought) & (data['rsi'].shift(1) < overbought)
    
    # 信号：1=买入，-1=卖出，0=持有
    data['signal'] = 0
    data.loc[data['rsi_oversold'], 'signal'] = 1
    data.loc[data['rsi_overbought'], 'signal'] = -1
    
    return data

def generate_bollinger_bands_signal(data):
    """基于布林带生成信号"""
    upper_band, middle_band, lower_band = calculate_bollinger_bands(data)
    data['upper_band'] = upper_band
    data['middle_band'] = middle_band
    data['lower_band'] = lower_band
    
    # 价格突破下轨：买入信号
    data['bb_buy'] = (data['close'] > data['lower_band']) & (data['close'].shift(1) <= data['lower_band'].shift(1))
    
    # 价格突破上轨：卖出信号
    data['bb_sell'] = (data['close'] < data['upper_band']) & (data['close'].shift(1) >= data['upper_band'].shift(1))
    
    # 信号：1=买入，-1=卖出，0=持有
    data['signal'] = 0
    data.loc[data['bb_buy'], 'signal'] = 1
    data.loc[data['bb_sell'], 'signal'] = -1
    
    return data

def generate_combined_signal(data, weights=None):
    """生成组合信号"""
    if weights is None:
        weights = {
            'ma': 0.25,
            'macd': 0.25,
            'rsi': 0.25,
            'bb': 0.25
        }
    
    # 生成各个信号
    data_ma = generate_ma_crossover_signal(data.copy())
    data_macd = generate_macd_signal(data.copy())
    data_rsi = generate_rsi_signal(data.copy())
    data_bb = generate_bollinger_bands_signal(data.copy())
    
    # 计算加权组合信号
    data['signal_ma'] = data_ma['signal']
    data['signal_macd'] = data_macd['signal']
    data['signal_rsi'] = data_rsi['signal']
    data['signal_bb'] = data_bb['signal']
    
    data['combined_signal'] = (
        data['signal_ma'] * weights['ma'] +
        data['signal_macd'] * weights['macd'] +
        data['signal_rsi'] * weights['rsi'] +
        data['signal_bb'] * weights['bb']
    )
    
    # 将组合信号转换为离散信号
    data['signal'] = 0
    data.loc[data['combined_signal'] > 0.5, 'signal'] = 1
    data.loc[data['combined_signal'] < -0.5, 'signal'] = -1
    
    return data

def generate_volume_signal(data, volume_threshold=1.5):
    """基于成交量生成信号"""
    # 计算成交量均值
    data['volume_ma'] = data['volume'].rolling(window=20).mean()
    
    # 成交量放大：超过均值的一定倍数
    data['volume_surge'] = data['volume'] > data['volume_ma'] * volume_threshold
    
    # 结合价格上涨生成买入信号
    data['price_rise'] = data['close'] > data['open']
    data['signal'] = 0
    data.loc[data['volume_surge'] & data['price_rise'], 'signal'] = 1
    
    return data