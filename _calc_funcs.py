"""
_calc_funcs.pyd - 内部计算函数模块
推测功能：calc_funcs.pyd的内部实现或优化版本，提供高性能的计算功能
"""

import numpy as np

def _calculate_ma(data, period):
    """内部计算移动平均线"""
    result = np.zeros_like(data, dtype=np.float64)
    
    # 计算移动平均线
    for i in range(len(data)):
        if i < period - 1:
            result[i] = np.nan
        else:
            result[i] = np.mean(data[i-period+1:i+1])
    
    return result

def _calculate_ema(data, period):
    """内部计算指数移动平均线"""
    result = np.zeros_like(data, dtype=np.float64)
    
    # 计算初始值
    result[period-1] = np.mean(data[:period])
    
    # 计算平滑因子
    alpha = 2.0 / (period + 1.0)
    
    # 计算EMA
    for i in range(period, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    
    return result

def _calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """内部计算MACD"""
    # 计算EMA
    ema_fast = _calculate_ema(data, fast_period)
    ema_slow = _calculate_ema(data, slow_period)
    
    # 计算DIF
    dif = ema_fast - ema_slow
    
    # 计算DEA
    dea = _calculate_ema(dif[slow_period-1:], signal_period)
    
    # 计算MACD柱状图
    macd_hist = (dif[slow_period+signal_period-2:] - dea) * 2
    
    return dif, dea, macd_hist

def _calculate_rsi(data, period=14):
    """内部计算RSI"""
    result = np.zeros_like(data, dtype=np.float64)
    gains = np.zeros_like(data, dtype=np.float64)
    losses = np.zeros_like(data, dtype=np.float64)
    
    # 计算涨跌
    changes = np.diff(data)
    
    # 计算初始平均涨跌
    gains[period] = np.mean(np.maximum(changes[:period], 0))
    losses[period] = np.mean(np.abs(np.minimum(changes[:period], 0)))
    
    # 计算RSI
    for i in range(period+1, len(data)):
        if changes[i-1] > 0:
            gains[i] = (gains[i-1] * (period - 1) + changes[i-1]) / period
            losses[i] = losses[i-1] * (period - 1) / period
        else:
            gains[i] = gains[i-1] * (period - 1) / period
            losses[i] = (losses[i-1] * (period - 1) + abs(changes[i-1])) / period
    
    # 计算RSI值
    rs = gains / losses
    result = 100 - (100 / (1 + rs))
    
    return result

def _calculate_bollinger_bands(data, period=20, std_dev=2):
    """内部计算布林带"""
    # 计算中轨（移动平均线）
    middle_band = _calculate_ma(data, period)
    
    # 计算标准差
    std = np.zeros_like(data, dtype=np.float64)
    for i in range(len(data)):
        if i < period - 1:
            std[i] = np.nan
        else:
            std[i] = np.std(data[i-period+1:i+1])
    
    # 计算上轨和下轨
    upper_band = middle_band + std * std_dev
    lower_band = middle_band - std * std_dev
    
    return upper_band, middle_band, lower_band

def _calculate_atr(high, low, close, period=14):
    """内部计算ATR"""
    result = np.zeros_like(close, dtype=np.float64)
    
    # 计算真实波动幅度
    tr = np.zeros_like(close, dtype=np.float64)
    tr[0] = high[0] - low[0]
    
    for i in range(1, len(close)):
        high_low = high[i] - low[i]
        high_close = abs(high[i] - close[i-1])
        low_close = abs(low[i] - close[i-1])
        tr[i] = max(high_low, high_close, low_close)
    
    # 计算ATR
    result[period-1] = np.mean(tr[:period])
    
    for i in range(period, len(close)):
        result[i] = (result[i-1] * (period - 1) + tr[i]) / period
    
    return result

def _calculate_kdj(high, low, close, period=9, slow_k_period=3, slow_d_period=3):
    """内部计算KDJ"""
    # 计算 RSV
    rsv = np.zeros_like(close, dtype=np.float64)
    
    for i in range(period-1, len(close)):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        rsv[i] = (close[i] - lowest_low) / (highest_high - lowest_low) * 100
    
    # 计算 K 值
    k = np.zeros_like(close, dtype=np.float64)
    k[period-1] = 50
    
    for i in range(period, len(close)):
        k[i] = (2/3) * k[i-1] + (1/3) * rsv[i]
    
    # 计算 D 值
    d = np.zeros_like(close, dtype=np.float64)
    d[period-1 + slow_k_period - 1] = 50
    
    for i in range(period + slow_k_period - 1, len(close)):
        d[i] = (2/3) * d[i-1] + (1/3) * k[i]
    
    # 计算 J 值
    j = 3 * d - 2 * k
    
    return k, d, j