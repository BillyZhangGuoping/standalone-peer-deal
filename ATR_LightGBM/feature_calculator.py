import pandas as pd
import numpy as np

def calculate_ma(data, window):
    """计算移动平均线"""
    return data['close'].rolling(window=window).mean()

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """计算MACD指标"""
    # 计算快速和慢速移动平均线
    ema_fast = data['close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow_period, adjust=False).mean()
    
    # 计算MACD线
    macd_line = ema_fast - ema_slow
    
    # 计算信号线
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # 计算柱状图
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_rsi(data, window=14):
    """计算RSI指标"""
    delta = data['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_bollinger_bands(data, window=20, num_std=2):
    """计算布林带"""
    # 计算中轨
    middle_band = data['close'].rolling(window=window).mean()
    
    # 计算标准差
    std = data['close'].rolling(window=window).std()
    
    # 计算上轨和下轨
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return upper_band, middle_band, lower_band

def calculate_atr(data, window=14):
    """计算ATR指标"""
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift(1))
    low_close = np.abs(data['low'] - data['close'].shift(1))
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    atr = true_range.rolling(window=window).mean()
    
    return atr

def calculate_volume_weighted_average_price(data, window=14):
    """计算成交量加权平均价格"""
    vwap = (data['close'] * data['volume']).rolling(window=window).sum() / data['volume'].rolling(window=window).sum()
    return vwap

def calculate_momentum(data, window=12):
    """计算动量指标"""
    return data['close'] - data['close'].shift(window)


def calculate_momentum_percentage(data, period=12):
    """计算百分比动量指标"""
    data['momentum'] = data['close'] / data['close'].shift(period) - 1
    return data


def calculate_dual_momentum(data, price_period=12, trend_period=6):
    """计算双重动量"""
    # 价格动量
    data['price_momentum'] = data['close'] / data['close'].shift(price_period) - 1
    
    # 趋势动量（短期均线）
    data['trend_momentum'] = data['close'].rolling(window=trend_period).mean() / data['close'].shift(trend_period) - 1
    
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
