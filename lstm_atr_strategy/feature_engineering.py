import pandas as pd
import numpy as np
import sys
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从utility/calc_funcs.py导入已有的技术指标计算函数
from utility.calc_funcs import (
    calculate_ma, calculate_ema, calculate_macd, calculate_rsi, 
    calculate_bollinger_bands, calculate_atr, calculate_volume_weighted_average_price
)

# 从utility/mom.py导入动量相关函数
from utility.mom import (
    calculate_momentum, calculate_relative_strength, 
    calculate_dual_momentum, generate_momentum_signal
)

def create_lstm_features(data, lookback=30):
    """创建LSTM路径特征"""
    df = data.copy()
    
    # 时序特征
    df['return_1'] = df['close'].pct_change(1)
    df['return_5'] = df['close'].pct_change(5)
    df['return_20'] = df['close'].pct_change(20)
    df['high_low_diff'] = (df['high'] - df['low']) / df['close']
    df['open_close_diff'] = (df['close'] - df['open']) / df['close']
    
    # 技术指标
    df['rsi'] = calculate_rsi(df)
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df)
    upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(df)
    df['bb_upper'] = upper_bb / df['close']
    df['bb_middle'] = middle_bb / df['close']
    df['bb_lower'] = lower_bb / df['close']
    df['bb_width'] = (upper_bb - lower_bb) / middle_bb
    
    # 移动平均线和指数移动平均线
    for window in [5, 20, 50]:
        df[f'ma_{window}'] = calculate_ma(df, window) / df['close']
        df[f'ema_{window}'] = calculate_ema(df, window) / df['close']
        df[f'ma_std_{window}'] = df['close'].rolling(window=window).std() / df['close']
    
    # 成交量加权平均价格
    df['vwap'] = calculate_volume_weighted_average_price(df) / df['close']
    
    # 动量指标
    df = calculate_momentum(df, period=12)
    df = calculate_dual_momentum(df, price_period=12, trend_period=6)
    df = calculate_absolute_momentum(df, period=12)
    
    # 波动率特征
    df['volatility_5'] = df['return_1'].rolling(window=5).std() * np.sqrt(252)
    df['volatility_20'] = df['return_1'].rolling(window=20).std() * np.sqrt(252)
    df['realized_vol'] = df['return_1'].abs().rolling(window=20).mean() * np.sqrt(252)
    
    # ATR波动率
    df['atr_14'] = calculate_atr(df, period=14) / df['close']
    
    # 成交量特征
    df['volume_change'] = df['volume'].pct_change(1)
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    return df

def create_atr_features(data):
    """创建ATR路径特征"""
    df = data.copy()
    
    # 核心波动率特征：不同窗口的ATR值
    atr_windows = [14, 20, 50]
    for window in atr_windows:
        df[f'atr_{window}'] = calculate_atr(df, period=window)
        df[f'atr_close_ratio_{window}'] = df[f'atr_{window}'] / df['close']
    
    # 波动率结构：短期ATR与长期ATR的比值
    df['atr_ratio'] = df['atr_14'] / df['atr_50']
    
    # 极值波动：当前ATR在历史中的分位数
    def calculate_quantile(series):
        return series.rank(pct=True).iloc[-1]
    
    df['atr_14_quantile'] = df['atr_14'].rolling(window=252).apply(calculate_quantile, raw=False)
    df['atr_close_ratio_quantile'] = df['atr_close_ratio_14'].rolling(window=252).apply(calculate_quantile, raw=False)
    
    # 添加ATR的移动平均线
    df['atr_ma_14'] = df['atr_14'].rolling(window=14).mean()
    df['atr_ema_14'] = df['atr_14'].ewm(span=14, adjust=False).mean()
    
    return df

def create_market_state_features(data, n_clusters=3):
    """创建市场状态特征"""
    df = data.copy()
    
    # 计算状态特征
    state_features = pd.DataFrame()
    state_features['return_1'] = df['return_1']
    state_features['volatility_20'] = df['volatility_20']
    state_features['rsi'] = df['rsi']
    state_features['bb_width'] = df['bb_width']
    
    # 填充缺失值
    state_features = state_features.fillna(0)
    
    # 标准化
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(state_features)
    
    # 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['market_state'] = kmeans.fit_predict(scaled_features)
    
    # 将状态转换为独热编码
    for i in range(n_clusters):
        df[f'state_{i}'] = (df['market_state'] == i).astype(int)
    
    return df

def prepare_lstm_input_data(data, lookback=30, features=None):
    """准备LSTM输入数据"""
    if features is None:
        features = [
            # 时序特征
            'return_1', 'return_5', 'return_20', 'high_low_diff', 'open_close_diff',
            
            # 技术指标
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            
            # 移动平均线和指数移动平均线
            'ma_5', 'ma_20', 'ma_50', 'ema_5', 'ema_20', 'ema_50',
            'ma_std_5', 'ma_std_20', 'ma_std_50',
            
            # 动量指标
            'momentum', 'price_momentum', 'trend_momentum', 'absolute_momentum',
            
            # 波动率特征
            'volatility_5', 'volatility_20', 'realized_vol',
            'atr_14', 'atr_20', 'atr_50', 'atr_ratio',
            
            # 成交量特征
            'vwap', 'volume_change', 'volume_ratio',
            
            # 市场状态特征
            'state_0', 'state_1', 'state_2'
        ]
    
    df = data.copy()
    df = df.fillna(0)
    
    # 确保只使用存在的特征
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < len(features):
        missing_features = set(features) - set(available_features)
        print(f"警告：以下特征不存在于数据中，已跳过：{missing_features}")
    
    # 准备时序输入数据
    X = []
    for i in range(lookback, len(df)):
        X.append(df[available_features].iloc[i-lookback:i].values)
    
    return np.array(X)
