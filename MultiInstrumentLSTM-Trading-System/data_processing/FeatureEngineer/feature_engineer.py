import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self, window_sizes=None):
        self.window_sizes = window_sizes if window_sizes else [5, 10, 20, 60]
    
    def engineer_all_varieties(self, all_data):
        """为所有品种生成特征"""
        engineered_data = {}
        
        for variety, data in all_data.items():
            engineered_data[variety] = self._engineer_single_variety(data)
        
        return engineered_data
    
    def _engineer_single_variety(self, data):
        """为单个品种生成特征"""
        df = data.copy()
        
        # 确保收盘价列存在
        if 'close' not in df.columns:
            raise ValueError("Data must contain 'close' column")
        
        # 价格变动特征
        df = self._add_price_features(df)
        
        # 技术指标特征
        df = self._add_technical_indicators(df)
        
        # 成交量相关特征
        if 'volume' in df.columns:
            df = self._add_volume_features(df)
        
        # 时间特征
        df = self._add_time_features(df)
        
        # 删除原始价格列和成交量列，只保留特征
        price_cols = ['open', 'high', 'low', 'close']
        volume_cols = ['volume'] if 'volume' in df.columns else []
        drop_cols = price_cols + volume_cols
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        # 删除包含缺失值的行
        df = df.dropna()
        
        return df
    
    def _add_price_features(self, df):
        """添加价格相关特征"""
        # 收益率
        df['return'] = df['close'].pct_change()
        
        # 对数收益率
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # 价格变动幅度
        if all(col in df.columns for col in ['high', 'low', 'open']):
            df['price_range'] = (df['high'] - df['low']) / df['open']
            df['intraday_return'] = (df['close'] - df['open']) / df['open']
        
        return df
    
    def _add_technical_indicators(self, df):
        """添加技术指标特征"""
        # 计算简单移动平均线 (SMA)
        for window in self.window_sizes:
            df[f'sma_{window}'] = talib.SMA(df['close'], timeperiod=window)
        
        # 计算指数移动平均线 (EMA)
        for window in self.window_sizes:
            df[f'ema_{window}'] = talib.EMA(df['close'], timeperiod=window)
            # 价格与均线的偏差
            df[f'price_sma_ratio_{window}'] = df['close'] / df[f'sma_{window}']
            df[f'price_ema_ratio_{window}'] = df['close'] / df[f'ema_{window}']
        
        # 计算RSI
        for window in self.window_sizes:
            df[f'rsi_{window}'] = talib.RSI(df['close'], timeperiod=window)
        
        # 计算MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        
        # 计算ATR (Average True Range)
        if all(col in df.columns for col in ['high', 'low', 'close']):
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        return df
    
    def _add_volume_features(self, df):
        """添加成交量相关特征"""
        # 成交量变化率
        df['volume_change'] = df['volume'].pct_change()
        
        # 成交量移动平均
        for window in self.window_sizes:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
        
        # 量价比
        if all(col in df.columns for col in ['volume', 'close']):
            df['volume_price_ratio'] = df['volume'] / df['close']
        
        return df
    
    def _add_time_features(self, df):
        """添加时间特征"""
        # 星期几（0-4，对应周一到周五）
        df['day_of_week'] = df.index.dayofweek
        
        # 月份（1-12）
        df['month'] = df.index.month
        
        # 是否为季度末
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        return df
    
    def standardize_features(self, engineered_data):
        """对每个品种的特征进行标准化"""
        standardized_data = {}
        scalers = {}
        
        for variety, data in engineered_data.items():
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(data)
            standardized_data[variety] = pd.DataFrame(
                scaled_features, 
                index=data.index, 
                columns=data.columns
            )
            scalers[variety] = scaler
        
        return standardized_data, scalers
