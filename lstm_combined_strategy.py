import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LSTMTrendPredictor:
    def __init__(self, sequence_length=60, re_train_interval=80, feature_window=360):
        self.sequence_length = sequence_length
        self.re_train_interval = re_train_interval
        self.feature_window = feature_window
        self.model = None
        self.scaler = MinMaxScaler()
        self.last_train_date = None
        self.model_path = os.path.join(os.path.dirname(__file__), 'lstm_models', 'trend_lstm_model.h5')
        self.scaler_path = os.path.join(os.path.dirname(__file__), 'lstm_models', 'trend_scaler.pkl')
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    
    def prepare_model_data(self, data):
        """准备模型数据"""
        # 特征列定义
        feature_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'ma_5', 'ma_20', 'ma_60', 'vwap', 'momentum',
                          'return_5', 'return_10', 'return_20', 'volatility_10', 'volatility_20']
        
        # 确保所有特征列存在
        missing_cols = [col for col in feature_columns if col not in data.columns]
        if missing_cols:
            return None, None
        
        # 提取特征数据
        X = data[feature_columns].values
        
        # 计算标签：1表示上涨，-1表示下跌，预测1个交易日后趋势
        y = np.sign(data['close'].shift(-1) - data['close'])
        
        # 移除NaN值
        valid_indices = ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]
        
        # 打印标签分布
        logger.info(f"原始标签分布：上涨: {(y == 1).sum()}, 下跌: {(y == -1).sum()}, 持平: {(y == 0).sum()}")
        
        # 将标签转换为0和1（用于分类）
        y = (y + 1) / 2  # -1→0，1→1
        
        # 打印转换后的标签分布
        logger.info(f"转换后标签分布：0: {(y == 0).sum()}, 1: {(y == 1).sum()}")
        
        # 转换y为numpy数组
        y = y.values if hasattr(y, 'values') else y
        
        return X, y
    
    def create_sequences(self, X, y):
        """创建LSTM所需的序列数据"""
        sequences = []
        labels = []
        
        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i:i+self.sequence_length])
            labels.append(y[i+self.sequence_length])
        
        return np.array(sequences), np.array(labels)
    
    def build_model(self, input_shape):
        """构建改进的LSTM模型"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy', tf.keras.metrics.AUC()])
        
        return model
    
    def train(self, data, symbol):
        """训练LSTM趋势预测模型"""
        # 准备模型数据
        X, y = self.prepare_model_data(data)
        
        if X is None or y is None or len(X) < self.sequence_length + 1:
            return None
        
        # 只使用最近feature_window天的数据
        if len(X) > self.feature_window:
            X = X[-self.feature_window:]
            y = y[-self.feature_window:]
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 创建序列数据
        X_sequences, y_sequences = self.create_sequences(X_scaled, y)
        
        if len(X_sequences) == 0:
            return None
        
        # 构建模型
        input_shape = (X_sequences.shape[1], X_sequences.shape[2])
        self.model = self.build_model(input_shape)
        
        # 确定验证集大小
        if len(X_sequences) < 2:
            # 如果样本太少，不使用验证集
            validation_split = 0.0
            callbacks = []
        else:
            # 正常使用验证集
            validation_split = 0.2
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(self.model_path.replace('trend_lstm_model.h5', f'trend_lstm_model_{symbol}.h5'), 
                              monitor='val_loss' if validation_split > 0 else 'loss', save_best_only=True)
            ]
        
        # 训练模型
        self.model.fit(
            X_sequences,
            y_sequences,
            epochs=50,
            batch_size=32,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # 保存缩放器
        joblib.dump(self.scaler, self.scaler_path.replace('trend_scaler.pkl', f'trend_scaler_{symbol}.pkl'))
        
        return self.model
    
    def predict(self, data, symbol):
        """预测趋势"""
        if self.model is None:
            # 尝试加载预训练模型
            try:
                self.model = load_model(self.model_path.replace('trend_lstm_model.h5', f'trend_lstm_model_{symbol}.h5'))
                self.scaler = joblib.load(self.scaler_path.replace('trend_scaler.pkl', f'trend_scaler_{symbol}.pkl'))
            except:
                # 如果加载失败，尝试训练模型
                logger.info(f"模型不存在，正在训练品种 {symbol} 的LSTM趋势预测模型...")
                self.train(data, symbol)
                # 如果训练后模型仍然不存在，返回默认值
                if self.model is None:
                    return 1.0  # 默认返回1，避免总是-1
        
        # 准备模型数据
        X, _ = self.prepare_model_data(data)
        
        if X is None or len(X) < self.sequence_length:
            return 1.0  # 默认返回1，避免总是-1
        
        # 使用最近的sequence_length天数据进行预测
        X_latest = X[-self.sequence_length:]
        
        # 数据标准化
        X_scaled = self.scaler.transform(X_latest)
        
        # 转换为LSTM输入格式
        X_sequence = X_scaled.reshape(1, self.sequence_length, X_scaled.shape[1])
        
        # 预测
        prediction = self.model.predict(X_sequence, verbose=0)[0][0]
        
        # 添加日志信息
        logger.info(f"品种 {symbol} 的预测概率: {prediction}")
        
        # 调整阈值，避免模型偏向某一类
        threshold = 0.5
        
        # 转换为-1或1
        signal = 1.0 if prediction > threshold else -1.0
        logger.info(f"品种 {symbol} 的生成信号: {signal}")
        
        return signal

class LSTMAllocationModel:
    def __init__(self, re_train_interval=80, feature_window=360, sequence_length=30):
        self.re_train_interval = re_train_interval
        self.feature_window = feature_window
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.last_train_date = None
        self.model_path = os.path.join(os.path.dirname(__file__), 'lstm_models', 'allocation_lstm_model.h5')
        self.scaler_path = os.path.join(os.path.dirname(__file__), 'lstm_models', 'allocation_scaler.pkl')
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    
    def calculate_atr(self, high, low, close, window=14):
        """计算ATR指标"""
        if len(high) < window:
            return 0.0
        # 转换为numpy数组
        high = np.array(high)
        low = np.array(low)
        close = np.array(close)
        
        tr1 = high[-window:] - low[-window:]
        tr2 = abs(high[-window:] - close[-window-1:-1])
        tr3 = abs(low[-window:] - close[-window-1:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return np.mean(tr)
    
    def calculate_volatility(self, returns, window=20):
        """计算波动率"""
        if len(returns) < window:
            return 0.0
        # 转换为numpy数组
        returns = np.array(returns)
        return np.std(returns[-window:]) * np.sqrt(252)
    
    def calculate_trend_strength(self, prices, window=20):
        """计算趋势强度"""
        if len(prices) < window:
            return 0.0
        # 转换为numpy数组
        prices = np.array(prices)
        returns = np.diff(prices) / prices[:-1]
        return np.mean(returns[-window:]) / (np.std(returns[-window:]) + 1e-8)
    
    def calculate_past_sharpe(self, returns, window=60):
        """计算过往夏普比率"""
        if len(returns) < window:
            return 0.0
        # 转换为numpy数组
        returns = np.array(returns)
        window_returns = returns[-window:]
        return np.mean(window_returns) / (np.std(window_returns) + 1e-8) * np.sqrt(252)
    
    def generate_features(self, instrument_data, market_data):
        """生成分配模型所需的特征"""
        features = []
        symbols = []
        
        for symbol, data in instrument_data.items():
            symbols.append(symbol)
            
            # 提取数据
            prices = data.get('price_history', [data['current_price']])
            high = data.get('high_history', [data['current_price']])
            low = data.get('low_history', [data['current_price']])
            
            # 计算特征
            atr = self.calculate_atr(high, low, prices)
            volatility = self.calculate_volatility(np.diff(prices) / prices[:-1] if len(prices) > 1 else [0])
            trend_strength = self.calculate_trend_strength(prices)
            
            # 添加到特征列表
            features.append([atr, volatility, trend_strength])
        
        return np.array(features), symbols
    
    def build_model(self, input_shape):
        """构建LSTM分配模型"""
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(16),
            Dropout(0.2),
            BatchNormalization(),
            Dense(8, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse')
        
        return model
    
    def train(self, historical_features, historical_targets):
        """训练LSTM分配模型"""
        # 准备训练数据
        X = []
        y = []
        
        for date, features in historical_features.items():
            if date in historical_targets:
                targets = historical_targets[date]
                X.append(features)
                y.append(list(targets.values()))
        
        if not X:
            return
        
        X = np.array(X)
        y = np.array(y)
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # 创建序列数据
        sequences = []
        labels = []
        
        for i in range(len(X_scaled) - self.sequence_length):
            sequences.append(X_scaled[i:i+self.sequence_length])
            labels.append(y[i+self.sequence_length])
        
        if not sequences:
            return
        
        X_sequences = np.array(sequences)
        y_sequences = np.array(labels)
        
        # 构建模型
        input_shape = (X_sequences.shape[1], X_sequences.shape[2])
        self.model = self.build_model(input_shape)
        
        # 定义回调函数
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True)
        ]
        
        # 训练模型
        self.model.fit(
            X_sequences,
            y_sequences,
            epochs=50,
            batch_size=16,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # 保存缩放器
        joblib.dump(self.scaler, self.scaler_path)
    
    def allocate(self, instrument_data, market_data):
        """使用LSTM模型进行资金分配"""
        # 生成当前特征
        current_features, symbols = self.generate_features(instrument_data, market_data)
        
        if len(current_features) == 0:
            return {symbol: 1/len(instrument_data) for symbol in instrument_data}
        
        if self.model is None:
            # 尝试加载预训练模型
            try:
                self.model = load_model(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
            except:
                return {symbol: 1/len(instrument_data) for symbol in instrument_data}
        
        # 数据标准化
        features_scaled = self.scaler.transform(current_features)
        
        # 转换为LSTM输入格式
        features_sequence = features_scaled.reshape(1, 1, features_scaled.shape[1])
        
        # 预测权重
        predictions = self.model.predict(features_sequence, verbose=0)[0]
        
        # 归一化预测结果为权重
        predictions = np.maximum(predictions, 0)  # 确保权重非负
        total = np.sum(predictions)
        
        if total == 0:
            return {symbol: 1/len(symbols) for symbol in symbols}
        
        weights = predictions / total
        
        return dict(zip(symbols, weights))

class LSTMCombinedStrategy:
    def __init__(self, backtest_days=100):
        self.backtest_days = backtest_days
        self.trend_predictor = LSTMTrendPredictor()
    
    def calculate_trend(self, data, symbol):
        """使用LSTM计算单个品种的趋势"""
        return self.trend_predictor.predict(data, symbol)
    
    def allocate_funds(self, instrument_data, market_data):
        """使用风险平价策略进行资金分配"""
        if not instrument_data:
            return {}
        
        # 计算每个品种的波动率（使用ATR作为风险度量）
        symbol_volatilities = {}
        for symbol, data in instrument_data.items():
            prices = data.get('price_history', [data['current_price']])
            high = data.get('high_history', [data['current_price']])
            low = data.get('low_history', [data['current_price']])
            
            # 计算ATR作为波动率的度量
            if len(prices) < 15:  # 需要至少15天数据来计算ATR
                volatility = 0.01  # 默认波动率
            else:
                # 转换为numpy数组
                high = np.array(high)
                low = np.array(low)
                close = np.array(prices)
                
                # 计算ATR
                tr1 = high[-14:] - low[-14:]
                tr2 = abs(high[-14:] - close[-15:-1])
                tr3 = abs(low[-14:] - close[-15:-1])
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                atr = np.mean(tr)
                
                # 使用ATR作为波动率
                volatility = atr / prices[-1]  # 归一化ATR
            
            symbol_volatilities[symbol] = volatility
            logger.info(f"品种 {symbol} 的归一化ATR波动率: {volatility}")
        
        # 计算风险平价权重（波动率的倒数）
        total_inverse_volatility = sum(1 / max(vol, 1e-8) for vol in symbol_volatilities.values())
        allocations = {}
        
        if total_inverse_volatility > 0:
            for symbol, volatility in symbol_volatilities.items():
                # 风险平价权重 = 1/volatility / sum(1/volatility)
                allocations[symbol] = (1 / max(volatility, 1e-8)) / total_inverse_volatility
        else:
            # 如果所有波动率都是0，平均分配
            equal_weight = 1.0 / len(instrument_data)
            allocations = {symbol: equal_weight for symbol in instrument_data}
        
        # 打印分配结果
        logger.info(f"风险平价分配结果: {allocations}")
        
        return allocations
    
    def run_strategy(self, historical_data, market_data):
        """运行完整策略"""
        # 获取最近backtest_days的数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.backtest_days)
        
        # 过滤数据
        filtered_data = {}
        for symbol, data in historical_data.items():
            if isinstance(data, pd.DataFrame):
                # 假设数据有date列
                data['date'] = pd.to_datetime(data['date'])
                filtered_data[symbol] = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        
        # 计算趋势
        trends = {}
        for symbol, data in filtered_data.items():
            trends[symbol] = self.calculate_trend(data, symbol)
        
        # 准备资金分配数据
        instrument_data = {}
        for symbol, data in filtered_data.items():
            instrument_data[symbol] = {
                'current_price': data['close'].iloc[-1],
                'price_history': data['close'].tolist(),
                'high_history': data['high'].tolist(),
                'low_history': data['low'].tolist()
            }
        
        # 进行资金分配
        allocations = self.allocate_funds(instrument_data, market_data)
        
        return trends, allocations

# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    strategy = LSTMCombinedStrategy(backtest_days=100)
    
    # 这里需要实际的历史数据和市场数据
    # 示例：
    # historical_data = {symbol: pd.DataFrame(...), ...}
    # market_data = {...}
    # trends, allocations = strategy.run_strategy(historical_data, market_data)
    # print("趋势预测:", trends)
    # print("资金分配:", allocations)
    
    print("LSTM组合策略已初始化，回测周期为100天")
