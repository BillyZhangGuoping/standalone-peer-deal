import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, TimeDistributed
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class LSTMModel:
    """LSTM模型类"""
    
    def __init__(self, lookback=30, feature_dim=10, hidden_units=64, dropout_rate=0.2, use_attention=False):
        """初始化LSTM模型"""
        self.lookback = lookback
        self.feature_dim = feature_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.model = self._build_model()
        self.scaler = MinMaxScaler()
    
    def _build_model(self):
        """构建LSTM模型"""
        inputs = Input(shape=(self.lookback, self.feature_dim))
        
        # LSTM层
        lstm_out = LSTM(self.hidden_units, return_sequences=self.use_attention, dropout=self.dropout_rate)(inputs)
        
        # 注意力机制
        if self.use_attention:
            attention_out = Attention()([lstm_out, lstm_out])
            attention_out = LSTM(self.hidden_units, dropout=self.dropout_rate)(attention_out)
        else:
            attention_out = lstm_out
        
        # 多任务输出
        # 方向预测
        direction_output = Dense(1, activation='tanh', name='direction')(attention_out)
        
        # 波动率预测
        volatility_output = Dense(1, activation='relu', name='volatility')(attention_out)
        
        # 概率分布输出
        prob_output = Dense(2, activation='softmax', name='probability')(attention_out)
        
        # 创建模型
        model = Model(inputs=inputs, outputs=[direction_output, volatility_output, prob_output])
        
        # 编译模型
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss={'direction': 'mse', 'volatility': 'mse', 'probability': 'categorical_crossentropy'},
                     loss_weights={'direction': 0.4, 'volatility': 0.4, 'probability': 0.2})
        
        return model
    
    def prepare_data(self, data, features, lookahead=5):
        """准备训练数据"""
        # 提取特征
        X = data[features].values
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 准备时序数据
        X_seq = []
        y_dir = []
        y_vol = []
        y_prob = []
        
        # 计算未来收益和波动率作为标签
        data['future_return'] = data['close'].pct_change(lookahead).shift(-lookahead)
        data['future_volatility'] = data['return_1'].rolling(window=lookahead).std().shift(-lookahead) * np.sqrt(252)
        
        # 生成方向标签和概率标签
        data['label_dir'] = data['future_return']
        data['label_vol'] = data['future_volatility'].fillna(0)
        data['label_prob_up'] = (data['future_return'] > 0.005).astype(int)
        data['label_prob_down'] = (data['future_return'] < -0.005).astype(int)
        data['label_prob'] = data[['label_prob_up', 'label_prob_down']].values
        
        for i in range(self.lookback, len(X_scaled)):
            X_seq.append(X_scaled[i-self.lookback:i])
            y_dir.append(data['label_dir'].iloc[i])
            y_vol.append(data['label_vol'].iloc[i])
            y_prob.append(data['label_prob'].iloc[i])
        
        return np.array(X_seq), np.array(y_dir), np.array(y_vol), np.array(y_prob)
    
    def train(self, X, y_dir, y_vol, y_prob, epochs=50, batch_size=32, validation_split=0.2):
        """训练模型"""
        self.model.fit(X, 
                      [y_dir, y_vol, y_prob],
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_split=validation_split,
                      verbose=1)
    
    def predict(self, X):
        """预测"""
        return self.model.predict(X)
    
    def save(self, filepath):
        """保存模型"""
        self.model.save(filepath)
    
    def load(self, filepath):
        """加载模型"""
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)

class LSTMModelManager:
    """模型管理器，用于管理不同品种的LSTM模型"""
    
    def __init__(self, model_params=None):
        """初始化模型管理器"""
        self.model_params = model_params if model_params else {}
        self.models = {}
        self.predict_counts = {}
        self.PREDICT_INTERVAL = 100
    
    def get_model(self, symbol):
        """获取指定品种的模型"""
        if symbol not in self.models:
            self.models[symbol] = LSTMModel(**self.model_params)
            self.predict_counts[symbol] = 0
        return self.models[symbol]
    
    def train_model(self, symbol, data, features):
        """训练指定品种的模型"""
        model = self.get_model(symbol)
        X, y_dir, y_vol, y_prob = model.prepare_data(data, features)
        
        if len(X) < 100:  # 确保有足够数据
            return None
        
        model.train(X, y_dir, y_vol, y_prob)
        return model
    
    def predict(self, symbol, data, features):
        """预测指定品种的结果"""
        model = self.get_model(symbol)
        
        # 准备预测数据
        X = data[features].values
        X_scaled = model.scaler.transform(X)
        
        # 确保有足够的历史数据
        if len(X_scaled) < model.lookback:
            return None
        
        # 取最近lookback天的数据
        X_seq = X_scaled[-model.lookback:].reshape(1, model.lookback, len(features))
        
        # 预测
        direction, volatility, probability = model.predict(X_seq)
        
        # 更新预测计数
        self.predict_counts[symbol] += 1
        
        return {
            'direction': direction[0][0],
            'volatility': volatility[0][0],
            'probability': probability[0]
        }
    
    def should_retrain(self, symbol):
        """判断是否需要重新训练模型"""
        if symbol not in self.models:
            return True
        return self.predict_counts[symbol] >= self.PREDICT_INTERVAL
