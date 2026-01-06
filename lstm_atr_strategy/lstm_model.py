import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class Attention(nn.Module):
    """注意力机制模块"""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
    
    def forward(self, lstm_output):
        batch_size = lstm_output.size(0)
        attention_weights = torch.tanh(self.attention(torch.cat([lstm_output, lstm_output], dim=2)))
        attention_weights = torch.matmul(attention_weights, self.v)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        context_vector = torch.sum(lstm_output * attention_weights.unsqueeze(2), dim=1)
        return context_vector

class LSTMModel(nn.Module):
    """LSTM模型类"""
    
    def __init__(self, lookback=30, feature_dim=10, hidden_units=64, dropout_rate=0.2, use_attention=False):
        """初始化LSTM模型"""
        super(LSTMModel, self).__init__()
        self.lookback = lookback
        self.feature_dim = feature_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        
        # 构建模型
        self.lstm1 = nn.LSTM(feature_dim, hidden_units, batch_first=True, dropout=dropout_rate, bidirectional=False)
        
        if use_attention:
            self.attention = Attention(hidden_units)
            self.lstm2 = nn.LSTM(hidden_units, hidden_units, batch_first=True, dropout=dropout_rate)
        else:
            self.fc_mid = nn.Linear(hidden_units, hidden_units)
        
        # 多任务输出层
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_direction = nn.Linear(hidden_units, 1)
        self.fc_volatility = nn.Linear(hidden_units, 1)
        self.fc_probability = nn.Linear(hidden_units, 2)
        
        # 激活函数
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 优化器，降低学习率防止梯度爆炸
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        
        # 标准化器，使用更鲁棒的标准化方法
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
    
    def forward(self, x):
        """前向传播"""
        # LSTM1层
        lstm_out, _ = self.lstm1(x)
        
        if self.use_attention:
            # 注意力机制
            attention_out = self.attention(lstm_out)
            # LSTM2层
            lstm_out, _ = self.lstm2(attention_out.unsqueeze(1))
            lstm_out = lstm_out[:, -1, :]
        else:
            # 取最后一个时间步的输出
            lstm_out = lstm_out[:, -1, :]
            lstm_out = self.fc_mid(lstm_out)
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # 多任务输出
        direction = self.tanh(self.fc_direction(lstm_out))
        volatility = self.relu(self.fc_volatility(lstm_out))
        probability = self.softmax(self.fc_probability(lstm_out))
        
        return direction, volatility, probability
    
    def prepare_data(self, data, features, lookahead=5):
        """准备训练数据"""
        # 提取特征
        feature_data = data[features].copy()
        
        # 数据清洗：处理缺失值和异常值
        feature_data = feature_data.fillna(0)
        
        # 去除无穷值
        feature_data = feature_data.replace([np.inf, -np.inf], 0)
        
        # 提取特征值
        X = feature_data.values
        
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
        data['label_dir'] = data['future_return'].fillna(0)
        data['label_vol'] = data['future_volatility'].fillna(0)
        data['label_prob_up'] = (data['future_return'] > 0.005).astype(int).fillna(0)
        data['label_prob_down'] = (data['future_return'] < -0.005).astype(int).fillna(0)
        
        # 数据清洗：去除含有NaN或无穷值的行
        valid_indices = []
        for i in range(self.lookback, len(X_scaled)):
            # 检查特征数据是否有效
            if not np.isnan(X_scaled[i-self.lookback:i]).any() and not np.isinf(X_scaled[i-self.lookback:i]).any():
                # 检查标签数据是否有效
                dir_val = data['label_dir'].iloc[i]
                vol_val = data['label_vol'].iloc[i]
                if not np.isnan(dir_val) and not np.isinf(dir_val) and not np.isnan(vol_val) and not np.isinf(vol_val):
                    valid_indices.append(i)
        
        # 只使用有效数据
        for i in valid_indices:
            X_seq.append(X_scaled[i-self.lookback:i])
            y_dir.append(data['label_dir'].iloc[i])
            y_vol.append(data['label_vol'].iloc[i])
            # 转换为类别标签（0或1）
            y_prob.append(0 if data['label_prob_down'].iloc[i] == 1 else 1)
        
        # 转换为PyTorch张量
        X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
        y_dir_tensor = torch.tensor(np.array(y_dir), dtype=torch.float32).unsqueeze(1)
        y_vol_tensor = torch.tensor(np.array(y_vol), dtype=torch.float32).unsqueeze(1)
        y_prob_tensor = torch.tensor(np.array(y_prob), dtype=torch.long)
        
        return X_tensor, y_dir_tensor, y_vol_tensor, y_prob_tensor
    
    def train_model(self, X, y_dir, y_vol, y_prob, epochs=50, batch_size=32, validation_split=0.2):
        """训练模型"""
        # 划分训练集和验证集
        val_size = int(len(X) * validation_split)
        X_train, X_val = X[:-val_size], X[-val_size:]
        y_dir_train, y_dir_val = y_dir[:-val_size], y_dir[-val_size:]
        y_vol_train, y_vol_val = y_vol[:-val_size], y_vol[-val_size:]
        y_prob_train, y_prob_val = y_prob[:-val_size], y_prob[-val_size:]
        
        # 训练循环
        for epoch in range(epochs):
            super().train()  # 设置模型为训练模式
            total_loss = 0
            
            # 打乱训练数据
            indices = torch.randperm(len(X_train))
            X_train = X_train[indices]
            y_dir_train = y_dir_train[indices]
            y_vol_train = y_vol_train[indices]
            y_prob_train = y_prob_train[indices]
            
            # 分批训练
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y_dir = y_dir_train[i:i+batch_size]
                batch_y_vol = y_vol_train[i:i+batch_size]
                batch_y_prob = y_prob_train[i:i+batch_size]
                
                # 前向传播
                self.optimizer.zero_grad()
                pred_dir, pred_vol, pred_prob = self(batch_X)
                
                # 计算损失
                loss_dir = self.mse_loss(pred_dir, batch_y_dir)
                loss_vol = self.mse_loss(pred_vol, batch_y_vol)
                loss_prob = self.ce_loss(pred_prob, batch_y_prob)
                
                # 加权总损失
                loss = 0.4 * loss_dir + 0.4 * loss_vol + 0.2 * loss_prob
                
                # 反向传播和优化
                loss.backward()
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # 验证
            super().eval()  # 设置模型为评估模式
            with torch.no_grad():
                val_dir, val_vol, val_prob = self(X_val)
                val_loss_dir = self.mse_loss(val_dir, y_dir_val)
                val_loss_vol = self.mse_loss(val_vol, y_vol_val)
                val_loss_prob = self.ce_loss(val_prob, y_prob_val)
                val_loss = 0.4 * val_loss_dir + 0.4 * val_loss_vol + 0.2 * val_loss_prob
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(X_train):.6f}, Val Loss: {val_loss:.6f}')
    
    def predict(self, X):
        """预测"""
        self.eval()
        with torch.no_grad():
            direction, volatility, probability = self(X)
            return direction.numpy(), volatility.numpy(), probability.numpy()
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

class LSTMModelManager:
    """模型管理器，用于管理不同品种的LSTM模型"""
    
    def __init__(self, model_params=None):
        """初始化模型管理器"""
        self.model_params = model_params if model_params else {}
        self.models = {}
        self.predict_counts = {}
        self.last_train_dates = {}
        self.PREDICT_INTERVAL = 100
        self.TRAIN_INTERVAL_TRADING_DAYS = 100  # 每100个交易日重新训练模型
    
    def _create_model(self, feature_dim):
        """创建模型实例"""
        model_params = self.model_params.copy()
        model_params['feature_dim'] = feature_dim
        return LSTMModel(**model_params)
    
    def get_model(self, symbol, feature_dim=None):
        """获取指定品种的模型"""
        if symbol not in self.models:
            if feature_dim is None:
                # 如果没有指定feature_dim，使用默认值
                self.models[symbol] = self._create_model(self.model_params.get('feature_dim', 10))
            else:
                # 使用指定的feature_dim
                self.models[symbol] = self._create_model(feature_dim)
            self.predict_counts[symbol] = 0
        return self.models[symbol]
    
    def train_model(self, symbol, data, features):
        """训练指定品种的模型"""
        # 获取模型，确保使用正确的feature_dim
        model = self.get_model(symbol, feature_dim=len(features))
        X, y_dir, y_vol, y_prob = model.prepare_data(data, features)
        
        if len(X) < 100:  # 确保有足够数据
            return None
        
        model.train_model(X, y_dir, y_vol, y_prob)
        
        # 记录最后训练的日期
        if not data.empty:
            self.last_train_dates[symbol] = data.index[-1]
        
        # 重置预测计数，确保每100个交易日重新训练
        self.predict_counts[symbol] = 0
        
        return model
    
    def predict(self, symbol, data, features):
        """预测指定品种的结果"""
        # 获取模型，确保使用正确的feature_dim
        model = self.get_model(symbol, feature_dim=len(features))
        
        # 准备预测数据
        X = data[features].values
        X_scaled = model.scaler.transform(X)
        
        # 确保有足够的历史数据
        if len(X_scaled) < model.lookback:
            return None
        
        # 取最近lookback天的数据
        X_seq = X_scaled[-model.lookback:].reshape(1, model.lookback, len(features))
        
        # 转换为PyTorch张量
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        
        # 预测
        direction, volatility, probability = model.predict(X_tensor)
        
        # 更新预测计数
        self.predict_counts[symbol] += 1
        
        return {
            'direction': direction[0][0],
            'volatility': volatility[0][0],
            'probability': probability[0]
        }
    
    def should_retrain(self, symbol, current_date=None):
        """判断是否需要重新训练模型"""
        if symbol not in self.models:
            return True
        
        # 基于预测次数的重新训练（每100个交易日）
        if self.predict_counts[symbol] >= self.TRAIN_INTERVAL_TRADING_DAYS:
            return True
        
        return False
