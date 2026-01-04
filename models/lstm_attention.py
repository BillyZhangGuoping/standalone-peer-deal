from models.base_model import BaseModel
import numpy as np
import pandas as pd

# 延迟导入tensorflow，仅在需要时导入

class LSTMAttentionModel(BaseModel):
    """LSTM+注意力机制模型，专注于时序建模"""
    
    def __init__(self, params=None):
        default_params = {
            'lstm_units': 64,
            'attention_units': 32,
            'dropout': 0.2,
            'batch_size': 32,
            'epochs': 50,
            'validation_split': 0.2,
            'random_state': 42
        }
        
        # 合并默认参数和用户提供的参数
        merged_params = {**default_params, **(params if params else {})}
        
        super().__init__(model_name='LSTM+Attention', params=merged_params)
        self.sequence_length = 20  # 时序窗口长度
        self.input_shape = None
    
    def create_model(self, input_shape):
        """创建LSTM+注意力机制模型结构
        
        参数：
        input_shape: 输入数据形状
        
        返回：
        model: 编译好的LSTM+注意力机制模型
        """
        # 动态导入tensorflow，仅在需要时导入
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import LSTM, Dense, Input, Attention, concatenate, Flatten
        
        # 输入层
        inputs = Input(shape=input_shape)
        
        # LSTM层，返回序列
        lstm_output = LSTM(self.params['lstm_units'], return_sequences=True, dropout=self.params['dropout'])(inputs)
        
        # 注意力层
        attention_output = Attention()([lstm_output, lstm_output])
        
        # 合并LSTM输出和注意力输出
        combined = concatenate([lstm_output, attention_output])
        
        # 展平层
        flattened = Flatten()(combined)
        
        # 全连接层
        dense_output = Dense(self.params['attention_units'], activation='relu')(flattened)
        
        # 输出层，3个类别：-1（下跌）, 0（不变）, 1（上涨）
        outputs = Dense(3, activation='softmax')(dense_output)
        
        # 创建模型
        model = Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def prepare_sequences(self, X, y):
        """准备时序序列数据
        
        参数：
        X: 特征数据
        y: 标签数据
        
        返回：
        X_sequences: 时序序列特征
        y_sequences: 时序序列标签
        """
        # 转换为numpy数组
        X_np = X.values
        y_np = y.values
        
        # 转换标签为0, 1, 2类别（原标签为-1, 0, 1）
        y_np = y_np + 1  # 转换为0, 1, 2
        
        X_sequences = []
        y_sequences = []
        
        # 创建序列
        for i in range(len(X_np) - self.sequence_length):
            X_sequences.append(X_np[i:i+self.sequence_length])
            y_sequences.append(y_np[i+self.sequence_length])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train(self, X_train, y_train):
        """训练LSTM+注意力机制模型
        
        参数：
        X_train: 训练特征数据
        y_train: 训练标签数据
        
        返回：
        model: 训练好的LSTM+注意力机制模型
        """
        # 准备时序序列数据
        X_sequences, y_sequences = self.prepare_sequences(X_train, y_train)
        
        # 保存特征列
        self.feature_columns = X_train.columns.tolist()
        
        # 获取输入形状
        self.input_shape = X_sequences.shape[1:]
        
        # 创建并编译模型
        self.model = self.create_model(self.input_shape)
        
        # 动态导入EarlyStopping
        from tensorflow.keras.callbacks import EarlyStopping
        
        # 早期停止回调
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # 训练模型
        self.model.fit(
            X_sequences,
            y_sequences,
            batch_size=self.params['batch_size'],
            epochs=self.params['epochs'],
            validation_split=self.params['validation_split'],
            callbacks=[early_stopping],
            verbose=0
        )
        
        return self.model
    
    def predict(self, X):
        """预测结果
        
        参数：
        X: 预测特征数据
        
        返回：
        predictions: 预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 确保X是DataFrame，保留特征名
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_columns)
        
        # 准备时序序列数据
        X_np = X.values
        
        # 创建单个序列（使用最新的sequence_length条数据）
        X_sequence = X_np[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        # 预测概率
        probabilities = self.model.predict(X_sequence, verbose=0)
        
        # 获取预测类别
        predicted_class = np.argmax(probabilities, axis=1)[0]
        
        # 转换回原标签：0→-1, 1→0, 2→1
        prediction = predicted_class - 1
        
        self.predict_count += 1
        
        return np.array([prediction])
    
    def get_feature_importance(self):
        """获取特征重要性
        
        对于LSTM+注意力机制模型，特征重要性的计算比较复杂，这里简化处理
        
        返回：
        feature_importance: 特征重要性字典
        """
        # 对于LSTM模型，特征重要性的计算比较复杂
        # 这里返回一个简化版本，基于输入特征的方差
        if self.feature_columns is None:
            raise ValueError("模型尚未训练")
        
        # 简化处理，返回均匀的特征重要性
        feature_importance = {feature: 1/len(self.feature_columns) for feature in self.feature_columns}
        
        return feature_importance
    
    def evaluate(self, X_test, y_test):
        """评估模型性能
        
        参数：
        X_test: 测试特征数据
        y_test: 测试标签数据
        
        返回：
        metrics: 评估指标字典
        """
        # 准备时序序列数据
        X_sequences, y_sequences = self.prepare_sequences(X_test, y_test)
        
        # 获取模型预测
        probabilities = self.model.predict(X_sequences, verbose=0)
        predicted_classes = np.argmax(probabilities, axis=1)
        
        # 转换回原标签
        predictions = predicted_classes - 1
        y_true = y_test.values[-len(predictions):]  # 确保标签长度匹配
        
        # 计算准确率
        accuracy = np.mean(predictions == y_true)
        
        # 计算其他指标
        metrics = {
            'accuracy': accuracy,
            'precision': self._calculate_precision(y_true, predictions),
            'recall': self._calculate_recall(y_true, predictions),
            'f1_score': self._calculate_f1_score(y_true, predictions)
        }
        
        self.predict_count += len(predictions)
        
        return metrics
