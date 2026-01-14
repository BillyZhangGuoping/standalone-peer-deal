import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

class LSTMModel:
    def __init__(self, input_shape, num_varieties, hidden_units=64, num_layers=2, dropout_rate=0.2):
        """
        初始化LSTM模型
        
        参数:
        - input_shape: 输入数据形状 (sequence_length, num_features)
        - num_varieties: 品种数量
        - hidden_units: LSTM隐藏单元数量
        - num_layers: LSTM层数
        - dropout_rate: Dropout率
        """
        self.input_shape = input_shape
        self.num_varieties = num_varieties
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
    
    def _build_model(self):
        """构建LSTM模型"""
        model = Sequential()
        
        # 第一层LSTM
        model.add(LSTM(
            units=self.hidden_units,
            return_sequences=True,
            input_shape=self.input_shape,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal'
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # 中间LSTM层
        for i in range(1, self.num_layers - 1):
            model.add(LSTM(
                units=self.hidden_units,
                return_sequences=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal'
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # 最后一层LSTM
        model.add(LSTM(
            units=self.hidden_units,
            return_sequences=False,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal'
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # 输出层，使用tanh激活函数将权重限制在[-1, 1]之间
        model.add(Dense(
            units=self.num_varieties,
            activation='tanh',
            kernel_initializer='glorot_uniform'
        ))
        
        return model
    
    def compile_model(self, loss_function, optimizer):
        """编译模型"""
        self.model.compile(
            loss=loss_function,
            optimizer=optimizer,
            metrics=['mean_squared_error']
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, callbacks=None):
        """训练模型"""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=False  # 时间序列数据不打乱
        )
        return history
    
    def predict(self, X):
        """预测持仓权重"""
        return self.model.predict(X)
    
    def save_model(self, file_path):
        """保存模型"""
        # 使用.keras扩展名保存模型
        model_file = os.path.join(file_path, 'model.keras')
        self.model.save(model_file)
    
    def load_model(self, file_path):
        """加载模型"""
        # 从目录加载模型文件
        model_file = os.path.join(file_path, 'model.keras')
        self.model = tf.keras.models.load_model(model_file)
    
    def get_model_summary(self):
        """获取模型结构摘要"""
        self.model.summary()
