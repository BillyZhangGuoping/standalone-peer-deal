import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """基础模型类，定义统一的模型接口"""
    
    def __init__(self, model_name, params=None):
        self.model_name = model_name
        self.params = params if params else {}
        self.model = None
        self.feature_columns = None
        self.last_trained_date = None
        self.predict_count = 0
        self.training_history = []
        
    @abstractmethod
    def train(self, X_train, y_train):
        """训练模型
        
        参数：
        X_train: 训练特征数据
        y_train: 训练标签数据
        
        返回：
        model: 训练好的模型
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """预测结果
        
        参数：
        X: 预测特征数据
        
        返回：
        predictions: 预测结果
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self):
        """获取特征重要性
        
        返回：
        feature_importance: 特征重要性字典
        """
        pass
    
    def evaluate(self, X_test, y_test):
        """评估模型性能
        
        参数：
        X_test: 测试特征数据
        y_test: 测试标签数据
        
        返回：
        metrics: 评估指标字典
        """
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        
        # 计算其他指标
        metrics = {
            'accuracy': accuracy,
            'precision': self._calculate_precision(y_test, predictions),
            'recall': self._calculate_recall(y_test, predictions),
            'f1_score': self._calculate_f1_score(y_test, predictions)
        }
        
        return metrics
    
    def _calculate_precision(self, y_true, y_pred):
        """计算精确率"""
        true_positive = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positive = np.sum(y_pred == 1)
        return true_positive / predicted_positive if predicted_positive != 0 else 0
    
    def _calculate_recall(self, y_true, y_pred):
        """计算召回率"""
        true_positive = np.sum((y_true == 1) & (y_pred == 1))
        actual_positive = np.sum(y_true == 1)
        return true_positive / actual_positive if actual_positive != 0 else 0
    
    def _calculate_f1_score(self, y_true, y_pred):
        """计算F1分数"""
        precision = self._calculate_precision(y_true, y_pred)
        recall = self._calculate_recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    def update_predict_count(self):
        """更新预测次数"""
        self.predict_count += 1
    
    def reset_predict_count(self):
        """重置预测次数"""
        self.predict_count = 0
    
    def should_retrain(self, retrain_interval=50):
        """判断是否需要重新训练模型
        
        参数：
        retrain_interval: 重训练间隔
        
        返回：
        bool: 是否需要重训练
        """
        return self.predict_count >= retrain_interval
    
    def save_training_history(self, metrics, training_date):
        """保存训练历史
        
        参数：
        metrics: 训练指标
        training_date: 训练日期
        """
        self.training_history.append({
            'training_date': training_date,
            'metrics': metrics,
            'predict_count': self.predict_count
        })
        self.last_trained_date = training_date
    
    def get_training_history(self):
        """获取训练历史"""
        return self.training_history

    def prepare_model_data(self, data):
        """准备模型数据，处理标签和特征
        
        参数：
        data: 原始数据
        
        返回：
        X: 特征数据
        y: 标签数据（如果是预测数据，可能为None）
        """
        # 特征列
        feature_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'ma_5', 'ma_20', 'ma_60', 
                          'macd', 'macd_signal', 'macd_histogram', 
                          'rsi', 'bb_upper', 'bb_middle', 'bb_lower', 
                          'vwap', 'momentum',
                          'signal_ma', 'signal_macd', 'signal_rsi', 'signal_bb', 'signal_combined']
        
        # 检查是否为预测数据（只有一行数据，通常是最新数据）
        is_prediction_data = len(data) == 1
        
        # 移除NaN值，只检查特征列（标签列可能在预测时不存在）
        model_data = data.dropna(subset=feature_columns)
        
        if len(model_data) == 0:
            return None, None
        
        X = model_data[feature_columns]
        
        # 只有在训练时才计算标签，预测时不计算
        if not is_prediction_data:
            # 标签：1表示上涨，-1表示下跌
            model_data['label'] = np.sign(model_data['close'].shift(-1) - model_data['close'])
            
            # 再次移除NaN值，包括标签列
            model_data = model_data.dropna(subset=['label'])
            
            if len(model_data) < 20:  # 数据量不足，无法训练模型
                return None, None
            
            X = model_data[feature_columns]
            y = model_data['label']
        else:
            y = None
        
        self.feature_columns = feature_columns
        
        return X, y
    
    def time_series_split(self, X, y, n_splits=5):
        """时间序列交叉验证分割
        
        参数：
        X: 特征数据
        y: 标签数据
        n_splits: 分割次数
        
        返回：
        splits: 分割后的训练集和测试集索引
        """
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(tscv.split(X))
        return splits
