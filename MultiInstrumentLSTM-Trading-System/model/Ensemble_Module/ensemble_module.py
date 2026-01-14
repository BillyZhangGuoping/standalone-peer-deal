import numpy as np
import tensorflow as tf

class EnsembleModule:
    def __init__(self, models):
        """
        初始化集成模块
        
        参数:
        - models: 模型列表
        """
        self.models = models
    
    def predict(self, X, method='average'):
        """
        集成预测
        
        参数:
        - X: 输入数据，形状为 (batch_size, sequence_length, num_features)
        - method: 集成方法，可选 'average'（平均）或 'weighted'（加权）
        
        返回:
        - predictions: 集成预测结果，形状为 (batch_size, num_varieties)
        """
        # 获取所有模型的预测结果
        all_predictions = []
        for model in self.models:
            pred = model.predict(X)
            all_predictions.append(pred)
        
        # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        
        # 根据集成方法计算最终预测
        if method == 'average':
            # 简单平均
            predictions = np.mean(all_predictions, axis=0)
        elif method == 'weighted':
            # 加权平均（这里使用相等权重，实际应用中可以根据模型表现调整权重）
            weights = np.ones(len(self.models)) / len(self.models)
            predictions = np.sum(all_predictions * weights[:, np.newaxis, np.newaxis], axis=0)
        else:
            raise ValueError(f"Unsupported ensemble method: {method}")
        
        return predictions
    
    def save_ensemble(self, file_path):
        """
        保存集成模型
        
        参数:
        - file_path: 保存路径
        """
        # 保存每个模型
        for i, model in enumerate(self.models):
            model.save_model(f"{file_path}_model_{i}")
    
    @classmethod
    def load_ensemble(cls, file_paths, model_class):
        """
        加载集成模型
        
        参数:
        - file_paths: 模型文件路径列表
        - model_class: 模型类
        
        返回:
        - ensemble: 集成模型实例
        """
        models = []
        for file_path in file_paths:
            model = model_class(None, None)  # 先创建空模型
            model.load_model(file_path)
            models.append(model)
        
        return cls(models)
