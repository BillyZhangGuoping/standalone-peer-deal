import pandas as pd
import numpy as np
import os
import logging
from models.boosting import BoostingModel

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LightGBMTrainer:
    """LightGBM模型训练器"""
    
    def __init__(self, params=None):
        """初始化训练器
        
        参数：
        params: 模型参数
        """
        self.params = params if params else {}
        self.model = None
    
    def train(self, data):
        """训练模型
        
        参数：
        data: 训练数据
        
        返回：
        model: 训练好的模型
        """
        logger.info("开始训练LightGBM模型...")
        
        # 创建模型实例
        self.model = BoostingModel(model_type='lightgbm', params=self.params)
        
        # 准备模型数据
        X, y = self.model.prepare_model_data(data)
        
        if X is None or y is None:
            logger.error("数据准备失败，无法训练模型")
            return None
        
        # 训练模型
        self.model.train(X, y)
        
        logger.info("LightGBM模型训练完成")
        return self.model
    
    def predict(self, data):
        """预测结果
        
        参数：
        data: 预测数据
        
        返回：
        prediction: 预测结果
        """
        if self.model is None:
            logger.error("模型尚未训练，无法预测")
            return None
        
        # 准备模型数据
        X, _ = self.model.prepare_model_data(data)
        
        if X is None:
            logger.error("数据准备失败，无法预测")
            return None
        
        # 预测
        return self.model.predict(X)
    
    def save_model(self, file_path):
        """保存模型
        
        参数：
        file_path: 模型保存路径
        """
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"模型已保存到 {file_path}")
    
    def load_model(self, file_path):
        """加载模型
        
        参数：
        file_path: 模型加载路径
        
        返回：
        model: 加载的模型
        """
        import pickle
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"模型已从 {file_path} 加载")
        return self.model
    
    def evaluate(self, data):
        """评估模型性能
        
        参数：
        data: 评估数据
        
        返回：
        metrics: 评估指标
        """
        if self.model is None:
            logger.error("模型尚未训练，无法评估")
            return None
        
        # 准备模型数据
        X, y = self.model.prepare_model_data(data)
        
        if X is None or y is None:
            logger.error("数据准备失败，无法评估")
            return None
        
        # 评估模型
        metrics = self.model.evaluate(X, y)
        logger.info(f"模型评估结果: {metrics}")
        return metrics
    
    def get_feature_importance(self):
        """获取特征重要性
        
        返回：
        feature_importance: 特征重要性字典
        """
        if self.model is None:
            logger.error("模型尚未训练，无法获取特征重要性")
            return None
        
        return self.model.get_feature_importance()
