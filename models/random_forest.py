from models.base_model import BaseModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

class RandomForestModel(BaseModel):
    """随机森林模型"""
    
    def __init__(self, params=None):
        default_params = {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }
        
        # 合并默认参数和用户提供的参数
        merged_params = {**default_params, **(params if params else {})}
        
        super().__init__(model_name='RandomForest', params=merged_params)
    
    def train(self, X_train, y_train):
        """训练随机森林模型
        
        参数：
        X_train: 训练特征数据
        y_train: 训练标签数据
        
        返回：
        model: 训练好的随机森林模型
        """
        # 创建并训练随机森林分类器
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_train, y_train)
        
        # 保存特征列
        self.feature_columns = X_train.columns.tolist()
        
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
        
        predictions = self.model.predict(X)
        self.predict_count += 1
        
        return predictions
    
    def get_feature_importance(self):
        """获取特征重要性
        
        返回：
        feature_importance: 特征重要性字典
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        importances = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_columns, importances))
        
        # 按重要性排序
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def evaluate(self, X_test, y_test):
        """评估模型性能，扩展基类方法，添加随机森林特有的指标"""
        base_metrics = super().evaluate(X_test, y_test)
        
        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, self.predict(X_test))
        
        # 添加随机森林特有的指标
        additional_metrics = {
            'confusion_matrix': cm.tolist(),
            'oob_score': self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None
        }
        
        return {**base_metrics, **additional_metrics}
