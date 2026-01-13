from models.base_model import BaseModel
import numpy as np
import pandas as pd

class BoostingModel(BaseModel):
    """LightGBM/XGBoost模型
    
    支持LightGBM和XGBoost，通过model_type参数指定
    """
    
    def __init__(self, model_type='lightgbm', params=None):
        # 验证模型类型
        if model_type not in ['lightgbm', 'xgboost']:
            raise ValueError("model_type必须是'lightgbm'或'xgboost'")
        
        # 默认参数
        default_params = {
            'lightgbm': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'random_state': 42,
                'verbose': -1
            },
            'xgboost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42,
                'verbosity': 0
            }
        }
        
        # 合并默认参数和用户提供的参数
        merged_params = {**default_params[model_type], **(params if params else {})}
        
        super().__init__(model_name=f'{model_type.capitalize()}', params=merged_params)
        self.model_type = model_type
    
    def train(self, X_train, y_train):
        """训练Boosting模型
        
        参数：
        X_train: 训练特征数据
        y_train: 训练标签数据
        
        返回：
        model: 训练好的Boosting模型
        """
        # 动态导入模型库，避免不必要的依赖
        if self.model_type == 'lightgbm':
            from lightgbm import LGBMClassifier
            self.model = LGBMClassifier(**self.params)
        else:  # xgboost
            from xgboost import XGBClassifier
            self.model = XGBClassifier(**self.params)
        
        # 训练模型
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
        
        # 根据模型类型获取特征重要性
        if self.model_type == 'lightgbm':
            importances = self.model.feature_importances_
        else:  # xgboost
            importances = self.model.feature_importances_
        
        feature_importance = dict(zip(self.feature_columns, importances))
        
        # 按重要性排序
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def evaluate(self, X_test, y_test):
        """评估模型性能，扩展基类方法，添加Boosting特有的指标"""
        base_metrics = super().evaluate(X_test, y_test)
        
        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, self.predict(X_test))
        
        # 添加Boosting特有的指标
        additional_metrics = {
            'confusion_matrix': cm.tolist(),
            'feature_importance_top_5': dict(list(self.get_feature_importance().items())[:5])
        }
        
        return {**base_metrics, **additional_metrics}
    
    def train_with_cv(self, X, y, n_splits=5):
        """使用交叉验证训练模型，选择最优参数
        
        参数：
        X: 特征数据
        y: 标签数据
        n_splits: 交叉验证折数
        
        返回：
        best_model: 最优模型
        best_params: 最优参数
        cv_scores: 交叉验证分数
        """
        from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
        
        # 使用时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # 定义参数网格
        param_grid = {
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [50, 100, 200],
        }
        
        # 初始化网格搜索
        if self.model_type == 'lightgbm':
            from lightgbm import LGBMClassifier
            grid_search = GridSearchCV(LGBMClassifier(random_state=42, verbose=-1), 
                                     param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
        else:  # xgboost
            from xgboost import XGBClassifier
            grid_search = GridSearchCV(XGBClassifier(random_state=42, verbosity=0), 
                                     param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
        
        # 执行网格搜索
        grid_search.fit(X, y)
        
        # 更新模型和参数
        self.model = grid_search.best_estimator_
        self.params = grid_search.best_params_
        self.feature_columns = X.columns.tolist()
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_['mean_test_score']
