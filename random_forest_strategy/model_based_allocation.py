import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

class ModelBasedAllocation:
    def __init__(self, re_train_interval=80, feature_window=360):
        self.re_train_interval = re_train_interval
        self.feature_window = feature_window
        self.model = None
        self.scaler = StandardScaler()
        self.last_train_date = None
        self.model_path = os.path.join(os.path.dirname(__file__), 'models', 'allocation_lgb_model.pkl')
        self.scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'allocation_scaler.pkl')
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
    def calculate_past_sharpe(self, returns, window=60):
        """计算过往夏普比率"""
        if len(returns) < window:
            return 0.0
        window_returns = returns[-window:]
        return np.mean(window_returns) / (np.std(window_returns) + 1e-8) * np.sqrt(252)
    
    def calculate_trend_strength(self, prices, window=20):
        """计算趋势强度"""
        if len(prices) < window:
            return 0.0
        returns = np.diff(prices) / prices[:-1]
        return np.mean(returns[-window:]) / (np.std(returns[-window:]) + 1e-8)
    
    def calculate_atr(self, high, low, close, window=14):
        """计算ATR指标"""
        if len(high) < window:
            return 0.0
        tr1 = high[-window:] - low[-window:]
        tr2 = abs(high[-window:] - close[-window-1:-1])
        tr3 = abs(low[-window:] - close[-window-1:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return np.mean(tr)
    
    def calculate_volatility(self, returns, window=20):
        """计算波动率"""
        if len(returns) < window:
            return 0.0
        return np.std(returns[-window:]) * np.sqrt(252)
    
    def calculate_correlation(self, returns, window=60):
        """计算相关性矩阵"""
        if len(returns) < window or returns.shape[1] < 2:
            return np.identity(returns.shape[1])
        window_returns = returns[-window:]
        corr_matrix = np.corrcoef(window_returns, rowvar=False)
        # 处理nan值
        corr_matrix[np.isnan(corr_matrix)] = 0.0
        return corr_matrix
    
    def generate_features(self, instrument_data, market_data, past_allocations, past_returns, current_date):
        """生成分配模型所需的特征"""
        features = {}
        
        # 提取市场数据
        market_returns = market_data.get('returns', [])
        market_volatility = self.calculate_volatility(market_returns)
        
        # 计算全局特征
        global_features = {
            'market_volatility': market_volatility,
            'market_sharpe': self.calculate_past_sharpe(market_returns),
            'avg_correlation': np.mean(np.abs(self.calculate_correlation(market_returns if len(market_returns) > 0 else np.array([[0]])) - np.eye(len(market_returns[0]) if len(market_returns) > 0 else 1)))
        }
        
        # 为每个品种生成特征
        for instrument, data in instrument_data.items():
            prices = data['close']
            returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else [0]
            
            atr = self.calculate_atr(data['high'], data['low'], data['close'])
            volatility = self.calculate_volatility(returns)
            trend_strength = self.calculate_trend_strength(prices)
            past_sharpe = self.calculate_past_sharpe(returns)
            
            # 计算与其他品种的平均相关性
            if len(instrument_data) > 1:
                instrument_correlations = []
                for other_instrument, other_data in instrument_data.items():
                    if other_instrument != instrument:
                        other_returns = np.diff(other_data['close']) / other_data['close'][:-1] if len(other_data['close']) > 1 else [0]
                        min_len = min(len(returns), len(other_returns))
                        if min_len > 0:
                            corr = np.corrcoef(returns[-min_len:], other_returns[-min_len:])[0, 1] if min_len > 1 else 0
                            instrument_correlations.append(corr)
                avg_correlation = np.mean(np.abs(instrument_correlations)) if instrument_correlations else 0
            else:
                avg_correlation = 0
            
            # 计算过往分配权重
            past_allocation = past_allocations.get(instrument, [0])[-60:] if instrument in past_allocations else [0]
            avg_past_allocation = np.mean(past_allocation)
            
            # 计算过往该品种的表现
            instrument_returns = past_returns.get(instrument, [0])[-60:] if instrument in past_returns else [0]
            avg_instrument_return = np.mean(instrument_returns)
            
            features[instrument] = {
                'atr': atr,
                'volatility': volatility,
                'trend_strength': trend_strength,
                'past_sharpe': past_sharpe,
                'avg_correlation': avg_correlation,
                'avg_past_allocation': avg_past_allocation,
                'avg_instrument_return': avg_instrument_return,
                **global_features
            }
        
        return features
    
    def prepare_training_data(self, historical_features, historical_targets):
        """准备训练数据"""
        X = []
        y = []
        
        for date, features in historical_features.items():
            if date in historical_targets:
                targets = historical_targets[date]
                for instrument, feat in features.items():
                    if instrument in targets:
                        X.append(list(feat.values()))
                        y.append(targets[instrument])
        
        if not X:
            return None, None
        
        return np.array(X), np.array(y)
    
    def train_model(self, X, y):
        """训练LightGBM模型"""
        if X is None or y is None or len(X) == 0:
            return
        
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 训练模型
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 使用正确的LightGBM API参数
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        
        # 保存模型
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
    
    def predict_allocation(self, features):
        """使用模型预测分配权重"""
        if self.model is None:
            # 如果模型未训练，返回等权重分配
            num_instruments = len(features)
            return {inst: 1/num_instruments for inst in features}
        
        # 准备预测数据
        X_pred = []
        instruments = list(features.keys())
        
        for instrument in instruments:
            X_pred.append(list(features[instrument].values()))
        
        X_pred = np.array(X_pred)
        
        # 使用模型预测
        predictions = self.model.predict(X_pred)
        
        # 归一化预测结果为权重
        predictions = np.maximum(predictions, 0)  # 确保权重非负
        total = np.sum(predictions)
        
        if total == 0:
            return {inst: 1/len(instruments) for inst in instruments}
        
        weights = predictions / total
        
        return dict(zip(instruments, weights))
    
    def should_retrain(self, current_date):
        """判断是否需要重新训练模型"""
        if self.last_train_date is None:
            return True
        
        days_since_last_train = (current_date - self.last_train_date).days
        return days_since_last_train >= self.re_train_interval
    
    def allocate(self, instrument_data, market_data, past_allocations, past_returns, current_date):
        """执行分配策略"""
        # 生成当前特征
        current_features = self.generate_features(instrument_data, market_data, past_allocations, past_returns, current_date)
        
        # 检查是否需要重新训练模型
        if self.should_retrain(current_date):
            # 准备训练数据（这里需要历史特征和目标值，实际应用中需要从历史数据中获取）
            # 暂时使用简单的历史数据模拟，实际应用中需要完善
            historical_features = {current_date - timedelta(days=i): current_features for i in range(self.feature_window)}
            historical_targets = {current_date - timedelta(days=i): {inst: 1/len(current_features) for inst in current_features} for i in range(self.feature_window)}
            
            X, y = self.prepare_training_data(historical_features, historical_targets)
            self.train_model(X, y)
            self.last_train_date = current_date
        
        # 预测分配权重
        allocation_weights = self.predict_allocation(current_features)
        
        return allocation_weights

# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    instrument_data = {
        'instrument1': {
            'close': np.random.rand(200),
            'high': np.random.rand(200) + 0.01,
            'low': np.random.rand(200) - 0.01
        },
        'instrument2': {
            'close': np.random.rand(200),
            'high': np.random.rand(200) + 0.01,
            'low': np.random.rand(200) - 0.01
        }
    }
    
    market_data = {
        'returns': np.random.randn(200, 2)
    }
    
    past_allocations = {
        'instrument1': np.random.rand(60),
        'instrument2': np.random.rand(60)
    }
    
    past_returns = {
        'instrument1': np.random.randn(60),
        'instrument2': np.random.randn(60)
    }
    
    # 测试模型
    model_allocator = ModelBasedAllocation(re_train_interval=80)
    allocation = model_allocator.allocate(instrument_data, market_data, past_allocations, past_returns, datetime.now())
    print("分配结果:", allocation)
