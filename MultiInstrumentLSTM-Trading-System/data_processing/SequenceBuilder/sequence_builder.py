import pandas as pd
import numpy as np
from datetime import timedelta

class SequenceBuilder:
    def __init__(self, sequence_length=60, prediction_horizon=1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
    
    def build_sequences(self, standardized_data, all_data):
        """为所有品种构建序列数据"""
        # 1. 对齐所有品种的时间索引
        aligned_data = self._align_time_indices(standardized_data)
        
        # 2. 获取所有有效日期
        valid_dates = aligned_data.index
        
        # 3. 构造序列数据
        X, y, date_list = [], [], []
        
        for i in range(self.sequence_length, len(valid_dates) - self.prediction_horizon + 1):
            # 获取序列起始和结束日期
            start_idx = i - self.sequence_length
            end_idx = i
            
            # 构建输入序列 (sequence_length, num_features)
            sequence = aligned_data.iloc[start_idx:end_idx].values
            X.append(sequence)
            
            # 构建标签：下一天的收益率
            next_day = valid_dates[i + self.prediction_horizon - 1]
            y.append(self._get_next_day_returns(all_data, next_day))
            
            # 记录日期
            date_list.append(valid_dates[i])
        
        # 转换为numpy数组
        X = np.array(X)
        y = np.array(y)
        
        return X, y, date_list
    
    def _align_time_indices(self, standardized_data):
        """对齐所有品种的时间索引"""
        # 获取所有品种的时间索引交集
        common_indices = None
        for variety, data in standardized_data.items():
            if common_indices is None:
                common_indices = set(data.index)
            else:
                common_indices = common_indices.intersection(set(data.index))
        
        # 按日期排序
        common_indices = sorted(common_indices)
        
        # 合并所有品种的特征，按品种顺序排列
        aligned_features = []
        variety_order = []
        
        for variety, data in standardized_data.items():
            # 筛选共同日期的数据
            variety_data = data.loc[common_indices]
            aligned_features.append(variety_data)
            variety_order.append(variety)
        
        # 水平拼接所有品种的特征
        aligned_data = pd.concat(aligned_features, axis=1)
        
        # 保存品种顺序，用于后续解释模型输出
        self.variety_order = variety_order
        
        return aligned_data
    
    def _get_next_day_returns(self, all_data, next_day):
        """获取下一天的收益率"""
        returns = []
        
        for variety in self.variety_order:
            if variety in all_data and next_day in all_data[variety].index:
                # 计算收益率：(next_close - prev_close) / prev_close
                prev_day = next_day - timedelta(days=1)
                if prev_day in all_data[variety].index:
                    prev_close = all_data[variety].loc[prev_day]['close']
                    next_close = all_data[variety].loc[next_day]['close']
                    returns.append((next_close - prev_close) / prev_close)
                else:
                    returns.append(0.0)
            else:
                returns.append(0.0)
        
        return np.array(returns)
    
    def split_train_val_test(self, X, y, date_list, train_ratio=0.7, val_ratio=0.15):
        """划分训练集、验证集和测试集"""
        # 计算划分点
        total_samples = len(X)
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)
        
        # 划分数据集
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        # 划分日期列表
        dates_train = date_list[:train_end]
        dates_val = date_list[train_end:val_end]
        dates_test = date_list[val_end:]
        
        return {
            'train': (X_train, y_train, dates_train),
            'val': (X_val, y_val, dates_val),
            'test': (X_test, y_test, dates_test)
        }
    
    def get_variety_order(self):
        """获取品种顺序"""
        return self.variety_order
