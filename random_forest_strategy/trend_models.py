# -*- coding: utf-8 -*-
"""
趋势信号模型的具体实现：
包含各种趋势信号模型的具体实现，都继承自BaseTrendModel接口
"""
import pandas as pd
import sys
import os

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interfaces import BaseTrendModel

# 导入趋势信号融合模块
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trend_signal_fusion_strategy'))
from trend_signal_fusion import generate_daily_trend_signal

# 导入随机森林信号生成模块
from random_forest_signal_generation import ModelManager, preprocess_data, generate_signals


class TrendSignalFusionModel(BaseTrendModel):
    """趋势信号融合模型的实现
    
    封装了trend_signal_fusion.py的功能
    """
    
    def generate_signals(self, data, date, variety_list):
        """生成趋势信号融合模型的信号
        
        参数：
        data: 历史数据，格式为pd.DataFrame，索引为日期，列为品种，值为收盘价
        date: 指定日期，格式为datetime对象
        variety_list: 品种列表
        
        返回：
        signals: 趋势信号，格式为pd.DataFrame
        """
        # 使用trend_signal_fusion.py的generate_daily_trend_signal函数生成信号
        try:
            # 获取data中的列名（保持原始大小写）
            data_columns = data.columns.tolist()
            data_columns_upper = [col.upper() for col in data_columns]
            
            # 创建品种名称映射，处理大小写问题
            variety_map = {}
            available_varieties = []
            column_mapping = {}
            
            for var in variety_list:
                var_upper = var.upper()
                if var_upper in data_columns_upper:
                    # 找到对应的原始列名
                    original_col = data_columns[data_columns_upper.index(var_upper)]
                    variety_map[original_col] = var  # 保存原始品种名称
                    available_varieties.append(original_col)
                    column_mapping[original_col] = var  # 用于重命名列
            
            if not available_varieties:
                # 如果没有可用品种，返回空信号
                return pd.DataFrame({
                    '品种': variety_list,
                    '趋势方向': [0] * len(variety_list),
                    '信号强度': [0.0] * len(variety_list)
                })
            
            # 只选择可用的品种列（原始名称）
            filtered_data = data[available_varieties]
            
            # 重命名列为配置中的品种名称（保持大小写）
            filtered_data = filtered_data.rename(columns=column_mapping)
            
            # 生成信号
            signals = generate_daily_trend_signal(filtered_data)
            
            # 确保所有请求的品种都在结果中
            all_results = []
            for var in variety_list:
                # 查找该品种的信号
                var_result = signals[signals['品种'] == var]
                if not var_result.empty:
                    all_results.append(var_result.iloc[0])
                else:
                    # 如果没有信号，添加默认值
                    all_results.append({
                        '品种': var,
                        '趋势方向': 0,
                        '信号强度': 0.0
                    })
            
            # 转换为DataFrame并返回
            return pd.DataFrame(all_results)[['品种', '趋势方向', '信号强度']]
        except Exception as e:
            raise Exception(f"趋势信号融合模型生成信号失败: {e}")


class RandomForestModel(BaseTrendModel):
    """随机森林模型的实现
    
    封装了random_forest_signal_generation.py的功能
    """
    
    def __init__(self, params=None):
        """初始化随机森林模型
        
        参数：
        params: 模型参数，包含：
            - predict_interval: 预测间隔
        """
        super().__init__(params)
        self.model_manager = ModelManager()
        self.predict_interval = self.params.get('predict_interval', 80)
        self.predict_counts = {}
    
    def generate_signals(self, data, date, variety_list):
        """生成随机森林模型的信号
        
        参数：
        data: 历史数据，格式为字典，键为品种代码，值为包含完整字段（high, low, close, volume等）的DataFrame
        date: 指定日期，格式为datetime对象
        variety_list: 品种列表
        
        返回：
        signals: 趋势信号，格式为pd.DataFrame
        """
        signals_list = []
        
        for base_symbol in variety_list:
            try:
                # 初始化预测次数计数器
                if base_symbol not in self.predict_counts:
                    self.predict_counts[base_symbol] = 0
                
                # 获取该品种的历史数据
                if base_symbol not in data:
                    continue
                
                # 提取该品种的完整历史数据
                symbol_data = data[base_symbol].copy()
                
                # 预处理数据
                preprocessed_data = preprocess_data(symbol_data)
                
                # 生成信号
                rf_signal, rf_strength = generate_signals(
                    model_manager=self.model_manager,
                    preprocessed_data=preprocessed_data,
                    date=date,
                    base_symbol=base_symbol,
                    predict_counts=self.predict_counts,
                    PREDICT_INTERVAL=self.predict_interval
                )
                
                if rf_signal is not None:
                    signals_list.append({
                        '品种': base_symbol,
                        '趋势方向': rf_signal,
                        '信号强度': rf_strength
                    })
                    self.predict_counts[base_symbol] += 1
                else:
                    signals_list.append({
                        '品种': base_symbol,
                        '趋势方向': 0,
                        '信号强度': 0.0
                    })
            except Exception as e:
                print(f"生成{base_symbol}随机森林信号时出错: {e}")
                signals_list.append({
                    '品种': base_symbol,
                    '趋势方向': 0,
                    '信号强度': 0.0
                })
                continue
        
        # 转换为DataFrame
        signals_df = pd.DataFrame(signals_list)
        
        return signals_df
