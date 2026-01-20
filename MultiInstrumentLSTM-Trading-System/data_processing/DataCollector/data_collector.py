import pandas as pd
import numpy as np
import os
from datetime import datetime

class DataCollector:
    def __init__(self, data_dir='../../History_Data/hot_daily_market_data', start_date='2015-01-01'):
        self.data_dir = data_dir
        self.start_date = start_date
        # 加载品种信息，包括合约乘数和保证金比率
        self.varieties_info = self._load_varieties_info()
    
    def _load_varieties_info(self):
        """加载品种信息，包括合约乘数和保证金比率"""
        # 加载根目录下的all_instruments_info.csv文件
        info_file = '../History_Data/all_instruments_info.csv'
        if os.path.exists(info_file):
            info_df = pd.read_csv(info_file)
            # 提取品种代码（从symbol列获取，如'A'、'AG'等）
            info_df['variety'] = info_df['symbol'].str.split('.').str[-1]
            # 提取交易所信息
            return info_df.set_index('variety')
        else:
            # 如果没有品种信息文件，返回一个空的DataFrame
            return pd.DataFrame()
    
    def collect_all_varieties(self):
        """收集所有品种的日线数据"""
        all_data = {}
        
        # 遍历History_Data目录下的所有CSV文件
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv') and filename != 'varieties_info.csv':
                variety = filename.split('.')[0]
                data = self._collect_single_variety(variety)
                if data is not None:
                    all_data[variety] = data
        
        return all_data
    
    def _collect_single_variety(self, variety):
        """收集单个品种的日线数据"""
        file_path = os.path.join(self.data_dir, f'{variety}.csv')
        
        try:
            # 读取CSV文件，第一列作为索引（日期）
            data = pd.read_csv(file_path, index_col=0)
            
            # 数据清洗
            data = self._clean_data(data)
            
            # 检查数据是否满足要求：至少60个交易日
            if len(data) < 60:
                return None
            
            # 确保交易量列存在
            if 'volume' not in data.columns:
                return None
            
            return data
        except Exception as e:
            print(f"Error collecting data for {variety}: {e}")
            return None
    
    def _clean_data(self, data):
        """数据清洗"""
        # 将索引转换为datetime
        data.index = pd.to_datetime(data.index)
        
        # 删除重复行
        data = data.drop_duplicates()
        
        # 删除包含缺失值的行
        data = data.dropna()
        
        # 筛选指定日期之后的数据
        data = data[data.index >= pd.to_datetime(self.start_date)]
        
        # 确保数据按日期排序
        data = data.sort_index()
        
        # 重命名列，确保与系统其他部分兼容
        if 'open' in data.columns and 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
            # 保留需要的列
            return data[['open', 'high', 'low', 'close', 'volume']]
        
        return data
    
    def get_valid_varieties(self, all_data):
        """获取满足条件的有效品种列表"""
        valid_varieties = []
        
        for variety, data in all_data.items():
            # 检查是否有足够的历史数据（60个交易日）
            if len(data) >= 60:
                valid_varieties.append(variety)
        
        return valid_varieties
