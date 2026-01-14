import pandas as pd
import numpy as np
import os
from datetime import datetime

class DataCollector:
    def __init__(self, data_dir='History_Data', start_date='2015-01-01'):
        self.data_dir = data_dir
        self.start_date = start_date
        self.varieties_info = self._load_varieties_info()
    
    def _load_varieties_info(self):
        """加载品种信息，包括合约乘数等"""
        # 这里假设品种信息存储在varieties_info.csv中
        info_file = os.path.join(self.data_dir, 'varieties_info.csv')
        if os.path.exists(info_file):
            return pd.read_csv(info_file, index_col='variety')
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
            data = pd.read_csv(file_path)
            
            # 数据清洗
            data = self._clean_data(data)
            
            # 检查数据是否满足要求：至少60个交易日，且每日交易量大于20000手
            if len(data) < 60:
                return None
            
            # 确保交易量列存在
            if 'volume' not in data.columns:
                return None
            
            # 检查最近60个交易日的交易量是否都大于20000手
            if (data['volume'].tail(60) < 20000).any():
                return None
            
            return data
        except Exception as e:
            print(f"Error collecting data for {variety}: {e}")
            return None
    
    def _clean_data(self, data):
        """数据清洗"""
        # 确保日期列格式正确
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data = data.set_index('date')
        # 处理索引列（当从CSV读取时，datetime索引会变成普通列）
        elif data.columns[0] == 'Unnamed: 0':
            data['date'] = pd.to_datetime(data[data.columns[0]])
            data = data.set_index('date')
            data = data.drop(columns=[data.columns[0]])
        
        # 删除重复行
        data = data.drop_duplicates()
        
        # 删除包含缺失值的行
        data = data.dropna()
        
        # 筛选指定日期之后的数据
        data = data[data.index >= pd.to_datetime(self.start_date)]
        
        # 确保数据按日期排序
        data = data.sort_index()
        
        return data
    
    def get_valid_varieties(self, all_data):
        """获取满足条件的有效品种列表"""
        valid_varieties = []
        
        for variety, data in all_data.items():
            # 检查是否有足够的历史数据（60个交易日）
            if len(data) >= 60:
                # 检查最近60个交易日的交易量是否都大于20000手
                if (data['volume'].tail(60) >= 20000).all():
                    valid_varieties.append(variety)
        
        return valid_varieties
