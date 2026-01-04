import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
DATA_DIR = 'History_Data/hot_daily_market_data'  # 历史数据目录
OUTPUT_DIR = 'analysis_results'  # 分析结果输出目录

# 品种列表
variety_list = ['i', 'rb', 'hc', 'jm', 'j', 'ss', 'ni', 'SF', 'SM', 'lc', 'sn', 'si', 'pb', 'cu', 'al', 'zn',
                'ao', 'SH', 'au', 'ag', 'OI', 'a', 'b', 'm', 'RM', 'p', 'y', 'CF', 'SR', 'c', 'cs', 'AP', 'PK',
                'jd', 'CJ', 'lh', 'sc', 'fu', 'lu', 'pg', 'bu', 'eg', 'TA', 'PF', 'PX', 'l', 'v', 'MA', 'pp',
                'eb', 'ru', 'nr', 'br', 'SA', 'FG', 'UR', 'sp', 'IF', 'IC', 'IH', 'IM', 'ec', 'T', 'TF'
                ]

class VarietyAnalyzer:
    """品种数据分析器"""
    
    def __init__(self, data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
        """初始化分析器
        
        参数：
        data_dir: 数据目录
        output_dir: 输出目录
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.all_data = None
        self.close_prices = None
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self):
        """加载所有品种的历史数据"""
        logger.info("开始加载历史数据...")
        
        all_data = {}
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        for file in files:
            file_path = os.path.join(self.data_dir, file)
            base_symbol = file.split('.')[0].upper()
            
            # 仅加载variety_list中的品种，忽略大小写
            if base_symbol.lower() not in [v.lower() for v in variety_list]:
                continue
            
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                all_data[base_symbol] = df
                logger.info(f"成功加载品种 {base_symbol}")
            except Exception as e:
                logger.error(f"加载品种 {base_symbol} 失败: {str(e)}")
                continue
        
        self.all_data = all_data
        logger.info(f"数据加载完成，共加载 {len(all_data)} 个品种")
    
    def preprocess_data(self):
        """预处理数据，提取收盘价并对齐日期"""
        logger.info("开始预处理数据...")
        
        if self.all_data is None:
            raise ValueError("请先调用load_data()加载数据")
        
        # 提取所有品种的收盘价
        close_prices_dict = {}
        for symbol, data in self.all_data.items():
            close_prices_dict[symbol] = data['close']
        
        # 合并为一个DataFrame，对齐日期
        self.close_prices = pd.DataFrame(close_prices_dict)
        
        # 删除完全空的行和列
        self.close_prices = self.close_prices.dropna(how='all')
        self.close_prices = self.close_prices.dropna(axis=1, how='all')
        
        # 填充缺失值
        self.close_prices = self.close_prices.ffill()
        
        logger.info(f"数据预处理完成，剩余 {len(self.close_prices.columns)} 个品种，数据时间范围: {self.close_prices.index.min()} 至 {self.close_prices.index.max()}")
    
    def calculate_returns(self):
        ""