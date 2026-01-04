import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import sharpe_ratio, max_error

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数
DATA_DIR = 'History_Data/hot_daily_market_data'  # 历史数据目录
OUTPUT_DIR = 'target_position'  # 输出目录
START_DATE = '2024-07-01'  # 开始日期
CAPITAL = 10000000  # 总资金
RISK_FREE_RATE = 0.03  # 无风险利率

# 品种列表
variety_list = ['i', 'rb', 'hc', 'jm', 'j', 'ss', 'ni', 'SF', 'SM', 'lc', 'sn', 'si', 'pb', 'cu', 'al', 'zn',
                'ao', 'SH', 'au', 'ag', 'OI', 'a', 'b', 'm', 'RM', 'p', 'y', 'CF', 'SR', 'c', 'cs', 'AP', 'PK',
                'jd', 'CJ', 'lh', 'sc', 'fu', 'lu', 'pg', 'bu', 'eg', 'TA', 'PF', 'PX', 'l', 'v', 'MA', 'pp',
                'eb', 'ru', 'nr', 'br', 'SA', 'FG', 'UR', 'sp', 'IF', 'IC', 'IH', 'IM', 'ec', 'T', 'TF'
                ]

class DynamicRebalancingStrategy:
    """每日动态再平衡策略"""
    
    def __init__(self, data_dir=DATA_DIR, output_dir=OUTPUT_DIR, start_date=START_DATE, capital=CAPITAL):
        """初始化策略
        
        参数：
        data_dir: 数据目录
        output_dir: 输出目录
        start_date: 开始日期
        capital: 总资金
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.start_date = pd.to_datetime(start_date)
        self.capital = capital
        
        self.all_data = None
        self.processed_data = None
        self.selected_varieties = None
        
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
                logger.info(f"  成功加载品种 {base_symbol}")
            except Exception as e:
                logger.error(f"  加载品种 {base_symbol} 失败: {str(e)}")
                continue
        
        self.all_data = all_data
        logger.info(f"数据加载完成，共加载 {len(all_data)} 个品种")
        return all_data
    
    def preprocess_data(self):
        """预处理数据，提取收盘价并对齐日期"""
        logger.info("开始预处理数据...")
        
        if self.all_data is None:
            self.load_data()
        
        # 提取所有品种的收盘价
        close_prices_dict = {}
        for symbol, data in self.all_data.items():
            close_prices_dict[symbol] = data['close']
        
        # 合并为一个DataFrame，对齐日期
        self.processed_data = pd.DataFrame(close_prices_dict)
        
        # 处理缺失值
        logger.info(f"原始数据形状: {self.processed_data.shape}")
        logger.info(f"缺失值数量: {self.processed_data.isnull().sum().sum()}")
        
        # 使用前向填充处理缺失值
        self.processed_data = self.processed_data.ffill()
        # 再使用后向填充处理剩余缺失值
        self.processed_data = self.processed_data.bfill()
        
        # 删除仍有缺失值的列
        self.processed_data = self.processed_data.dropna(axis=1)
        
        logger.info(f"预处理后数据形状: {self.processed_data.shape}")
        logger.info(f"预处理后缺失值数量: {self.processed_data.isnull().sum().sum()}")
        
        return self.processed_data
    
    def select_varieties(self, corr_threshold=0.7, lookback_days=60):
        """基于相关性和绩效筛选品种
        
        参数：
        corr_threshold: 相关性阈值，超过该阈值的品种将被排除
        lookback_days: 回溯天数，用于计算绩效
        
        返回：
        selected_varieties: 筛选后的品种列表
        """
        logger.info(f"开始筛选品种，相关性阈值: {corr_threshold}，回溯天数: {lookback_days}")
        
        if self.processed_data is None:
            self.preprocess_data()
        
        # 获取回溯期数据
        lookback_start = self.start_date - pd.Timedelta(days=lookback_days)
        lookback_data = self.processed_data[(self.processed_data.index >= lookback_start) & (self.processed_data.index < self.start_date)]
        
        if lookback_data.empty:
            logger.warning(f"回溯期数据不足，使用全部数据进行筛选")
            lookback_data = self.processed_data
        
        # 计算品种的绩效（年化收益率）
        returns = lookback_data.pct_change().dropna()
        annual_returns = returns.mean() * 252
        
        # 计算品种的波动率
        volatility = returns.std()
        
        # 计算夏普比率
        sharpe_ratios = (annual_returns - RISK_FREE_RATE) / volatility
        
        # 计算相关系数矩阵
        corr_matrix = lookback_data.corr()
        
        # 基于相关性和绩效筛选品种
        selected = []
        
        # 按夏普比率降序排序
        sorted_varieties = sharpe_ratios.sort_values(ascending=False