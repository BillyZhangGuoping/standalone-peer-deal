import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import logging

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_training import ModelTraining
from correlation_matrix import CorrelationMatrix
from portfolio_allocation import PortfolioAllocation

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
CAPITAL = 10000000  # 总资金为一千万
START_DATE = '2024-01-01'  # 开始日期
RISK_PER_TRADE = 0.02  # 每笔交易风险比例
# 使用绝对路径指向主目录下的History_Data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'History_Data', 'hot_daily_market_data')  # 历史数据目录

# 获取当前脚本所在目录的绝对路径，确保所有目录都在LightGBM_strategy文件夹下
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 使用绝对路径构建目录
MODEL_DIR = os.path.join(SCRIPT_DIR, 'models')  # 模型保存目录
CORRELATION_DIR = os.path.join(SCRIPT_DIR, 'correlation')  # 相关性矩阵保存目录
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'target_position')  # 目标头寸输出目录

# 品种列表（与random_forest_strategy保持一致）
variety_list = ['i', 'rb', 'hc', 'jm', 'j', 'ss', 'ni', 'SF', 'SM', 'lc', 'sn', 'si', 'pb', 'cu', 'al', 'zn',
                'ao', 'SH', 'au', 'ag', 'OI', 'a', 'b', 'm', 'RM', 'p', 'y', 'CF', 'SR', 'c', 'cs', 'AP', 'PK',
                'jd', 'CJ', 'lh', 'sc', 'fu', 'lu', 'pg', 'bu', 'eg', 'TA', 'PF', 'PX', 'l', 'v', 'MA', 'pp',
                'eb', 'ru', 'nr', 'br', 'SA', 'FG', 'UR', 'sp', 'IF', 'IC', 'IH', 'IM', 'ec', 'T', 'TF'
                ]

class LightGBMStrategy:
    def __init__(self):
        self.model_training = None
        self.correlation_matrix = None
        self.portfolio_allocation = None
        self.all_data = None
        self.models = {}
    
    def initialize(self):
        """初始化各个模块"""
        # 创建必要的目录
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(CORRELATION_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 初始化模块
        self.model_training = ModelTraining(DATA_DIR)
        self.correlation_matrix = CorrelationMatrix()
        self.portfolio_allocation = PortfolioAllocation(CAPITAL, RISK_PER_TRADE)
        
        logger.info("LightGBM策略初始化完成")
    
    def load_data(self):
        """加载历史数据"""
        logger.info("开始加载历史数据...")
        self.all_data = self.model_training.load_data(variety_list)
        
        # 预处理数据
        logger.info("开始预处理数据...")
        self.all_data = self.model_training.preprocess_data(self.all_data)
        
        logger.info("历史数据加载和预处理完成")
    
    def train_models(self, retrain=False, end_date=None):
        """训练模型（或加载已有模型）
        
        参数：
        retrain: 是否重新训练模型
        end_date: 截止日期，只使用该日期之前的数据进行训练
        """
        if retrain or not os.listdir(MODEL_DIR):
            logger.info("开始训练LightGBM模型...")
            self.models = self.model_training.train_all_models(self.all_data, MODEL_DIR, end_date)
        else:
            logger.info("加载已有模型...")
            self.models = self.portfolio_allocation.load_models(MODEL_DIR)
    
    def calculate_correlation(self):
        """计算相关性矩阵、波动率和ATR"""
        logger.info("开始计算相关性矩阵、波动率和ATR...")
        correlation_results = self.correlation_matrix.calculate_all(self.all_data, window=60, use_rolling=True)
        
        # 保存计算结果
        self.correlation_matrix.save_results(CORRELATION_DIR)
        
        logger.info("相关性矩阵、波动率和ATR计算完成")
        return correlation_results
    
    def run_portfolio_allocation(self, past_data):
        """进行资金分配和手数计算"""
        logger.info("开始进行资金分配和手数计算...")
        
        # 加载相关性矩阵、波动率和ATR
        correlation_matrix = pd.read_csv(os.path.join(CORRELATION_DIR, 'correlation_matrix.csv'), index_col=0)
        volatility = pd.read_csv(os.path.join(CORRELATION_DIR, 'volatility.csv'), index_col=0)['volatility'].to_dict()
        atr = pd.read_csv(os.path.join(CORRELATION_DIR, 'atr.csv'), index_col=0)['atr'].to_dict()
        
        # 运行资金分配，使用当前日期之前的数据
        allocation_results = self.portfolio_allocation.run_allocation(
            self.models, past_data, volatility, correlation_matrix, atr
        )
        
        logger.info("资金分配和手数计算完成")
        return allocation_results
    
    def simulate_daily_rebalance(self):
        """模拟每日调仓流程，确保每次训练只使用当前日期之前的数据"""
        logger.info("开始模拟每日调仓流程...")
        
        # 获取所有交易日期
        all_dates = []
        for df in self.all_data.values():
            all_dates.extend(df.index.tolist())
        all_dates = sorted(list(set(all_dates)))
        all_dates = [date for date in all_dates if date >= datetime.strptime(START_DATE, '%Y-%m-%d')]
        
        # 初始化资产记录
        portfolio_value = [CAPITAL]
        rebalance_dates = []
        
        # 初始化模型训练计数器
        model_retrain_counter = 0
        
        # 模拟每日调仓
        for i, date in enumerate(all_dates):
            logger.info(f"\n处理日期: {date.strftime('%Y-%m-%d')}")
            
            # 获取当前日期及之前的历史数据
            past_data = {}
            for symbol, df in self.all_data.items():
                # 使用当前日期及之前的数据
                past_data[symbol] = df[df.index <= date]
            
            # 初始化模型（第一次运行）或每100天重新训练一次模型
            if i == 0 or model_retrain_counter >= 100:
                logger.info("重新训练模型...")
                # 使用当前日期及之前的数据训练模型
                self.models = self.model_training.train_all_models(past_data, MODEL_DIR, end_date=date)
                model_retrain_counter = 0
            
            # 每次都重新计算相关性矩阵，确保使用最新数据
            logger.info("重新计算相关性矩阵...")
            correlation_results = self.correlation_matrix.calculate_all(past_data, window=60, use_rolling=True)
            self.correlation_matrix.save_results(CORRELATION_DIR)
            
            # 运行资金分配
            allocation_results = self.run_portfolio_allocation(past_data)
            
            # 保存每日目标头寸
            target_positions = allocation_results['target_positions']
            if target_positions:
                positions_df = pd.DataFrame(target_positions)
                # 只保留position_size不为0的品种
                positions_df = positions_df[positions_df['position_size'] != 0]
                
                # 使用固定文件夹名称，所有目标头寸放在一个文件夹中
                daily_output_dir = os.path.join(OUTPUT_DIR, 'all_target_positions')
                os.makedirs(daily_output_dir, exist_ok=True)
                
                # 保存到文件
                positions_file = os.path.join(daily_output_dir, f'target_positions_{date.strftime("%Y%m%d")}.csv')
                positions_df.to_csv(positions_file, index=False)
                logger.info(f"目标头寸已保存到 {positions_file}")
                logger.info(f"生成了 {len(positions_df)} 个品种的目标头寸")
            
            # 记录调仓日期
            rebalance_dates.append(date)
            
            # 简单模拟资产变化（这里仅作示例，实际应根据交易结果计算）
            if i > 0:
                # 假设资产变化为0（实际应根据交易结果计算）
                portfolio_value.append(portfolio_value[-1])
            
            # 增加模型训练计数器
            model_retrain_counter += 1
        
        # 保存资产记录
        portfolio_df = pd.DataFrame({
            'date': rebalance_dates,
            'portfolio_value': portfolio_value
        })
        portfolio_file = os.path.join(OUTPUT_DIR, 'portfolio_value.csv')
        portfolio_df.to_csv(portfolio_file, index=False)
        logger.info(f"资产记录已保存到 {portfolio_file}")
        
        logger.info("每日调仓流程模拟完成")
    
    def run(self, retrain_models=False):
        """运行完整的策略流程"""
        logger.info("开始运行LightGBM策略...")
        
        # 初始化
        self.initialize()
        
        # 加载数据
        self.load_data()
        
        # 训练或加载模型
        self.train_models(retrain=retrain_models)
        
        # 计算相关性矩阵
        self.calculate_correlation()
        
        # 模拟每日调仓
        self.simulate_daily_rebalance()
        
        logger.info("LightGBM策略运行完成")

def main():
    """主函数"""
    strategy = LightGBMStrategy()
    strategy.run(retrain_models=True)

if __name__ == "__main__":
    main()
