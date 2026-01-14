import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import logging

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入必要的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import create_lstm_features, create_atr_features, create_market_state_features
from lstm_model import LSTMModelManager
from signal_generation import generate_lstm_signals, filter_signals, normalize_signals
from risk_allocation import calculate_dynamic_atr, calculate_risk_adjusted_position
from position_calculator import calculate_target_positions
from utility.instrument_utils import get_contract_multiplier

# 配置参数
CAPITAL = 10000000  # 总资金为一千万
START_DATE = '2024-01-01'  # 开始日期
RISK_PER_TRADE = 0.02  # 每笔交易风险比例
DATA_DIR = 'History_Data/hot_daily_market_data'  # 历史数据目录
OUTPUT_DIR = 'lstm_atr_strategy/target_position'  # 输出目录

# 品种列表（与random_forest_strategy保持一致）
variety_list = ['i', 'rb', 'hc', 'jm', 'j', 'ss', 'ni', 'SF', 'SM', 'lc', 'sn', 'si', 'pb', 'cu', 'al', 'zn',
                'ao', 'SH', 'au', 'ag', 'OI', 'a', 'b', 'm', 'RM', 'p', 'y', 'CF', 'SR', 'c', 'cs', 'AP', 'PK',
                'jd', 'CJ', 'lh', 'sc', 'fu', 'lu', 'pg', 'bu', 'eg', 'TA', 'PF', 'PX', 'l', 'v', 'MA', 'pp',
                'eb', 'ru', 'nr', 'br', 'SA', 'FG', 'UR', 'sp', 'IF', 'IC', 'IH', 'IM', 'ec', 'T', 'TF'
                ] 

def load_all_data(data_dir):
    """加载所有历史数据"""
    all_data = {}
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        base_symbol = file.split('.')[0].upper()
        
        # 仅加载variety_list中的品种，忽略大小写
        if base_symbol.lower() not in [v.lower() for v in variety_list]:
            continue
        
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # 确保symbol列存在
        if 'symbol' not in df.columns:
            df['symbol'] = df.index.map(lambda x: f"{base_symbol.lower()}{x.strftime('%y%m')}.{get_exchange(base_symbol)}")
        
        all_data[base_symbol] = df
    
    return all_data

def get_exchange(base_symbol):
    """根据基础代码获取交易所"""
    # 简化处理，实际需要根据合约规则映射
    if base_symbol in ['A', 'B', 'C', 'CS', 'J', 'JD', 'JM', 'L', 'M', 'P', 'V', 'Y']:
        return 'DCE'
    elif base_symbol in ['CU', 'AL', 'ZN', 'PB', 'NI', 'SN', 'AG', 'AU', 'RB', 'HC', 'BU', 'RU', 'WR', 'SP']:
        return 'SHFE'
    elif base_symbol in ['CF', 'SR', 'OI', 'RM', 'MA', 'TA', 'ZC', 'FG', 'RS', 'RI', 'JR', 'LR', 'PM', 'WH', 'CY', 'AP', 'CJ', 'UR', 'SA', 'SF', 'SM', 'PF', 'PK']:
        return 'CZCE'
    elif base_symbol in ['IC', 'IF', 'IH', 'IM', 'T', 'TF', 'TS']:
        return 'CFFEX'
    elif base_symbol in ['SC', 'LU', 'NR', 'EB', 'EG', 'PG', 'BC', 'BD', 'BR', 'LPG', 'PF', 'SA']:
        return 'INE'
    elif base_symbol in ['LC', 'SI', 'PT', 'PD', 'PS']:
        return 'GFEX'
    else:
        return 'DCE'

def main():
    """主函数"""
    logger.info("开始运行LSTM-ATR策略")
    
    # 加载所有历史数据
    logger.info("开始加载历史数据...")
    all_data = load_all_data(DATA_DIR)
    logger.info(f"历史数据加载完成，共加载 {len(all_data)} 个品种")
    
    # 创建模型管理器
    model_manager = LSTMModelManager()
    
    # 生成每日目标头寸
    calculate_target_positions(model_manager, all_data, START_DATE, CAPITAL, OUTPUT_DIR)
    
    logger.info("LSTM-ATR策略运行完成")

if __name__ == "__main__":
    main()
