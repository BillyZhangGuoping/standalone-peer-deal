import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import logging
import json

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from lstm_combined_strategy import LSTMCombinedStrategy, LSTMTrendPredictor, LSTMAllocationModel
from utility.instrument_utils import get_contract_multiplier
from utility.calc_funcs import calculate_ma, calculate_macd, calculate_rsi, calculate_bollinger_bands, calculate_atr, calculate_volume_weighted_average_price
from utility.data_process import clean_data
from random_forest_strategy.risk_allocation import signal_strength_based_allocation

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
START_DATE = '2025-01-01'
END_DATE = '2025-12-31'  # 设为2025-12-31，只生成2025年的头寸
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'History_Data', 'hot_daily_market_data')

# 创建新的输出目录结构，使用YYMMDD_hhmm格式
current_datetime = datetime.now().strftime('%y%m%d_%H%M')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lstm_combine_strategy_target_postion', current_datetime)
VAL_RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lstm_validation_result', current_datetime)

# 品种列表
variety_list = ['i', 'rb', 'hc', 'jm', 'j', 'ss', 'ni', 'SF', 'SM', 'lc', 'sn', 'si', 'pb', 'cu', 'al', 'zn',
                'ao', 'SH', 'au', 'ag', 'OI', 'a', 'b', 'm', 'RM', 'p', 'y', 'CF', 'SR', 'c', 'cs', 'AP', 'PK',
                'jd', 'CJ', 'lh', 'sc', 'fu', 'lu', 'pg', 'bu', 'eg', 'TA', 'PF', 'PX', 'l', 'v', 'MA', 'pp',
                'eb', 'ru', 'nr', 'br', 'SA', 'FG', 'UR', 'sp', 'IF', 'IC', 'IH', 'IM', 'ec','T', 'TF']

def load_variety_data(variety, start_date, end_date):
    """加载单个品种的历史数据"""
    file_path = os.path.join(DATA_DIR, f'{variety}.csv')
    if not os.path.exists(file_path):
        logger.warning(f"品种 {variety} 的数据文件不存在: {file_path}")
        return None
    
    try:
        # 读取CSV文件，将第一列（索引列）作为日期
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        # 将索引列重命名为'date'
        data = data.reset_index().rename(columns={'index': 'date'})
        # 转换日期列
        data['date'] = pd.to_datetime(data['date'])
        # 过滤日期范围
        data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        
        # 计算技术指标
        data = calculate_technical_indicators(data)
        
        return data
    except Exception as e:
        logger.error(f"加载品种 {variety} 数据失败: {e}")
        return None

def calculate_technical_indicators(data):
    """计算技术指标"""
    # 计算移动平均线
    data['ma_5'] = calculate_ma(data, 5)
    data['ma_20'] = calculate_ma(data, 20)
    data['ma_60'] = calculate_ma(data, 60)
    
    # 计算ATR
    data['atr'] = calculate_atr(data, 14)
    
    # 计算动量
    data['momentum'] = data['close'].pct_change(10)
    
    # 计算收益率
    data['return_5'] = data['close'].pct_change(5)
    data['return_10'] = data['close'].pct_change(10)
    data['return_20'] = data['close'].pct_change(20)
    
    # 计算波动率
    data['volatility_10'] = data['return_5'].rolling(window=10).std() * np.sqrt(252)
    data['volatility_20'] = data['return_10'].rolling(window=20).std() * np.sqrt(252)
    
    # 计算VWAP（简单实现）
    data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
    
    return data

def generate_target_positions():
    """使用LSTM组合策略生成目标头寸"""
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 初始化LSTM策略
    lstm_strategy = LSTMCombinedStrategy(backtest_days=100)
    
    # 加载所有品种的数据
    historical_data = {}
    for variety in variety_list:
        logger.info(f"加载品种 {variety} 的历史数据...")
        data = load_variety_data(variety, START_DATE, END_DATE)
        if data is not None:
            historical_data[variety] = data
    
    logger.info(f"成功加载 {len(historical_data)} 个品种的数据")
    
    # 获取所有交易日期
    all_dates = []
    for data in historical_data.values():
        all_dates.extend(data['date'].tolist())
    all_dates = sorted(list(set(all_dates)))
    
    # 为每个交易日生成目标头寸
    for target_date in all_dates:
        logger.info(f"\n处理日期: {target_date.strftime('%Y-%m-%d')}")
        
        # 准备当日及之前的数据
        day_data = {}
        instrument_data = {}
        
        for symbol, data in historical_data.items():
            # 获取当日及之前的数据
            symbol_data = data[data['date'] <= target_date]
            if len(symbol_data) < 60:  # 需要足够的数据来计算特征
                continue
            
            day_data[symbol] = symbol_data
            
            # 准备资金分配数据
            latest_data = symbol_data.iloc[-1]
            instrument_data[symbol] = {
                'current_price': latest_data['close'],
                'price_history': symbol_data['close'].tolist(),
                'high_history': symbol_data['high'].tolist(),
                'low_history': symbol_data['low'].tolist()
            }
        
        if not instrument_data:
            logger.warning(f"日期 {target_date.strftime('%Y-%m-%d')} 没有足够的数据生成头寸")
            continue
        
        # 使用LSTM计算趋势
        logger.info(f"共有 {len(day_data)} 个品种有足够数据，开始计算趋势...")
        trends = {}
        for symbol, data in day_data.items():
            trend = lstm_strategy.calculate_trend(data, symbol)
            trends[symbol] = trend
        
        # 准备用于信号强度风险分配的数据
        logger.info("开始准备信号强度风险分配数据...")
        risk_allocation_data = {}
        for symbol, data in day_data.items():
            latest_data = data.iloc[-1]
            risk_allocation_data[symbol] = {
                'current_price': latest_data['close'],
                'prices': data['close'].tolist(),
                'price_history': data['close'].tolist(),
                'atr': latest_data['atr'],
                'contract_multiplier': get_contract_multiplier(symbol)[0],
                'trend_strength': trends[symbol]
            }
        
        # 使用基于信号强度的风险分配
        logger.info("开始进行基于信号强度的风险分配...")
        capital = 10000000  # 总资金1000万
        allocation_result, risk_units = signal_strength_based_allocation(capital, risk_allocation_data)
        
        # 生成目标头寸
        target_positions = []
        
        for symbol, trend in trends.items():
            if symbol not in allocation_result:
                continue
                
            # 获取品种数据
            data = historical_data[symbol]
            latest_data = data[data['date'] == target_date].iloc[-1]
            
            # 计算分配资金
            allocated_capital = allocation_result[symbol]
            
            # 计算合约乘数
            multiplier, _ = get_contract_multiplier(symbol)
            
            # 计算目标手数
            margin_rate = 0.1  # 假设保证金率为10%
            price_multiplier = latest_data['close'] * multiplier * margin_rate
            
            if price_multiplier > 0:
                base_quantity = int(allocated_capital / price_multiplier)
                # 根据趋势调整方向
                position_size = base_quantity * trend
            else:
                position_size = 0
            
            # 获取正确的合约月份
            def get_contract_month(date):
                """根据日期获取正确的合约月份"""
                year = date.year
                month = date.month
                # 对于大多数品种，使用主力合约月份（1,5,9月）
                if month <= 3:
                    # 1-3月，使用当年5月合约
                    contract_year = year % 100
                    contract_month = '5'
                elif month <= 7:
                    # 4-7月，使用当年9月合约
                    contract_year = year % 100
                    contract_month = '9'
                elif month <= 11:
                    # 8-11月，使用次年1月合约
                    contract_year = (year + 1) % 100
                    contract_month = '1'
                else:
                    # 12月，使用次年1月合约
                    contract_year = (year + 1) % 100
                    contract_month = '1'
                return f"{contract_year}{contract_month}"
            
            # 基础品种到交易所的映射
            base_symbol_to_exchange = {
                # 上海期货交易所 (SHFE)
                'cu': 'SHFE', 'al': 'SHFE', 'zn': 'SHFE', 'pb': 'SHFE', 'ni': 'SHFE', 
                'sn': 'SHFE', 'au': 'SHFE', 'ag': 'SHFE', 'rb': 'SHFE', 'hc': 'SHFE', 
                'ss': 'SHFE', 'bu': 'SHFE', 'ru': 'SHFE', 'nr': 'SHFE', 'sp': 'SHFE',
                # 大连商品交易所 (DCE)
                'i': 'DCE', 'j': 'DCE', 'jm': 'DCE', 'l': 'DCE', 'v': 'DCE',
                'pp': 'DCE', 'ma': 'DCE', 'y': 'DCE', 'p': 'DCE',
                'm': 'DCE', 'a': 'DCE', 'b': 'DCE', 'jd': 'DCE', 'eg': 'DCE',
                'rr': 'DCE', 'eb': 'DCE', 'lu': 'DCE', 'pg': 'DCE', 'sc': 'DCE',
                'fu': 'DCE',
                # 郑州商品交易所 (CZCE)
                'cf': 'CZCE', 'sr': 'CZCE', 'ta': 'CZCE', 'ma': 'CZCE', 'wh': 'CZCE',
                'ri': 'CZCE', 'oi': 'CZCE', 'rm': 'CZCE', 'rs': 'CZCE', 'lr': 'CZCE',
                'sf': 'CZCE', 'sm': 'CZCE', 'fg': 'CZCE', 'tc': 'CZCE', 'cy': 'CZCE',
                'pm': 'CZCE', 'lh': 'CZCE', 'ap': 'CZCE', 'jr': 'CZCE',
                'sa': 'CZCE', 'ur': 'CZCE', 'lc': 'CZCE',
                # 中国金融期货交易所 (CFFEX)
                'if': 'CFFEX', 'ic': 'CFFEX', 'ih': 'CFFEX', 'im': 'CFFEX', 't': 'CFFEX',
                'tf': 'CFFEX', 'ts': 'CFFEX',
                # 上海国际能源交易中心 (INE)
                'sc': 'INE', 'lu': 'INE', 'pg': 'INE', 'br': 'INE', 'eb': 'INE',
                # 其他
                'ao': 'CZCE', 'sh': 'SHFE', 'cj': 'DCE',
                'pk': 'DCE', 'ec': 'CFFEX',
                # 明确添加PX的映射
                'px': 'CZCE',
            }
            
            # 从历史数据中获取当日的实际合约代码
            data = historical_data[symbol]
            target_date_str = target_date.strftime('%Y-%m-%d')
            day_data = data[data['date'] == target_date_str]
            if not day_data.empty:
                # 如果有当日数据，使用实际的合约代码
                full_contract_code = day_data['symbol'].iloc[-1]
            else:
                # 否则使用计算的合约代码
                contract_month = get_contract_month(target_date)
                base_symbol_lower = symbol.lower()
                exchange = base_symbol_to_exchange.get(base_symbol_lower, 'SHFE')  # 默认使用SHFE
                full_contract_code = f"{symbol}{contract_month}.{exchange}"
            
            # 创建头寸字典
            position_dict = {
                'date': target_date.strftime('%Y-%m-%d'),
                'symbol': full_contract_code,  # 使用完整的合约代码，包括交易所后缀
                'base_symbol': symbol,
                'signal': trend,
                'allocated_capital': allocated_capital,
                'current_price': latest_data['close'],
                'contract_multiplier': multiplier,
                'margin_rate': margin_rate,
                'position_size': position_size,
                'position_value': abs(position_size) * latest_data['close'] * multiplier,
                'margin_usage': abs(position_size) * latest_data['close'] * multiplier * margin_rate,
                'risk_amount': allocated_capital,
                'atr': latest_data['atr'],
                'volatility': latest_data['volatility_20'],
                'trend_strength': trend,
                'notional_value': 0,
                'total_notional': 0,
                'price_multiplier': price_multiplier,
                'target_quantity': position_size
            }
            
            target_positions.append(position_dict)
        
        # 保存目标头寸到文件
        if target_positions:
            positions_df = pd.DataFrame(target_positions)
            file_name = f"target_positions_{target_date.strftime('%Y%m%d')}.csv"
            file_path = os.path.join(OUTPUT_DIR, file_name)
            positions_df.to_csv(file_path, index=False)
            logger.info(f"目标头寸已保存到 {file_path}")
            logger.info(f"生成了 {len(target_positions)} 个品种的目标头寸")

def run_backtest():
    """运行回测"""
    # 更新backtest.py的配置
    backtest_config = {
        'TARGET_POSITION_DIR': OUTPUT_DIR,
        'SPECIFIC_TARGET_DIR': '',
        'PRIMARY_DATA_DIR': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'History_Data', 'hot_daily_market_data'),
        'SECONDARY_DATA_DIR': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'History_Data', 'secondary_daily_market_data'),
        'OVER_DATA_DIR': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'History_Data', 'over_daily_market_data'),
        'START_DATE': START_DATE,
        'END_DATE': END_DATE
    }
    
    # 保存配置到临时文件
    config_file = 'backtest_config_temp.json'
    with open(config_file, 'w') as f:
        json.dump(backtest_config, f)
    
    # 运行回测
    logger.info("开始运行回测...")
    os.system(f"python backtest.py --config {config_file}")
    
    # 删除临时配置文件
    os.remove(config_file)

def main():
    """主函数"""
    logger.info("=== LSTM组合策略运行开始 ===")
    
    # 生成目标头寸
    generate_target_positions()
    
    # 运行回测
    run_backtest()
    
    logger.info("=== LSTM组合策略运行结束 ===")

if __name__ == "__main__":
    main()
