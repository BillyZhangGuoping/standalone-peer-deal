# -*- coding: utf-8 -*-
"""
主处理流程模块：
负责整个策略的流程控制，包括数据加载、信号融合、资金分配和目标头寸生成
"""
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta
import logging

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from position_calculator import calculate_position_size
from utility.instrument_utils import get_contract_multiplier
from position import calculate_portfolio_metrics

# 导入配置管理模块
from config_manager import get_config

# 导入标准化接口
from interfaces import SignalFusion

# 导入趋势模型实现
from trend_models import TrendSignalFusionModel, RandomForestModel

# 导入资金分配方法实现
from allocation_methods import RiskParityAllocation
from risk_allocation import calculate_atr_allocation, floor_asset_tilt_allocation

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局配置
config = get_config()

# 加载配置
GENERAL_CONFIG = config.get_general_config()
VARIETY_LIST = config.get_variety_list()
TREND_MODEL_CONFIG = config.get_trend_model_config()
ALLOCATION_CONFIG = config.get_allocation_config()

# 解析配置参数
CAPITAL = GENERAL_CONFIG.get('capital', 3000000)
START_DATE = GENERAL_CONFIG.get('start_date', '2025-01-01')
CLOSE_DATE = GENERAL_CONFIG.get('end_date', '2025-12-31')
RISK_PER_TRADE = GENERAL_CONFIG.get('risk_per_trade', 0.02)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'History_Data', 'hot_daily_market_data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), GENERAL_CONFIG.get('output_dir', 'target_position'))

# 模型缓存，键为品种，值为模型对象
model_cache = {}

# 模型预测次数计数器
predict_counts = {symbol: 0 for symbol in VARIETY_LIST}


def create_variety_mapping():
    """创建品种代码映射，确保base_symbol与历史数据文件的大写保持一致
    
    返回：
    mapping: 品种代码映射字典，格式为 {原始品种代码: 大写品种代码}
    """
    mapping = {}
    for var in VARIETY_LIST:
        # 保存原始品种代码到大写的映射
        mapping[var] = var.upper()
    return mapping


def load_all_data(data_dir):
    """加载所有历史数据"""
    all_data = {}
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # 创建品种代码映射
    variety_mapping = create_variety_mapping()
    
    for file in files:
        # 提取品种代码（文件名的前部分，如A.csv -> A，AP601.CZCE.csv -> AP）
        file_name = file.split('.')[0]
        
        # 处理完整合约代码格式，如AP601.CZCE -> AP
        base_symbol = file_name
        if len(file_name) > 2 and file_name[2:].isdigit():
            # 提取基础品种代码（前2个字符）
            base_symbol = file_name[:2].upper()
        else:
            # 普通格式，如A.csv -> A
            base_symbol = file_name.upper()
        
        # 仅加载VARIETY_LIST中的品种，使用映射确保一致性
        found = False
        for original_var, mapped_var in variety_mapping.items():
            if mapped_var == base_symbol:
                found = True
                final_symbol = mapped_var  # 使用映射后的大写品种代码
                break
        
        if not found:
            continue
        
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # 使用CSV文件中已有的symbol列，不需要手动生成
        # 确保symbol列存在
        if 'symbol' not in df.columns:
            # 如果没有symbol列，使用当前代码生成（保留大写）
            df['symbol'] = df.index.map(lambda x: f"{final_symbol}{x.strftime('%y%m')}.{get_exchange(final_symbol)}")
        
        all_data[final_symbol] = df
    
    return all_data



def get_exchange(base_symbol):
    """根据基础代码获取交易所"""
    # 统一转换为大写，确保匹配正确
    base_symbol = base_symbol.upper()
    
    # 简化处理，实际需要根据合约规则映射
    if base_symbol in ['A', 'B', 'C', 'CS', 'J', 'JD', 'JM', 'L', 'M', 'P', 'V', 'Y']:
        return 'DCE'
    elif base_symbol in ['CU', 'AL', 'ZN', 'PB', 'NI', 'SN', 'AG', 'AU', 'RB', 'HC', 'BU', 'RU', 'WR', 'SP']:
        return 'SHFE'
    elif base_symbol in ['CF', 'SR', 'OI', 'RM', 'MA', 'TA', 'ZC', 'FG', 'RS', 'RI', 'JR', 'LR', 'PM', 'WH', 'CY', 'AP', 'CJ', 'UR', 'SA', 'SF', 'SM', 'PF', 'PK', 'BO']:
        return 'CZCE'
    elif base_symbol in ['IC', 'IF', 'IH', 'IM', 'T', 'TF', 'TS']:
        return 'CFFEX'
    elif base_symbol in ['SC', 'LU', 'NR', 'EB', 'EG', 'PG', 'BC', 'BD', 'BR', 'LPG']:
        return 'INE'
    elif base_symbol in ['LC', 'SI', 'PT', 'PD', 'PS']:
        return 'GFEX'
    else:
        return 'DCE'


def generate_daily_target_positions(all_data, start_date, capital, close_date=None):
    """生成每日目标头寸"""
    logger.info(f"开始生成每日目标头寸，起始日期: {start_date}, 结束日期: {close_date if close_date else '最新'}")
    
    # 创建带时间戳的输出文件夹
    from datetime import datetime
    timestamp = datetime.now().strftime('%y%m%d_%H%M')
    output_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"目标头寸将保存到: {output_dir}")
    
    # 保存配置文件到输出目录
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config.config, f, ensure_ascii=False, indent=2)
    logger.info(f"配置文件已保存到: {config_file}")
    
    # 初始化趋势模型
    trend_model = None
    if TREND_MODEL_CONFIG.get('type') == 'fusion':
        # 信号融合模型
        fusion_model = SignalFusion()
        
        # 添加各个趋势模型
        for model_config in TREND_MODEL_CONFIG.get('models', []):
            model_name = model_config.get('name')
            weight = model_config.get('weight', 1.0)
            params = model_config.get('params', {})
            
            # 根据模型名称创建对应的模型实例
            if model_name == 'trend_signal_fusion':
                from trend_models import TrendSignalFusionModel
                model = TrendSignalFusionModel(params)
            elif model_name == 'random_forest':
                from trend_models import RandomForestModel
                model = RandomForestModel(params)
            else:
                logger.warning(f"不支持的趋势模型: {model_name}")
                continue
            
            fusion_model.add_model(model, weight)
        
        trend_model = fusion_model
    else:
        # 单个趋势模型
        model_name = TREND_MODEL_CONFIG.get('type')
        params = TREND_MODEL_CONFIG.get('params', {})
        
        if model_name == 'trend_signal_fusion':
            from trend_models import TrendSignalFusionModel
            trend_model = TrendSignalFusionModel(params)
        elif model_name == 'random_forest':
            from trend_models import RandomForestModel
            trend_model = RandomForestModel(params)
        else:
            raise ValueError(f"不支持的趋势模型: {model_name}")
    
    # 初始化资金分配方法 - 直接使用calculate_atr_allocation函数
    logger.info(f"使用的趋势模型: {trend_model}")
    logger.info(f"使用的资金分配方法: calculate_atr_allocation")
    
    # 获取分配参数
    allocation_params = ALLOCATION_CONFIG.get('params', {})
    
    # 获取所有交易日期
    all_dates = []
    for data in all_data.values():
        all_dates.extend(data.index.tolist())
    all_dates = sorted(list(set(all_dates)))
    
    # 转换日期格式
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    close_dt = datetime.strptime(close_date, '%Y-%m-%d') if close_date else None
    
    # 筛选交易日期
    if close_dt:
        all_dates = [date for date in all_dates if start_dt <= date <= close_dt]
    else:
        all_dates = [date for date in all_dates if date >= start_dt]
    
    # 分配策略优化间隔
    allocation_reoptimize_interval = allocation_params.get('reoptimize_interval', 100)
    last_optimize_date = None
    allocation_optimize_count = 0
    
    # 遍历每个交易日
    for date_idx, date in enumerate(all_dates):
        logger.info(f"\n处理日期: {date.strftime('%Y-%m-%d')}")
        
        # 检查是否需要重新优化分配策略
        if last_optimize_date is None or (date - last_optimize_date).days >= allocation_reoptimize_interval:
            logger.info(f"\n开始重新优化分配策略...")
            
            # 收集过去180天的历史数据用于优化
            lookback_days = 180
            lookback_date = date - timedelta(days=lookback_days)
            
            # 收集这段时间内所有品种的收益率数据
            historical_returns = pd.DataFrame()
            for base_symbol, data in all_data.items():
                # 获取该品种在回测期内的数据
                symbol_data = data[(data.index >= lookback_date) & (data.index <= date)]
                if len(symbol_data) > 0:
                    # 计算收益率
                    if 'return' not in symbol_data.columns:
                        symbol_data['return'] = symbol_data['close'].pct_change()
                    historical_returns[base_symbol] = symbol_data['return'].dropna()
            
            # 计算市场统计指标
            if not historical_returns.empty:
                # 计算市场平均波动率
                market_volatility = historical_returns.std().mean()
                # 计算平均相关性
                correlation_matrix = historical_returns.corr()
                avg_correlation = correlation_matrix.stack().mean()
                
                logger.info(f"市场分析结果 - 平均波动率: {market_volatility:.4f}, 平均相关性: {avg_correlation:.4f}")
            
            last_optimize_date = date
            allocation_optimize_count += 1
            logger.info(f"分配策略重新优化完成，这是第{allocation_optimize_count}次优化")
        
        # 初始化每日目标头寸和品种数据
        daily_target_positions = []
        varieties_data = {}
        
        # 准备用于趋势信号生成的数据
        # 对于RandomForestModel，我们需要传递包含完整字段（high, low, close, volume等）的品种数据字典
        variety_mapping = create_variety_mapping()
        
        # 创建品种列表的大写映射
        variety_list_upper = [variety_mapping.get(var, var.upper()) for var in VARIETY_LIST]
        
        # 生成趋势信号
        logger.info("=== 生成趋势信号 ===")
        try:
            # 使用all_data字典直接传递给generate_signals方法，该字典包含完整的品种数据
            trend_signal_df = trend_model.generate_signals(all_data, date, variety_list_upper)
            logger.info(f"生成了{len(trend_signal_df)}个品种的趋势信号")
            
            # 将趋势信号转换为字典，便于后续使用
            trend_signal_dict = {}
            for _, row in trend_signal_df.iterrows():
                variety = row['品种']
                trend_signal_dict[variety] = {
                    'trend_dir': row['趋势方向'],
                    'signal_strength': row['信号强度']
                }
        except Exception as e:
            logger.error(f"生成趋势信号失败: {e}")
            continue
        
        # 收集所有有数据的品种信息
        logger.info("=== 收集品种数据 ===")
        # 创建品种列表的小写映射，用于检查
        variety_list_lower = [var.lower() for var in VARIETY_LIST]
        
        for base_symbol, data in all_data.items():
            # 检查该品种在该日期是否有数据
            if date not in data.index:
                continue
            
            # 检查品种是否在配置的品种列表中（忽略大小写）
            if base_symbol.lower() not in variety_list_lower:
                continue
            
            # 获取当前日期的品种数据
            current_data = data.loc[date]
            current_price = current_data['close']
            contract_symbol = current_data['symbol']
            
            # 获取合约乘数和保证金率
            contract_multiplier, margin_rate = get_contract_multiplier(contract_symbol)
            
            # 获取价格历史数据
            prices = data['close'].tail(360).tolist()
            
            # 从趋势信号结果中获取信号和强度
            if base_symbol in trend_signal_dict:
                signal = trend_signal_dict[base_symbol]['trend_dir']
                trend_strength = trend_signal_dict[base_symbol]['signal_strength']
            else:
                # 如果没有趋势信号，默认使用0
                signal = 0
                trend_strength = 0
            
            # 收集品种数据用于分配策略，使用大写的base_symbol
            varieties_data[base_symbol] = {
                'current_price': current_price,
                'contract_multiplier': contract_multiplier,
                'margin_rate': margin_rate,
                'contract_symbol': contract_symbol,  # 保持原有合约代码，应该是大写的
                'signal': signal,  # 趋势方向（-1/0/1）
                'trend_strength': trend_strength,  # 信号强度（0~1）
                'prices': prices
            }
        
        # 资金分配
        if varieties_data:
            logger.info(f"共有 {len(varieties_data)} 个品种有交易信号，开始进行资金分配...")
            
            try:
                # 使用calculate_atr_allocation函数分配资金
                allocation_dict, risk_units = calculate_atr_allocation(
                    total_capital=capital,
                    varieties_data=varieties_data,
                    target_volatility=0.01
                )
                
                if not allocation_dict:
                    logger.warning("资金分配返回空结果")
                    continue
            except Exception as e:
                logger.error(f"资金分配失败: {e}")
                continue
            
            # 生成目标头寸
            target_positions = []
            for base_symbol, allocated_capital in allocation_dict.items():
                data = varieties_data[base_symbol]
                
                # 计算目标手数：目标手数 = 分配资金 / (当前价格 * 合约乘数 * 保证金率)
                price_multiplier = data['current_price'] * data['contract_multiplier'] * data['margin_rate']
                if price_multiplier > 0:
                    base_quantity = int(allocated_capital / price_multiplier)
                    # 根据信号调整方向：信号为0时不持仓
                    direction = 1 if data['signal'] > 0 else (-1 if data['signal'] < 0 else 0)
                    position_size = base_quantity * direction
                else:
                    position_size = 0
                
                position_dict = {
                    'symbol': data['contract_symbol'],
                    'current_price': data['current_price'],
                    'contract_multiplier': data['contract_multiplier'],
                    'position_size': position_size,
                    'position_value': abs(position_size) * data['current_price'] * data['contract_multiplier'],
                    'margin_usage': abs(position_size) * data['current_price'] * data['contract_multiplier'] * data['margin_rate'],
                    'risk_amount': allocated_capital,
                    'margin_rate': data['margin_rate'],
                    'total_capital': capital,
                    'signal': data['signal'],
                    'risk_unit': risk_units[base_symbol],
                    'notional_value': 0,  # 暂不使用notional_value概念
                    'total_notional': 0,  # 暂不使用total_notional概念
                    'price_multiplier': price_multiplier,
                    'target_quantity': position_size
                }
                target_positions.append(position_dict)
            
            # 完善目标头寸信息
            for position in target_positions:
                symbol = position['symbol']
                base_symbol = symbol.split('.')[0] if '.' in symbol else symbol
                
                # 提取品种的基础信息
                if base_symbol in varieties_data:
                    data = varieties_data[base_symbol]
                    position['trend_direction'] = 1 if data['signal'] > 0 else -1
                    position['trend_strength'] = data['trend_strength']
                    position['trend_strength_reference'] = data['trend_strength']
                    position['model_type'] = str(trend_model)
                    position['market_value'] = position['position_value'] if position['trend_direction'] == 1 else -position['position_value']
                
                daily_target_positions.append(position)
        
        # 过滤掉position_size为0的品种
        if daily_target_positions:
            daily_target_positions = [position for position in daily_target_positions if position['position_size'] != 0]
        
        # 保存每日目标头寸到文件
        logger.info(f"每日目标头寸长度: {len(daily_target_positions)}")
        if daily_target_positions:
            positions_df = pd.DataFrame(daily_target_positions)
            logger.info(f"positions_df形状: {positions_df.shape}")
            
            # 只保留position_size不为0的品种
            positions_df = positions_df[positions_df['position_size'] != 0]
            logger.info(f"过滤后positions_df形状: {positions_df.shape}")
            
            if not positions_df.empty:
                # 保存到文件
                positions_file = os.path.join(output_dir, f'target_positions_{date.strftime("%Y%m%d")}.csv')
                positions_df.to_csv(positions_file, index=False)
                logger.info(f"目标头寸已保存到 {positions_file}")
                logger.info(f"生成了 {len(positions_df)} 个品种的目标头寸")
            else:
                logger.info(f"所有品种的position_size均为0，跳过保存")
        else:
            logger.info(f"当日没有生成目标头寸")


def main():
    """主函数"""
    logger.info("开始运行随机森林策略")
    
    # 加载所有历史数据
    logger.info("开始加载历史数据...")
    all_data = load_all_data(DATA_DIR)
    logger.info(f"历史数据加载完成，共加载 {len(all_data)} 个品种")
    logger.info(f"配置的品种列表包含 {len(VARIETY_LIST)} 个品种")
    
    # 生成每日目标头寸
    generate_daily_target_positions(all_data, START_DATE, CAPITAL, CLOSE_DATE)
    
    logger.info("随机森林策略运行完成")


if __name__ == "__main__":
    main()
