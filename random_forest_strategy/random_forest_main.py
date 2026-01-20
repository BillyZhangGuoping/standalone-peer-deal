import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import logging
from sklearn.metrics import confusion_matrix

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入必要的模块
import sys
import os

# 添加当前目录到Python路径，确保可以导入position_calculator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from position_calculator import calculate_position_size
from utility.instrument_utils import get_contract_multiplier
from position import calculate_portfolio_metrics
from risk_allocation import calculate_atr_allocation, atr_momentum_composite_allocation, enhanced_atr_allocation, cluster_risk_parity_allocation, enhanced_atr_cluster_risk_allocation, enhanced_sharpe_atr_allocation, model_based_allocation, signal_strength_based_allocation, floor_asset_tilt_allocation
from risk_parity_allocation import risk_parity_allocation
from utility.data_process import clean_data, normalize_data, standardize_data
from utility.calc_funcs import calculate_ma, calculate_macd, calculate_rsi, calculate_bollinger_bands, calculate_atr, calculate_volume_weighted_average_price
from utility.long_short_signals import generate_combined_signal, generate_ma_crossover_signal, generate_macd_signal, generate_rsi_signal, generate_bollinger_bands_signal
from utility.mom import generate_cross_sectional_momentum_signal, calculate_momentum, generate_momentum_signal

# 导入随机森林模型和训练器
from random_forest_strategy.models.random_forest import RandomForestModel
from random_forest_strategy.trade_model.random_forest_trainer import RandomForestTrainer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模型缓存，键为品种，值为模型对象
model_cache = {}
PREDICT_INTERVAL = 80  # 每80次预测后重新训练模型

class ModelManager:
    """模型管理器，用于管理不同模型的训练、预测和重训练"""
    
    def __init__(self, model_type='random_forest', params=None):
        """初始化模型管理器
        
        参数：
        model_type: 模型类型，可选值：'random_forest', 'lightgbm', 'xgboost', 'lstm_attention'
        params: 模型参数
        """
        self.model_type = model_type
        self.params = params if params else {}
        self.models = {}
    
    def get_model(self, symbol):
        """获取指定品种的模型
        
        参数：
        symbol: 品种代码
        
        返回：
        model: 模型对象
        """
        if symbol not in self.models:
            self.models[symbol] = self._create_model()
        return self.models[symbol]
    
    def _create_model(self):
        """创建模型实例
        
        返回：
        model: 模型实例
        """
        if self.model_type == 'random_forest':
            return RandomForestTrainer(self.params)
        else:
            raise ValueError(f"不支持的模型类型：{self.model_type}")
    
    def train_model(self, symbol, data):
        """训练指定品种的模型
        
        参数：
        symbol: 品种代码
        data: 包含特征和标签的数据
        
        返回：
        model: 训练好的模型
        """
        model = self.get_model(symbol)
        model.train(data)
        return model
    
    def predict(self, symbol, data):
        """预测指定品种的结果
        
        参数：
        symbol: 品种代码
        data: 包含特征数据
        
        返回：
        prediction: 预测结果
        """
        model = self.get_model(symbol)
        return model.predict(data)
    
    def predict_with_proba(self, symbol, data):
        """预测指定品种的结果和概率
        
        参数：
        symbol: 品种代码
        data: 包含特征数据
        
        返回：
        tuple: (prediction, probability)
            prediction: 预测结果
            probability: 预测概率
        """
        model = self.get_model(symbol)
        return model.predict_with_proba(data)
    
    def should_retrain(self, symbol):
        """判断是否需要重新训练模型
        
        参数：
        symbol: 品种代码
        
        返回：
        bool: 是否需要重新训练
        """
        if symbol not in self.models:
            return True
        return False  # 简化处理，实际应根据预测次数判断
    
    def get_model_metrics(self, symbol, X_test, y_test):
        """获取模型评估指标
        
        参数：
        symbol: 品种代码
        X_test: 测试特征数据
        y_test: 测试标签数据
        
        返回：
        metrics: 评估指标字典
        """
        model = self.get_model(symbol)
        return model.evaluate(X_test, y_test)
    
    def get_feature_importance(self, symbol):
        """获取特征重要性
        
        参数：
        symbol: 品种代码
        
        返回：
        feature_importance: 特征重要性字典
        """
        model = self.get_model(symbol)
        return model.get_feature_importance()

# 配置参数
CAPITAL = 3000000  # 总资金为三百万
START_DATE = '2025-01-01'  # 开始日期
CLOSE_DATE = '2025-12-31'  # 结束日期
RISK_PER_TRADE = 0.02  # 每笔交易风险比例
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'History_Data', 'hot_daily_market_data')  # 历史数据目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'target_position')  # 输出目录，使用绝对路径确保在random_forest_strategy目录下
ALLOCATION_STRATEGY = 'risk_parity'  # 分配策略: 'var' 或 'sharpe' 或 'cvar' 或 'calculate_atr' 或 'enhanced_atr' 或 'cluster_risk_parity' 或 'enhanced_atr_cluster_risk' 或 'enhanced_sharpe_atr' 或 'model_based' 或 'signal_strength' 或 'floor_asset_tilt' 或 'risk_parity'
ALLOCATION_REOPTIMIZE_INTERVAL = 100  # 分配策略重新优化间隔（天）
variety_list = ['i', 'rb', 'hc', 'jm', 'j', 'ss', 'ni', 'SF', 'SM', 'lc', 'sn', 'si', 'pb', 'cu', 'al', 'zn',
                'ao', 'SH', 'au', 'ag', 'OI', 'a', 'b', 'm', 'RM', 'p', 'y', 'CF', 'SR', 'c', 'cs', 'AP', 'PK',
                'jd', 'CJ', 'lh', 'sc', 'fu', 'lu', 'pg', 'bu', 'eg', 'TA', 'PF', 'PX', 'l', 'v', 'MA', 'pp',
                'eb', 'ru', 'nr', 'br', 'SA', 'FG', 'UR', 'sp', 'IF', 'IC', 'IH', 'IM', 'ec','T', 'TF' 
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
        
        # 使用CSV文件中已有的symbol列，不需要手动生成
        # 确保symbol列存在
        if 'symbol' not in df.columns:
            # 如果没有symbol列，使用当前代码生成（保留原有逻辑作为备份）
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
    elif base_symbol in ['CF', 'SR', 'OI', 'RM', 'MA', 'TA', 'ZC', 'FG', 'RS', 'RI', 'JR', 'LR', 'PM', 'WH', 'CF', 'CY', 'AP', 'CJ', 'UR', 'SA', 'SF', 'SM', 'PF', 'PK', 'BO', 'UR', 'LR', 'RS', 'RI', 'JR', 'PM', 'WH']:
        return 'CZCE'
    elif base_symbol in ['IC', 'IF', 'IH', 'IM', 'T', 'TF', 'TS']:
        return 'CFFEX'
    elif base_symbol in ['SC', 'LU', 'NR', 'EB', 'EG', 'PG', 'BC', 'BD', 'BR', 'LPG', 'PF', 'SA', 'EB', 'EG', 'PG', 'BC', 'BD', 'BR', 'LPG']:
        return 'INE'
    elif base_symbol in ['LC', 'SI', 'PT', 'PD', 'PS']:
        return 'GFEX'
    else:
        return 'DCE'


def preprocess_data(data):
    """预处理数据"""
    import numpy as np
    # 检查记录条数是否足够生成指标（至少需要60天数据生成MA60）
    if len(data) < 60:
        logger.warning(f"数据不足，无法生成完整指标，仅有{len(data)}条记录")
    
    # 计算技术指标 - 使用与BaseModel一致的列名
    data['ma_5'] = calculate_ma(data, 5)
    data['ma_20'] = calculate_ma(data, 20)
    data['ma_60'] = calculate_ma(data, 60)
    
    # 计算RSI
    data['rsi'] = calculate_rsi(data)
    
    # 计算布林带（返回元组：upper_band, sma, lower_band）
    bollinger_upper, bollinger_mid, bollinger_lower = calculate_bollinger_bands(data)
    data['bollinger_mid'] = bollinger_mid
    data['bollinger_upper'] = bollinger_upper
    data['bollinger_lower'] = bollinger_lower
    data['bollinger_percent'] = (data['close'] - bollinger_lower) / (bollinger_upper - bollinger_lower)
    
    # 计算MACD（返回元组：macd, signal, histogram）
    macd, macd_signal, macd_histogram = calculate_macd(data)
    data['macd'] = macd
    data['macd_signal'] = macd_signal
    data['macd_histogram'] = macd_histogram
    
    # 计算ATR
    data['atr'] = calculate_atr(data)
    
    # 计算成交量加权平均价格
    data['vwap'] = calculate_volume_weighted_average_price(data)
    
    # 计算动量指标
    data['momentum'] = calculate_ma(data, 12)  # 使用12日动量，与BaseModel一致
    
    # 计算趋势类特征
    # 价格在均线上方/下方的天数
    data['days_above_ma5'] = (data['close'] > data['ma_5']).rolling(window=10).sum()
    data['days_above_ma20'] = (data['close'] > data['ma_20']).rolling(window=20).sum()
    data['days_above_ma60'] = (data['close'] > data['ma_60']).rolling(window=60).sum()
    
    # 价格与均线的偏离程度
    data['price_ma5_diff'] = (data['close'] - data['ma_5']) / data['ma_5']
    data['price_ma20_diff'] = (data['close'] - data['ma_20']) / data['ma_20']
    data['price_ma60_diff'] = (data['close'] - data['ma_60']) / data['ma_60']
    
    # 趋势强度指标
    data['trend_strength'] = (data['ma_5'] - data['ma_60']) / data['ma_60']
    
    # 计算收益率
    data['return_1'] = data['close'].pct_change(1)
    data['return_5'] = data['close'].pct_change(5)
    data['return_10'] = data['close'].pct_change(10)
    data['return_20'] = data['close'].pct_change(20)
    data['return_60'] = data['close'].pct_change(60)
    
    # 计算波动率
    data['volatility_5'] = data['return_1'].rolling(window=5).std() * np.sqrt(252)
    data['volatility_10'] = data['return_1'].rolling(window=10).std() * np.sqrt(252)
    data['volatility_20'] = data['return_1'].rolling(window=20).std() * np.sqrt(252)
    data['volatility_60'] = data['return_1'].rolling(window=60).std() * np.sqrt(252)
    
    # 计算量价关系
    data['volume_change'] = data['volume'].pct_change(1)
    data['price_volume_corr'] = data['close'].rolling(window=20).corr(data['volume'])
    
    # 新增交易量相关指数指标
    # 交易量移动平均线
    data['volume_ma5'] = data['volume'].rolling(window=5).mean()
    data['volume_ma20'] = data['volume'].rolling(window=20).mean()
    data['volume_ma60'] = data['volume'].rolling(window=60).mean()
    
    # 交易量指数移动平均线（EMA）
    data['volume_ema5'] = data['volume'].ewm(span=5, adjust=False).mean()
    data['volume_ema20'] = data['volume'].ewm(span=20, adjust=False).mean()
    
    # 交易量比率（当前成交量与N日平均成交量的比值）
    data['volume_ratio_5'] = data['volume'] / data['volume_ma5']
    data['volume_ratio_10'] = data['volume'] / data['volume'].rolling(window=10).mean()
    
    # 交易量波动率
    data['volume_volatility_5'] = data['volume_change'].rolling(window=5).std() * np.sqrt(252)
    data['volume_volatility_20'] = data['volume_change'].rolling(window=20).std() * np.sqrt(252)
    
    # 新增趋势跟踪指标
    # 趋势强度指标 - 加强趋势特征权重
    data['trend_strength_ma'] = (data['ma_5'] - data['ma_60']) / data['ma_60'] * 100
    data['trend_strength_ema'] = (data['close'].ewm(span=20, adjust=False).mean() - data['close'].ewm(span=100, adjust=False).mean()) / data['close'].ewm(span=100, adjust=False).mean() * 100
    
    # 价格斜率 - 衡量价格变化速率
    data['price_slope_5'] = data['close'].rolling(window=5).apply(lambda x: np.polyfit(range(5), x, 1)[0], raw=True)
    data['price_slope_20'] = data['close'].rolling(window=20).apply(lambda x: np.polyfit(range(20), x, 1)[0], raw=True)
    
    # 趋势过滤器 - 检测明显趋势
    data['is_strong_up_trend'] = ((data['ma_5'] > data['ma_20']) & (data['ma_20'] > data['ma_60']) & (data['close'] > data['ma_5']) & (data['trend_strength'] > 0.02)).astype(int)
    data['is_strong_down_trend'] = ((data['ma_5'] < data['ma_20']) & (data['ma_20'] < data['ma_60']) & (data['close'] < data['ma_5']) & (data['trend_strength'] < -0.02)).astype(int)
    data['is_sideways'] = ((data['ma_5'] - data['ma_60']).abs() / data['ma_60'] < 0.015).astype(int)
    
    # ADX指标（平均趋向指数）- 衡量趋势强度
    # 计算真实波幅
    data['tr'] = np.maximum(data['high'] - data['low'], 
                          np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                   abs(data['low'] - data['close'].shift(1))))
    # 计算上升趋向和下降趋向
    data['dm_plus'] = np.where(data['high'] > data['high'].shift(1), data['high'] - data['high'].shift(1), 0)
    data['dm_minus'] = np.where(data['low'] < data['low'].shift(1), data['low'].shift(1) - data['low'], 0)
    # 计算14日平均真实波幅、上升趋向和下降趋向
    period = 14
    data['tr_smooth'] = data['tr'].ewm(span=period, adjust=False).mean()
    data['dm_plus_smooth'] = data['dm_plus'].ewm(span=period, adjust=False).mean()
    data['dm_minus_smooth'] = data['dm_minus'].ewm(span=period, adjust=False).mean()
    # 计算趋向指标
    data['di_plus'] = 100 * (data['dm_plus_smooth'] / data['tr_smooth'])
    data['di_minus'] = 100 * (data['dm_minus_smooth'] / data['tr_smooth'])
    # 计算DX和ADX
    data['dx'] = 100 * (abs(data['di_plus'] - data['di_minus']) / (data['di_plus'] + data['di_minus']))
    data['adx'] = data['dx'].ewm(span=period, adjust=False).mean()
    
    # 新增趋势确认指标 - 增强强趋势识别
    # 趋势一致性指标：短期、中期、长期均线方向一致
    data['trend_consistency'] = ((np.sign(data['ma_5'] - data['ma_5'].shift(1)) == np.sign(data['ma_20'] - data['ma_20'].shift(1))) & \
                                 (np.sign(data['ma_20'] - data['ma_20'].shift(1)) == np.sign(data['ma_60'] - data['ma_60'].shift(1)))).astype(int)
    # 价格创新高指标：最近30天内的最高价
    data['is_new_high'] = (data['close'] == data['close'].rolling(window=30).max()).astype(int)
    # 增强版趋势强度：结合价格变化率和成交量
    data['enhanced_trend_strength'] = (data['return_20'] * data['volume_ratio_10'] * (data['di_plus'] - data['di_minus']))
    # 趋势持续时间：连续上涨/下跌的天数 - 使用向量化操作优化
    price_changes = np.sign(data['close'].diff())
    trend_changes = price_changes.diff().fillna(0)
    trend_duration = np.zeros(len(data))
    current_duration = 0
    current_trend = 0
    
    # 使用NumPy数组进行计算，比Pandas Series的apply方法更快
    price_changes_np = price_changes.values
    
    for i in range(len(price_changes_np)):
        if i == 0:
            trend_duration[i] = 0
            continue
        
        change = price_changes_np[i]
        
        if change == 1:
            if current_trend == 1:
                current_duration += 1
            else:
                current_duration = 1
                current_trend = 1
        elif change == -1:
            if current_trend == -1:
                current_duration += 1
            else:
                current_duration = 1
                current_trend = -1
        else:
            current_duration = 0
            current_trend = 0
        
        trend_duration[i] = current_duration * current_trend
    
    data['trend_duration'] = trend_duration
    
    # 处理缺失值
    data = data.dropna()
    
    return data


def generate_daily_target_positions(model_manager, all_data, start_date, capital, close_date=None):
    """生成每日目标头寸"""
    logger.info(f"开始生成每日目标头寸，起始日期: {start_date}, 结束日期: {close_date if close_date else '最新'}")
    
    # 创建带时间戳的输出文件夹
    from datetime import datetime
    timestamp = datetime.now().strftime('%y%m%d_%H%M')
    output_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"目标头寸将保存到: {output_dir}")
    
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
    
    # 初始化模型预测次数计数器
    predict_counts = {symbol: 0 for symbol in all_data.keys()}
    
    # 分配策略优化计数器
    allocation_optimize_count = 0
    last_optimize_date = None
    
    # 预先为每个品种预处理一次完整的数据，避免重复计算
    logger.info("开始预先预处理所有品种的数据...")
    preprocessed_data = {}
    for base_symbol, data in all_data.items():
        logger.debug(f"预处理{base_symbol}的数据...")
        preprocessed_data[base_symbol] = preprocess_data(data.copy())
    logger.info("所有品种数据预处理完成")
    
    # 遍历每个交易日
    for date_idx, date in enumerate(all_dates):
        logger.info(f"\n处理日期: {date.strftime('%Y-%m-%d')}")
        
        # 检查是否需要重新优化分配策略
        if last_optimize_date is None or (date - last_optimize_date).days >= ALLOCATION_REOPTIMIZE_INTERVAL:
            logger.info(f"\n开始重新优化分配策略...")
            
            # 收集过去180天的历史数据用于优化
            lookback_days = 180
            lookback_date = date - timedelta(days=lookback_days)
            
            # 收集这段时间内所有品种的收益率数据
            historical_returns = pd.DataFrame()
            for base_symbol, data in preprocessed_data.items():
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
                
                # 根据市场条件调整分配策略参数
                # 这里可以添加更复杂的优化逻辑，例如：
                # - 高波动市场：降低风险暴露
                # - 高相关性市场：增加多样性权重
                # - 低波动市场：增加趋势跟踪权重
                
                # 我们将这些参数存储在全局变量中，供分配函数使用
                global allocation_optimization_params
                allocation_optimization_params = {
                    'market_volatility': market_volatility,
                    'avg_correlation': avg_correlation,
                    'optimize_date': date,
                    'lookback_days': lookback_days
                }
            
            last_optimize_date = date
            allocation_optimize_count += 1
            logger.info(f"分配策略重新优化完成，这是第{allocation_optimize_count}次优化")
        
        # 初始化每日目标头寸和品种数据
        daily_target_positions = []
        varieties_data = {}
        predicted_signals = {}
        
        # 第一步：收集所有有数据的品种信息
        for base_symbol, data in all_data.items():
            # 检查该品种在该日期是否有数据
            if date not in data.index:
                continue
            
            # 获取预处理后的数据
            full_processed_data = preprocessed_data[base_symbol]
            
            # 筛选出用于训练的数据（不包括当天）
            training_data = full_processed_data[full_processed_data.index < date]
            
            if training_data.empty:
                continue
            
            # 获取模型
            trainer = model_manager.get_model(base_symbol)
            
            # 检查是否需要训练模型
            if trainer.model is None or predict_counts[base_symbol] >= PREDICT_INTERVAL:
                logger.info(f"训练{base_symbol}模型...")
                # 确保至少有360个交易日数据用于训练
                if len(training_data) < 360:
                    logger.warning(f"{base_symbol}训练数据不足360个交易日，当前仅有{len(training_data)}个交易日，跳过训练")
                    continue
                # 使用训练数据训练模型（历史开始点到生成头寸前一天）
                model = model_manager.train_model(base_symbol, training_data)
                if model is None:
                    logger.warning(f"{base_symbol}模型训练失败，跳过该品种")
                    continue
                predict_counts[base_symbol] = 0
            
            # 检查模型是否已训练
            if trainer.model is None:
                logger.warning(f"{base_symbol}模型尚未训练，跳过该品种")
                continue
            
            # 筛选出用于预测的数据（包括当天）
            predict_data = full_processed_data[full_processed_data.index <= date]
            
            if predict_data.empty:
                continue
            
            # 使用最新数据（当天）进行预测
            latest_data = predict_data.iloc[-1:]
            prediction, probabilities = model_manager.predict_with_proba(base_symbol, latest_data)
            
            # 获取预测结果 - 处理None情况
            if prediction is None or probabilities is None:
                signal = 0
                trend_strength = 0
            else:
                signal = prediction[0]
                # 使用预测概率作为趋势强度
                # 提取正确类别的概率（信号为1则取类别1的概率，信号为-1则取类别-1的概率）
                if signal == 1:
                    trend_strength = probabilities[0][1] if probabilities.shape[1] > 1 else 0.5
                elif signal == -1:
                    trend_strength = probabilities[0][0] if probabilities.shape[1] > 1 else 0.5
                else:
                    trend_strength = 0
            
            # 获取当前价格和主力合约代码
            current_data = data.loc[date]
            current_price = current_data['close']
            contract_symbol = current_data['symbol']
            
            # 获取合约乘数和保证金率
            contract_multiplier, margin_rate = get_contract_multiplier(contract_symbol)
            
            # 获取价格历史数据（用于动量计算）
            # 取最近30天的收盘价
            prices = predict_data['close'].tail(30).tolist()
            
            # 收集品种数据用于分配策略
            varieties_data[base_symbol] = {
                'current_price': current_price,
                'contract_multiplier': contract_multiplier,
                'margin_rate': margin_rate,
                'contract_symbol': contract_symbol,
                'signal': signal,  # 保留原始信号（-1.0到1.0）
                'trend_strength': trend_strength,  # 预测概率作为趋势强度
                'prices': prices
            }
                
            # 更新预测次数
            predict_counts[base_symbol] += 1
        
        # 第二步：资金分配
        if varieties_data:
            # 收集所有品种的收益率数据
            returns_data = pd.DataFrame()
            for base_symbol, data in preprocessed_data.items():
                # 提取品种的收益率数据
                if 'return' not in data.columns:
                    # 计算日收益率
                    data['return'] = data['close'].pct_change()
                
                # 提取收益率数据
                returns = data['return'].dropna()
                
                # 将收益率数据添加到DataFrame中
                returns_data[base_symbol] = returns
            
            # 根据配置选择分配策略
            if ALLOCATION_STRATEGY == 'calculate_atr':
                logger.info(f"共有 {len(varieties_data)} 个品种有交易信号，开始进行基于ATR的资金分配...")
                
                # 使用ATR分配策略，返回元组：(allocation_dict, risk_units)
                allocation_dict, risk_units = calculate_atr_allocation(capital, varieties_data)
                
                # 生成目标头寸
                target_positions = []
                for base_symbol, allocated_capital in allocation_dict.items():
                    data = varieties_data[base_symbol]
                    # 计算目标手数：目标手数 = 分配资金 / (当前价格 * 合约乘数 * 保证金率)
                    price_multiplier = data['current_price'] * data['contract_multiplier'] * data['margin_rate']
                    if price_multiplier > 0:
                        base_quantity = int(allocated_capital / price_multiplier)
                        # 根据信号调整方向
                        direction = 1 if data['signal'] > 0 else -1
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
            elif ALLOCATION_STRATEGY == 'enhanced_atr':
                logger.info(f"共有 {len(varieties_data)} 个品种有交易信号，开始进行基于增强ATR的资金分配...")
                
                # 使用增强ATR分配策略，返回元组：(allocation_dict, risk_units)
                allocation_dict, risk_units = enhanced_atr_allocation(capital, varieties_data)
                
                # 生成目标头寸
                target_positions = []
                for base_symbol, allocated_capital in allocation_dict.items():
                    data = varieties_data[base_symbol]
                    # 计算目标手数：目标手数 = 分配资金 / (当前价格 * 合约乘数 * 保证金率)
                    price_multiplier = data['current_price'] * data['contract_multiplier'] * data['margin_rate']
                    if price_multiplier > 0:
                        base_quantity = int(allocated_capital / price_multiplier)
                        # 根据信号调整方向
                        direction = 1 if data['signal'] > 0 else -1
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
            elif ALLOCATION_STRATEGY == 'cluster_risk_parity':
                logger.info(f"共有 {len(varieties_data)} 个品种有交易信号，开始进行基于相关性聚类和风险平价的资金分配...")
                
                # 使用基于相关性聚类和风险平价的分配策略
                # 返回元组：(allocation_dict, risk_units, cluster_weights)
                allocation_dict, risk_units, cluster_weights = cluster_risk_parity_allocation(capital, varieties_data)
                
                logger.info(f"聚类分配权重：{cluster_weights}")
                
                # 生成目标头寸
                target_positions = []
                for base_symbol, allocated_capital in allocation_dict.items():
                    data = varieties_data[base_symbol]
                    # 计算目标手数：目标手数 = 分配资金 / (当前价格 * 合约乘数 * 保证金率)
                    price_multiplier = data['current_price'] * data['contract_multiplier'] * data['margin_rate']
                    if price_multiplier > 0:
                        base_quantity = int(allocated_capital / price_multiplier)
                        # 根据信号调整方向
                        direction = 1 if data['signal'] > 0 else -1
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
            elif ALLOCATION_STRATEGY == 'enhanced_atr_cluster_risk':
                logger.info(f"共有 {len(varieties_data)} 个品种有交易信号，开始进行基于增强ATR聚类风险的资金分配...")
                
                # 使用增强型ATR聚类风险分配策略
                # 返回元组：(allocation_dict, risk_units, cluster_weights)
                allocation_dict, risk_units, cluster_weights = enhanced_atr_cluster_risk_allocation(capital, varieties_data)
                
                logger.info(f"聚类分配权重：{cluster_weights}")
                
                # 生成目标头寸
                target_positions = []
                for base_symbol, allocated_capital in allocation_dict.items():
                    data = varieties_data[base_symbol]
                    # 计算目标手数：目标手数 = 分配资金 / (当前价格 * 合约乘数 * 保证金率)
                    price_multiplier = data['current_price'] * data['contract_multiplier'] * data['margin_rate']
                    if price_multiplier > 0:
                        base_quantity = int(allocated_capital / price_multiplier)
                        # 根据信号调整方向
                        direction = 1 if data['signal'] > 0 else -1
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
            elif ALLOCATION_STRATEGY == 'enhanced_sharpe_atr':
                logger.info(f"共有 {len(varieties_data)} 个品种有交易信号，开始进行基于夏普比率优化的增强型ATR资金分配...")
                
                # 使用基于夏普比率优化的增强型ATR分配策略
                # 返回元组：(allocation_dict, risk_units)
                # 传递市场参数，使策略能够根据市场条件动态调整
                allocation_dict, risk_units = enhanced_sharpe_atr_allocation(capital, varieties_data, market_params=globals().get('allocation_optimization_params', None))
                
                # 生成目标头寸
                target_positions = []
                for base_symbol, allocated_capital in allocation_dict.items():
                    data = varieties_data[base_symbol]
                    # 计算目标手数：目标手数 = 分配资金 / (当前价格 * 合约乘数 * 保证金率)
                    price_multiplier = data['current_price'] * data['contract_multiplier'] * data['margin_rate']
                    if price_multiplier > 0:
                        base_quantity = int(allocated_capital / price_multiplier)
                        # 根据信号调整方向
                        direction = 1 if data['signal'] > 0 else -1
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
            elif ALLOCATION_STRATEGY == 'model_based':
                logger.info(f"共有 {len(varieties_data)} 个品种有交易信号，开始进行基于LightGBM模型的资金分配...")
                
                # 使用基于LightGBM模型的分配策略
                # 返回元组：(allocation_dict, risk_units)
                # 传递市场参数，使策略能够根据市场条件动态调整
                allocation_dict, risk_units = model_based_allocation(capital, varieties_data, market_params=globals().get('allocation_optimization_params', None))
                
                # 生成目标头寸
                target_positions = []
                for base_symbol, allocated_capital in allocation_dict.items():
                    data = varieties_data[base_symbol]
                    # 计算目标手数：目标手数 = 分配资金 / (当前价格 * 合约乘数 * 保证金率)
                    price_multiplier = data['current_price'] * data['contract_multiplier'] * data['margin_rate']
                    if price_multiplier > 0:
                        base_quantity = int(allocated_capital / price_multiplier)
                        # 根据信号调整方向
                        direction = 1 if data['signal'] > 0 else -1
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
            elif ALLOCATION_STRATEGY == 'signal_strength':
                logger.info(f"共有 {len(varieties_data)} 个品种有交易信号，开始进行基于信号强度的资金分配...")
                
                # 使用基于信号强度的分配策略
                # 返回元组：(allocation_dict, risk_units)
                allocation_dict, risk_units = signal_strength_based_allocation(capital, varieties_data)
                
                # 生成目标头寸
                target_positions = []
                for base_symbol, allocated_capital in allocation_dict.items():
                    data = varieties_data[base_symbol]
                    # 计算目标手数：目标手数 = 分配资金 / (当前价格 * 合约乘数 * 保证金率)
                    price_multiplier = data['current_price'] * data['contract_multiplier'] * data['margin_rate']
                    if price_multiplier > 0:
                        base_quantity = int(allocated_capital / price_multiplier)
                        # 根据信号调整方向
                        direction = 1 if data['signal'] > 0 else -1
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
                        'atr': data['atr'],
                        'risk_unit': risk_units[base_symbol],
                        'notional_value': 0,  # 暂不使用notional_value概念
                        'total_notional': 0,  # 暂不使用total_notional概念
                        'price_multiplier': price_multiplier,
                        'target_quantity': position_size
                    }
                    target_positions.append(position_dict)
            elif ALLOCATION_STRATEGY == 'floor_asset_tilt':
                logger.info(f"共有 {len(varieties_data)} 个品种有交易信号，开始进行地板资产倾斜 (sign/Vol) 资金分配...")
                
                # 使用地板资产倾斜 (sign/Vol) 分配策略
                # 返回元组：(allocation_dict, risk_units)
                allocation_dict, risk_units = floor_asset_tilt_allocation(capital, varieties_data)
                
                # 生成目标头寸
                target_positions = []
                for base_symbol, allocated_capital in allocation_dict.items():
                    data = varieties_data[base_symbol]
                    # 计算目标手数：目标手数 = 分配资金 / (当前价格 * 合约乘数 * 保证金率)
                    price_multiplier = data['current_price'] * data['contract_multiplier'] * data['margin_rate']
                    if price_multiplier > 0:
                        base_quantity = int(allocated_capital / price_multiplier)
                        # 根据信号调整方向
                        direction = 1 if data['signal'] > 0 else -1
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
            elif ALLOCATION_STRATEGY == 'risk_parity':
                logger.info(f"共有 {len(varieties_data)} 个品种有交易信号，开始进行风险平价资金分配...")
                
                # 使用风险平价分配策略
                # 返回元组：(allocation_dict, risk_units)
                allocation_dict, risk_units = risk_parity_allocation(capital, varieties_data, date, all_data)
                
                # 生成目标头寸
                target_positions = []
                for base_symbol, allocated_capital in allocation_dict.items():
                    data = varieties_data[base_symbol]
                    # 计算目标手数：目标手数 = 分配资金 / (当前价格 * 合约乘数 * 保证金率)
                    price_multiplier = data['current_price'] * data['contract_multiplier'] * data['margin_rate']
                    if price_multiplier > 0:
                        base_quantity = int(allocated_capital / price_multiplier)
                        # 根据信号调整方向
                        direction = 1 if data['signal'] > 0 else -1
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
            
            # 其他分配策略已弃用，仅保留enhanced_atr策略
            # elif ALLOCATION_STRATEGY == 'sharpe':
            #     logger.info(f"共有 {len(varieties_data)} 个品种有交易信号，开始进行基于夏普比率的资金分配...")
            #     
            #     # 创建夏普比率分配实例

            #     
            #     # 使用夏普比率分配策略生成目标头寸
            #     target_positions = sharpe_allocator.generate_target_positions(varieties_data, returns_data)
            # 
            # elif ALLOCATION_STRATEGY == 'cvar':
            #     logger.info(f"共有 {len(varieties_data)} 个品种有交易信号，开始进行基于CVaR的资金分配...")
            #     
            #     # 创建CVaR分配实例

            #     
            #     # 使用CVaR分配策略生成目标头寸
            #     target_positions = cvar_allocator.generate_target_positions(varieties_data, returns_data)
            # 
            # else:  # 默认使用VaR
            #     logger.info(f"共有 {len(varieties_data)} 个品种有交易信号，开始进行基于VaR的资金分配...")
            #     
            #     # 创建VaR分配实例

            #     
            #     # 使用VaR分配策略生成目标头寸
            #     target_positions = var_allocator.generate_target_positions(varieties_data, returns_data)
            
            # 将生成的头寸添加到每日目标头寸列表中
            for position in target_positions:
                symbol = position['symbol']
                base_symbol = symbol.split('.')[0] if '.' in symbol else symbol
                
                # 提取品种的基础信息
                if base_symbol in varieties_data:
                    data = varieties_data[base_symbol]
                    position['trend_direction'] = 1 if data['signal'] > 0 else -1
                    position['trend_strength'] = data['trend_strength']
                    position['trend_strength_reference'] = data['trend_strength']
                    position['model_type'] = 'random_forest_strategy'
                    position['market_value'] = position['position_value'] if position['trend_direction'] == 1 else -position['position_value']
                
                daily_target_positions.append(position)
        
        # 第四步：过滤掉position_size为0的品种
        if daily_target_positions:
            daily_target_positions = [position for position in daily_target_positions if position['position_size'] != 0]
        
        # 保存每日目标头寸到文件
        logger.info(f"每日目标头寸长度: {len(daily_target_positions)}")
        if daily_target_positions:
            positions_df = pd.DataFrame(daily_target_positions)
            logger.info(f"positions_df形状: {positions_df.shape}")
            logger.info(f"positions_df内容: {positions_df.head()}")
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
    
    # 创建模型管理器
    model_manager = ModelManager(model_type='random_forest')
    
    # 生成每日目标头寸
    generate_daily_target_positions(model_manager, all_data, START_DATE, CAPITAL, CLOSE_DATE)
    
    logger.info("随机森林策略运行完成")


if __name__ == "__main__":
    main()
