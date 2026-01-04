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
from risk_allocation import calculate_atr_allocation, atr_momentum_composite_allocation
from data_process import clean_data, normalize_data, standardize_data
from calc_funcs import calculate_ma, calculate_ema, calculate_macd, calculate_rsi, calculate_bollinger_bands, calculate_atr, calculate_volume_weighted_average_price
from long_short_signals import generate_combined_signal, generate_ma_crossover_signal, generate_macd_signal, generate_rsi_signal, generate_bollinger_bands_signal
from mom import generate_cross_sectional_momentum_signal, calculate_momentum, generate_momentum_signal

# 导入随机森林模型和训练器
from models.random_forest import RandomForestModel
from trade_model.random_forest_trainer import RandomForestTrainer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模型缓存，键为品种，值为模型对象
model_cache = {}
PREDICT_INTERVAL = 100  # 每100次预测后重新训练模型

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
CAPITAL = 10000000  # 总资金为一千万
START_DATE = '2024-07-01'  # 开始日期
RISK_PER_TRADE = 0.02  # 每笔交易风险比例
DATA_DIR = 'History_Data/hot_daily_market_data'  # 历史数据目录
OUTPUT_DIR = 'random_forest_strategy/target_position'  # 输出目录
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
    # 检查记录条数是否足够生成指标（至少需要60天数据生成MA60）
    if len(data) < 60:
        logger.warning(f"数据不足，无法生成完整指标，仅有{len(data)}条记录")
    
    # 计算技术指标 - 使用与BaseModel一致的列名
    data['ma_5'] = calculate_ma(data, 5)
    data['ma_10'] = calculate_ma(data, 10)
    data['ma_20'] = calculate_ma(data, 20)
    data['ma_60'] = calculate_ma(data, 60)
    
    # 计算指数移动平均线
    data['ema_12'] = calculate_ema(data, 12)
    data['ema_26'] = calculate_ema(data, 26)
    data['ema_9'] = calculate_ema(data, 9)
    
    # 计算MACD
    data['macd'], data['macd_signal'], data['macd_histogram'] = calculate_macd(data)
    
    # 计算RSI（多个周期）
    data['rsi_7'] = calculate_rsi(data, 7)
    data['rsi_14'] = calculate_rsi(data, 14)
    data['rsi_21'] = calculate_rsi(data, 21)
    
    # 计算布林带
    data['bb_upper'], data['bb_middle'], data['bb_lower'] = calculate_bollinger_bands(data)
    # 布林带宽度和价格偏离程度
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
    data['bb_distance'] = (data['close'] - data['bb_middle']) / (data['bb_upper'] - data['bb_lower'])
    
    # 计算ATR（多个周期）
    data['atr_10'] = calculate_atr(data, 10)
    data['atr_14'] = calculate_atr(data, 14)
    data['atr_20'] = calculate_atr(data, 20)
    
    # 计算成交量加权平均价格
    data['vwap'] = calculate_volume_weighted_average_price(data)
    
    # 计算动量指标
    # 1. 经典动量指标
    data['momentum_5'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_12'] = data['close'] / data['close'].shift(12) - 1
    data['momentum_20'] = data['close'] / data['close'].shift(20) - 1
    
    # 2. 相对强弱指标
    data['relative_strength'] = data['close'] / data['close'].shift(14) - 1
    
    # 3. 双重动量
    data['price_momentum'] = data['close'] / data['close'].shift(12) - 1
    data['trend_momentum'] = data['close'].rolling(window=6).mean() / data['close'].shift(6) - 1
    
    # 4. 绝对动量
    data['absolute_momentum'] = data['close'] / data['close'].shift(12) - 1
    
    # 保持与base_model兼容的特征列
    data['momentum'] = data['close'] / data['close'].shift(12) - 1  # 基础动量指标
    data['rsi'] = data['rsi_14']  # 使用14日RSI作为基础RSI
    
    # 生成信号列
    # 1. 均线交叉信号
    data['ma_golden_cross'] = 0
    data['ma_death_cross'] = 0
    data['ma_golden_cross'] = ((data['ma_5'] > data['ma_20']) & (data['ma_5'].shift(1) <= data['ma_20'].shift(1))).astype(int)
    data['ma_death_cross'] = ((data['ma_5'] < data['ma_20']) & (data['ma_5'].shift(1) >= data['ma_20'].shift(1))).astype(int)
    
    # 2. MACD交叉信号
    data['macd_golden_cross'] = 0
    data['macd_death_cross'] = 0
    data['macd_golden_cross'] = ((data['macd'] > data['macd_signal']) & (data['macd'].shift(1) <= data['macd_signal'].shift(1))).astype(int)
    data['macd_death_cross'] = ((data['macd'] < data['macd_signal']) & (data['macd'].shift(1) >= data['macd_signal'].shift(1))).astype(int)
    
    # 3. RSI超买超卖信号
    data['rsi_overbought'] = (data['rsi_14'] >= 70).astype(int)
    data['rsi_oversold'] = (data['rsi_14'] <= 30).astype(int)
    
    # 4. 布林带突破信号
    data['bb_buy_signal'] = ((data['close'] > data['bb_lower']) & (data['close'].shift(1) <= data['bb_lower'].shift(1))).astype(int)
    data['bb_sell_signal'] = ((data['close'] < data['bb_upper']) & (data['close'].shift(1) >= data['bb_upper'].shift(1))).astype(int)
    
    # 5. 成交量信号
    data['volume_ma'] = data['volume'].rolling(window=20).mean()
    data['volume_surge'] = (data['volume'] > data['volume_ma'] * 1.5).astype(int)
    data['price_volume_buy'] = ((data['volume_surge'] == 1) & (data['close'] > data['open'])).astype(int)
    
    # 生成base_model期望的信号列
    # 1. 生成各个单信号
    # 直接计算信号，避免使用复杂的布尔索引
    # 均线信号：1=金叉，-1=死叉，0=持有
    data['signal_ma'] = 0
    data.loc[data['ma_golden_cross'] == 1, 'signal_ma'] = 1
    data.loc[data['ma_death_cross'] == 1, 'signal_ma'] = -1
    
    # MACD信号：1=金叉，-1=死叉，0=持有
    data['signal_macd'] = 0
    data.loc[data['macd_golden_cross'] == 1, 'signal_macd'] = 1
    data.loc[data['macd_death_cross'] == 1, 'signal_macd'] = -1
    
    # RSI信号：1=超卖，-1=超买，0=持有
    data['signal_rsi'] = 0
    data.loc[data['rsi_oversold'] == 1, 'signal_rsi'] = 1
    data.loc[data['rsi_overbought'] == 1, 'signal_rsi'] = -1
    
    # 布林带信号：1=突破下轨，-1=突破上轨，0=持有
    data['signal_bb'] = 0
    data.loc[data['bb_buy_signal'] == 1, 'signal_bb'] = 1
    data.loc[data['bb_sell_signal'] == 1, 'signal_bb'] = -1
    
    # 组合信号（等权重）
    data['signal_combined'] = (data['signal_ma'] + data['signal_macd'] + data['signal_rsi'] + data['signal_bb']) / 4
    # 转换为离散信号
    data['signal_combined'] = np.sign(data['signal_combined'])
    
    # 计算收益率
    data['return_1'] = data['close'].pct_change(1)
    data['return_5'] = data['close'].pct_change(5)
    data['return_10'] = data['close'].pct_change(10)
    data['return_20'] = data['close'].pct_change(20)
    
    # 计算波动率
    data['volatility_5'] = data['return_1'].rolling(window=5).std() * np.sqrt(252)
    data['volatility_10'] = data['return_1'].rolling(window=10).std() * np.sqrt(252)
    data['volatility_20'] = data['return_1'].rolling(window=20).std() * np.sqrt(252)
    
    # 计算量价关系
    data['volume_change'] = data['volume'].pct_change(1)
    data['price_volume_corr'] = data['close'].rolling(window=20).corr(data['volume'])
    
    # 处理缺失值
    data = data.dropna()
    
    return data


def prepare_features(data):
    """准备特征数据"""
    # 选择特征列
    feature_columns = [
        # 移动平均线相关
        'ma_5', 'ma_10', 'ma_20', 'ma_60', 
        'ema_12', 'ema_26', 'ema_9',
        
        # MACD相关
        'macd', 'macd_signal', 'macd_histogram',
        'macd_golden_cross', 'macd_death_cross',
        
        # RSI相关
        'rsi_7', 'rsi_14', 'rsi_21',
        'rsi_overbought', 'rsi_oversold',
        
        # 布林带相关
        'bb_upper', 'bb_middle', 'bb_lower',
        'bb_width', 'bb_distance',
        'bb_buy_signal', 'bb_sell_signal',
        
        # ATR相关（多个周期）
        'atr_10', 'atr_14', 'atr_20',
        
        # 成交量相关
        'vwap', 'volume_ma', 'volume_surge', 'price_volume_buy',
        
        # 动量相关
        'momentum_5', 'momentum_12', 'momentum_20',
        'relative_strength', 'price_momentum', 'trend_momentum',
        'absolute_momentum',
        
        # 均线交叉信号
        'ma_golden_cross', 'ma_death_cross',
        
        # 价格和收益率相关
        'return_1', 'return_5', 'return_10', 'return_20',
        
        # 波动率相关
        'volatility_5', 'volatility_10', 'volatility_20',
        
        # 量价关系
        'volume_change', 'price_volume_corr'
    ]
    
    # 确保所有特征列存在
    available_features = [col for col in feature_columns if col in data.columns]
    
    X = data[available_features]
    return X


def prepare_labels(data, lookahead=5):
    """准备标签数据"""
    # 计算未来5天的收益率
    data['future_return'] = data['close'].pct_change(lookahead).shift(-lookahead)
    
    # 生成标签：1表示上涨，-1表示下跌
    data['label'] = 0
    data.loc[data['future_return'] > 0.01, 'label'] = 1
    data.loc[data['future_return'] < -0.01, 'label'] = -1
    
    y = data['label']
    return y


def generate_daily_target_positions(model_manager, all_data, start_date, capital):
    """生成每日目标头寸"""
    logger.info(f"开始生成每日目标头寸，起始日期: {start_date}")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取所有交易日期
    all_dates = []
    for data in all_data.values():
        all_dates.extend(data.index.tolist())
    all_dates = sorted(list(set(all_dates)))
    all_dates = [date for date in all_dates if date >= datetime.strptime(start_date, '%Y-%m-%d')]
    
    # 初始化模型预测次数计数器
    predict_counts = {symbol: 0 for symbol in all_data.keys()}
    
    # 遍历每个交易日
    for date in all_dates:
        logger.info(f"\n处理日期: {date.strftime('%Y-%m-%d')}")
        
        # 初始化每日目标头寸和品种数据
        daily_target_positions = []
        varieties_data = {}
        predicted_signals = {}
        
        # 第一步：收集所有有数据的品种信息
        for base_symbol, data in all_data.items():
            # 检查该品种在该日期是否有数据
            if date not in data.index:
                continue
            
            # 获取该品种在该日期之前的数据
            past_data = data[data.index <= date]
            
            # 检查数据量是否足够：需要至少360天数据（300天训练+60天验证+技术指标计算）
            if len(past_data) < 360:  # 需要足够的数据生成指标和训练模型
                continue
            
            # 预处理数据
            processed_data = preprocess_data(past_data.copy())
            
            if processed_data.empty:
                continue
            
            # 确保有足够的处理后数据用于训练和验证
            if len(processed_data) < 360:
                continue
            
            # 准备特征和标签
            X = prepare_features(processed_data)
            y = prepare_labels(processed_data)
            
            # 确保特征和标签长度一致
            min_length = min(len(X), len(y))
            X = X.iloc[:min_length]
            y = y.iloc[:min_length]
            
            # 使用最近360天的数据，其中300天用于训练，60天用于验证
            if len(X) < 360:
                continue
            
            # 截取最近360天的数据
            X = X.iloc[-360:]
            y = y.iloc[-360:]
            
            # 检查是否需要重新训练模型
            if model_manager.should_retrain(base_symbol) or predict_counts[base_symbol] >= PREDICT_INTERVAL:
                # 训练模型
                logger.info(f"训练{base_symbol}模型...")
                # Combine X and y back into a single DataFrame
                train_data = processed_data.copy()
                model_manager.train_model(base_symbol, train_data)
                predict_counts[base_symbol] = 0
            
            # 使用最新数据进行预测
            latest_data = processed_data.iloc[-1:]
            prediction = model_manager.predict(base_symbol, latest_data)
            
            # 获取预测结果 - 处理None情况
            if prediction is None:
                signal = 0
            else:
                signal = prediction[0]
            
            # 只处理有信号的品种
            if signal == 1 or signal == -1:
                # 获取当前价格和主力合约代码
                current_data = data.loc[date]
                current_price = current_data['close']
                contract_symbol = current_data['symbol']
                
                # 获取合约乘数和保证金率
                contract_multiplier, margin_rate = get_contract_multiplier(contract_symbol)
                
                # 获取ATR值（从processed_data的最后一行）
                atr = processed_data['atr'].iloc[-1] if 'atr' in processed_data.columns else 0
                
                # 获取价格历史数据（用于动量计算）
                # 取最近30天的收盘价
                prices = processed_data['close'].tail(30).tolist()
                
                # 收集品种数据用于ATR动量复合分配
                varieties_data[base_symbol] = {
                    'current_price': current_price,
                    'atr': atr,
                    'contract_multiplier': contract_multiplier,
                    'margin_rate': margin_rate,
                    'contract_symbol': contract_symbol,
                    'signal': signal,
                    'prices': prices
                }
                
            # 更新预测次数
            predict_counts[base_symbol] += 1
        
        # 第二步：基于ATR动量复合分配进行资金分配
        if varieties_data:
            logger.info(f"共有 {len(varieties_data)} 个品种有交易信号，开始进行ATR动量复合分配...")
            
            # 使用ATR动量复合分配
            allocation = atr_momentum_composite_allocation(capital, varieties_data, momentum_window=20)
            
            # 第三步：为每个品种计算目标头寸
            for base_symbol, data in varieties_data.items():
                # 获取品种信息
                current_price = data['current_price']
                contract_symbol = data['contract_symbol']
                contract_multiplier = data['contract_multiplier']
                margin_rate = data['margin_rate']
                signal = data['signal']
                atr = data['atr']
                
                # 获取分配的资金
                allocated_capital = allocation[base_symbol]
                
                # 计算目标手数
                # 充分利用保证金杠杆：手数 = 分配资金 / (当前价格 * 合约乘数 * 保证金率)
                # 这样可以用更少的资金（保证金）持有更大的头寸
                price_multiplier = current_price * contract_multiplier * margin_rate
                target_quantity = int(allocated_capital / price_multiplier)
                
                # 根据信号调整方向
                target_quantity = target_quantity * int(signal)
                
                # 调试信息：打印计算过程
                logger.debug(f"品种 {base_symbol}: 分配资金={allocated_capital:.2f}, 当前价格={current_price}, 合约乘数={contract_multiplier}, 价格*乘数={price_multiplier:.2f}, 计算手数={target_quantity}")
                
                # 确保至少有1手（如果有信号的话）
                if abs(target_quantity) < 1 and signal != 0:
                    logger.debug(f"品种 {base_symbol}: 计算手数小于1，调整为1手")
                    target_quantity = int(signal)  # 至少1手
                
                # 确保手数不为0
                if target_quantity == 0:
                    continue
                
                # 计算持仓价值
                position_value = abs(target_quantity) * current_price * contract_multiplier
                
                # 计算保证金占用
                margin_usage = position_value * margin_rate
                
                # 计算风险暴露
                risk_exposure = position_value / capital
                
                # 计算实际风险金额
                # 风险金额 = 持仓价值 * 保证金率
                actual_risk_amount = margin_usage
                
                # 构建头寸字典
                position_dict = {
                    'symbol': contract_symbol,
                    'current_price': current_price,
                    'contract_multiplier': contract_multiplier,
                    'position_size': target_quantity,
                    'position_value': position_value,
                    'margin_usage': margin_usage,
                    'risk_amount': actual_risk_amount,  # 使用实际风险金额（保证金占用）
                    'margin_rate': margin_rate,
                    'total_capital': capital,
                    'signal': signal,
                    'model_type': 'random_forest_strategy',
                    'market_value': position_value if signal == 1 else -position_value,
                    'allocated_capital': allocated_capital,
                    'atr': atr
                }
                
                daily_target_positions.append(position_dict)
        
        # 保存每日目标头寸到文件
        if daily_target_positions:
            positions_df = pd.DataFrame(daily_target_positions)
            # 只保留position_size不为0的品种
            positions_df = positions_df[positions_df['position_size'] != 0]
            
            # 保存到文件
            positions_file = os.path.join(OUTPUT_DIR, f'target_positions_{date.strftime("%Y%m%d")}.csv')
            positions_df.to_csv(positions_file, index=False)
            logger.info(f"目标头寸已保存到 {positions_file}")
            logger.info(f"生成了 {len(positions_df)} 个品种的目标头寸")


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
    generate_daily_target_positions(model_manager, all_data, START_DATE, CAPITAL)
    
    logger.info("随机森林策略运行完成")


if __name__ == "__main__":
    main()
