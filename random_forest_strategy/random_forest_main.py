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
from risk_allocation import calculate_atr_allocation, atr_momentum_composite_allocation, enhanced_atr_allocation
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
CAPITAL = 3000000  # 总资金为三百万
START_DATE = '2024-01-01'  # 开始日期
RISK_PER_TRADE = 0.02  # 每笔交易风险比例
DATA_DIR = 'History_Data/hot_daily_market_data'  # 历史数据目录
OUTPUT_DIR = 'random_forest_strategy/target_position'  # 输出目录
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
    # 检查记录条数是否足够生成指标（至少需要60天数据生成MA60）
    if len(data) < 60:
        logger.warning(f"数据不足，无法生成完整指标，仅有{len(data)}条记录")
    
    # 计算技术指标 - 使用与BaseModel一致的列名
    data['ma_5'] = calculate_ma(data, 5)
    data['ma_20'] = calculate_ma(data, 20)
    data['ma_60'] = calculate_ma(data, 60)
    
    # 移除RSI和布林带等超买超卖指标，避免在长期趋势中发出错误信号
    
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
    # 趋势持续时间：连续上涨/下跌的天数
    def calculate_trend_duration(series):
        duration = []
        current_duration = 0
        current_trend = 0
        for i in range(len(series)):
            if i == 0:
                duration.append(0)
                continue
            if series.iloc[i] > series.iloc[i-1]:
                if current_trend == 1:
                    current_duration += 1
                else:
                    current_duration = 1
                    current_trend = 1
            elif series.iloc[i] < series.iloc[i-1]:
                if current_trend == -1:
                    current_duration += 1
                else:
                    current_duration = 1
                    current_trend = -1
            else:
                current_duration = 0
                current_trend = 0
            duration.append(current_duration * current_trend)
        return duration
    data['trend_duration'] = calculate_trend_duration(data['close'])
    
    # 处理缺失值
    data = data.dropna()
    
    return data


def generate_daily_target_positions(model_manager, all_data, start_date, capital):
    """生成每日目标头寸"""
    logger.info(f"开始生成每日目标头寸，起始日期: {start_date}")
    
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
            
            # 获取该品种在该日期之前的数据（不包括当天）用于训练模型
            training_data = data[data.index < date]
            
            # 预处理训练数据
            processed_training_data = preprocess_data(training_data.copy())
            
            if processed_training_data.empty:
                continue
            
            # 获取模型
            trainer = model_manager.get_model(base_symbol)
            
            # 检查是否需要训练模型
            if trainer.model is None or predict_counts[base_symbol] >= PREDICT_INTERVAL:
                logger.info(f"训练{base_symbol}模型...")
                # 确保至少有360个交易日数据用于训练
                if len(processed_training_data) < 360:
                    logger.warning(f"{base_symbol}训练数据不足360个交易日，当前仅有{len(processed_training_data)}个交易日，跳过训练")
                    continue
                # 使用训练数据训练模型（历史开始点到生成头寸前一天）
                model = model_manager.train_model(base_symbol, processed_training_data)
                if model is None:
                    logger.warning(f"{base_symbol}模型训练失败，跳过该品种")
                    continue
                predict_counts[base_symbol] = 0
            
            # 检查模型是否已训练
            if trainer.model is None:
                logger.warning(f"{base_symbol}模型尚未训练，跳过该品种")
                continue
            
            # 获取包含当天数据的完整数据，用于预测
            full_data = data[data.index <= date]
            full_processed_data = preprocess_data(full_data.copy())
            
            if full_processed_data.empty:
                continue
            
            # 使用最新数据（当天）进行预测
            latest_data = full_processed_data.iloc[-1:]
            prediction = model_manager.predict(base_symbol, latest_data)
            
            # 获取预测结果 - 处理None情况
            if prediction is None:
                signal = 0
            else:
                signal = prediction[0]
            
            # 获取当前价格和主力合约代码
            current_data = data.loc[date]
            current_price = current_data['close']
            contract_symbol = current_data['symbol']
            
            # 获取合约乘数和保证金率
            contract_multiplier, margin_rate = get_contract_multiplier(contract_symbol)
            
            # 获取ATR值（从full_processed_data的最后一行）
            atr = full_processed_data['atr'].iloc[-1] if 'atr' in full_processed_data.columns else 0
            
            # 获取价格历史数据（用于动量计算）
            # 取最近30天的收盘价
            prices = full_processed_data['close'].tail(30).tolist()
            
            # 收集品种数据用于ATR动量复合分配
            varieties_data[base_symbol] = {
                'current_price': current_price,
                'atr': atr,
                'contract_multiplier': contract_multiplier,
                'margin_rate': margin_rate,
                'contract_symbol': contract_symbol,
                'signal': signal,  # 保留原始信号（趋势强度，-1.0到1.0）
                'prices': prices
            }
                
            # 更新预测次数
            predict_counts[base_symbol] += 1
        
        # 第二步：基于ATR的等风险资金分配
        if varieties_data:
            logger.info(f"共有 {len(varieties_data)} 个品种有交易信号，开始进行基于ATR的等风险资金分配...")
            
            # 使用增强型ATR分配策略
            allocation, _ = enhanced_atr_allocation(capital, varieties_data)
            
            # 第三步：为每个品种计算目标头寸
            for base_symbol, data in varieties_data.items():
                # 获取品种信息
                current_price = data['current_price']
                contract_symbol = data['contract_symbol']
                contract_multiplier = data['contract_multiplier']
                margin_rate = data['margin_rate']
                trend_strength = data['signal']  # 现在是趋势强度（-1.0到1.0）
                atr = data['atr']
                
                # 获取分配的资金
                allocated_capital = allocation[base_symbol]
                
                # 计算目标手数
                # 充分利用保证金杠杆：手数 = 分配资金 / (当前价格 * 合约乘数 * 保证金率)
                # 这样可以用更少的资金（保证金）持有更大的头寸
                price_multiplier = current_price * contract_multiplier * margin_rate
                target_quantity = int(allocated_capital / price_multiplier)
                
                # 根据趋势强度调整方向和大小
                # 方向由趋势强度的符号决定
                direction = 1 if trend_strength > 0 else -1
                # 大小可以考虑趋势强度的绝对值，趋势越强，仓位越大
                # 这里使用基础手数乘以趋势强度的绝对值来调整仓位大小
                target_quantity = int(target_quantity * abs(trend_strength) * direction)
                
                # 调试信息：打印计算过程
                logger.debug(f"品种 {base_symbol}: 分配资金={allocated_capital:.2f}, 当前价格={current_price}, 合约乘数={contract_multiplier}, 价格*乘数={price_multiplier:.2f}, 趋势强度={trend_strength:.4f}, 计算手数={target_quantity}")
                
                # 确保至少有1手（如果趋势强度足够强）
                if abs(target_quantity) < 1 and abs(trend_strength) > 0.3:  # 趋势强度超过0.3才考虑开仓
                    logger.debug(f"品种 {base_symbol}: 计算手数小于1，调整为1手")
                    target_quantity = direction  # 至少1手
                
                # 确保手数不为0
                if target_quantity == 0:
                    continue
                
                # 计算持仓价值
                position_value = abs(target_quantity) * current_price * contract_multiplier
                
                # 计算保证金占用
                margin_usage = position_value * margin_rate
                
                # 取消单品种保证金占用约束
                
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
                    'signal': trend_strength,  # 保存趋势强度，-1.0到1.0
                    'trend_direction': direction,  # 保存趋势方向
                    'trend_strength': abs(trend_strength),  # 保存趋势强度绝对值
                    'model_type': 'random_forest_strategy',
                    'market_value': position_value if direction == 1 else -position_value,
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
            positions_file = os.path.join(output_dir, f'target_positions_{date.strftime("%Y%m%d")}.csv')
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
