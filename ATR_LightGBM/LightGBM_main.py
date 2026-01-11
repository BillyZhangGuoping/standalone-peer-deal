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
from instrument_utils import get_contract_multiplier
from position import calculate_portfolio_metrics
from risk_allocation import calculate_atr_allocation, atr_momentum_composite_allocation


# 使用本地特征计算函数
from feature_calculator import calculate_ma, calculate_macd, calculate_rsi, calculate_bollinger_bands, calculate_atr, calculate_volume_weighted_average_price, calculate_momentum

# 导入LightGBM模型和训练器
from models.boosting import BoostingModel
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
        elif self.model_type == 'lightgbm':
            return RandomForestTrainer(self.params, model_class='lightgbm')
        elif self.model_type == 'multi_time_period':
            return RandomForestTrainer(self.params, model_class='multi_time_period')
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
START_DATE = '2025-10-01'  # 开始日期
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
    # 检查记录条数是否足够生成指标（至少需要250天数据生成所有指标）
    if len(data) < 250:
        logger.warning(f"数据不足，无法生成完整指标，仅有{len(data)}条记录")
    
    # 计算技术指标 - 使用与BaseModel一致的列名
    data['ma_5'] = calculate_ma(data, 5)
    data['ma_20'] = calculate_ma(data, 20)
    data['ma_60'] = calculate_ma(data, 60)
    data['ma_120'] = calculate_ma(data, 120)  # 增加120日均线
    data['ma_250'] = calculate_ma(data, 250)  # 增加250日均线
    
    # 计算MACD
    data['macd'], data['macd_signal'], data['macd_histogram'] = calculate_macd(data)
    
    # 计算RSI
    data['rsi'] = calculate_rsi(data)
    
    # 计算布林带
    data['bb_upper'], data['bb_middle'], data['bb_lower'] = calculate_bollinger_bands(data)
    
    # 计算ATR
    data['atr'] = calculate_atr(data)
    
    # 计算成交量加权平均价格
    data['vwap'] = calculate_volume_weighted_average_price(data)
    
    # 计算动量指标
    data['momentum_3'] = calculate_momentum(data, 3)  # 增加3日短期动量
    data['momentum_5'] = calculate_momentum(data, 5)  # 增加5日短期动量
    data['momentum_10'] = calculate_momentum(data, 10)  # 增加10日短期动量
    data['momentum_12'] = calculate_momentum(data, 12)  # 12日动量
    data['momentum_60'] = calculate_momentum(data, 60)  # 60日动量
    
    # 计算短期动量变化率
    data['momentum_3_change'] = data['momentum_3'].pct_change(3)
    data['momentum_5_change'] = data['momentum_5'].pct_change(5)
    
    # 计算短期动量与价格的相关性
    data['momentum_price_corr_5'] = data['momentum_5'].rolling(window=5).corr(data['close'])
    data['momentum_price_corr_10'] = data['momentum_10'].rolling(window=10).corr(data['close']) 
    
    # 计算短期动量突破指标
    data['momentum_5_above_10'] = (data['momentum_5'] > data['momentum_10']).astype(int)
    data['momentum_3_above_5'] = (data['momentum_3'] > data['momentum_5']).astype(int)
    
    # 计算趋势类特征
    # 价格在均线上方/下方的天数
    data['days_above_ma5'] = (data['close'] > data['ma_5']).rolling(window=10).sum()
    data['days_above_ma20'] = (data['close'] > data['ma_20']).rolling(window=20).sum()
    data['days_above_ma60'] = (data['close'] > data['ma_60']).rolling(window=60).sum()
    data['days_above_ma120'] = (data['close'] > data['ma_120']).rolling(window=120).sum()  # 增加120日均线天数
    data['days_above_ma250'] = (data['close'] > data['ma_250']).rolling(window=250).sum()  # 增加250日均线天数
    
    # 价格与均线的偏离程度
    data['price_ma5_diff'] = (data['close'] - data['ma_5']) / data['ma_5']
    data['price_ma20_diff'] = (data['close'] - data['ma_20']) / data['ma_20']
    data['price_ma60_diff'] = (data['close'] - data['ma_60']) / data['ma_60']
    data['price_ma120_diff'] = (data['close'] - data['ma_120']) / data['ma_120']  # 增加120日均线偏离
    data['price_ma250_diff'] = (data['close'] - data['ma_250']) / data['ma_250']  # 增加250日均线偏离
    
    # 趋势强度指标
    data['trend_strength'] = (data['ma_5'] - data['ma_60']) / data['ma_60']
    data['long_term_trend_strength'] = (data['ma_60'] - data['ma_250']) / data['ma_250']  # 增加长期趋势强度
    
    # 计算收益率
    data['return_1'] = data['close'].pct_change(1)
    data['return_3'] = data['close'].pct_change(3)  # 增加3日收益率
    data['return_5'] = data['close'].pct_change(5)
    data['return_10'] = data['close'].pct_change(10)
    data['return_20'] = data['close'].pct_change(20)
    data['return_60'] = data['close'].pct_change(60)
    data['return_120'] = data['close'].pct_change(120)  # 增加120日收益率
    
    # 计算波动率
    data['volatility_5'] = data['return_1'].rolling(window=5).std() * np.sqrt(252)
    data['volatility_10'] = data['return_1'].rolling(window=10).std() * np.sqrt(252)
    data['volatility_20'] = data['return_1'].rolling(window=20).std() * np.sqrt(252)
    data['volatility_60'] = data['return_1'].rolling(window=60).std() * np.sqrt(252)
    data['volatility_120'] = data['return_1'].rolling(window=120).std() * np.sqrt(252)  # 增加120日波动率
    
    # 计算量价关系
    data['volume_change'] = data['volume'].pct_change(1)
    data['price_volume_corr'] = data['close'].rolling(window=20).corr(data['volume'])
    data['price_volume_corr_60'] = data['close'].rolling(window=60).corr(data['volume'])  # 增加60日量价相关
    
    # 计算价格变化速率
    data['price_change_rate'] = data['close'].diff(5) / data['close'].shift(5)  # 5日价格变化速率
    
    # 计算相对强弱指标RSI与价格的背离
    data['rsi_price_divergence'] = data['rsi'] - data['close'] / data['close'].rolling(window=20).mean()
    
    # 计算均线排列特征
    data['ma_5_above_20'] = (data['ma_5'] > data['ma_20']).astype(int)  # 5日均线上穿20日均线
    data['ma_20_above_60'] = (data['ma_20'] > data['ma_60']).astype(int)  # 20日均线上穿60日均线
    data['ma_60_above_120'] = (data['ma_60'] > data['ma_120']).astype(int)  # 60日均线上穿120日均线
    
    # 计算趋势方向的稳定性
    data['trend_stability'] = data['trend_strength'].rolling(window=20).std()  # 趋势强度的稳定性
    
    # 添加趋势类特征
    
    # 1. 趋势线突破特征
    # 计算价格与不同周期均线的突破
    data['price_break_above_ma5'] = (data['close'] > data['ma_5']) & (data['close'].shift(1) <= data['ma_5'].shift(1)).astype(int)
    data['price_break_below_ma5'] = (data['close'] < data['ma_5']) & (data['close'].shift(1) >= data['ma_5'].shift(1)).astype(int)
    data['price_break_above_ma20'] = (data['close'] > data['ma_20']) & (data['close'].shift(1) <= data['ma_20'].shift(1)).astype(int)
    data['price_break_below_ma20'] = (data['close'] < data['ma_20']) & (data['close'].shift(1) >= data['ma_20'].shift(1)).astype(int)
    data['price_break_above_ma60'] = (data['close'] > data['ma_60']) & (data['close'].shift(1) <= data['ma_60'].shift(1)).astype(int)
    data['price_break_below_ma60'] = (data['close'] < data['ma_60']) & (data['close'].shift(1) >= data['ma_60'].shift(1)).astype(int)
    
    # 2. 布林带突破特征
    data['price_break_above_bb'] = (data['close'] > data['bb_upper']).astype(int)
    data['price_break_below_bb'] = (data['close'] < data['bb_lower']).astype(int)
    
    # 3. 趋势强度特征
    # 计算趋势斜率，添加值范围限制
    def safe_slope(series, window):
        """安全计算斜率，处理异常值"""
        try:
            slope = np.polyfit(range(window), series, 1)[0]
            return slope
        except:
            return 0
    
    data['ma5_slope'] = data['ma_5'].rolling(window=5).apply(lambda x: safe_slope(x, 5))
    data['ma20_slope'] = data['ma_20'].rolling(window=10).apply(lambda x: safe_slope(x, 10))
    data['ma60_slope'] = data['ma_60'].rolling(window=20).apply(lambda x: safe_slope(x, 20))
    
    # 限制斜率值范围
    data['ma5_slope'] = data['ma5_slope'].clip(-100, 100)
    data['ma20_slope'] = data['ma20_slope'].clip(-50, 50)
    data['ma60_slope'] = data['ma60_slope'].clip(-20, 20)
    
    # 4. 价格形态特征
    # 计算最高价与最低价的比值，添加除以零保护和值范围限制
    data['high_low_ratio'] = (data['high'] / (data['low'].replace(0, 1e-8))).clip(1, 5)
    # 计算收盘价相对于最高价/最低价的位置，添加除以零保护
    price_range = data['high'] - data['low']
    price_range = price_range.replace(0, 1e-8)
    data['close_position'] = ((data['close'] - data['low']) / price_range).clip(0, 1)
    # 计算价格波动范围
    data['price_range'] = data['high'] - data['low']
    # 计算范围比率，添加除以零保护和值范围限制
    prev_close = data['close'].shift(1).replace(0, 1e-8)
    data['range_ratio'] = (data['price_range'] / prev_close).clip(0, 1)
    
    # 5. 支撑位和阻力位相关特征
    # 计算最近20天的最高价和最低价作为临时阻力位和支撑位
    data['resistance_20d'] = data['high'].rolling(window=20).max()
    data['support_20d'] = data['low'].rolling(window=20).min()
    # 计算价格与支撑阻力位的距离，添加除以零保护和值范围限制
    data['distance_to_resistance'] = ((data['resistance_20d'] - data['close']) / (data['close'].replace(0, 1e-8))).clip(0, 1)
    data['distance_to_support'] = ((data['close'] - data['support_20d']) / (data['close'].replace(0, 1e-8))).clip(0, 1)
    
    # 6. 动量震荡指标
    # 更安全的动量振荡器计算，添加除以零保护和值范围限制
    momentum_12 = data['momentum_12'].replace(0, 1e-8)  # 替换0为很小的值
    momentum_osc = (data['momentum_3'] - momentum_12) / momentum_12 * 100
    data['momentum_oscillator'] = momentum_osc.clip(-200, 200)  # 限制值范围在-200到200之间
    
    # 7. 量价配合特征
    # 上涨/下跌时的成交量变化
    data['volume_on_up'] = data['volume'] * (data['close'] > data['open']).astype(int)
    data['volume_on_down'] = data['volume'] * (data['close'] < data['open']).astype(int)
    # 更安全的成交量比率计算，添加值范围限制
    data['volume_ratio'] = (data['volume_on_up'] / (data['volume_on_down'] + 1e-8)).clip(0, 10)  # 限制在0到10之间
    
    # 8. 相对强弱指标的趋势
    data['rsi_trend'] = data['rsi'].rolling(window=10).mean()
    data['rsi_overbought'] = (data['rsi'] > 70).astype(int)
    data['rsi_oversold'] = (data['rsi'] < 30).astype(int)
    
    # 处理缺失值
    data = data.dropna()
    
    # 全面清理数据，处理无限值和过大值
    # 替换无限值为NaN
    data = data.replace([np.inf, -np.inf], np.nan)
    
    # 使用中位数填充NaN值
    for col in data.columns:
        if data[col].dtype in ['float64', 'int64']:
            data[col] = data[col].fillna(data[col].median())
    
    # 限制所有数值列的值范围
    for col in data.columns:
        if data[col].dtype in ['float64', 'int64']:
            # 计算列的上下限（基于四分位数）
            Q1 = data[col].quantile(0.01)
            Q3 = data[col].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # 限制值范围
            data[col] = data[col].clip(lower_bound, upper_bound)
    
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
                # 使用训练数据训练模型
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
                
                # 调试：打印信号
                logger.debug(f"信号 - {base_symbol}: {signal}")
            
            # 只处理有信号的品种
            if signal == 1 or signal == -1:
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
    
    # 创建模型管理器，使用LightGBM模型
    model_manager = ModelManager(model_type='lightgbm')
    
    # 生成每日目标头寸
    generate_daily_target_positions(model_manager, all_data, START_DATE, CAPITAL)
    
    logger.info("随机森林策略运行完成")


if __name__ == "__main__":
    main()
