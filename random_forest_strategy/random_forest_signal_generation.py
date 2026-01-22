# -*- coding: utf-8 -*-
"""
随机森林信号生成模块：
负责随机森林模型的训练、预测和信号生成
"""
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import confusion_matrix

# 导入随机森林模型和训练器
from random_forest_strategy.models.random_forest import RandomForestModel
from random_forest_strategy.trade_model.random_forest_trainer import RandomForestTrainer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


def preprocess_data(data):
    """预处理数据"""
    import numpy as np
    from utility.calc_funcs import calculate_ma, calculate_macd, calculate_rsi, calculate_bollinger_bands, calculate_atr, calculate_volume_weighted_average_price
    from utility.mom import calculate_momentum
    
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


def generate_signals(model_manager, preprocessed_data, date, base_symbol, predict_counts, PREDICT_INTERVAL):
    """生成单个品种的信号
    
    参数：
    model_manager: 模型管理器实例
    preprocessed_data: 预处理后的数据
    date: 当前日期
    base_symbol: 品种代码
    predict_counts: 预测次数计数器
    PREDICT_INTERVAL: 预测间隔
    
    返回：
    tuple: (signal, trend_strength) or (None, None) if no signal generated
    """
    # 筛选出用于训练的数据（不包括当天）
    training_data = preprocessed_data[preprocessed_data.index < date]
    
    if training_data.empty:
        return None, None
    
    # 获取模型
    trainer = model_manager.get_model(base_symbol)
    
    # 检查是否需要训练模型
    if trainer.model is None or predict_counts[base_symbol] >= PREDICT_INTERVAL:
        logger.info(f"训练{base_symbol}模型...")
        # 确保至少有360个交易日数据用于训练
        if len(training_data) < 360:
            logger.warning(f"{base_symbol}训练数据不足360个交易日，当前仅有{len(training_data)}个交易日，跳过训练")
            return None, None
        # 使用训练数据训练模型（历史开始点到生成头寸前一天）
        model = model_manager.train_model(base_symbol, training_data)
        if model is None:
            logger.warning(f"{base_symbol}模型训练失败，跳过该品种")
            return None, None
        predict_counts[base_symbol] = 0
    
    # 检查模型是否已训练
    if trainer.model is None:
        logger.warning(f"{base_symbol}模型尚未训练，跳过该品种")
        return None, None
    
    # 筛选出用于预测的数据（包括当天）
    predict_data = preprocessed_data[preprocessed_data.index <= date]
    
    if predict_data.empty:
        return None, None
    
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
    
    return signal, trend_strength