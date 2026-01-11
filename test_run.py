import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
from sklearn.metrics import confusion_matrix
from position import calculate_position_size, get_contract_multiplier, calculate_portfolio_metrics
from utility.data_process import clean_data, normalize_data, standardize_data
from utility.calc_funcs import calculate_ma, calculate_macd, calculate_rsi, calculate_bollinger_bands, calculate_atr, calculate_volume_weighted_average_price
from utility.long_short_signals import generate_combined_signal, generate_ma_crossover_signal, generate_macd_signal, generate_rsi_signal, generate_bollinger_bands_signal
from utility.mom import generate_cross_sectional_momentum_signal, calculate_momentum, generate_momentum_signal

# 导入模型训练器
from trade_model.random_forest_trainer import RandomForestTrainer
from trade_model.lightgbm_trainer import LightGBMTrainer
from trade_model.xgboost_trainer import XGBoostTrainer
from trade_model.lstm_attention_trainer import LSTMAttentionTrainer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模型缓存，键为品种，值为模型对象
model_cache = {}
PREDICT_INTERVAL = 50  # 每50次预测后重新训练模型

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
            return RandomForestModel(self.params)
        elif self.model_type in ['lightgbm', 'xgboost']:
            return BoostingModel(model_type=self.model_type, params=self.params)
        elif self.model_type == 'lstm_attention':
            return LSTMAttentionModel(self.params)
        else:
            raise ValueError(f"不支持的模型类型：{self.model_type}")
    
    def train_model(self, symbol, X, y):
        """训练指定品种的模型
        
        参数：
        symbol: 品种代码
        X: 特征数据
        y: 标签数据
        
        返回：
        model: 训练好的模型
        """
        model = self.get_model(symbol)
        model.train(X, y)
        return model
    
    def predict(self, symbol, X):
        """预测指定品种的结果
        
        参数：
        symbol: 品种代码
        X: 特征数据
        
        返回：
        prediction: 预测结果
        """
        model = self.get_model(symbol)
        return model.predict(X)
    
    def should_retrain(self, symbol):
        """判断是否需要重新训练模型
        
        参数：
        symbol: 品种代码
        
        返回：
        bool: 是否需要重新训练
        """
        if symbol not in self.models:
            return True
        return self.models[symbol].should_retrain(PREDICT_INTERVAL)
    
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

# 创建模型管理器实例
model_manager = ModelManager(model_type='random_forest')

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
        print(f"  警告：数据记录条数不足60条，无法生成所有指标，跳过该品种")
        return None
    
    # 清洗数据
    cleaned_data = clean_data(data)
    
    # 保存原始价格数据用于计算仓位大小
    original_prices = cleaned_data[['close']].copy()
    
    # 计算技术指标
    # 移动平均线
    cleaned_data['ma_5'] = calculate_ma(cleaned_data, 5)
    cleaned_data['ma_20'] = calculate_ma(cleaned_data, 20)
    cleaned_data['ma_60'] = calculate_ma(cleaned_data, 60)
    
    # MACD指标
    macd, signal, histogram = calculate_macd(cleaned_data)
    cleaned_data['macd'] = macd
    cleaned_data['macd_signal'] = signal
    cleaned_data['macd_histogram'] = histogram
    
    # RSI指标
    cleaned_data['rsi'] = calculate_rsi(cleaned_data)
    
    # 布林带
    upper_band, middle_band, lower_band = calculate_bollinger_bands(cleaned_data)
    cleaned_data['bb_upper'] = upper_band
    cleaned_data['bb_middle'] = middle_band
    cleaned_data['bb_lower'] = lower_band
    
    # ATR指标（使用实际值，不进行归一化）
    cleaned_data['atr'] = calculate_atr(cleaned_data, 14)
    
    # 成交量加权平均价格
    cleaned_data['vwap'] = calculate_volume_weighted_average_price(cleaned_data, 14)
    
    # 动量指标
    cleaned_data = calculate_momentum(cleaned_data, 12)
    
    # 生成多个信号
    ma_signal = generate_ma_crossover_signal(cleaned_data.copy())
    macd_signal = generate_macd_signal(cleaned_data.copy())
    rsi_signal = generate_rsi_signal(cleaned_data.copy())
    bb_signal = generate_bollinger_bands_signal(cleaned_data.copy())
    combined_signal = generate_combined_signal(cleaned_data.copy())
    
    # 合并信号
    cleaned_data['signal_ma'] = ma_signal['signal']
    cleaned_data['signal_macd'] = macd_signal['signal']
    cleaned_data['signal_rsi'] = rsi_signal['signal']
    cleaned_data['signal_bb'] = bb_signal['signal']
    cleaned_data['signal_combined'] = combined_signal['signal']
    
    # 归一化处理 - 只对用于机器学习的特征进行归一化，保留实际价格
    # 选择需要归一化的列（不包括close和atr，这些用于计算仓位大小）
    columns_to_normalize = ['open', 'high', 'low', 'volume', 
                           'ma_5', 'ma_20', 'ma_60', 
                           'macd', 'macd_signal', 'macd_histogram', 
                           'rsi', 'bb_upper', 'bb_middle', 'bb_lower', 
                           'vwap', 'momentum']
    
    # 归一化数据
    normalized_data = normalize_data(cleaned_data, columns=columns_to_normalize)
    
    # 将原始收盘价和实际ATR添加回数据中
    normalized_data['actual_close'] = original_prices['close']
    normalized_data['actual_atr'] = cleaned_data['atr']
    
    return normalized_data


def train_machine_learning_model(data, model_type='random_forest', params=None):
    """训练机器学习模型，使用统一的模型接口
    
    参数：
    data: 原始数据
    model_type: 模型类型
    params: 模型参数
    
    返回：
    model: 训练好的模型
    """
    # 创建模型训练器实例
    if model_type == 'random_forest':
        trainer = RandomForestTrainer(params)
    elif model_type == 'lightgbm':
        trainer = LightGBMTrainer(params)
    elif model_type == 'xgboost':
        trainer = XGBoostTrainer(params)
    elif model_type == 'lstm_attention':
        trainer = LSTMAttentionTrainer(params)
    else:
        raise ValueError(f"不支持的模型类型：{model_type}")
    
    # 训练模型
    model = trainer.train(data)
    
    return model


def calculate_target_positions(data_dict, capital, risk_per_trade, target_date, model_type='random_forest'):
    """计算目标仓位，使用统一的模型接口，避免数据泄露
    
    参数：
    data_dict: 原始数据字典
    capital: 总资金
    risk_per_trade: 每笔交易风险比例
    target_date: 目标日期
    model_type: 模型类型
    
    返回：
    positions_df: 目标仓位数据框
    """
    # 筛选目标日期及之前的数据
    filtered_data_dict = {}
    for symbol, data in data_dict.items():
        filtered_data = data[data.index <= target_date]
        if not filtered_data.empty:
            filtered_data_dict[symbol] = filtered_data
    
    if not filtered_data_dict:
        logger.warning(f"没有{target_date}之前的数据")
        return pd.DataFrame()
    
    # 预处理每个品种的数据
    processed_data_dict = {}
    for symbol, data in filtered_data_dict.items():
        logger.info(f"正在预处理品种 {symbol}...")
        processed_data = preprocess_data(data)
        if processed_data is not None:  # 只有有效的预处理数据才会被使用
            processed_data_dict[symbol] = processed_data
            logger.info(f"  成功预处理品种 {symbol}")
        else:
            logger.warning(f"  预处理失败，跳过品种 {symbol}")
    
    # 打印处理结果统计
    logger.info(f"预处理完成，成功处理 {len(processed_data_dict)}/{len(filtered_data_dict)} 个品种")
    
    # 打印CZCE品种的处理情况
    czce_processed = [symbol for symbol in processed_data_dict.keys() if get_exchange(symbol) == 'CZCE']
    czce_total = [symbol for symbol in filtered_data_dict.keys() if get_exchange(symbol) == 'CZCE']
    logger.info(f"CZCE品种处理情况：成功 {len(czce_processed)}/{len(czce_total)} 个")
    if czce_total:
        logger.info(f"  CZCE品种列表：{czce_total}")
    if czce_processed:
        logger.info(f"  成功处理的CZCE品种：{czce_processed}")
    else:
        logger.warning(f"  没有成功处理任何CZCE品种")
    
    if not processed_data_dict:
        logger.warning(f"没有有效的预处理数据")
        return pd.DataFrame()
    
    # 生成横截面动量信号
    cross_sectional_signals = generate_cross_sectional_momentum_signal(processed_data_dict)
    
    # 计算每个品种的目标仓位
    positions = []
    
    # 使用指定模型计算每个品种的仓位
    for base_symbol, data in processed_data_dict.items():
        try:
            # 获取最新数据
            latest_data = data.iloc[-1]
            
            # 获取当前合约代码
            symbol = latest_data['symbol']
            
            # 准备特征列
            feature_columns = ['open', 'high', 'low', 'close', 'volume', 
                              'ma_5', 'ma_20', 'ma_60', 
                              'macd', 'macd_signal', 'macd_histogram', 
                              'rsi', 'bb_upper', 'bb_middle', 'bb_lower', 
                              'vwap', 'momentum',
                              'signal_ma', 'signal_macd', 'signal_rsi', 'signal_bb', 'signal_combined']
            
            # 生成信号
            signal = 0
            
            # 检查缓存中是否有该品种的模型，以及是否需要重新训练
            cache_key = f"{base_symbol}_{model_type}"
            model = None
            if cache_key in model_cache:
                model = model_cache[cache_key]
            
            # 如果没有模型，或者预测次数达到阈值，重新训练模型
            if model is None or model.should_retrain(PREDICT_INTERVAL):
                # 训练机器学习模型
                model = train_machine_learning_model(data, model_type)
                logger.info(f"重新训练模型 {model_type} 用于品种 {base_symbol}")
            
            # 使用机器学习模型预测
            if model is not None:
                # 使用DataFrame而不是numpy数组，保留特征名
                latest_features_df = pd.DataFrame([latest_data[feature_columns]])
                ml_signal = model.predict(latest_features_df)[0]
                signal = ml_signal
                # 更新模型缓存
                model_cache[cache_key] = model
            else:
                # 回退到横截面动量信号
                signal = cross_sectional_signals.get(base_symbol, 0)
                logger.warning(f"品种 {base_symbol} 模型训练失败，使用横截面动量信号")
            
            # 结合多个信号
            combined_signal = signal
            
            if combined_signal == 0:
                continue  # 不持仓
            
            # 使用实际价格和实际ATR计算仓位
            current_price = latest_data['actual_close'].item() if hasattr(latest_data['actual_close'], 'item') else latest_data['actual_close']
            atr = latest_data['actual_atr'].item() if hasattr(latest_data['actual_atr'], 'item') else latest_data['actual_atr']
            
            # 使用ATR作为止损
            stop_loss_price = current_price - atr if combined_signal == 1 else current_price + atr
            
            # 计算仓位大小
            position_size, risk_amount = calculate_position_size(
                capital=capital,
                risk_per_trade=risk_per_trade,
                entry_price=current_price,
                stop_loss_price=stop_loss_price,
                symbol=symbol
            )
            
            # 根据信号调整仓位方向
            position_size = position_size * combined_signal
            
            # 获取合约乘数和保证金率
            contract_multiplier, margin_rate = get_contract_multiplier(symbol)
            
            # 计算持仓价值和保证金占用
            position_value = abs(position_size) * current_price * contract_multiplier
            margin_usage = abs(position_value * margin_rate)  # 使用绝对值
            
            positions.append({
                'symbol': symbol,
                'current_price': current_price,
                'stop_loss_price': stop_loss_price,
                'contract_multiplier': contract_multiplier,
                'position_size': position_size,
                'position_value': position_value,
                'margin_usage': margin_usage,
                'risk_amount': risk_amount,
                'margin_rate': margin_rate,
                'total_capital': capital,
                'signal': combined_signal,
                'model_type': model_type
            })
        except Exception as e:
            logger.error(f"处理品种 {base_symbol} 时出错: {str(e)}")
            continue
    
    positions_df = pd.DataFrame(positions)
    return positions_df


def generate_daily_positions(data_dict, capital, risk_per_trade, start_date):
    """生成从开始日期到最后一天的每日目标仓位，包含模型性能对比和可视化
    
    参数：
    data_dict: 原始数据字典
    capital: 总资金
    risk_per_trade: 每笔交易风险比例
    start_date: 开始日期
    """
    # 模型类型列表
    model_types = ['random_forest', 'lightgbm', 'xgboost', 'lstm_attention']
    
    # 获取所有日期并排序
    all_dates = []
    for data in data_dict.values():
        all_dates.extend(data.index.tolist())
    all_dates = sorted(list(set(all_dates)))
    
    # 筛选从开始日期之后的日期
    target_dates = [date for date in all_dates if date >= pd.to_datetime(start_date)]
    
    if not target_dates:
        logger.warning(f"没有{start_date}之后的数据日期")
        return
    
    logger.info(f"开始生成从{start_date}到{target_dates[-1].strftime('%Y-%m-%d')}的每日目标仓位...")
    
    # 用于模型性能对比的字典
    model_performance_history = {}
    
    # 为每个模型类型生成目标仓位
    for model_type in model_types:
        logger.info(f"\n===== 使用模型 {model_type} 生成目标仓位 =====")
        
        # 创建模型特定的输出目录
        model_output_dir = os.path.join(OUTPUT_DIR, model_type)
        os.makedirs(model_output_dir, exist_ok=True)
        
        daily_metrics = []
        
        # 为每个日期生成目标仓位
        for date in target_dates:
            date_str = date.strftime('%Y-%m-%d')
            logger.info(f"正在处理{date_str}...")
            
            # 计算目标仓位
            positions_df = calculate_target_positions(data_dict, capital, risk_per_trade, date, model_type)
            
            if positions_df.empty:
                logger.warning(f"{date_str}没有生成任何仓位")
                continue
            
            # 计算投资组合指标
            portfolio_metrics = calculate_portfolio_metrics(positions_df)
            
            # 添加market_value列
            positions_df['market_value'] = positions_df['position_value']
            
            # 生成文件名
            output_file = os.path.join(model_output_dir, f'target_positions_{date.strftime("%Y%m%d")}.csv')
            
            # 保存为CSV文件
            positions_df.to_csv(output_file, index=False)
            logger.info(f"  已保存到{output_file}")
            
            # 打印投资组合指标
            logger.info(f"  投资组合指标：")
            for key, value in portfolio_metrics.items():
                logger.info(f"    {key}: {value:.2f}")
            
            # 记录每日指标
            daily_metrics.append({
                'date': date,
                'total_position_value': portfolio_metrics.get('total_position_value', 0),
                'total_margin_usage': portfolio_metrics.get('total_margin_usage', 0),
                'total_risk_amount': portfolio_metrics.get('total_risk_amount', 0),
                'position_count': len(positions_df)
            })
            
            # 记录模型性能
            if model_type not in model_performance_history:
                model_performance_history[model_type] = []
            
            model_performance_history[model_type].append({
                'date': date,
                'position_count': len(positions_df),
                'avg_position_size': positions_df['position_size'].abs().mean() if len(positions_df) > 0 else 0,
                'total_position_value': positions_df['position_value'].sum() if len(positions_df) > 0 else 0
            })
        
        # 投资组合指标可视化（每个模型单独生成）
        visualize_portfolio_metrics(daily_metrics, model_type)
    
    # 模型性能对比可视化（所有模型一起生成）
    visualize_model_performance(model_performance_history, target_dates)
    
    logger.info("\n所有模型的每日目标仓位生成完成！")

def visualize_model_performance(model_performance_history, target_dates):
    """可视化不同模型的性能对比
    
    参数：
    model_performance_history: 模型性能历史数据
    target_dates: 目标日期列表
    """
    try:
        import matplotlib.pyplot as plt
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 绘制模型仓位数量对比
        plt.figure(figsize=(12, 6))
        
        for model_type, performance in model_performance_history.items():
            dates = [p['date'] for p in performance]
            position_counts = [p['position_count'] for p in performance]
            plt.plot(dates, position_counts, label=model_type)
        
        plt.title('不同模型的仓位数量对比')
        plt.xlabel('日期')
        plt.ylabel('仓位数量')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # 保存图表到总目录
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'model_position_count_comparison.png'))
        plt.close()
        
        # 绘制模型持仓价值对比
        plt.figure(figsize=(12, 6))
        
        for model_type, performance in model_performance_history.items():
            dates = [p['date'] for p in performance]
            total_values = [p['total_position_value'] for p in performance]
            plt.plot(dates, total_values, label=model_type)
        
        plt.title('不同模型的持仓价值对比')
        plt.xlabel('日期')
        plt.ylabel('持仓价值 (元)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # 保存图表到总目录
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'model_position_value_comparison.png'))
        plt.close()
        
        logger.info("模型性能对比图表生成完成")
    except Exception as e:
        logger.error(f"生成模型性能对比图表时出错: {str(e)}")

def visualize_portfolio_metrics(daily_metrics, model_type=None):
    """可视化投资组合指标
    
    参数：
    daily_metrics: 每日指标数据
    model_type: 模型类型，用于确定保存路径
    """
    try:
        import matplotlib.pyplot as plt
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 转换为DataFrame
        metrics_df = pd.DataFrame(daily_metrics)
        metrics_df.set_index('date', inplace=True)
        
        # 确定保存目录
        if model_type:
            save_dir = os.path.join(OUTPUT_DIR, model_type)
        else:
            save_dir = OUTPUT_DIR
        
        # 绘制投资组合总价值
        plt.figure(figsize=(12, 6))
        plt.plot(metrics_df.index, metrics_df['total_position_value'], label='总持仓价值')
        plt.plot(metrics_df.index, metrics_df['total_margin_usage'], label='总保证金占用')
        
        title_suffix = f' - {model_type}' if model_type else ''
        plt.title(f'投资组合总价值变化{title_suffix}')
        plt.xlabel('日期')
        plt.ylabel('金额 (元)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # 保存图表
        plt.tight_layout()
        filename = f'portfolio_total_value{"_" + model_type if model_type else ""}.png'
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()
        
        # 绘制每日仓位数量
        plt.figure(figsize=(12, 6))
        plt.bar(metrics_df.index, metrics_df['position_count'], alpha=0.7)
        plt.title(f'每日仓位数量变化{title_suffix}')
        plt.xlabel('日期')
        plt.ylabel('仓位数量')
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # 保存图表
        plt.tight_layout()
        filename = f'daily_position_count{"_" + model_type if model_type else ""}.png'
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()
        
        logger.info(f"投资组合指标图表生成完成{title_suffix}")
    except Exception as e:
        logger.error(f"生成投资组合指标图表时出错: {str(e)}")


def main():
    # 加载所有数据
    print("正在加载历史数据...")
    all_data = load_all_data(DATA_DIR)
    
    # 生成每日目标仓位
    generate_daily_positions(all_data, CAPITAL, RISK_PER_TRADE, START_DATE)


if __name__ == "__main__":
    main()