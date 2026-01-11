import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import logging

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from random_forest_strategy.random_forest_main import preprocess_data
from models.random_forest import RandomForestModel
from trade_model.random_forest_trainer import RandomForestTrainer
from utility.data_process import clean_data, normalize_data, standardize_data
from utility.calc_funcs import calculate_ma, calculate_macd, calculate_rsi, calculate_bollinger_bands, calculate_atr, calculate_volume_weighted_average_price
from utility.long_short_signals import generate_combined_signal, generate_ma_crossover_signal, generate_macd_signal, generate_rsi_signal, generate_bollinger_bands_signal
from utility.mom import generate_cross_sectional_momentum_signal, calculate_momentum, generate_momentum_signal

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义常量
DATA_DIR = 'History_Data/hot_daily_market_data'
START_DATE = '2025-10-01'
AG_SYMBOL = 'ag'


def load_ag_data(data_dir):
    """加载ag品种的历史数据"""
    ag_data = {}
    for filename in os.listdir(data_dir):
        # 不区分大小写检查ag符号
        if AG_SYMBOL.upper() in filename.upper() and filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            # 读取CSV文件，第一列作为日期索引
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            # 提取基础品种名称并转换为小写（如ag）
            base_symbol = filename.split('.')[0].split('_')[0].lower()
            if base_symbol not in ag_data:
                ag_data[base_symbol] = data
            else:
                ag_data[base_symbol] = pd.concat([ag_data[base_symbol], data])
    
    # 按日期排序
    for symbol, data in ag_data.items():
        ag_data[symbol] = data.sort_index()
    
    return ag_data


def analyze_ag_predictions():
    """分析ag品种的每日预测"""
    logger.info("开始分析ag品种的每日预测")
    
    # 加载ag数据
    ag_data = load_ag_data(DATA_DIR)
    if not ag_data:
        logger.error("未找到ag品种数据")
        return
    
    # 获取完整ag数据
    full_data = ag_data[AG_SYMBOL]
    
    # 获取START_DATE之后的数据用于分析
    analysis_data = full_data[full_data.index >= START_DATE]
    if analysis_data.empty:
        logger.error(f"{START_DATE}之后没有ag数据")
        return
    
    # 创建模型
    trainer = RandomForestTrainer()
    
    # 准备结果列表
    results = []
    
    # 获取所有分析日期
    trade_dates = analysis_data.index.unique()
    
    for date in trade_dates:
        logger.info(f"分析{date.strftime('%Y-%m-%d')}的ag预测")
        
        # 获取截止到前一天的完整历史数据用于训练
        training_data = full_data[full_data.index < date]
        if len(training_data) < 360:
            logger.warning(f"{date}前的历史数据不足360条，跳过")
            continue
        
        # 预处理训练数据
        processed_training_data = preprocess_data(training_data.copy())
        if processed_training_data.empty:
            logger.warning(f"{date}的训练数据预处理失败，跳过")
            continue
        
        # 训练模型
        model = trainer.train(processed_training_data)
        if model is None:
            logger.warning(f"{date}的模型训练失败，跳过")
            continue
        
        # 获取包含当天数据的完整数据，用于预测
        prediction_data = full_data[full_data.index <= date]
        full_processed_data = preprocess_data(prediction_data.copy())
        if full_processed_data.empty:
            logger.warning(f"{date}的完整数据预处理失败，跳过")
            continue
        
        # 使用最新数据（当天）进行预测
        latest_data = full_processed_data.iloc[-1:]
        prediction = trainer.predict(latest_data)
        
        # 获取预测结果
        if prediction is None:
            signal = 0
        else:
            signal = prediction[0]
        
        # 获取当天的原始行情数据
        current_data = analysis_data.loc[date]
        
        # 获取模型特征 - 与base_model.py保持一致，加强趋势类特征
        feature_columns = ['open', 'high', 'low', 'close', 'volume',
                          # 趋势类特征（加强权重）
                          'ma_5', 'ma_20', 'ma_60',
                          'days_above_ma5', 'days_above_ma20', 'days_above_ma60',
                          'price_ma5_diff', 'price_ma20_diff', 'price_ma60_diff',
                          'trend_strength', 'trend_strength_ma', 'trend_strength_ema',
                          'price_slope_5', 'price_slope_20',
                          'is_strong_up_trend', 'is_strong_down_trend', 'is_sideways',
                          'adx', 'di_plus', 'di_minus',
                          # 核心指标，移除RSI和布林带等超买超卖指标
                          'vwap', 'momentum', 'atr',
                          # 成交量相关特征
                          'volume_ma5', 'volume_ma20', 'volume_ma60',
                          'volume_ema5', 'volume_ema20',
                          'volume_ratio_5', 'volume_ratio_10',
                          'volume_change', 'price_volume_corr',
                          # 收益和波动率特征
                          'return_5', 'return_10', 'return_20', 'return_60',
                          'volatility_10', 'volatility_20', 'volatility_60']
        
        # 提取特征值
        features = latest_data[feature_columns].iloc[0].to_dict()
        
        # 计算实际趋势（5天后的涨跌）
        future_dates = analysis_data[analysis_data.index > date].index
        actual_trend = 0
        if len(future_dates) >= 5:
            next_date = future_dates[4]  # 取第5个未来日期（索引从0开始）
            next_close = analysis_data.loc[next_date]['close']
            current_close = current_data['close']
            actual_trend = np.sign(next_close - current_close)
        
        # 构建结果字典
        result = {
            'trade_date': date.strftime('%Y-%m-%d'),
            'symbol': current_data['symbol'],
            'open': current_data['open'],
            'high': current_data['high'],
            'low': current_data['low'],
            'close': current_data['close'],
            'volume': current_data['volume'],
            'predicted_signal': signal,
            'actual_trend': actual_trend,
            'signal_correct': 1 if signal == actual_trend else 0
        }
        
        # 添加特征值
        result.update(features)
        
        results.append(result)
    
    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存到CSV文件，使用新文件名避免权限问题
    output_file = 'ag_prediction_analysis_trend_strength.csv'
    results_df.to_csv(output_file, index=False, encoding='gbk')
    logger.info(f"分析结果已保存到{output_file}")
    
    return results_df


if __name__ == "__main__":
    analyze_ag_predictions()