import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

from feature_engineering import create_lstm_features, create_atr_features, create_market_state_features
from risk_allocation import calculate_dynamic_atr, calculate_risk_adjusted_position, apply_position_constraints
from utility.instrument_utils import get_contract_multiplier

logger = logging.getLogger(__name__)

def calculate_target_positions(model_manager, all_data, start_date, capital, output_dir, risk_per_trade=0.02):
    """生成每日目标头寸"""
    logger.info(f"开始生成每日目标头寸，起始日期: {start_date}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有交易日期
    all_dates = []
    for data in all_data.values():
        all_dates.extend(data.index.tolist())
    all_dates = sorted(list(set(all_dates)))
    all_dates = [date for date in all_dates if date >= datetime.strptime(start_date, '%Y-%m-%d')]
    
    # 定义特征列表
    features = [
        'return_1', 'rsi', 'macd', 'bb_width', 'volatility_20',
        'atr_14', 'atr_ratio', 'state_0', 'state_1', 'state_2',
        'return_5', 'high_low_diff', 'open_close_diff', 'realized_vol',
        'ma_5', 'ma_20', 'ma_50', 'ma_std_5', 'ma_std_20', 'ma_std_50'
    ]
    
    # 遍历每个交易日
    for date in all_dates:
        logger.info(f"\n处理日期: {date.strftime('%Y-%m-%d')}")
        
        # 初始化每日目标头寸和品种数据
        daily_target_positions = []
        varieties_data = {}
        
        # 第一步：预处理数据并生成特征
        for base_symbol, data in all_data.items():
            # 检查该品种在该日期是否有数据
            if date not in data.index:
                continue
            
            # 获取该品种在该日期之前的数据
            past_data = data[data.index <= date]
            
            # 检查数据量是否足够：需要至少360天数据
            if len(past_data) < 360:
                continue
            
            # 1. 创建LSTM特征
            lstm_featured_data = create_lstm_features(past_data.copy())
            
            # 2. 创建ATR特征
            atr_featured_data = create_atr_features(lstm_featured_data.copy())
            
            # 3. 创建市场状态特征
            final_data = create_market_state_features(atr_featured_data.copy())
            
            # 检查是否需要重新训练模型
            if model_manager.should_retrain(base_symbol):
                logger.info(f"训练{base_symbol}模型...")
                model_manager.train_model(base_symbol, final_data, features)
            
            # 预测
            pred = model_manager.predict(base_symbol, final_data, features)
            if pred is None:
                continue
            
            # 获取当前数据
            current_data = final_data.loc[date]
            current_price = current_data['close']
            contract_symbol = data.loc[date]['symbol']
            
            # 获取合约乘数和保证金率
            contract_multiplier, margin_rate = get_contract_multiplier(contract_symbol)
            
            # 计算动态ATR
            dynamic_atr = calculate_dynamic_atr(final_data).loc[date]
            
            # 收集品种数据
            varieties_data[base_symbol] = {
                'current_price': current_price,
                'atr': dynamic_atr,
                'contract_multiplier': contract_multiplier,
                'margin_rate': margin_rate,
                'contract_symbol': contract_symbol,
                'pred': pred,
                'atr_14': current_data['atr_14'],
                'atr_quantile': current_data['atr_14_quantile'],
                'final_data': final_data
            }
        
        # 第二步：生成信号
        if varieties_data:
            # 生成组合信号
            from signal_generation import generate_combined_signals
            signals = generate_combined_signals(model_manager, all_data, features, date)
            
            # 第三步：计算风险调整后的头寸
            positions = {}
            for symbol, signal_info in signals.items():
                if symbol not in varieties_data:
                    continue
                
                data = varieties_data[symbol]
                signal = signal_info['signal']
                pred_volatility = data['pred']['volatility']
                
                # 获取实际波动率（最近20日）
                actual_volatility = data['final_data']['volatility_20'].loc[date]
                
                # 计算风险调整后的头寸
                position = calculate_risk_adjusted_position(
                    capital=capital,
                    signal=signal,
                    atr=data['atr'],
                    current_price=data['current_price'],
                    contract_multiplier=data['contract_multiplier'],
                    margin_rate=data['margin_rate'],
                    risk_per_trade=risk_per_trade,
                    pred_volatility=pred_volatility,
                    actual_volatility=actual_volatility
                )
                
                positions[symbol] = position
            
            # 第四步：应用头寸约束
            adjusted_positions = apply_position_constraints(
                positions=positions,
                varieties_data=varieties_data,
                total_capital=capital
            )
            
            # 第五步：生成最终头寸
            for symbol, position in adjusted_positions.items():
                if position == 0:
                    continue
                
                data = varieties_data[symbol]
                signal_info = signals[symbol]
                
                # 计算持仓价值
                position_value = abs(position) * data['current_price'] * data['contract_multiplier']
                
                # 计算保证金占用
                margin_usage = position_value * data['margin_rate']
                
                # 构建头寸字典
                position_dict = {
                    'symbol': data['contract_symbol'],
                    'current_price': data['current_price'],
                    'contract_multiplier': data['contract_multiplier'],
                    'position_size': position,
                    'position_value': position_value,
                    'margin_usage': margin_usage,
                    'risk_amount': margin_usage,
                    'margin_rate': data['margin_rate'],
                    'total_capital': capital,
                    'signal': signal_info['signal'],
                    'confidence': signal_info['confidence'],
                    'model_type': 'lstm_atr_strategy',
                    'market_value': position_value if position > 0 else -position_value,
                    'atr': data['atr'],
                    'predicted_direction': data['pred']['direction'],
                    'predicted_volatility': data['pred']['volatility'],
                    'up_probability': data['pred']['probability'][0],
                    'down_probability': data['pred']['probability'][1]
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
