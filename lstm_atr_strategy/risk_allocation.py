import numpy as np
import pandas as pd


def calculate_dynamic_atr(data, windows=[14, 20, 50], weights=[0.5, 0.3, 0.2]):
    """计算动态ATR，使用多窗口加权平均"""
    # 计算各窗口ATR
    atr_values = []
    for window in windows:
        # 计算TR (真实波动幅度)
        tr = np.max([
            data['high'] - data['low'],
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        ], axis=0)
        
        # 计算ATR
        atr = tr.rolling(window=window).mean()
        atr_values.append(atr)
    
    # 加权平均
    dynamic_atr = sum(w * atr for w, atr in zip(weights, atr_values))
    
    return dynamic_atr


def calculate_risk_adjusted_position(capital, signal, atr, current_price, contract_multiplier, margin_rate, 
                                    risk_per_trade=0.02, pred_volatility=None, actual_volatility=None):
    """计算风险调整后的头寸规模"""
    # 核心公式：头寸规模 = (总资金 * 单笔风险比例) / (ATR * 合约乘数)
    base_position = (capital * risk_per_trade) / (atr * contract_multiplier)
    
    # 根据预测波动率与实际波动率的差异调整风险比例
    if pred_volatility is not None and actual_volatility is not None and actual_volatility > 0:
        # 波动率调整因子：如果预测波动率高于实际波动率，增加风险暴露，反之减少
        volatility_factor = min(max(pred_volatility / actual_volatility, 0.5), 2.0)
    else:
        volatility_factor = 1.0
    
    # 应用波动率调整
    adjusted_position = base_position * volatility_factor
    
    # 根据信号方向和强度调整
    final_position = adjusted_position * signal
    
    # 转换为整数手数
    final_position = int(round(final_position))
    
    return final_position


def calculate_atr_based_allocation(total_capital, varieties_data, risk_per_trade=0.02):
    """基于ATR的资金分配"""
    allocation = {}
    
    # 计算总ATR风险
    total_atr_risk = sum(data['atr'] * data['contract_multiplier'] for data in varieties_data.values())
    
    if total_atr_risk <= 0:
        # 等权分配
        n_symbols = len(varieties_data)
        return {symbol: total_capital / n_symbols for symbol in varieties_data.keys()}
    
    for symbol, data in varieties_data.items():
        # 计算单个品种的风险占比
        risk_ratio = (data['atr'] * data['contract_multiplier']) / total_atr_risk
        
        # 分配资金
        allocation[symbol] = total_capital * risk_per_trade * (1 / risk_ratio)
    
    return allocation


def adjust_for_volatility_state(positions, varieties_data, atr_quantile_threshold=0.8):
    """根据波动率状态调整头寸"""
    adjusted_positions = {}
    
    for symbol, position in positions.items():
        data = varieties_data[symbol]
        atr_quantile = data.get('atr_quantile', 0.5)
        
        # 当市场处于高波动状态（ATR分位数超过阈值）时，降低头寸
        if atr_quantile > atr_quantile_threshold:
            # 降低头寸规模，最高降低50%
            adjustment_factor = 1 - (atr_quantile - atr_quantile_threshold) * 2
            adjusted_position = int(position * adjustment_factor)
        else:
            adjusted_position = position
        
        adjusted_positions[symbol] = adjusted_position
    
    return adjusted_positions


def calculate_portfolio_risk(positions, varieties_data, correlations=None):
    """计算组合风险"""
    if not positions or not varieties_data:
        return 0.0
    
    # 简化的风险计算：考虑品种间相关性
    total_risk = 0.0
    
    # 如果没有相关性数据，假设品种间完全不相关
    if correlations is None:
        for symbol, position in positions.items():
            data = varieties_data[symbol]
            risk = abs(position) * data['atr'] * data['contract_multiplier']
            total_risk += risk ** 2
        total_risk = np.sqrt(total_risk)
    else:
        # 考虑相关性的风险计算
        symbols = list(positions.keys())
        n = len(symbols)
        
        # 计算协方差矩阵
        cov_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                symbol_i = symbols[i]
                symbol_j = symbols[j]
                
                # 单个品种的风险贡献
                risk_i = abs(positions[symbol_i]) * varieties_data[symbol_i]['atr'] * varieties_data[symbol_i]['contract_multiplier']
                risk_j = abs(positions[symbol_j]) * varieties_data[symbol_j]['atr'] * varieties_data[symbol_j]['contract_multiplier']
                
                # 相关性系数
                corr = correlations.get((symbol_i, symbol_j), 0.0) if (symbol_i, symbol_j) in correlations else \
                       correlations.get((symbol_j, symbol_i), 0.0) if (symbol_j, symbol_i) in correlations else 0.0
                
                cov_matrix[i, j] = risk_i * risk_j * corr
        
        # 组合风险为协方差矩阵的平方根
        total_risk = np.sqrt(np.sum(cov_matrix))
    
    return total_risk


def apply_position_constraints(positions, varieties_data, total_capital, max_lever_ratio=3.0, max_position_percent=0.05):
    """应用头寸约束"""
    adjusted_positions = positions.copy()
    
    # 1. 计算当前杠杆率
    total_margin = 0.0
    for symbol, position in adjusted_positions.items():
        data = varieties_data[symbol]
        margin = abs(position) * data['current_price'] * data['contract_multiplier'] * data['margin_rate']
        total_margin += margin
    
    current_lever = total_margin / total_capital
    
    # 2. 如果杠杆率超过上限，按比例缩减所有头寸
    if current_lever > max_lever_ratio:
        reduction_factor = max_lever_ratio / current_lever
        for symbol in adjusted_positions:
            adjusted_positions[symbol] = int(adjusted_positions[symbol] * reduction_factor)
    
    # 3. 检查单个品种头寸上限（相对于总资金）
    for symbol, position in adjusted_positions.items():
        data = varieties_data[symbol]
        position_value = abs(position) * data['current_price'] * data['contract_multiplier']
        position_percent = position_value / total_capital
        
        if position_percent > max_position_percent:
            # 缩减至上限
            max_position = int((total_capital * max_position_percent) / (data['current_price'] * data['contract_multiplier']))
            adjusted_positions[symbol] = np.sign(position) * max_position
    
    return adjusted_positions
