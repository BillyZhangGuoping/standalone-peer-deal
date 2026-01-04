"""
rules.pyd - 交易规则模块
推测功能：实现各种交易规则和策略逻辑
"""

import pandas as pd
import numpy as np

def apply_stop_loss(data, stop_loss_percent=0.02):
    """应用止损规则"""
    # 跟踪持仓成本
    data['position_cost'] = 0.0
    data['stop_loss_price'] = 0.0
    
    position = 0
    cost = 0
    
    for i in range(len(data)):
        if position == 0 and data['signal'].iloc[i] == 1:
            # 买入
            position = 1
            cost = data['close'].iloc[i]
            data['position_cost'].iloc[i] = cost
            data['stop_loss_price'].iloc[i] = cost * (1 - stop_loss_percent)
        elif position == 1:
            # 持有多头
            data['position_cost'].iloc[i] = cost
            data['stop_loss_price'].iloc[i] = cost * (1 - stop_loss_percent)
            
            # 检查止损
            if data['low'].iloc[i] <= data['stop_loss_price'].iloc[i]:
                data['signal'].iloc[i] = -1  # 触发止损，卖出
                position = 0
        elif position == 0 and data['signal'].iloc[i] == -1:
            # 做空
            position = -1
            cost = data['close'].iloc[i]
            data['position_cost'].iloc[i] = cost
            data['stop_loss_price'].iloc[i] = cost * (1 + stop_loss_percent)
        elif position == -1:
            # 持有空头
            data['position_cost'].iloc[i] = cost
            data['stop_loss_price'].iloc[i] = cost * (1 + stop_loss_percent)
            
            # 检查止损
            if data['high'].iloc[i] >= data['stop_loss_price'].iloc[i]:
                data['signal'].iloc[i] = 1  # 触发止损，买入平仓
                position = 0
    
    return data

def apply_take_profit(data, take_profit_percent=0.05):
    """应用止盈规则"""
    # 跟踪持仓成本
    data['position_cost'] = 0.0
    data['take_profit_price'] = 0.0
    
    position = 0
    cost = 0
    
    for i in range(len(data)):
        if position == 0 and data['signal'].iloc[i] == 1:
            # 买入
            position = 1
            cost = data['close'].iloc[i]
            data['position_cost'].iloc[i] = cost
            data['take_profit_price'].iloc[i] = cost * (1 + take_profit_percent)
        elif position == 1:
            # 持有多头
            data['position_cost'].iloc[i] = cost
            data['take_profit_price'].iloc[i] = cost * (1 + take_profit_percent)
            
            # 检查止盈
            if data['high'].iloc[i] >= data['take_profit_price'].iloc[i]:
                data['signal'].iloc[i] = -1  # 触发止盈，卖出
                position = 0
        elif position == 0 and data['signal'].iloc[i] == -1:
            # 做空
            position = -1
            cost = data['close'].iloc[i]
            data['position_cost'].iloc[i] = cost
            data['take_profit_price'].iloc[i] = cost * (1 - take_profit_percent)
        elif position == -1:
            # 持有空头
            data['position_cost'].iloc[i] = cost
            data['take_profit_price'].iloc[i] = cost * (1 - take_profit_percent)
            
            # 检查止盈
            if data['low'].iloc[i] <= data['take_profit_price'].iloc[i]:
                data['signal'].iloc[i] = 1  # 触发止盈，买入平仓
                position = 0
    
    return data

def apply_trailing_stop_loss(data, trailing_stop_percent=0.03):
    """应用移动止损规则"""
    # 跟踪持仓成本和最高/最低价
    data['position_cost'] = 0.0
    data['trailing_stop_price'] = 0.0
    data['highest_high'] = 0.0
    data['lowest_low'] = 0.0
    
    position = 0
    cost = 0
    highest_high = 0
    lowest_low = 0
    
    for i in range(len(data)):
        if position == 0 and data['signal'].iloc[i] == 1:
            # 买入
            position = 1
            cost = data['close'].iloc[i]
            highest_high = data['high'].iloc[i]
            data['position_cost'].iloc[i] = cost
            data['highest_high'].iloc[i] = highest_high
            data['trailing_stop_price'].iloc[i] = highest_high * (1 - trailing_stop_percent)
        elif position == 1:
            # 持有多头
            highest_high = max(highest_high, data['high'].iloc[i])
            data['position_cost'].iloc[i] = cost
            data['highest_high'].iloc[i] = highest_high
            data['trailing_stop_price'].iloc[i] = highest_high * (1 - trailing_stop_percent)
            
            # 检查移动止损
            if data['low'].iloc[i] <= data['trailing_stop_price'].iloc[i]:
                data['signal'].iloc[i] = -1  # 触发止损，卖出
                position = 0
                highest_high = 0
        elif position == 0 and data['signal'].iloc[i] == -1:
            # 做空
            position = -1
            cost = data['close'].iloc[i]
            lowest_low = data['low'].iloc[i]
            data['position_cost'].iloc[i] = cost
            data['lowest_low'].iloc[i] = lowest_low
            data['trailing_stop_price'].iloc[i] = lowest_low * (1 + trailing_stop_percent)
        elif position == -1:
            # 持有空头
            lowest_low = min(lowest_low, data['low'].iloc[i])
            data['position_cost'].iloc[i] = cost
            data['lowest_low'].iloc[i] = lowest_low
            data['trailing_stop_price'].iloc[i] = lowest_low * (1 + trailing_stop_percent)
            
            # 检查移动止损
            if data['high'].iloc[i] >= data['trailing_stop_price'].iloc[i]:
                data['signal'].iloc[i] = 1  # 触发止损，买入平仓
                position = 0
                lowest_low = 0
    
    return data

def apply_position_sizing(data, capital, risk_per_trade=0.02):
    """应用仓位管理规则"""
    # 计算ATR
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()
    
    # 计算每笔交易的风险金额
    risk_amount = capital * risk_per_trade
    
    # 计算仓位大小（假设1个ATR止损）
    data['atr'] = atr
    data['position_size'] = (risk_amount / atr) / data['close']
    data['position_size'] = data['position_size'].fillna(0).astype(int)
    
    return data

def apply_risk_management(data, max_position=0.5, max_drawdown=0.2):
    """应用风险管理规则"""
    # 计算累计收益率
    data['returns'] = data['close'].pct_change()
    data['cumulative_returns'] = (1 + data['returns']).cumprod()
    
    # 计算最大回撤
    data['peak'] = data['cumulative_returns'].expanding(min_periods=1).max()
    data['drawdown'] = (data['cumulative_returns'] - data['peak']) / data['peak']
    
    # 应用最大仓位限制
    data['position_limit'] = max_position
    
    # 应用最大回撤限制
    data['drawdown_limit'] = max_drawdown
    data['risk_management_signal'] = 1
    data.loc[data['drawdown'] <= -max_drawdown, 'risk_management_signal'] = 0
    
    # 调整信号
    data['signal'] = data['signal'] * data['risk_management_signal']
    
    return data