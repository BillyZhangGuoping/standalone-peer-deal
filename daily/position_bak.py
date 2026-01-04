"""
position_bak.py - 仓位管理模块(备份版本)
功能：实现仓位计算、管理和风险控制
"""

import pandas as pd
import numpy as np

# 从position.py复制核心功能
def calculate_position_size(capital, risk_per_trade, entry_price, stop_loss_price, symbol):
    """计算仓位大小"""
    # 计算风险金额
    risk_amount = capital * risk_per_trade
    
    # 计算每手风险
    risk_per_unit = abs(entry_price - stop_loss_price)
    
    if risk_per_unit <= 0:
        return 0, 0
    
    # 计算仓位大小
    position_size = risk_amount / risk_per_unit
    
    # 根据合约乘数调整
    contract_multiplier = get_contract_multiplier(symbol)
    position_size = position_size / contract_multiplier
    
    # 取整
    position_size = int(position_size)
    
    return position_size, risk_amount

def get_contract_multiplier(symbol):
    """获取合约乘数"""
    multiplier_dict = {
        'IF': 300,
        'IC': 200,
        'IH': 300,
        'AU': 1000,
        'AG': 1500,
        'CU': 5,
        'AL': 5,
        'RB': 10,
        'HC': 10,
        'BU': 10,
        'RU': 10,
        'ZN': 5,
        'PB': 5,
        'NI': 1,
        'SN': 1,
        'SS': 5,
        'SC': 1000,
        'FU': 5000,
        'BU': 10,
        'FG': 20,
        'MA': 10,
        'TA': 5,
        'RM': 10,
        'JR': 20,
        'OI': 10,
        'RS': 5,
        'LR': 10,
        'CF': 5,
        'SR': 10,
        'ZC': 100,
        'FG': 20,
        'CY': 5,
        'AP': 10,
        'CJ': 5,
        'PF': 5,
        'SA': 50,
        'UR': 20,
        'PG': 20,
        'EB': 5,
        'EG': 10,
        'V': 5,
        'P': 10,
        'B': 10,
        'M': 10,
        'Y': 10,
        'JD': 5,
        'JM': 60,
        'J': 100,
        'I': 100,
        'A': 10,
        'L': 5,
        'PP': 5,
        'CS': 10,
        'C': 10,
        'ST': 10,
        'SM': 5,
        'SF': 5,
    }
    
    # 提取合约代码的基础部分
    base_symbol = symbol[:2] if symbol[0].isalpha() else symbol[:1]
    
    return multiplier_dict.get(base_symbol, 10)

def calculate_position_value(position_size, current_price, symbol):
    """计算当前持仓价值"""
    contract_multiplier = get_contract_multiplier(symbol)
    position_value = position_size * current_price * contract_multiplier
    return position_value

def calculate_margin_usage(position_size, current_price, symbol, margin_rate=0.1):
    """计算保证金占用"""
    position_value = calculate_position_value(position_size, current_price, symbol)
    margin_usage = position_value * margin_rate
    return margin_usage

def calculate_portfolio_metrics(portfolio_data):
    """计算投资组合指标"""
    # 计算总市值
    portfolio_data['market_value'] = portfolio_data['position_size'] * portfolio_data['current_price'] * portfolio_data['contract_multiplier']
    
    # 计算总持仓价值
    total_market_value = portfolio_data['market_value'].sum()
    
    # 计算总保证金占用
    portfolio_data['margin_usage'] = portfolio_data['market_value'] * portfolio_data['margin_rate']
    total_margin_usage = portfolio_data['margin_usage'].sum()
    
    # 计算可用资金
    total_capital = portfolio_data['total_capital'].iloc[0]
    available_capital = total_capital - total_margin_usage
    
    # 计算风险敞口
    risk_exposure = total_market_value / total_capital
    
    return {
        'total_market_value': total_market_value,
        'total_margin_usage': total_margin_usage,
        'available_capital': available_capital,
        'risk_exposure': risk_exposure
    }

def update_position(position, signal, entry_price, stop_loss_price, capital, risk_per_trade, symbol):
    """更新仓位"""
    if signal == 0:
        return position, 0  # 无信号，保持仓位
    
    # 平仓
    if (position > 0 and signal == -1) or (position < 0 and signal == 1):
        close_position = position
        return 0, close_position
    
    # 开仓或加仓
    position_size, risk_amount = calculate_position_size(capital, risk_per_trade, entry_price, stop_loss_price, symbol)
    
    if position_size <= 0:
        return position, 0
    
    # 根据信号方向确定仓位
    if signal == 1:
        new_position = position + position_size
    else:  # signal == -1
        new_position = position - position_size
    
    return new_position, position_size

def run_position_management(data, signals, capital, risk_per_trade=0.02, max_position=0.5):
    """运行仓位管理"""
    # 合并数据和信号
    df = data.copy()
    df['signal'] = signals
    
    # 初始化变量
    position = 0
    portfolio_value = [capital]
    positions = []
    
    for i in range(len(df)):
        current_price = df['close'].iloc[i]
        signal = df['signal'].iloc[i]
        symbol = df['symbol'].iloc[i] if 'symbol' in df.columns else 'UNKNOWN'
        
        # 计算止损价格（简化处理，使用ATR）
        if i > 0:
            atr = df['atr'].iloc[i-1] if 'atr' in df.columns else 0
        else:
            atr = 0
        
        stop_loss_price = current_price - atr if signal == 1 else current_price + atr
        
        # 更新仓位
        position, trade_size = update_position(
            position, signal, current_price, stop_loss_price, capital, risk_per_trade, symbol
        )
        
        positions.append(position)
        
        # 更新账户价值
        if i > 0:
            price_change = df['close'].iloc[i] - df['close'].iloc[i-1]
            contract_multiplier = get_contract_multiplier(symbol)
            pnl = position * price_change * contract_multiplier
            new_portfolio_value = portfolio_value[-1] + pnl
            portfolio_value.append(new_portfolio_value)
        
    # 更新DataFrame
    df['position'] = positions
    df['portfolio_value'] = portfolio_value
    
    return df

def run():
    """主函数入口"""
    print("Running position_bak...")
    # 这里可以添加具体的仓位管理逻辑
    pass