"""
position.pyd - 仓位管理模块
推测功能：实现仓位计算、管理和风险控制
"""

import pandas as pd
import numpy as np

def atr_inverse_weight_allocation(total_capital, varieties_data, current_price, contract_symbol, risk_per_trade=0.02, min_weight=0.01, max_weight=0.15): 
    """ 
    ATR倒数加权法 
    权重 ∝ 1/ATR 
    
    参数：
    total_capital: 总资金
    varieties_data: 品种数据，包含每个品种的ATR
    current_price: 当前价格
    contract_symbol: 合约代码
    risk_per_trade: 每笔交易风险比例
    min_weight: 最小权重
    max_weight: 最大权重
    
    返回：
    position_size: 仓位大小
    risk_amount: 风险金额
    capital_allocation: 资金分配
    """ 
    weights = {} 
    atr_inverse_sum = 0 
    
    # 计算各品种1/ATR 
    for symbol, data in varieties_data.items(): 
        atr = data['atr'] 
        if atr > 0: 
            atr_inverse = 1 / atr 
            weights[symbol] = atr_inverse 
            atr_inverse_sum += atr_inverse 
        else: 
            weights[symbol] = 0 
    
    # 归一化权重 
    if atr_inverse_sum > 0: 
        for symbol in weights: 
            weights[symbol] = weights[symbol] / atr_inverse_sum 
    else: 
        # 默认等权 
        n_symbols = len(varieties_data) 
        for symbol in varieties_data: 
            weights[symbol] = 1 / n_symbols 
    
    # 应用权重上下限 
    for symbol in weights: 
        weights[symbol] = max(min(weights[symbol], max_weight), min_weight) 
    
    # 重新归一化 
    weight_sum = sum(weights.values()) 
    allocation = {} 
    for symbol, weight in weights.items(): 
        allocation[symbol] = (weight / weight_sum) * total_capital 
    
    # 计算风险金额
    risk_amount = total_capital * risk_per_trade
    
    # 获取合约乘数
    contract_multiplier, _ = get_contract_multiplier(contract_symbol)
    
    # 计算基础ATR（当前品种的ATR）
    base_symbol = contract_symbol.split('.')[0] if '.' in contract_symbol else contract_symbol
    first_digit_index = None
    for i, char in enumerate(base_symbol):
        if char.isdigit():
            first_digit_index = i
            break
    if first_digit_index is not None:
        base_symbol = base_symbol[:first_digit_index]
    
    # 获取当前品种的ATR
    current_atr = varieties_data.get(base_symbol, {}).get('atr', 0) if base_symbol in varieties_data else 0
    
    # 计算每手风险
    if current_atr > 0:
        risk_per_unit = current_atr * contract_multiplier
        # 计算仓位大小
        position_size = risk_amount / risk_per_unit
        position_size = int(position_size)
    else:
        # 如果ATR为0，使用默认值
        position_size = 0
    
    return position_size, risk_amount, allocation

def calculate_position_size(capital, risk_per_trade, entry_price, stop_loss_price, symbol):
    """计算仓位大小
    
    参数：
    capital: 总资金
    risk_per_trade: 每笔交易风险比例
    entry_price: 入场价格
    stop_loss_price: 止损价格
    symbol: 合约代码
    
    返回：
    position_size: 仓位大小
    risk_amount: 风险金额
    """
    # 计算风险金额
    risk_amount = capital * risk_per_trade
    
    # 获取合约乘数
    contract_multiplier, _ = get_contract_multiplier(symbol)
    
    # 计算每手风险（考虑合约乘数）
    risk_per_unit = abs(entry_price - stop_loss_price) * contract_multiplier
    
    if risk_per_unit <= 0:
        return 0, 0
    
    # 计算仓位大小（合约数量）
    position_size = risk_amount / risk_per_unit
    
    # 取整
    position_size = int(position_size)
    
    return position_size, risk_amount

# 全局缓存，存储合约乘数和保证金率信息
_instrument_info_cache = None


def get_contract_multiplier(symbol):
    """获取合约乘数和保证金率
    
    参数：
    symbol: 合约代码
    
    返回：
    multiplier: 合约乘数
    margin_ratio: 保证金率
    """
    import pandas as pd
    import os
    import logging
    global _instrument_info_cache
    
    logger = logging.getLogger(__name__)
    
    # 如果缓存为空，读取合约信息
    if _instrument_info_cache is None:
        csv_path = os.path.join(os.path.dirname(__file__), 'all_instruments_info.csv')
        
        # 如果文件存在，从文件中读取并缓存
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # 构建合约信息字典，使用小写sec_id作为键，确保大小写不敏感
            _instrument_info_cache = {}
            for _, row in df.iterrows():
                # 使用小写sec_id作为键，确保大小写不敏感
                sec_id_lower = row['sec_id'].lower()
                _instrument_info_cache[sec_id_lower] = (row['multiplier'], row['margin_ratio'])
            logger.info(f"成功加载合约信息，共 {len(_instrument_info_cache)} 个合约")
        else:
            # 如果文件不存在，初始化空缓存
            _instrument_info_cache = {}
            logger.error(f"合约信息文件不存在: {csv_path}")
    
    # 提取合约代码的基础部分（如IF, IC, IH, A, B等）
    # 处理不同交易所的合约代码格式：
    # - DCE/SHFE: 如 "a2409.DCE" 或 "fu2505.SHFE"（4位数字）
    # - CZCE: 如 "sa505.CZCE"（3位数字）
    # - CFFEX: 如 "IF2409.CFFEX" 或 "IC"
    
    # 先移除交易所后缀，如 ".CZCE"
    symbol_without_exchange = symbol.split('.')[0] if '.' in symbol else symbol
    
    # 查找第一个数字的位置
    first_digit_index = None
    for i, char in enumerate(symbol_without_exchange):
        if char.isdigit():
            first_digit_index = i
            break
    
    if first_digit_index is not None:
        # 提取数字前的字母作为基础品种代码
        base_symbol = symbol_without_exchange[:first_digit_index]
    else:
        # 没有数字，使用完整符号作为基础品种代码
        base_symbol = symbol_without_exchange
    
    # 转换为小写，确保大小写不敏感
    base_symbol_lower = base_symbol.lower()
    
    # 从缓存中获取合约乘数和保证金率，不存在则返回默认值
    if base_symbol_lower in _instrument_info_cache:
        multiplier, margin_ratio = _instrument_info_cache[base_symbol_lower]
        logger.debug(f"合约 {symbol} 的基础代码 {base_symbol} 匹配到 {base_symbol_lower}，乘数: {multiplier}, 保证金率: {margin_ratio}")
        return multiplier, margin_ratio
    else:
        logger.warning(f"合约 {symbol} 的基础代码 {base_symbol}（小写: {base_symbol_lower}）未找到，使用默认值")
        return 10, 0.1  # 默认乘数为10，保证金率为10%

def calculate_position_value(position_size, current_price, symbol):
    """计算当前持仓价值"""
    contract_multiplier, _ = get_contract_multiplier(symbol)
    position_value = position_size * current_price * contract_multiplier
    return position_value

def calculate_margin_usage(position_size, current_price, symbol, margin_rate=0.1):
    """计算保证金占用"""
    position_value = calculate_position_value(position_size, current_price, symbol)
    margin_usage = position_value * margin_rate
    return margin_usage

def calculate_portfolio_metrics(portfolio_data):
    """计算投资组合指标"""
    # 计算总市值（使用绝对值，无论多空仓都计算为正值）
    portfolio_data['market_value'] = abs(portfolio_data['position_size']) * portfolio_data['current_price'] * portfolio_data['contract_multiplier']
    
    # 计算总持仓价值
    total_market_value = portfolio_data['market_value'].sum()
    
    # 计算总保证金占用（使用绝对值）
    portfolio_data['margin_usage'] = abs(portfolio_data['market_value'] * portfolio_data['margin_rate'])
    total_margin_usage = portfolio_data['margin_usage'].sum()
    
    # 计算可用资金
    total_capital = portfolio_data['total_capital'].iloc[0]
    available_capital = total_capital - total_margin_usage
    
    # 计算风险敞口（使用绝对值，避免负号影响）
    risk_exposure = abs(total_market_value) / total_capital
    
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