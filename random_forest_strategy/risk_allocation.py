"""
risk_allocation.py - 风险分配模块
提供基于ATR的等风险资金分配等功能
"""

def calculate_atr_allocation(total_capital, varieties_data, target_volatility=0.01): 
    """ 
    基于ATR的等风险资金分配 
    
    参数： 
    total_capital: 总资金（1000万） 
    varieties_data: 各品种数据字典 
    target_volatility: 目标单位风险（默认1%） 
    
    返回： 
    allocation: 各品种分配资金 
    risk_units: 各品种风险单位数
    """ 
    allocation = {} 
    
    # 步骤1：计算各品种的ATR价值 
    atr_values = {} 
    for symbol, data in varieties_data.items(): 
        # 获取当前价格、ATR、合约乘数 
        current_price = data['current_price'] 
        atr = data['atr']  # 比如20日ATR 
        contract_multiplier = data['contract_multiplier'] 
        
        # 计算ATR的货币价值 
        atr_value = atr * contract_multiplier 
        atr_values[symbol] = atr_value 
    
    # 步骤2：计算风险单位数量 
    # 公式：风险单位 = 目标风险金额 / 单风险单位价值 
    # 目标：每个品种每日波动不超过总资金的target_volatility 
    target_risk_amount = total_capital * target_volatility 
    risk_units = {} 
    
    for symbol, atr_value in atr_values.items(): 
        if atr_value > 0: 
            # 计算可承受的风险单位数 
            risk_units[symbol] = target_risk_amount / atr_value 
        else: 
            risk_units[symbol] = 0 
    
    # 步骤3：计算名义市值 
    # 名义市值 = 风险单位数 × 当前价格 × 合约乘数 
    notional_values = {} 
    for symbol, data in varieties_data.items(): 
        if risk_units[symbol] > 0: 
            notional_value = ( 
                risk_units[symbol] * 
                data['current_price'] * 
                data['contract_multiplier'] 
            ) 
            notional_values[symbol] = notional_value 
        else: 
            notional_values[symbol] = 0 
    
    # 步骤4：归一化分配 
    total_notional = sum(notional_values.values()) 
    if total_notional > 0: 
        for symbol, notional_value in notional_values.items(): 
            allocation[symbol] = (notional_value / total_notional) * total_capital 
    else: 
        # 默认等权分配 
        n_symbols = len(varieties_data) 
        for symbol in varieties_data: 
            allocation[symbol] = total_capital / n_symbols 
    
    return allocation, risk_units

def calculate_simple_atr(prices, window=20):
    """
    从价格列表计算简单ATR（基于收盘价差异）
    用于自适应ATR窗口策略
    
    参数：
    prices: 价格列表
    window: 计算窗口
    
    返回：
    atr: 平均真实波幅
    """
    import numpy as np
    if len(prices) < window + 1:
        return 0
    
    # 计算每日价格变动
    price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
    
    # 计算ATR
    atr = np.mean(price_changes[-window:])
    return atr

def adaptive_atr_allocation(total_capital, varieties_data, volatility_regime_threshold=0.3): 
    """ 
    自适应ATR窗口的分配策略 
    根据市场波动率状态调整ATR计算窗口 
    """ 
    import numpy as np
    allocations = {} 
    
    for symbol, data in varieties_data.items(): 
        prices = data['prices'] 
        
        # 检测市场波动率状态 
        if len(prices) < 60:  # 确保有足够数据计算波动率
            continue
            
        recent_vol = np.std(prices[-20:]) / np.mean(prices[-20:])  # 最近20日波动率 
        long_term_vol = np.std(prices[-60:]) / np.mean(prices[-60:])  # 长期波动率 
        
        # 选择ATR计算窗口 
        if recent_vol / long_term_vol > 1 + volatility_regime_threshold: 
            # 高波动状态：使用短窗口ATR（10日） 
            atr_window = 10 
        elif recent_vol / long_term_vol < 1 - volatility_regime_threshold: 
            # 低波动状态：使用长窗口ATR（30日） 
            atr_window = 30 
        else: 
            # 正常状态：使用标准窗口（20日） 
            atr_window = 20 
        
        # 计算自适应ATR 
        atr = calculate_simple_atr(prices, window=atr_window) 
        
        # 使用ATR进行资金分配
        # 这里使用简化的ATR分配逻辑，与现有的ATR动量复合分配兼容
        allocations[symbol] = {'atr': atr, 'atr_window': atr_window}
    
    return allocations

def atr_momentum_composite_allocation(total_capital, varieties_data, momentum_window=20): 
    """ 
    ATR权重 + 动量信号复合分配 
    波动率低的品种 + 动量强的品种 = 高权重 
    """ 
    import numpy as np
    weights = {} 
    
    # 第一步：获取自适应ATR
    adaptive_atr_data = adaptive_atr_allocation(total_capital, varieties_data)
    
    for symbol, data in varieties_data.items(): 
        # 1. ATR权重部分 (50%)
        # 使用自适应ATR，如果没有则使用原始ATR
        if symbol in adaptive_atr_data:
            atr = adaptive_atr_data[symbol]['atr']
        else:
            atr = data['atr']
        
        avg_atr = np.mean([(adaptive_atr_data.get(s, {'atr': d['atr']})['atr']) for s, d in varieties_data.items()])
        
        if atr > 0: 
            atr_weight = avg_atr / atr  # 波动率低 → 权重高 
        else: 
            atr_weight = 1 
        
        # 2. 动量权重部分 (50%) 
        if 'prices' in data and len(data['prices']) >= momentum_window: 
            # 计算20日动量 
            prices = data['prices'] 
            momentum = (prices[-1] / prices[-momentum_window] - 1) * 100 
            # 动量强的品种权重高 
            momentum_weight = 1 + abs(momentum) * 0.1 
        else: 
            # 如果没有价格数据，使用默认权重
            momentum_weight = 1 
        
        # 3. 复合权重 
        composite_weight = 0.3 * atr_weight + 0.7 * momentum_weight 
        weights[symbol] = composite_weight 
    
    # 归一化 
    total_weight = sum(weights.values()) 
    allocation = {} 
    for symbol, weight in weights.items(): 
        allocation[symbol] = (weight / total_weight) * total_capital 
    
    return allocation
