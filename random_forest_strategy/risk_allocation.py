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
    根据市场波动率状态调整ATR计算窗口，设定ATR最小值为1
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
        
        # 设定ATR最小值为1
        atr = max(atr, 1.0)
        
        # 直接使用ATR进行资金分配，不再乘以合约乘数
        allocations[symbol] = {'atr': atr, 'original_atr': atr, 'atr_window': atr_window}
    
    return allocations

def atr_momentum_composite_allocation(total_capital, varieties_data, momentum_window=20): 
    """ 
    ATR动量复合分配策略
    结合ATR风险分配和动量加权
    
    参数：
    total_capital: 总资金
    varieties_data: 各品种数据字典
    momentum_window: 动量计算窗口
    
    返回：
    allocation: 各品种分配资金
    """ 
    import numpy as np
    
    # 1. 获取自适应ATR
    adaptive_atr_data = adaptive_atr_allocation(total_capital, varieties_data)
    
    # 2. 计算各品种的动量
    momentum_values = {}
    for symbol, data in varieties_data.items():
        prices = data['prices']
        if len(prices) < momentum_window:
            momentum_values[symbol] = 0
        else:
            # 计算动量：最近收盘价 / momentum_window前收盘价 - 1
            momentum = (prices[-1] / prices[-momentum_window]) - 1
            momentum_values[symbol] = momentum
    
    # 3. 计算ATR权重
    atr_weights = {}
    all_atrs = []
    
    # 收集所有ATR值
    for symbol, data in varieties_data.items():
        if symbol in adaptive_atr_data:
            all_atrs.append(adaptive_atr_data[symbol]['atr'])
        else:
            all_atrs.append(data['atr'])
    
    avg_atr = np.mean(all_atrs) if all_atrs else 1
    
    for symbol, data in varieties_data.items():
        # 使用自适应ATR，如果没有则使用原始ATR
        if symbol in adaptive_atr_data:
            atr = adaptive_atr_data[symbol]['atr']
        else:
            atr = data['atr']
        
        if atr > 0:
            # 波动率低 → 权重高
            atr_weights[symbol] = avg_atr / atr
        else:
            atr_weights[symbol] = 1.0
    
    # 4. 计算动量权重（标准化动量）
    # 处理动量值，确保都是正数
    momentum_abs = [abs(m) for m in momentum_values.values()]
    avg_momentum_abs = np.mean(momentum_abs) if momentum_abs else 1
    
    momentum_weights = {}
    for symbol, momentum in momentum_values.items():
        # 动量越大，权重越高
        momentum_weights[symbol] = abs(momentum) / avg_momentum_abs if avg_momentum_abs > 0 else 1.0
    
    # 5. 计算复合权重（ATR权重 * 0.3 + 动量权重 * 0.7）
    composite_weights = {}
    for symbol in varieties_data:
        composite_weights[symbol] = atr_weights[symbol] * 0.3 + momentum_weights[symbol] * 0.7
    
    # 6. 归一化权重，分配资金
    total_weight = sum(composite_weights.values())
    allocation = {}
    
    if total_weight > 0:
        for symbol, weight in composite_weights.items():
            allocation[symbol] = (weight / total_weight) * total_capital
    else:
        # 如果总权重为0，等权分配
        for symbol in varieties_data:
            allocation[symbol] = total_capital / len(varieties_data)
    
    return allocation

def enhanced_atr_allocation(total_capital, varieties_data, target_volatility=0.01, momentum_window=20):
    """
    增强型ATR分配策略
    综合calculate_atr_allocation和atr_momentum_composite_allocation的优点
    考虑合约乘数、当前价格和趋势强度
    
    参数：
    total_capital: 总资金
    varieties_data: 各品种数据字典
    target_volatility: 目标单位风险
    momentum_window: 动量计算窗口
    
    返回：
    allocation: 各品种分配资金
    risk_units: 各品种风险单位数
    """
    import numpy as np
    
    # 1. 获取自适应ATR
    adaptive_atr_data = adaptive_atr_allocation(total_capital, varieties_data)
    
    # 2. 计算各品种的动量
    momentum_values = {}
    for symbol, data in varieties_data.items():
        prices = data['prices']
        if len(prices) < momentum_window:
            momentum_values[symbol] = 0
        else:
            # 计算动量：最近收盘价 / momentum_window前收盘价 - 1
            momentum = (prices[-1] / prices[-momentum_window]) - 1
            momentum_values[symbol] = momentum
    
    # 3. 计算风险单位数
    risk_units = {}
    notional_values = {}
    
    for symbol, data in varieties_data.items():
        # 获取当前价格、ATR、合约乘数和趋势强度
        current_price = data['current_price']
        contract_multiplier = data['contract_multiplier']
        trend_strength = abs(data.get('trend_strength', 1))  # 默认趋势强度为1
        
        # 使用自适应ATR，如果没有则使用原始ATR
        if symbol in adaptive_atr_data:
            atr = adaptive_atr_data[symbol]['atr']
        else:
            atr = data['atr']
        
        # 设定ATR最小值为1
        atr = max(atr, 1.0)
        
        # 计算每手ATR价值
        atr_per_lot = atr * contract_multiplier
        
        if atr_per_lot > 0 and trend_strength > 0:
            # 计算风险单位数：风险单位数 = (1 / 趋势强度) ÷ 每手ATR价值
            risk_unit = (1 / trend_strength) / atr_per_lot
            risk_units[symbol] = risk_unit
            
            # 计算名义市值：名义市值 = 风险单位数 × 当前价格 × 合约乘数
            notional_value = risk_unit * current_price * contract_multiplier
            notional_values[symbol] = notional_value
        else:
            risk_units[symbol] = 0
            notional_values[symbol] = 0
    
    # 4. 计算动量权重（用于调整名义市值）
    momentum_abs = [abs(m) for m in momentum_values.values()]
    avg_momentum_abs = np.mean(momentum_abs) if momentum_abs else 1
    
    momentum_weights = {}
    for symbol, momentum in momentum_values.items():
        # 动量越大，权重越高
        momentum_weights[symbol] = abs(momentum) / avg_momentum_abs if avg_momentum_abs > 0 else 1.0
    
    # 5. 应用动量权重调整名义市值
    adjusted_notional_values = {}
    for symbol, notional_value in notional_values.items():
        adjusted_notional_values[symbol] = notional_value * momentum_weights[symbol]
    
    # 6. 归一化分配
    total_adjusted_notional = sum(adjusted_notional_values.values())
    allocation = {}
    
    if total_adjusted_notional > 0:
        for symbol, notional_value in adjusted_notional_values.items():
            allocation[symbol] = (notional_value / total_adjusted_notional) * total_capital
    else:
        # 默认等权分配
        n_symbols = len(varieties_data)
        for symbol in varieties_data:
            allocation[symbol] = total_capital / n_symbols
    
    return allocation, risk_units
