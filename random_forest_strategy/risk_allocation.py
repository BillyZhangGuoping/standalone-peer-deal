"""
risk_allocation.py - 风险分配模块
提供基于ATR的等风险资金分配等功能
"""
import numpy as np

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
        prices = data.get('prices', [current_price])  # 获取价格列表
        contract_multiplier = data['contract_multiplier'] 
        
        # 使用现有的calculate_simple_atr函数计算ATR
        atr = calculate_simple_atr(prices, window=20)
        
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
    
    # 添加日志
    print(f"最终分配资金: {allocation}")
    print(f"风险单位: {risk_units}")
    
    return allocation, risk_units

def floor_asset_tilt_allocation(total_capital, varieties_data, target_volatility=0.01, volatility_window=20):
    """
    地板资产倾斜 (sign/Vol) 分配策略
    倾向于给低波动品种分配更多头寸
    权重 = 信号强度的绝对值 / 历史波动率（收益率标准差）
    
    参数：
    total_capital: 总资金
    varieties_data: 各品种数据字典
    target_volatility: 目标单位风险
    volatility_window: 历史波动率计算窗口（默认20天）
    
    返回：
    allocation: 各品种分配资金
    risk_units: 各品种风险单位数
    """
    import numpy as np
    
    # 计算历史波动率（收益率标准差）
    def calculate_historical_volatility(prices, window=20):
        """
        计算历史波动率
        
        参数：
        prices: 价格列表
        window: 计算窗口
        
        返回：
        volatility: 历史波动率（年化）
        """
        if len(prices) < window + 1:
            return 0.0
        
        # 计算收益率
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        
        # 计算窗口内的收益率标准差
        recent_returns = returns[-window:]
        if len(recent_returns) < window:
            return 0.0
        
        # 计算日波动率并年化（假设252个交易日）
        daily_vol = np.std(recent_returns)
        annual_vol = daily_vol * np.sqrt(252)
        
        return annual_vol
    
    # 步骤1：计算各品种的历史波动率和信号强度
    volatilities = {}
    signal_strengths = {}
    
    for symbol, data in varieties_data.items():
        # 获取价格历史数据
        prices = data.get('prices', data.get('price_history', [data['current_price']]))
        trend_strength = abs(data.get('trend_strength', 1.0))  # 默认趋势强度为1
        
        # 计算历史波动率
        volatility = calculate_historical_volatility(prices, window=volatility_window)
        volatilities[symbol] = volatility
        
        # 使用趋势强度的绝对值作为信号强度
        signal_strengths[symbol] = trend_strength
    
    # 步骤2：计算基础权重（sign/Vol）
    base_weights = {}
    for symbol in varieties_data:
        volatility = volatilities[symbol]
        signal_strength = signal_strengths[symbol]
        
        if volatility > 0:
            # sign/Vol策略：权重与信号强度成正比，与波动率成反比
            base_weights[symbol] = signal_strength / volatility
        else:
            base_weights[symbol] = 1.0  # 避免除以零
    
    # 步骤3：归一化基础权重
    total_base_weight = sum(base_weights.values())
    normalized_weights = {}
    
    if total_base_weight > 0:
        for symbol, weight in base_weights.items():
            normalized_weights[symbol] = weight / total_base_weight
    else:
        # 默认等权分配
        n_symbols = len(varieties_data)
        for symbol in varieties_data:
            normalized_weights[symbol] = 1.0 / n_symbols
    
    # 步骤4：计算各品种的分配资金
    allocation = {}
    for symbol, weight in normalized_weights.items():
        allocation[symbol] = weight * total_capital
    
    # 步骤5：计算风险单位数（基于历史波动率）
    risk_units = {}
    for symbol, data in varieties_data.items():
        current_price = data['current_price']
        contract_multiplier = data['contract_multiplier']
        volatility = volatilities[symbol]
        
        if volatility > 0:
            # 使用历史波动率计算风险单位数
            # 风险单位数 = 分配资金 × 目标波动率 / (当前价格 × 合约乘数 × 历史波动率)
            risk_unit = (allocation[symbol] * target_volatility) / (current_price * contract_multiplier * volatility)
            risk_units[symbol] = risk_unit
        else:
            risk_units[symbol] = 0
    
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
    
    # 2. 使用模型预测的趋势强度作为动量
    momentum_values = {}
    for symbol, data in varieties_data.items():
        # 使用模型预测的趋势强度作为动量值
        momentum_values[symbol] = data['trend_strength']
    
    # 3. 计算ATR权重（考虑合约乘数）
    atr_weights = {}
    all_adjusted_atrs = []
    
    # 收集所有调整后的ATR值（考虑合约乘数）
    for symbol, data in varieties_data.items():
        # 使用自适应ATR，如果没有则使用原始ATR
        if symbol in adaptive_atr_data:
            atr = adaptive_atr_data[symbol]['atr']
        else:
            atr = data['atr']
        
        # 不考虑合约乘数，直接使用原始ATR
        adjusted_atr = atr
        all_adjusted_atrs.append(adjusted_atr)
    
    avg_adjusted_atr = np.mean(all_adjusted_atrs) if all_adjusted_atrs else 1
    
    for symbol, data in varieties_data.items():
        # 使用自适应ATR，如果没有则使用原始ATR
        if symbol in adaptive_atr_data:
            atr = adaptive_atr_data[symbol]['atr']
        else:
            atr = data['atr']
        
        # 不考虑合约乘数，直接使用原始ATR
        adjusted_atr = atr
        
        if adjusted_atr > 0:
            # 波动率低 → 权重高
            atr_weights[symbol] = avg_adjusted_atr / adjusted_atr
        else:
            atr_weights[symbol] = 1.0
    
    # 4. 直接使用趋势强度作为动量权重，确保为正
    momentum_weights = {}
    for symbol, momentum in momentum_values.items():
        # 确保趋势强度为正，直接使用绝对值
        momentum_weights[symbol] = abs(momentum)
    
    # 5. 计算复合权重（ATR权重 * (1 + 动量权重)）
    composite_weights = {}
    for symbol in varieties_data:
        composite_weights[symbol] = atr_weights[symbol] * (1+ momentum_weights[symbol] )
    
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

def enhanced_atr_allocation(total_capital, varieties_data, target_volatility=0.01):
    """
    增强型ATR分配策略
    综合calculate_atr_allocation和atr_momentum_composite_allocation的优点
    考虑合约乘数、当前价格、趋势强度和保证金比率
    
    参数：
    total_capital: 总资金
    varieties_data: 各品种数据字典
    target_volatility: 目标单位风险
    
    返回：
    allocation: 各品种分配资金
    risk_units: 各品种风险单位数
    """
    import numpy as np
    
    # 3. 计算风险单位数
    risk_units = {}
    notional_values = {}
    margin_adjusted_values = {}
    
    for symbol, data in varieties_data.items():
        # 获取当前价格、ATR、合约乘数、趋势强度和保证金率
        current_price = data['current_price']
        contract_multiplier = data['contract_multiplier']
        trend_strength = abs(data.get('trend_strength', 1))  # 默认趋势强度为1
        margin_rate = data.get('margin_rate', 0.05)  # 默认保证金率为5%
        prices = data.get('prices', [current_price])  # 获取价格列表
        
        # 从价格列表计算ATR
        def calculate_atr_from_prices(prices, window=20):
            """
            从价格列表计算ATR
            """
            if len(prices) < window + 1:
                return 0.0
            
            # 计算每日真实波幅
            tr_values = []
            for i in range(1, len(prices)):
                # 由于我们只有收盘价，使用简单的价格变动作为TR
                tr = abs(prices[i] - prices[i-1])
                tr_values.append(tr)
            
            # 计算ATR
            atr = np.mean(tr_values[-window:])
            return atr
        
        # 计算ATR
        atr = calculate_atr_from_prices(prices, window=20)
        
        # 计算每手ATR价值，降低合约乘数的权重
        atr_per_lot = atr * (contract_multiplier)
        
        if atr_per_lot > 0:
            # 计算风险单位数：风险单位数 = 1 ÷ 每手ATR价值
            risk_unit = (1+ trend_strength)/ atr_per_lot
            risk_units[symbol] = risk_unit
            
            # 计算名义市值：名义市值 = 风险单位数 × 当前价格 × 合约乘数
            notional_value = risk_unit * current_price * contract_multiplier
            notional_values[symbol] = notional_value
            
            # 计算保证金调整后的价值：考虑保证金比率，确保不同保证金比率的品种生成的手数占比一致
            # 思路：保证金率越高，需要分配的资金越多，才能获得相同的手数
            margin_adjusted_value = notional_value * margin_rate
            margin_adjusted_values[symbol] = margin_adjusted_value
        else:
            risk_units[symbol] = 0
            notional_values[symbol] = 0
            margin_adjusted_values[symbol] = 0
    
    # 4. 归一化分配
    total_notional = sum(notional_values.values())
    allocation = {}
    
    if total_notional > 0:
        for symbol, notional_value in notional_values.items():
            # 计算基于名义市值的资金分配比例
            allocation_ratio = notional_value / total_notional
            # 分配资金 = 总资金 × 分配比例
            allocation[symbol] = allocation_ratio * total_capital
    else:
        # 默认等权分配
        n_symbols = len(varieties_data)
        for symbol in varieties_data:
            allocation[symbol] = total_capital / n_symbols
    
    return allocation, risk_units


def enhanced_atr_cluster_risk_allocation(total_capital, varieties_data, target_volatility=0.01):
    """
    增强型ATR聚类风险分配策略
    基于enhanced_atr_allocation获得每个品种的名义市值，按聚类累计，确保每个聚类的名义市值不超过总名义市值的1/5
    
    参数：
    total_capital: 总资金
    varieties_data: 各品种数据字典
    target_volatility: 目标单位风险
    
    返回：
    allocation: 各品种分配资金
    risk_units: 各品种风险单位数
    cluster_weights: 各聚类分配权重
    """
    import numpy as np
    import json
    import os
    
    # 步骤1：读取相关性聚类数据
    correlation_file = "Market_Inform/correlation_analysis_result.json"
    
    if not os.path.exists(correlation_file):
        # 如果相关性分析文件不存在，直接使用增强型ATR分配
        print(f"相关性分析文件不存在: {correlation_file}，使用增强型ATR分配")
        return enhanced_atr_allocation(total_capital, varieties_data, target_volatility)
    
    with open(correlation_file, 'r', encoding='utf-8') as f:
        correlation_data = json.load(f)
    
    clusters = correlation_data.get("clusters", {})
    
    # 步骤2：将品种映射到聚类
    symbol_to_cluster = {}
    for cluster_id, symbols in clusters.items():
        for symbol in symbols:
            symbol_to_cluster[symbol] = cluster_id
    
    # 步骤3：调用enhanced_atr_allocation获取每个品种的名义市值
    # 注意：enhanced_atr_allocation返回的是分配资金和风险单位数，我们需要重新计算名义市值
    allocation, risk_units = enhanced_atr_allocation(total_capital, varieties_data, target_volatility)
    
    # 重新计算每个品种的名义市值
    notional_values = {}
    for symbol, data in varieties_data.items():
        current_price = data['current_price']
        contract_multiplier = data['contract_multiplier']
        atr = data['atr']
        trend_strength = abs(data.get('trend_strength', 1))
        
        if atr > 0:
            atr_per_lot = atr * contract_multiplier
            risk_unit = (1 + trend_strength) / atr_per_lot
            notional_values[symbol] = risk_unit * current_price * contract_multiplier
        else:
            notional_values[symbol] = 0
    
    # 步骤4：按照聚类累计名义市值
    cluster_notional_values = {}
    cluster_symbols = {}
    
    # 初始化聚类符号映射
    for cluster_id in clusters:
        cluster_symbols[cluster_id] = []
        cluster_notional_values[cluster_id] = 0.0
    
    # 添加临时聚类
    cluster_symbols["temporary_cluster"] = []
    cluster_notional_values["temporary_cluster"] = 0.0
    
    # 分配品种到聚类并累计名义市值
    for symbol in varieties_data:
        if symbol in symbol_to_cluster:
            cluster_id = symbol_to_cluster[symbol]
            cluster_symbols[cluster_id].append(symbol)
            cluster_notional_values[cluster_id] += notional_values.get(symbol, 0.0)
        else:
            # 不在任何聚类中的品种放入临时聚类
            cluster_symbols["temporary_cluster"].append(symbol)
            cluster_notional_values["temporary_cluster"] += notional_values.get(symbol, 0.0)
    
    # 步骤5：计算总名义市值并确保每个聚类不超过1/5限制
    total_notional = sum(notional_values.values())
    max_cluster_notional = total_notional / 5.0  # 每个聚类的最大名义市值
    
    # 调整聚类名义市值，确保不超过限制
    adjusted_cluster_notional = {}
    excess_notional = 0.0
    
    # 第一次遍历：计算超过限制的部分
    for cluster_id, notional in cluster_notional_values.items():
        if notional > max_cluster_notional:
            adjusted_cluster_notional[cluster_id] = max_cluster_notional
            excess_notional += notional - max_cluster_notional
        else:
            adjusted_cluster_notional[cluster_id] = notional
    
    # 第二次遍历：分配超额部分
    if excess_notional > 0:
        # 计算未达到限制的聚类可分配的比例
        available_clusters = [cluster_id for cluster_id, notional in adjusted_cluster_notional.items() if notional < max_cluster_notional]
        if available_clusters:
            # 计算每个可用聚类可接收的比例
            total_available_space = sum(max_cluster_notional - adjusted_cluster_notional[cluster_id] for cluster_id in available_clusters)
            if total_available_space > 0:
                for cluster_id in available_clusters:
                    available_space = max_cluster_notional - adjusted_cluster_notional[cluster_id]
                    allocation_ratio = available_space / total_available_space
                    adjusted_cluster_notional[cluster_id] += excess_notional * allocation_ratio
    
    # 步骤6：计算聚类权重
    adjusted_total_notional = sum(adjusted_cluster_notional.values())
    cluster_weights = {}
    
    if adjusted_total_notional > 0:
        for cluster_id, notional in adjusted_cluster_notional.items():
            cluster_weights[cluster_id] = notional / adjusted_total_notional
    else:
        # 如果总调整后名义市值为0，等权分配
        n_clusters = len(adjusted_cluster_notional)
        for cluster_id in adjusted_cluster_notional:
            cluster_weights[cluster_id] = 1.0 / n_clusters
    
    # 步骤7：在每个聚类内分配资金
    final_allocation = {}
    
    for cluster_id, cluster_weight in cluster_weights.items():
        cluster_capital = total_capital * cluster_weight
        symbols = cluster_symbols[cluster_id]
        
        if not symbols:
            continue
        
        # 计算聚类内各品种的名义市值占比
        cluster_notional = sum(notional_values.get(symbol, 0.0) for symbol in symbols)
        
        if cluster_notional > 0:
            for symbol in symbols:
                if symbol in varieties_data:
                    symbol_notional = notional_values.get(symbol, 0.0)
                    symbol_allocation = (symbol_notional / cluster_notional) * cluster_capital
                    final_allocation[symbol] = symbol_allocation
        else:
            # 如果聚类内名义市值为0，等权分配
            for symbol in symbols:
                if symbol in varieties_data:
                    final_allocation[symbol] = cluster_capital / len(symbols)
    
    # 确保所有品种都有分配
    for symbol in varieties_data:
        if symbol not in final_allocation:
            final_allocation[symbol] = 0.0
    
    return final_allocation, risk_units, cluster_weights

def cluster_risk_parity_allocation(total_capital, varieties_data, target_volatility=0.01):
    """
    基于相关性聚类和风险平价的分配策略
    结合相关性聚类关系、品种相关性和ATR进行风险平价优化分配
    
    参数：
    total_capital: 总资金
    varieties_data: 各品种数据字典
    target_volatility: 目标单位风险
    
    返回：
    allocation: 各品种分配资金
    risk_units: 各品种风险单位数
    cluster_weights: 各聚类分配权重
    """
    import numpy as np
    import json
    import os
    
    # 步骤1：读取相关性聚类数据
    correlation_file = "Market_Inform/correlation_analysis_result.json"
    
    if not os.path.exists(correlation_file):
        # 如果相关性分析文件不存在，使用增强型ATR分配作为备选
        print(f"相关性分析文件不存在: {correlation_file}，使用增强型ATR分配")
        return enhanced_atr_allocation(total_capital, varieties_data, target_volatility)
    
    with open(correlation_file, 'r', encoding='utf-8') as f:
        correlation_data = json.load(f)
    
    clusters = correlation_data.get("clusters", {})
    instrument_correlations = correlation_data.get("instrument_correlations", {})
    
    # 步骤2：将品种映射到聚类
    symbol_to_cluster = {}
    for cluster_id, symbols in clusters.items():
        for symbol in symbols:
            symbol_to_cluster[symbol] = cluster_id
    
    # 步骤3：分离品种到聚类和临时聚类
    # 创建聚类符号映射
    cluster_symbols = {}
    temporary_cluster_symbols = []
    
    for symbol in varieties_data:
        if symbol in symbol_to_cluster:
            cluster_id = symbol_to_cluster[symbol]
            if cluster_id not in cluster_symbols:
                cluster_symbols[cluster_id] = []
            cluster_symbols[cluster_id].append(symbol)
        else:
            # 不在任何聚类中的品种放入临时聚类
            temporary_cluster_symbols.append(symbol)
    
    # 添加临时聚类
    if temporary_cluster_symbols:
        cluster_symbols["temporary_cluster"] = temporary_cluster_symbols
    
    # 步骤4：计算每个品种的权重（基于ATR和相关性）
    symbol_base_weights = {}
    for symbol, data in varieties_data.items():
        atr = data['atr']
        if atr > 0:
            # 初始权重：ATR越小，权重越高
            symbol_base_weights[symbol] = 1.0 / atr
        else:
            symbol_base_weights[symbol] = 1.0
    
    # 步骤5：计算每个聚类的平均ATR、平均价格和名义价值
    cluster_notional_values = {}
    
    for cluster_id, symbols in cluster_symbols.items():
        if not symbols:
            continue
        
        # 计算聚类内各品种的权重和
        total_weight = sum(symbol_base_weights.get(symbol, 1.0) for symbol in symbols if symbol in varieties_data)
        if total_weight <= 0:
            # 如果权重和为0，使用等权
            normalized_weights = {symbol: 1.0 / len(symbols) for symbol in symbols}
        else:
            # 归一化权重
            normalized_weights = {symbol: symbol_base_weights.get(symbol, 1.0) / total_weight for symbol in symbols}
        
        # 计算聚类的平均ATR、平均价格和平均趋势强度（直接平均，不考虑权重）
        atr_list = []
        price_list = []
        trend_strength_list = []
        valid_count = 0
        
        for symbol in symbols:
            if symbol in varieties_data:
                data = varieties_data[symbol]
                atr = data['atr']
                current_price = data['current_price']
                trend_strength = data.get('trend_strength', 0.0)
                
                atr_list.append(atr)
                price_list.append(current_price)
                trend_strength_list.append(trend_strength)
                valid_count += 1
        
        if valid_count == 0:
            continue
        
        # 直接计算平均值
        avg_atr = sum(atr_list) / valid_count
        avg_price = sum(price_list) / valid_count
        avg_trend_strength = sum(trend_strength_list) / valid_count
        
        # 计算聚类的名义价值：平均价格 / 平均ATR * (1 + |平均趋势强度|)
        if avg_atr > 0:
            notional_value = (avg_price / avg_atr) * (1 + abs(avg_trend_strength))
        else:
            notional_value = 1.0
        
        cluster_notional_values[cluster_id] = {
            "notional_value": notional_value,
            "avg_atr": avg_atr,
            "avg_price": avg_price,
            "normalized_weights": normalized_weights
        }
    
    # 步骤6：计算聚类权重（基于名义价值归一化）
    total_notional = sum(cluster_data["notional_value"] for cluster_data in cluster_notional_values.values())
    cluster_weights = {}
    
    if total_notional > 0:
        for cluster_id, cluster_data in cluster_notional_values.items():
            cluster_weights[cluster_id] = cluster_data["notional_value"] / total_notional
    else:
        # 如果总名义价值为0，等权分配
        n_clusters = len(cluster_notional_values)
        for cluster_id in cluster_notional_values:
            cluster_weights[cluster_id] = 1.0 / n_clusters
    
    # 步骤7：在每个聚类内二次分配资金
    allocation = {}
    risk_units = {}
    
    for cluster_id, cluster_weight in cluster_weights.items():
        symbols = cluster_symbols[cluster_id]
        cluster_data = cluster_notional_values[cluster_id]
        normalized_weights = cluster_data["normalized_weights"]
        
        # 计算聚类分配的总资金
        cluster_capital = total_capital * cluster_weight
        
        # 直接考虑所有品种，不检查方向集中度
        selected_symbols = [symbol for symbol in symbols if symbol in varieties_data]
        
        if not selected_symbols:
            continue
        
        # 重新计算所选品种的权重和
        selected_total_weight = sum(normalized_weights.get(symbol, 1.0) for symbol in selected_symbols)
        if selected_total_weight <= 0:
            # 等权分配
            selected_weights = {symbol: 1.0 / len(selected_symbols) for symbol in selected_symbols}
        else:
            # 归一化权重
            selected_weights = {symbol: normalized_weights.get(symbol, 1.0) / selected_total_weight for symbol in selected_symbols}
        
        # 二次分配：考虑ATR调整
        final_weights = {}
        total_adjusted_weight = 0.0
        
        for symbol in selected_symbols:
            data = varieties_data[symbol]
            atr = data['atr']
            weight = selected_weights[symbol]
            
            # ATR调整：ATR越小，权重越高
            if atr > 0:
                adjusted_weight = weight / atr
            else:
                adjusted_weight = weight
            
            final_weights[symbol] = adjusted_weight
            total_adjusted_weight += adjusted_weight
        
        # 最终归一化
        if total_adjusted_weight > 0:
            final_normalized_weights = {symbol: weight / total_adjusted_weight for symbol, weight in final_weights.items()}
        else:
            final_normalized_weights = {symbol: 1.0 / len(selected_symbols) for symbol in selected_symbols}
        
        # 分配聚类资金到每个品种
        for symbol, weight in final_normalized_weights.items():
            symbol_capital = cluster_capital * weight
            allocation[symbol] = symbol_capital
            
            # 计算风险单位数
            data = varieties_data[symbol]
            current_price = data['current_price']
            atr = data['atr']
            contract_multiplier = data['contract_multiplier']
            
            if atr > 0:
                atr_value = atr * contract_multiplier
                risk_unit = symbol_capital * target_volatility / atr_value
                risk_units[symbol] = risk_unit
            else:
                risk_units[symbol] = 0
    
    # 归一化总资金，确保总和等于总资金
    total_allocated = sum(allocation.values())
    if total_allocated > 0:
        for symbol in allocation:
            allocation[symbol] = (allocation[symbol] / total_allocated) * total_capital
    else:
        # 如果总分配为0，使用增强型ATR分配
        return enhanced_atr_allocation(total_capital, varieties_data, target_volatility)
    
    return allocation, risk_units, cluster_weights


def enhanced_sharpe_atr_allocation(total_capital, varieties_data, target_volatility=0.01, market_params=None):
    """
    基于夏普比率优化的增强型ATR分配策略
    结合ATR分配、夏普比率优化和相关性分析，目标是最大化夏普比率和收益率
    
    参数：
    total_capital: 总资金
    varieties_data: 各品种数据字典
    target_volatility: 目标单位风险
    market_params: 市场条件参数，包含波动率和相关性等信息
    
    返回：
    allocation: 各品种分配资金
    risk_units: 各品种风险单位数
    """
    # 初始化市场参数
    if market_params is None:
        market_params = {
            'market_volatility': 0.02,  # 默认波动率
            'avg_correlation': 0.3,      # 默认相关性
            'optimize_date': None,
            'lookback_days': 180
        }
    import numpy as np
    import json
    import os
    
    # 步骤1：初始化
    risk_units = {}
    notional_values = {}
    margin_adjusted_values = {}
    sharpe_scores = {}
    
    # 根据市场波动率动态调整目标波动率
    market_volatility = market_params['market_volatility']
    avg_correlation = market_params['avg_correlation']
    
    # 高波动率市场：降低目标波动率，减少风险暴露
    # 低波动率市场：提高目标波动率，增加收益机会
    adjusted_volatility = target_volatility * (1 - min(0.5, market_volatility / 0.03))
    
    # 步骤2：遍历品种数据，计算风险单位数、名义市值和夏普比率
    for symbol, data in varieties_data.items():
        # 获取每个品种的当前价格、ATR、合约乘数、趋势强度和保证金率
        current_price = data['current_price']
        contract_multiplier = data['contract_multiplier']
        trend_strength = abs(data.get('trend_strength', 1))  # 默认趋势强度为1
        margin_rate = data.get('margin_rate', 0.05)  # 默认保证金率为5%
        
        # 直接使用默认20日ATR，不设最小值
        atr = data['atr']
        
        # 计算每手ATR价值
        atr_per_lot = atr * contract_multiplier
        
        if atr_per_lot > 0:
            # 添加止损机制：ATR过高的品种直接降低权重
            if atr > 0.1 * current_price:  # ATR超过当前价格10%的品种，降低权重
                trend_strength *= 0.5  # 降低趋势强度权重
            
            # 根据市场条件调整趋势强度权重
            # 高相关性市场：降低趋势强度权重，增加多样性
            # 低相关性市场：提高趋势强度权重，利用趋势跟踪
            trend_strength *= (1 + (0.5 - avg_correlation))
            
            # 计算风险单位数：风险单位数 = (1 + 趋势强度) ÷ 每手ATR价值
            risk_unit = (1 + trend_strength) / atr_per_lot
            risk_units[symbol] = risk_unit
            
            # 计算名义市值：名义市值 = 风险单位数 × 当前价格 × 合约乘数
            notional_value = risk_unit * current_price * contract_multiplier
            notional_values[symbol] = notional_value
            
            # 计算保证金调整后的价值
            margin_adjusted_value = notional_value * margin_rate
            margin_adjusted_values[symbol] = margin_adjusted_value
            
            # 计算夏普比率得分：结合趋势强度和ATR（越低风险越高）
            # 公式：夏普比率得分 = (1 + 趋势强度) / (atr + 0.0001)  # 避免除以零
            sharpe_score = (1 + trend_strength) / (atr + 0.0001)
            sharpe_scores[symbol] = sharpe_score
        else:
            risk_units[symbol] = 0
            notional_values[symbol] = 0
            margin_adjusted_values[symbol] = 0
            sharpe_scores[symbol] = 0
    
    # 步骤3：读取相关性数据
    correlation_file = "Market_Inform/correlation_analysis_result.json"
    correlations = {}
    clusters = {}
    symbol_to_cluster = {}
    
    if os.path.exists(correlation_file):
        with open(correlation_file, 'r', encoding='utf-8') as f:
            correlation_data = json.load(f)
        
        correlations = correlation_data.get("instrument_correlations", {})
        clusters = correlation_data.get("clusters", {})
        
        # 构建品种到聚类的映射
        for cluster_id, symbols in clusters.items():
            for symbol in symbols:
                symbol_to_cluster[symbol] = cluster_id
    
    # 步骤4：计算初始权重
    initial_weights = {}
    total_sharpe_score = sum(sharpe_scores.values())
    
    if total_sharpe_score > 0:
        for symbol in varieties_data:
            initial_weights[symbol] = sharpe_scores[symbol] / total_sharpe_score
    else:
        # 默认等权
        n_symbols = len(varieties_data)
        for symbol in varieties_data:
            initial_weights[symbol] = 1.0 / n_symbols
    
    # 步骤5：基于相关性调整权重
    correlation_adjusted_weights = initial_weights.copy()
    
    if correlations:
        # 获取所有品种列表
        symbols = list(varieties_data.keys())
        n = len(symbols)
        
        # 创建相关矩阵
        corr_matrix = np.eye(n)
        for i in range(n):
            for j in range(i+1, n):
                s1 = symbols[i]
                s2 = symbols[j]
                corr = correlations.get(s1, {}).get(s2, 0.0)
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
        
        # 计算每个品种的平均相关性
        avg_correlations = np.mean(corr_matrix, axis=1)
        
        # 调整权重：平均相关性越低，权重越高；波动率越低，权重越高
        for i, symbol in enumerate(symbols):
            # 相关性调整因子：使用更平滑的指数函数，使得低相关性品种获得更高权重
            corr_factor = (1 - avg_correlations[i]) ** 4
            
            # 添加波动率调整：使用对数处理，避免极端值影响
            atr = varieties_data[symbol]['atr']
            # 波动率调整因子：使用对数处理，确保波动率越低权重越高
            vol_factor = np.log(1 + 100 / (atr + 0.0001))
            
            # 结合相关性和波动率调整因子
            combined_factor = corr_factor * vol_factor
            correlation_adjusted_weights[symbol] *= combined_factor
    
    # 步骤6：基于聚类调整权重，确保聚类多样性
    cluster_adjusted_weights = correlation_adjusted_weights.copy()
    
    if clusters:
        # 计算每个聚类的总权重
        cluster_weights = {}
        for cluster_id in clusters:
            cluster_weights[cluster_id] = sum(correlation_adjusted_weights.get(symbol, 0) for symbol in clusters[cluster_id] if symbol in correlation_adjusted_weights)
        
        # 确保每个聚类的权重不超过总权重的1/5，更严格的聚类风险控制
        max_cluster_weight = 0.20
        excess_weights = {}
        
        # 计算超额权重
        for cluster_id, weight in cluster_weights.items():
            if weight > max_cluster_weight:
                excess_weights[cluster_id] = weight - max_cluster_weight
                cluster_weights[cluster_id] = max_cluster_weight
        
        # 分配超额权重到权重不足的聚类，优先分配给低相关性聚类
        total_excess = sum(excess_weights.values())
        if total_excess > 0:
            # 计算可分配的聚类
            under_clusters = [cluster_id for cluster_id, weight in cluster_weights.items() if weight < max_cluster_weight]
            if under_clusters:
                # 计算每个可分配聚类的分配比例，基于现有权重和
                total_available = sum(max_cluster_weight - cluster_weights[cluster_id] for cluster_id in under_clusters)
                if total_available > 0:
                    for cluster_id in under_clusters:
                        available_space = max_cluster_weight - cluster_weights[cluster_id]
                        allocation_ratio = available_space / total_available
                        cluster_weights[cluster_id] += total_excess * allocation_ratio
        
        # 根据调整后的聚类权重调整品种权重
        for symbol in cluster_adjusted_weights:
            if symbol in symbol_to_cluster:
                cluster_id = symbol_to_cluster[symbol]
                # 调整品种权重，使其所在聚类的总权重符合限制
                original_cluster_weight = sum(correlation_adjusted_weights.get(s, 0) for s in clusters[cluster_id] if s in correlation_adjusted_weights)
                if original_cluster_weight > 0:
                    adjustment_factor = cluster_weights[cluster_id] / original_cluster_weight
                    cluster_adjusted_weights[symbol] *= adjustment_factor
    
    # 步骤7：计算最终分配
    total_adjusted_weight = sum(cluster_adjusted_weights.values())
    allocation = {}
    
    if total_adjusted_weight > 0:
        for symbol in varieties_data:
            allocation[symbol] = (cluster_adjusted_weights[symbol] / total_adjusted_weight) * total_capital
    else:
        # 默认等权分配
        n_symbols = len(varieties_data)
        for symbol in varieties_data:
            allocation[symbol] = total_capital / n_symbols
    
def calculate_trend_signal_strength(prices, window=20):
    """
    计算价格序列的趋势信号强度
    使用价格序列回归斜率的t统计量
    
    参数：
    prices: 价格列表
    window: 计算窗口
    
    返回：
    t_statistic: 回归斜率的t统计量
    """
    import numpy as np
    from scipy import stats
    
    if len(prices) < window:
        return 0.0
    
    # 取最近window个价格数据
    recent_prices = prices[-window:]
    
    # 创建时间序列（1, 2, ..., window）
    x = np.arange(1, window + 1)
    y = np.array(recent_prices)
    
    # 执行线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # 计算t统计量
    if std_err > 0:
        t_statistic = slope / std_err
    else:
        # 对于完美线性趋势，std_err为0，t统计量应为很大的值
        # 使用r_value来判断趋势强度，r_value=1或-1表示完美趋势
        t_statistic = slope * 1000 * np.sign(slope)  # 给一个大的t统计量
    
    return t_statistic

def signal_strength_based_allocation(total_capital, varieties_data, target_volatility=0.01, volatility_window=20):
    """
    基于信号强度的风险分配
    每个品种的风险贡献与信号强度的绝对值成正比
    信号强度使用价格序列回归斜率的t统计量
    使用历史波动率（收益率标准差）而非ATR
    
    参数：
    total_capital: 总资金
    varieties_data: 各品种数据字典
    target_volatility: 目标单位风险
    volatility_window: 历史波动率计算窗口（默认20天）
    
    返回：
    allocation: 各品种分配资金
    risk_units: 各品种风险单位数
    """
    import numpy as np
    
    # 计算历史波动率（收益率标准差）
    def calculate_historical_volatility(prices, window=20):
        """
        计算历史波动率
        
        参数：
        prices: 价格列表
        window: 计算窗口
        
        返回：
        volatility: 历史波动率（年化）
        """
        if len(prices) < window + 1:
            return 0.0
        
        # 计算收益率
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        
        # 计算窗口内的收益率标准差
        recent_returns = returns[-window:]
        if len(recent_returns) < window:
            return 0.0
        
        # 计算日波动率并年化（假设252个交易日）
        daily_vol = np.std(recent_returns)
        annual_vol = daily_vol * np.sqrt(252)
        
        return annual_vol
    
    # 步骤1：计算各品种的信号强度（t统计量）
    signal_strengths = {}
    volatilities = {}
    
    for symbol, data in varieties_data.items():
        prices = data.get('prices', data.get('price_history', [data['current_price']]))
        t_statistic = calculate_trend_signal_strength(prices)
        signal_strengths[symbol] = abs(t_statistic)  # 使用绝对值，只考虑信号强度
        
        # 计算历史波动率
        volatility = calculate_historical_volatility(prices, window=volatility_window)
        volatilities[symbol] = volatility
    
    # 步骤2：计算初始风险单位数（基础风险分配）
    base_risk_units = {}
    for symbol, volatility in volatilities.items():
        current_price = varieties_data[symbol]['current_price']
        contract_multiplier = varieties_data[symbol]['contract_multiplier']
        
        if volatility > 0:
            # 计算波动率的货币价值
            # 年化波动率转换为日波动率，然后计算每手的风险价值
            daily_vol = volatility / np.sqrt(252)
            risk_value_per_lot = current_price * daily_vol * contract_multiplier
            base_risk_units[symbol] = 1 / risk_value_per_lot  # 基础风险单位：1/风险价值
        else:
            base_risk_units[symbol] = 0
    
    # 步骤3：根据信号强度调整风险单位数
    adjusted_risk_units = {}
    for symbol in varieties_data:
        base_unit = base_risk_units[symbol]
        signal_strength = signal_strengths[symbol]
        # 信号强度越高，风险单位数越大，给予更高的风险预算
        adjusted_risk_units[symbol] = base_unit * (1 + signal_strength)
    
    # 步骤4：计算名义市值
    notional_values = {}
    for symbol, data in varieties_data.items():
        if adjusted_risk_units[symbol] > 0:
            current_price = data['current_price']
            contract_multiplier = data['contract_multiplier']
            notional_value = adjusted_risk_units[symbol] * current_price * contract_multiplier
            notional_values[symbol] = notional_value
        else:
            notional_values[symbol] = 0
    
    # 步骤5：归一化分配
    total_notional = sum(notional_values.values())
    allocation = {}
    
    if total_notional > 0:
        for symbol, notional_value in notional_values.items():
            allocation[symbol] = (notional_value / total_notional) * total_capital
    else:
        # 默认等权分配
        n_symbols = len(varieties_data)
        for symbol in varieties_data:
            allocation[symbol] = total_capital / n_symbols
    
    # 步骤6：计算最终风险单位数
    risk_units = {}
    for symbol, data in varieties_data.items():
        current_price = data['current_price']
        contract_multiplier = data['contract_multiplier']
        volatility = volatilities[symbol]
        
        if volatility > 0:
            # 计算波动率的货币价值
            daily_vol = volatility / np.sqrt(252)
            risk_value_per_lot = current_price * daily_vol * contract_multiplier
            
            # 风险单位数 = 分配资金 × 目标波动率 / 风险价值
            risk_unit = (allocation[symbol] * target_volatility) / risk_value_per_lot
            risk_units[symbol] = risk_unit
        else:
            risk_units[symbol] = 0
    
    return allocation, risk_units

def model_based_allocation(total_capital, varieties_data, target_volatility=0.01, market_params=None):
    """
    基于LightGBM模型的智能分配策略
    每80天重新训练模型，使用历史波动率、相关性、趋势强度和过往夏普值等参数
    
    参数：
    total_capital: 总资金
    varieties_data: 各品种数据字典
    target_volatility: 目标单位风险
    market_params: 市场条件参数，包含波动率、相关性、过往分配和回报等信息
    
    返回：
    allocation: 各品种分配资金
    risk_units: 各品种风险单位数
    """
    from random_forest_strategy.model_based_allocation import ModelBasedAllocation
    import numpy as np
    
    # 计算历史波动率（收益率标准差）
    def calculate_historical_volatility(prices, window=20):
        """
        计算历史波动率
        
        参数：
        prices: 价格列表
        window: 计算窗口
        
        返回：
        volatility: 历史波动率（年化）
        """
        if len(prices) < window + 1:
            return 0.0
        
        # 计算收益率
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        
        # 计算窗口内的收益率标准差
        recent_returns = returns[-window:]
        if len(recent_returns) < window:
            return 0.0
        
        # 计算日波动率并年化（假设252个交易日）
        daily_vol = np.std(recent_returns)
        annual_vol = daily_vol * np.sqrt(252)
        
        return annual_vol
    
    # 初始化市场参数
    if market_params is None:
        market_params = {
            'market_volatility': 0.02,
            'avg_correlation': 0.3,
            'optimize_date': None,
            'lookback_days': 180,
            'past_allocations': {},
            'past_returns': {},
            'market_returns': []
        }
    
    # 过滤掉没有足够历史数据的品种
    filtered_varieties_data = {}
    min_history_length = 20  # 最小历史数据长度要求
    for symbol, data in varieties_data.items():
        price_history = data.get('price_history', [data['current_price']])
        # 检查是否有足够的历史数据
        if len(price_history) >= min_history_length:
            filtered_varieties_data[symbol] = data
    
    # 添加日志
    print(f"原始品种数量: {len(varieties_data)}")
    print(f"过滤后品种数量: {len(filtered_varieties_data)}")
    
    # 如果过滤后没有品种，返回空分配
    if not filtered_varieties_data:
        return {}, {}
    
    # 初始化模型基分配器
    model_allocator = ModelBasedAllocation(re_train_interval=80)
    
    # 准备模型所需的输入数据
    # 1. 构建instrument_data
    instrument_data = {}
    for symbol, data in filtered_varieties_data.items():
        instrument_data[symbol] = {
            'close': data.get('price_history', [data['current_price']]),
            'high': data.get('high_history', [data['current_price']]),
            'low': data.get('low_history', [data['current_price']])
        }
    
    # 2. 构建market_data
    market_data = {
        'returns': market_params.get('market_returns', [])
    }
    
    # 3. 获取past_allocations和past_returns
    past_allocations = market_params.get('past_allocations', {})
    past_returns = market_params.get('past_returns', {})
    
    # 4. 获取当前日期
    current_date = market_params.get('optimize_date', None)
    
    # 调用模型进行分配
    allocation_weights = model_allocator.allocate(
        instrument_data, 
        market_data, 
        past_allocations, 
        past_returns, 
        current_date
    )
    
    # 添加日志
    print(f"模型分配权重: {allocation_weights}")
    
    # 计算历史波动率
    volatilities = {}
    for symbol, data in filtered_varieties_data.items():
        prices = data.get('price_history', [data['current_price']])
        volatility = calculate_historical_volatility(prices, window=20)
        volatilities[symbol] = volatility
    
    # 计算风险单位数量
    target_risk_amount = total_capital * target_volatility
    risk_units = {}
    
    for symbol, data in filtered_varieties_data.items():
        current_price = data['current_price']
        contract_multiplier = data['contract_multiplier']
        volatility = volatilities[symbol]
        
        if volatility > 0:
            # 计算波动率的货币价值
            daily_vol = volatility / np.sqrt(252)
            risk_value_per_lot = current_price * daily_vol * contract_multiplier
            risk_units[symbol] = target_risk_amount / risk_value_per_lot
        else:
            risk_units[symbol] = 0
    
    # 计算名义市值
    notional_values = {}
    for symbol, data in filtered_varieties_data.items():
        if risk_units[symbol] > 0:
            notional_value = risk_units[symbol] * data['current_price'] * data['contract_multiplier']
            notional_values[symbol] = notional_value
        else:
            notional_values[symbol] = 0
    
    # 根据模型预测的权重分配资金
    allocation = {}
    for symbol in filtered_varieties_data:
        # 使用模型预测的权重
        weight = allocation_weights.get(symbol, 1.0 / len(filtered_varieties_data))
        allocation[symbol] = weight * total_capital
    
    # 归一化资金分配，确保总和等于总资金
    total_allocated = sum(allocation.values())
    if total_allocated > 0:
        for symbol in allocation:
            allocation[symbol] = (allocation[symbol] / total_allocated) * total_capital
    else:
        # 默认等权分配
        n_symbols = len(filtered_varieties_data)
        for symbol in filtered_varieties_data:
            allocation[symbol] = total_capital / n_symbols
    
    return allocation, risk_units
