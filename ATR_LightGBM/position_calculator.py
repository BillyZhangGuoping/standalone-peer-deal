"""
position_calculator.py - 持仓手数计算核心逻辑
"""

import pandas as pd
import numpy as np
import os
import logging

# 从本地模块导入合约工具函数
import sys
import os

# 添加当前目录到Python路径，确保可以导入instrument_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from instrument_utils import get_contract_multiplier

def calculate_position_size(capital, risk_per_trade, entry_price, stop_loss_price, symbol, margin_rate=None):
    """计算仓位大小
    
    参数：
    capital: 总资金
    risk_per_trade: 每笔交易风险比例
    entry_price: 入场价格
    stop_loss_price: 止损价格
    symbol: 合约代码
    margin_rate: 保证金率（可选，默认从合约信息中获取）
    
    返回：
    position_size: 仓位大小
    risk_amount: 风险金额
    """
    # 计算风险金额
    risk_amount = capital * risk_per_trade
    
    # 获取合约乘数和保证金率
    contract_multiplier, default_margin_rate = get_contract_multiplier(symbol)
    margin_rate = margin_rate if margin_rate is not None else default_margin_rate
    
    # 计算每手风险（基于止损）
    risk_per_unit = abs(entry_price - stop_loss_price) * contract_multiplier
    
    if risk_per_unit <= 0:
        return 0, 0
    
    # 基于止损风险计算的手数
    position_size_by_risk = risk_amount / risk_per_unit
    
    # 基于保证金占用计算的手数
    # 保证金占用 = 手数 * 入场价格 * 合约乘数 * 保证金率
    # 为了确保保证金占用不超过风险金额，我们有：
    # 手数 * 入场价格 * 合约乘数 * 保证金率 <= 风险金额
    # 因此：
    position_size_by_margin = risk_amount / (entry_price * contract_multiplier * margin_rate)
    
    # 取两种计算方式中的较小值
    position_size = min(position_size_by_risk, position_size_by_margin)
    
    # 取整
    position_size = int(position_size)
    
    return position_size, risk_amount
