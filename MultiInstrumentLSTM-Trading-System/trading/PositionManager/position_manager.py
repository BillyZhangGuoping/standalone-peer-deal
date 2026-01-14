import numpy as np
import pandas as pd

class PositionManager:
    def __init__(self, variety_order, initial_capital=1000000, max_leverage=3.0, max_position_percent=0.05):
        """
        初始化仓位管理器
        
        参数:
        - variety_order: 品种顺序列表
        - initial_capital: 初始资金
        - max_leverage: 最大杠杆率
        - max_position_percent: 单个品种最大持仓比例
        """
        self.variety_order = variety_order
        self.initial_capital = initial_capital
        self.max_leverage = max_leverage
        self.max_position_percent = max_position_percent
        
        # 初始化当前仓位
        self.current_positions = np.zeros(len(variety_order))
        self.current_capital = initial_capital
    
    def update_positions(self, signals, prices, contract_multipliers):
        """
        根据信号更新仓位
        
        参数:
        - signals: 交易信号，形状为 (num_varieties,)
        - prices: 当前价格，形状为 (num_varieties,)
        - contract_multipliers: 合约乘数列表，形状为 (num_varieties,)
        
        返回:
        - new_positions: 新的仓位，形状为 (num_varieties,)
        - position_changes: 仓位变化，形状为 (num_varieties,)
        """
        # 计算目标仓位
        target_positions = self._calculate_target_positions(signals, prices, contract_multipliers)
        
        # 应用仓位约束
        target_positions = self._apply_position_constraints(target_positions, prices, contract_multipliers)
        
        # 计算仓位变化
        position_changes = target_positions - self.current_positions
        
        # 更新当前仓位
        self.current_positions = target_positions
        
        return target_positions, position_changes
    
    def _calculate_target_positions(self, signals, prices, contract_multipliers):
        """
        计算目标仓位
        
        参数:
        - signals: 交易信号，形状为 (num_varieties,)
        - prices: 当前价格，形状为 (num_varieties,)
        - contract_multipliers: 合约乘数列表，形状为 (num_varieties,)
        
        返回:
        - target_positions: 目标仓位，形状为 (num_varieties,)
        """
        target_positions = []
        
        for i in range(len(self.variety_order)):
            signal = signals[i]
            price = prices[i]
            multiplier = contract_multipliers[i]
            
            # 计算目标持仓数量
            target_weight = signal
            target_position_value = self.current_capital * target_weight
            target_position = target_position_value / (price * multiplier)
            
            target_positions.append(target_position)
        
        return np.array(target_positions)
    
    def _apply_position_constraints(self, positions, prices, contract_multipliers):
        """
        应用仓位约束
        
        参数:
        - positions: 目标仓位，形状为 (num_varieties,)
        - prices: 当前价格，形状为 (num_varieties,)
        - contract_multipliers: 合约乘数列表，形状为 (num_varieties,)
        
        返回:
        - constrained_positions: 应用约束后的仓位，形状为 (num_varieties,)
        """
        constrained_positions = positions.copy()
        
        # 计算每个品种的持仓价值
        position_values = constrained_positions * prices * contract_multipliers
        
        # 单个品种最大持仓比例约束
        max_position_value = self.current_capital * self.max_position_percent
        for i in range(len(constrained_positions)):
            if abs(position_values[i]) > max_position_value:
                # 调整仓位到最大允许值
                constrained_positions[i] = (max_position_value * np.sign(constrained_positions[i])) / (prices[i] * contract_multipliers[i])
        
        # 总杠杆约束
        total_position_value = np.sum(abs(position_values))
        max_total_position_value = self.current_capital * self.max_leverage
        
        if total_position_value > max_total_position_value:
            # 按比例调整所有仓位
            adjustment_ratio = max_total_position_value / (total_position_value + 1e-8)
            constrained_positions *= adjustment_ratio
        
        return constrained_positions
    
    def calculate_position_value(self, prices, contract_multipliers):
        """
        计算当前持仓价值
        
        参数:
        - prices: 当前价格，形状为 (num_varieties,)
        - contract_multipliers: 合约乘数列表，形状为 (num_varieties,)
        
        返回:
        - total_position_value: 总持仓价值
        - position_values: 每个品种的持仓价值，形状为 (num_varieties,)
        """
        position_values = self.current_positions * prices * contract_multipliers
        total_position_value = np.sum(position_values)
        
        return total_position_value, position_values
    
    def update_capital(self, new_capital):
        """
        更新当前资金
        
        参数:
        - new_capital: 新的资金额
        """
        self.current_capital = new_capital
    
    def get_current_positions(self):
        """
        获取当前仓位
        
        返回:
        - current_positions: 当前仓位，形状为 (num_varieties,)
        """
        return self.current_positions
