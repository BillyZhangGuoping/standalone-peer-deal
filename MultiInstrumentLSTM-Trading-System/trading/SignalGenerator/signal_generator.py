import numpy as np
import pandas as pd

class SignalGenerator:
    def __init__(self, variety_order):
        """
        初始化信号生成器
        
        参数:
        - variety_order: 品种顺序列表
        """
        self.variety_order = variety_order
    
    def generate_signals(self, predictions, threshold=0.05):
        """
        根据模型预测生成交易信号
        
        参数:
        - predictions: 预测持仓权重，形状为 (batch_size, num_varieties)
        - threshold: 信号阈值，当预测权重绝对值大于该值时生成信号
        
        返回:
        - signals: 交易信号，形状为 (batch_size, num_varieties)
        """
        signals = []
        
        for pred in predictions:
            # 生成信号：-1 做空，0 不操作，1 做多
            signal = np.where(pred > threshold, 1, np.where(pred < -threshold, -1, 0))
            signals.append(signal)
        
        return np.array(signals)
    
    def generate_weight_signals(self, predictions):
        """
        直接使用预测权重作为信号
        
        参数:
        - predictions: 预测持仓权重，形状为 (batch_size, num_varieties)
        
        返回:
        - signals: 交易信号，形状为 (batch_size, num_varieties)
        """
        return predictions
    
    def smooth_signals(self, signals, window=5):
        """
        平滑交易信号，减少信号频繁切换
        
        参数:
        - signals: 交易信号，形状为 (batch_size, num_varieties)
        - window: 平滑窗口大小
        
        返回:
        - smoothed_signals: 平滑后的交易信号，形状为 (batch_size, num_varieties)
        """
        smoothed_signals = []
        
        for i in range(len(signals)):
            if i < window - 1:
                # 前window-1个信号直接使用原始信号
                smoothed_signals.append(signals[i])
            else:
                # 计算最近window个信号的平均值
                window_signals = signals[i-window+1:i+1]
                avg_signal = np.mean(window_signals, axis=0)
                smoothed_signals.append(avg_signal)
        
        return np.array(smoothed_signals)
    
    def generate_trade_orders(self, signals, previous_positions, capital, prices, contract_multipliers):
        """
        根据信号和当前持仓生成交易订单
        
        参数:
        - signals: 交易信号，形状为 (num_varieties,)
        - previous_positions: 前一天持仓，形状为 (num_varieties,)
        - capital: 当前资金
        - prices: 当前价格，形状为 (num_varieties,)
        - contract_multipliers: 合约乘数列表，形状为 (num_varieties,)
        
        返回:
        - orders: 交易订单，形状为 (num_varieties,)
        """
        orders = []
        
        for i in range(len(self.variety_order)):
            variety = self.variety_order[i]
            signal = signals[i]
            prev_position = previous_positions[i]
            price = prices[i]
            multiplier = contract_multipliers[i]
            
            # 计算目标持仓数量
            target_weight = signal
            target_position = capital * target_weight / (price * multiplier)
            
            # 计算需要交易的数量
            order = target_position - prev_position
            orders.append(order)
        
        return np.array(orders)
