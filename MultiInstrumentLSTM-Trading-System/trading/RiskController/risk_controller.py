import numpy as np
import pandas as pd

class RiskController:
    def __init__(self, variety_order, max_drawdown=0.2, max_volatility=0.3, stop_loss=-0.05, take_profit=0.1):
        """
        初始化风险控制器
        
        参数:
        - variety_order: 品种顺序列表
        - max_drawdown: 最大允许回撤
        - max_volatility: 最大允许波动率
        - stop_loss: 单个品种止损阈值
        - take_profit: 单个品种止盈阈值
        """
        self.variety_order = variety_order
        self.max_drawdown = max_drawdown
        self.max_volatility = max_volatility
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # 初始化风险监控变量
        self.high_watermark = 0.0
        self.current_drawdown = 0.0
        self.rolling_volatility = 0.0
    
    def check_risk(self, portfolio_value, daily_returns, positions, prices, contract_multipliers):
        """
        检查投资组合风险
        
        参数:
        - portfolio_value: 当前投资组合价值
        - daily_returns: 每日收益率序列
        - positions: 当前仓位，形状为 (num_varieties,)
        - prices: 当前价格，形状为 (num_varieties,)
        - contract_multipliers: 合约乘数列表，形状为 (num_varieties,)
        
        返回:
        - risk_alerts: 风险警报字典
        - risk_adjustments: 风险调整建议
        """
        risk_alerts = {}
        risk_adjustments = {}
        
        # 计算最大回撤
        drawdown_alert, drawdown_adjustment = self._check_drawdown(portfolio_value)
        if drawdown_alert:
            risk_alerts['drawdown'] = drawdown_alert
            risk_adjustments['drawdown'] = drawdown_adjustment
        
        # 计算波动率
        volatility_alert, volatility_adjustment = self._check_volatility(daily_returns)
        if volatility_alert:
            risk_alerts['volatility'] = volatility_alert
            risk_adjustments['volatility'] = volatility_adjustment
        
        # 检查单个品种止损止盈
        stop_loss_alert, stop_loss_adjustment = self._check_stop_loss_take_profit(positions, prices, contract_multipliers)
        if stop_loss_alert:
            risk_alerts['stop_loss_take_profit'] = stop_loss_alert
            risk_adjustments['stop_loss_take_profit'] = stop_loss_adjustment
        
        return risk_alerts, risk_adjustments
    
    def _check_drawdown(self, portfolio_value):
        """
        检查最大回撤
        
        参数:
        - portfolio_value: 当前投资组合价值
        
        返回:
        - alert: 风险警报
        - adjustment: 风险调整建议
        """
        # 更新高水位线
        if portfolio_value > self.high_watermark:
            self.high_watermark = portfolio_value
        
        # 计算当前回撤
        self.current_drawdown = (self.high_watermark - portfolio_value) / self.high_watermark
        
        alert = None
        adjustment = None
        
        if self.current_drawdown > self.max_drawdown:
            alert = f"Max drawdown exceeded: {self.current_drawdown:.2%} > {self.max_drawdown:.2%}"
            # 建议降低仓位
            adjustment = {"action": "reduce_position", "ratio": 0.5}
        
        return alert, adjustment
    
    def _check_volatility(self, daily_returns, window=20):
        """
        检查波动率
        
        参数:
        - daily_returns: 每日收益率序列
        - window: 计算波动率的窗口大小
        
        返回:
        - alert: 风险警报
        - adjustment: 风险调整建议
        """
        if len(daily_returns) < window:
            return None, None
        
        # 计算滚动波动率
        self.rolling_volatility = daily_returns.tail(window).std() * np.sqrt(252)
        
        alert = None
        adjustment = None
        
        if self.rolling_volatility > self.max_volatility:
            alert = f"Max volatility exceeded: {self.rolling_volatility:.2%} > {self.max_volatility:.2%}"
            # 建议降低仓位
            adjustment = {"action": "reduce_position", "ratio": 0.3}
        
        return alert, adjustment
    
    def _check_stop_loss_take_profit(self, positions, prices, contract_multipliers):
        """
        检查单个品种的止损止盈
        
        参数:
        - positions: 当前仓位，形状为 (num_varieties,)
        - prices: 当前价格，形状为 (num_varieties,)
        - contract_multipliers: 合约乘数列表，形状为 (num_varieties,)
        
        返回:
        - alert: 风险警报
        - adjustment: 风险调整建议
        """
        alerts = []
        adjustments = []
        
        for i in range(len(self.variety_order)):
            variety = self.variety_order[i]
            position = positions[i]
            price = prices[i]
            multiplier = contract_multipliers[i]
            
            if position == 0:
                continue
            
            # 这里简化处理，实际应该跟踪每个品种的开仓价格
            # 假设开仓价格为前一天的收盘价
            # 这里需要根据实际情况调整
            open_price = price  # 简化处理
            current_pnl = (price - open_price) * position * multiplier
            position_value = abs(open_price * position * multiplier)
            pnl_ratio = current_pnl / position_value
            
            if pnl_ratio <= self.stop_loss:
                alerts.append(f"Stop loss triggered for {variety}: {pnl_ratio:.2%} <= {self.stop_loss:.2%}")
                adjustments.append({"variety": variety, "action": "close_position"})
            elif pnl_ratio >= self.take_profit:
                alerts.append(f"Take profit triggered for {variety}: {pnl_ratio:.2%} >= {self.take_profit:.2%}")
                adjustments.append({"variety": variety, "action": "close_position"})
        
        alert = alerts if alerts else None
        adjustment = adjustments if adjustments else None
        
        return alert, adjustment
    
    def adjust_positions_for_risk(self, positions, risk_adjustments):
        """
        根据风险调整建议调整仓位
        
        参数:
        - positions: 当前仓位，形状为 (num_varieties,)
        - risk_adjustments: 风险调整建议
        
        返回:
        - adjusted_positions: 调整后的仓位，形状为 (num_varieties,)
        """
        adjusted_positions = positions.copy()
        
        # 处理回撤调整
        if 'drawdown' in risk_adjustments:
            adjustment = risk_adjustments['drawdown']
            if adjustment['action'] == 'reduce_position':
                ratio = adjustment['ratio']
                adjusted_positions *= (1 - ratio)
        
        # 处理波动率调整
        if 'volatility' in risk_adjustments:
            adjustment = risk_adjustments['volatility']
            if adjustment['action'] == 'reduce_position':
                ratio = adjustment['ratio']
                adjusted_positions *= (1 - ratio)
        
        # 处理止损止盈调整
        if 'stop_loss_take_profit' in risk_adjustments:
            adjustments = risk_adjustments['stop_loss_take_profit']
            for adj in adjustments:
                if adj['action'] == 'close_position':
                    variety = adj['variety']
                    idx = self.variety_order.index(variety)
                    adjusted_positions[idx] = 0
        
        return adjusted_positions
