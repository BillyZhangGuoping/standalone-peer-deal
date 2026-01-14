import tensorflow as tf
import numpy as np

def portfolio_return_loss(contract_multipliers=None, l2_regularization=0.001, turnover_penalty=0.0):
    """
    投资组合收益损失函数
    
    参数:
    - contract_multipliers: 合约乘数列表，形状为 (num_varieties,)
    - l2_regularization: L2正则化系数
    - turnover_penalty: 换手率惩罚系数
    
    返回:
    - loss_fn: 损失函数
    """
    def loss_fn(y_true, y_pred):
        """
        计算损失
        
        参数:
        - y_true: 真实收益率，形状为 (batch_size, num_varieties)
        - y_pred: 预测持仓权重，形状为 (batch_size, num_varieties)
        
        返回:
        - loss: 损失值
        """
        # 计算投资组合收益
        if contract_multipliers is not None:
            # 考虑合约乘数
            portfolio_return = tf.reduce_sum(y_pred * y_true * contract_multipliers, axis=1)
        else:
            # 不考虑合约乘数
            portfolio_return = tf.reduce_sum(y_pred * y_true, axis=1)
        
        # 计算平均投资组合收益
        avg_return = tf.reduce_mean(portfolio_return)
        
        # L2正则化
        l2_loss = l2_regularization * tf.reduce_mean(tf.square(y_pred))
        
        # 换手率惩罚 (如果有前一天的持仓)
        turnover_loss = 0.0
        
        # 总损失：负收益（因为我们要最大化收益，所以最小化负收益）+ L2正则化 + 换手率惩罚
        total_loss = -avg_return + l2_loss + turnover_loss
        
        return total_loss
    
    return loss_fn

def sharpe_ratio_loss(risk_free_rate=0.0, contract_multipliers=None):
    """
    夏普比率损失函数
    
    参数:
    - risk_free_rate: 无风险利率
    - contract_multipliers: 合约乘数列表，形状为 (num_varieties,)
    
    返回:
    - loss_fn: 损失函数
    """
    def loss_fn(y_true, y_pred):
        """
        计算损失
        
        参数:
        - y_true: 真实收益率，形状为 (batch_size, num_varieties)
        - y_pred: 预测持仓权重，形状为 (batch_size, num_varieties)
        
        返回:
        - loss: 损失值
        """
        # 计算投资组合收益
        if contract_multipliers is not None:
            portfolio_returns = tf.reduce_sum(y_pred * y_true * contract_multipliers, axis=1)
        else:
            portfolio_returns = tf.reduce_sum(y_pred * y_true, axis=1)
        
        # 计算超额收益
        excess_returns = portfolio_returns - risk_free_rate
        
        # 计算夏普比率
        avg_excess_return = tf.reduce_mean(excess_returns)
        std_excess_return = tf.math.reduce_std(excess_returns)
        
        # 防止除以零
        std_excess_return = tf.clip_by_value(std_excess_return, 1e-8, tf.float32.max)
        
        sharpe_ratio = avg_excess_return / std_excess_return
        
        # 返回负夏普比率（因为我们要最大化夏普比率，所以最小化负夏普比率）
        return -sharpe_ratio
    
    return loss_fn
