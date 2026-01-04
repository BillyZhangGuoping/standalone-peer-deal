"""
functions.pyd - 通用函数模块
推测功能：包含各种通用的工具函数和辅助功能
"""

import pandas as pd
import numpy as np
import os
import datetime
import json

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def save_config(config, config_path):
    """保存配置文件"""
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
    """计算夏普比率"""
    excess_returns = returns - risk_free_rate / 252  # 假设252个交易日
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    return sharpe

def calculate_max_drawdown(returns):
    """计算最大回撤"""
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_sortino_ratio(returns, risk_free_rate=0.03):
    """计算索提诺比率"""
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    sortino = excess_returns.mean() * 252 / downside_deviation if downside_deviation > 0 else 0
    return sortino

def calculate_win_rate(returns):
    """计算胜率"""
    winning_trades = returns[returns > 0]
    win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
    return win_rate

def calculate_profit_factor(returns):
    """计算盈利因子"""
    gross_profit = returns[returns > 0].sum()
    gross_loss = -returns[returns < 0].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    return profit_factor

def calculate_information_ratio(returns, benchmark_returns=None, risk_free_rate=0.03):
    """计算信息比率"""
    if benchmark_returns is None:
        # 如果没有基准收益率，使用无风险利率
        benchmark_returns = pd.Series(risk_free_rate / 252, index=returns.index)
    
    # 计算超额收益率
    excess_returns = returns - benchmark_returns
    
    # 计算年化平均超额收益率
    annualized_excess_return = excess_returns.mean() * 252
    
    # 计算跟踪误差（超额收益率的年化波动率）
    tracking_error = excess_returns.std() * np.sqrt(252)
    
    # 计算信息比率
    information_ratio = annualized_excess_return / tracking_error if tracking_error != 0 else 0
    
    return information_ratio

def get_current_date():
    """获取当前日期（YYYYMMDD格式）"""
    return datetime.datetime.now().strftime('%Y%m%d')

def get_current_datetime():
    """获取当前日期时间"""
    return datetime.datetime.now()

def ensure_directory_exists(dir_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def format_number(num, decimals=2):
    """格式化数字"""
    return round(num, decimals)

def calculate_position_size(capital, risk_per_trade, stop_loss_percent, current_price):
    """计算仓位大小"""
    risk_amount = capital * risk_per_trade
    shares = risk_amount / (current_price * stop_loss_percent)
    return int(shares)