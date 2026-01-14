import pandas as pd
import numpy as np

class PerformanceMetrics:
    def __init__(self, backtest_results):
        """
        初始化绩效评估模块
        
        参数:
        - backtest_results: 回测结果DataFrame
        """
        self.backtest_results = backtest_results
        self.daily_returns = backtest_results['daily_return']
    
    def calculate_cumulative_return(self):
        """
        计算累积收益率
        
        返回:
        - cumulative_return: 累积收益率
        """
        cumulative_return = (1 + self.daily_returns).cumprod() - 1
        return cumulative_return
    
    def calculate_annualized_return(self):
        """
        计算年化收益率
        
        返回:
        - annualized_return: 年化收益率
        """
        total_return = (1 + self.daily_returns).prod() - 1
        num_days = len(self.daily_returns)
        annualized_return = (1 + total_return) ** (252 / num_days) - 1
        return annualized_return
    
    def calculate_annualized_volatility(self):
        """
        计算年化波动率
        
        返回:
        - annualized_volatility: 年化波动率
        """
        daily_volatility = self.daily_returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)
        return annualized_volatility
    
    def calculate_sharpe_ratio(self, risk_free_rate=0.0):
        """
        计算夏普比率
        
        参数:
        - risk_free_rate: 无风险利率
        
        返回:
        - sharpe_ratio: 夏普比率
        """
        annualized_return = self.calculate_annualized_return()
        annualized_volatility = self.calculate_annualized_volatility()
        sharpe_ratio = (annualized_return - risk_free_rate) / (annualized_volatility + 1e-8)
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, risk_free_rate=0.0, target_return=0.0):
        """
        计算索提诺比率
        
        参数:
        - risk_free_rate: 无风险利率
        - target_return: 目标收益率
        
        返回:
        - sortino_ratio: 索提诺比率
        """
        excess_returns = self.daily_returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        annualized_return = self.calculate_annualized_return()
        sortino_ratio = (annualized_return - risk_free_rate) / (downside_deviation + 1e-8)
        return sortino_ratio
    
    def calculate_max_drawdown(self):
        """
        计算最大回撤
        
        返回:
        - max_drawdown: 最大回撤
        """
        cumulative_return = self.calculate_cumulative_return()
        peak = cumulative_return.cummax()
        drawdown = (cumulative_return - peak) / (peak + 1e-8)
        max_drawdown = drawdown.min()
        return max_drawdown
    
    def calculate_calmar_ratio(self):
        """
        计算卡玛比率
        
        返回:
        - calmar_ratio: 卡玛比率
        """
        annualized_return = self.calculate_annualized_return()
        max_drawdown = abs(self.calculate_max_drawdown())
        calmar_ratio = annualized_return / (max_drawdown + 1e-8)
        return calmar_ratio
    
    def calculate_win_rate(self):
        """
        计算胜率
        
        返回:
        - win_rate: 胜率
        """
        winning_days = len(self.daily_returns[self.daily_returns > 0])
        total_days = len(self.daily_returns)
        win_rate = winning_days / total_days
        return win_rate
    
    def calculate_profit_loss_ratio(self):
        """
        计算盈亏比
        
        返回:
        - profit_loss_ratio: 盈亏比
        """
        winning_returns = self.daily_returns[self.daily_returns > 0]
        losing_returns = self.daily_returns[self.daily_returns < 0]
        avg_win = winning_returns.mean()
        avg_loss = abs(losing_returns.mean())
        profit_loss_ratio = avg_win / (avg_loss + 1e-8)
        return profit_loss_ratio
    
    def calculate_alpha_beta(self, benchmark_returns, risk_free_rate=0.0):
        """
        计算阿尔法和贝塔系数
        
        参数:
        - benchmark_returns: 基准收益率序列
        - risk_free_rate: 无风险利率
        
        返回:
        - alpha: 阿尔法系数
        - beta: 贝塔系数
        """
        # 确保基准收益率与投资组合收益率长度相同
        if len(benchmark_returns) != len(self.daily_returns):
            raise ValueError("Benchmark returns and portfolio returns must have the same length")
        
        # 计算超额收益率
        excess_portfolio_returns = self.daily_returns - risk_free_rate
        excess_benchmark_returns = benchmark_returns - risk_free_rate
        
        # 计算协方差和基准方差
        covariance = np.cov(excess_portfolio_returns, excess_benchmark_returns)[0, 1]
        benchmark_variance = np.var(excess_benchmark_returns)
        
        # 计算贝塔系数
        beta = covariance / benchmark_variance
        
        # 计算阿尔法系数
        alpha = (excess_portfolio_returns.mean() - beta * excess_benchmark_returns.mean()) * 252
        
        return alpha, beta
    
    def run_performance_analysis(self, benchmark_returns=None, risk_free_rate=0.0):
        """
        运行完整的绩效分析
        
        参数:
        - benchmark_returns: 基准收益率序列（可选）
        - risk_free_rate: 无风险利率
        
        返回:
        - performance_results: 绩效分析结果
        """
        # 计算各种绩效指标
        cumulative_return = self.calculate_cumulative_return()
        annualized_return = self.calculate_annualized_return()
        annualized_volatility = self.calculate_annualized_volatility()
        sharpe_ratio = self.calculate_sharpe_ratio(risk_free_rate)
        sortino_ratio = self.calculate_sortino_ratio(risk_free_rate)
        max_drawdown = self.calculate_max_drawdown()
        calmar_ratio = self.calculate_calmar_ratio()
        win_rate = self.calculate_win_rate()
        profit_loss_ratio = self.calculate_profit_loss_ratio()
        
        # 计算阿尔法和贝塔系数（如果提供了基准收益率）
        alpha = None
        beta = None
        if benchmark_returns is not None:
            alpha, beta = self.calculate_alpha_beta(benchmark_returns, risk_free_rate)
        
        # 构建绩效分析结果
        performance_results = {
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'alpha': alpha,
            'beta': beta
        }
        
        return performance_results
