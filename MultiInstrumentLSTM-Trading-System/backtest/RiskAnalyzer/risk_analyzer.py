import pandas as pd
import numpy as np
import scipy.stats as stats

class RiskAnalyzer:
    def __init__(self, backtest_results):
        """
        初始化风险分析器
        
        参数:
        - backtest_results: 回测结果DataFrame
        """
        self.backtest_results = backtest_results
        self.daily_returns = backtest_results['daily_return']
    
    def calculate_value_at_risk(self, confidence_level=0.95, method='historical'):
        """
        计算风险价值（VaR）
        
        参数:
        - confidence_level: 置信水平
        - method: 计算方法，可选 'historical'（历史模拟法）或 'parametric'（参数法）
        
        返回:
        - var: 风险价值
        """
        if method == 'historical':
            # 历史模拟法
            var = -np.percentile(self.daily_returns, 100 * (1 - confidence_level))
        elif method == 'parametric':
            # 参数法（假设收益率服从正态分布）
            mu = self.daily_returns.mean()
            sigma = self.daily_returns.std()
            var = - (mu - sigma * stats.norm.ppf(confidence_level))
        else:
            raise ValueError(f"Unsupported VaR method: {method}")
        
        return var
    
    def calculate_conditional_value_at_risk(self, confidence_level=0.95, method='historical'):
        """
        计算条件风险价值（CVaR）
        
        参数:
        - confidence_level: 置信水平
        - method: 计算方法，可选 'historical'（历史模拟法）或 'parametric'（参数法）
        
        返回:
        - cvar: 条件风险价值
        """
        if method == 'historical':
            # 历史模拟法
            var = self.calculate_value_at_risk(confidence_level, method='historical')
            cvar = -self.daily_returns[self.daily_returns <= -var].mean()
        elif method == 'parametric':
            # 参数法（假设收益率服从正态分布）
            sigma = self.daily_returns.std()
            z_score = stats.norm.ppf(confidence_level)
            cvar = sigma * stats.norm.pdf(z_score) / (1 - confidence_level)
        else:
            raise ValueError(f"Unsupported CVaR method: {method}")
        
        return cvar
    
    def calculate_drawdown(self):
        """
        计算最大回撤和回撤序列
        
        返回:
        - drawdown: 回撤序列
        - max_drawdown: 最大回撤
        """
        # 计算累积收益率
        cumulative_returns = (1 + self.daily_returns).cumprod()
        
        # 计算峰值
        peak = cumulative_returns.cummax()
        
        # 计算回撤
        drawdown = (cumulative_returns - peak) / peak
        
        # 计算最大回撤
        max_drawdown = drawdown.min()
        
        return drawdown, max_drawdown
    
    def calculate_volatility(self, window=20):
        """
        计算滚动波动率
        
        参数:
        - window: 滚动窗口大小
        
        返回:
        - volatility: 滚动波动率序列
        """
        volatility = self.daily_returns.rolling(window=window).std() * np.sqrt(252)
        
        return volatility
    
    def calculate_beta(self, benchmark_returns):
        """
        计算投资组合相对于基准的贝塔系数
        
        参数:
        - benchmark_returns: 基准收益率序列
        
        返回:
        - beta: 贝塔系数
        """
        # 确保基准收益率与投资组合收益率长度相同
        if len(benchmark_returns) != len(self.daily_returns):
            raise ValueError("Benchmark returns and portfolio returns must have the same length")
        
        # 计算协方差和基准方差
        covariance = np.cov(self.daily_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        # 计算贝塔系数
        beta = covariance / benchmark_variance
        
        return beta
    
    def calculate_skewness(self):
        """
        计算收益率的偏度
        
        返回:
        - skewness: 偏度
        """
        return stats.skew(self.daily_returns)
    
    def calculate_kurtosis(self):
        """
        计算收益率的峰度
        
        返回:
        - kurtosis: 峰度
        """
        return stats.kurtosis(self.daily_returns)
    
    def run_risk_analysis(self, benchmark_returns=None):
        """
        运行完整的风险分析
        
        参数:
        - benchmark_returns: 基准收益率序列（可选）
        
        返回:
        - risk_results: 风险分析结果
        """
        # 计算各种风险指标
        var_95 = self.calculate_value_at_risk(confidence_level=0.95, method='historical')
        cvar_95 = self.calculate_conditional_value_at_risk(confidence_level=0.95, method='historical')
        drawdown, max_drawdown = self.calculate_drawdown()
        volatility = self.calculate_volatility()
        skewness = self.calculate_skewness()
        kurtosis = self.calculate_kurtosis()
        
        # 计算贝塔系数（如果提供了基准收益率）
        beta = None
        if benchmark_returns is not None:
            beta = self.calculate_beta(benchmark_returns)
        
        # 构建风险分析结果
        risk_results = {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'drawdown': drawdown,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'beta': beta
        }
        
        return risk_results
