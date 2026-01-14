import pandas as pd
import numpy as np

class PortfolioSimulator:
    def __init__(self, variety_order, initial_capital=1000000, transaction_cost=0.001):
        """
        初始化组合模拟器
        
        参数:
        - variety_order: 品种顺序列表
        - initial_capital: 初始资金
        - transaction_cost: 交易成本率（单边）
        """
        self.variety_order = variety_order
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # 初始化回测结果
        self.backtest_results = {
            'date': [],
            'capital': [],
            'portfolio_value': [],
            'daily_return': [],
            'positions': [],
            'weights': []
        }
    
    def simulate(self, dates, predictions, returns, contract_multipliers=None, prices=None):
        """
        运行回测模拟
        
        参数:
        - dates: 日期列表
        - predictions: 预测持仓权重，形状为 (num_days, num_varieties)
        - returns: 实际收益率，形状为 (num_days, num_varieties)
        - contract_multipliers: 合约乘数列表，形状为 (num_varieties,)
        - prices: 价格数据，形状为 (num_days, num_varieties)，用于计算持仓价值
        
        返回:
        - backtest_results: 回测结果
        """
        # 初始化资金和持仓
        capital = self.initial_capital
        previous_weights = np.zeros(len(self.variety_order))
        
        for i in range(len(dates)):
            date = dates[i]
            weight_pred = predictions[i]
            ret = returns[i]
            
            # 计算交易成本
            if i > 0:
                # 换手率：|当前权重 - 前一天权重| 的总和
                turnover = np.sum(np.abs(weight_pred - previous_weights))
                transaction_cost = capital * turnover * self.transaction_cost
            else:
                transaction_cost = 0
            
            # 计算当日收益
            if contract_multipliers is not None:
                # 考虑合约乘数
                daily_return = capital * np.sum(weight_pred * ret * contract_multipliers)
            else:
                # 不考虑合约乘数
                daily_return = capital * np.sum(weight_pred * ret)
            
            # 更新资金
            capital = capital + daily_return - transaction_cost
            
            # 计算投资组合价值（如果有价格数据）
            portfolio_value = capital
            if prices is not None:
                # 这里简化处理，实际应该考虑每个品种的持仓数量和价格
                portfolio_value = capital
            
            # 记录回测结果
            self.backtest_results['date'].append(date)
            self.backtest_results['capital'].append(capital)
            self.backtest_results['portfolio_value'].append(portfolio_value)
            self.backtest_results['daily_return'].append(daily_return / self.initial_capital)
            self.backtest_results['positions'].append(weight_pred * capital)
            self.backtest_results['weights'].append(weight_pred)
            
            # 更新前一天权重
            previous_weights = weight_pred
        
        # 转换为DataFrame
        self.backtest_df = pd.DataFrame(self.backtest_results)
        self.backtest_df.set_index('date', inplace=True)
        
        return self.backtest_df
    
    def get_portfolio_metrics(self):
        """
        获取投资组合绩效指标
        
        返回:
        - metrics: 绩效指标字典
        """
        if not hasattr(self, 'backtest_df'):
            raise ValueError("No backtest results available. Please run simulate() first.")
        
        # 计算累计收益率
        cumulative_return = (self.backtest_df['portfolio_value'].iloc[-1] - self.initial_capital) / self.initial_capital
        
        # 计算年化收益率
        num_days = len(self.backtest_df)
        annualized_return = (1 + cumulative_return) ** (252 / num_days) - 1
        
        # 计算年化波动率
        daily_returns = self.backtest_df['daily_return']
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        
        # 计算夏普比率（假设无风险利率为0）
        sharpe_ratio = annualized_return / (annualized_volatility + 1e-8)
        
        # 计算最大回撤
        max_drawdown = self._calculate_max_drawdown(self.backtest_df['portfolio_value'])
        
        # 计算胜率
        winning_days = len(daily_returns[daily_returns > 0])
        win_rate = winning_days / num_days
        
        # 计算盈亏比
        avg_win = daily_returns[daily_returns > 0].mean()
        avg_loss = daily_returns[daily_returns < 0].mean()
        profit_loss_ratio = avg_win / (abs(avg_loss) + 1e-8)
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': self.backtest_df['portfolio_value'].iloc[-1],
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'num_trading_days': num_days
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, portfolio_values):
        """
        计算最大回撤
        
        参数:
        - portfolio_values: 投资组合价值序列
        
        返回:
        - max_drawdown: 最大回撤
        """
        if len(portfolio_values) == 0:
            return 0
        
        peak = portfolio_values.iloc[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    def plot_results(self, save_path=None):
        """
        绘制回测结果
        
        参数:
        - save_path: 保存路径，如果为None则显示图表
        """
        import matplotlib.pyplot as plt
        
        if not hasattr(self, 'backtest_df'):
            raise ValueError("No backtest results available. Please run simulate() first.")
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 投资组合价值
        axes[0].plot(self.backtest_df.index, self.backtest_df['portfolio_value'])
        axes[0].set_title('Portfolio Value')
        axes[0].set_ylabel('Value')
        axes[0].grid(True)
        
        # 每日收益
        axes[1].bar(self.backtest_df.index, self.backtest_df['daily_return'])
        axes[1].set_title('Daily Return')
        axes[1].set_ylabel('Return')
        axes[1].grid(True)
        
        # 持仓权重（前5个品种）
        num_varieties = min(5, len(self.variety_order))
        for i in range(num_varieties):
            weights = [w[i] for w in self.backtest_df['weights']]
            axes[2].plot(self.backtest_df.index, weights, label=self.variety_order[i])
        axes[2].set_title('Top 5 Variety Weights')
        axes[2].set_ylabel('Weight')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
