import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 品种列表
variety_list = ['i', 'rb', 'hc', 'jm', 'j', 'ss', 'ni', 'SF', 'SM', 'lc', 'sn', 'si', 'pb', 'cu', 'al', 'zn',
                'ao', 'SH', 'au', 'ag', 'OI', 'a', 'b', 'm', 'RM', 'p', 'y', 'CF', 'SR', 'c', 'cs', 'AP', 'PK',
                'jd', 'CJ', 'lh', 'sc', 'fu', 'lu', 'pg', 'bu', 'eg', 'TA', 'PF', 'PX', 'l', 'v', 'MA', 'pp',
                'eb', 'ru', 'nr', 'br', 'SA', 'FG', 'UR', 'sp', 'IF', 'IC', 'IH', 'IM', 'ec', 'T', 'TF']

# 数据目录
DATA_DIR = 'History_Data/hot_daily_market_data'

class VarietyCorrelationAnalyzer:
    """品种相关性分析器，用于分析variety_list中各个品种的历史数据"""
    
    def __init__(self, data_dir=DATA_DIR):
        """初始化分析器
        
        参数：
        data_dir: 数据目录
        """
        self.data_dir = data_dir
        self.all_data = None
        self.processed_data = None
        self.correlation_matrix = None
    
    def load_data(self):
        """加载所有品种的历史数据"""
        logger.info("开始加载数据...")
        
        all_data = {}
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        for file in files:
            file_path = os.path.join(self.data_dir, file)
            base_symbol = file.split('.')[0].upper()
            
            # 仅加载variety_list中的品种，忽略大小写
            if base_symbol.lower() not in [v.lower() for v in variety_list]:
                continue
            
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                all_data[base_symbol] = df
                logger.info(f"  成功加载品种 {base_symbol}")
            except Exception as e:
                logger.error(f"  加载品种 {base_symbol} 失败: {str(e)}")
                continue
        
        self.all_data = all_data
        logger.info(f"数据加载完成，共加载 {len(all_data)} 个品种")
        return all_data
    
    def preprocess_data(self):
        """预处理数据，提取收盘价并进行标准化"""
        logger.info("开始预处理数据...")
        
        if self.all_data is None:
            self.load_data()
        
        # 提取收盘价数据
        close_data = {}
        for symbol, df in self.all_data.items():
            if 'close' in df.columns:
                close_data[symbol] = df['close']
        
        # 合并为一个DataFrame
        combined_data = pd.DataFrame(close_data)
        
        # 处理缺失值
        logger.info(f"原始数据形状: {combined_data.shape}")
        logger.info(f"缺失值数量: {combined_data.isnull().sum().sum()}")
        
        # 使用前向填充处理缺失值
        combined_data = combined_data.ffill()
        # 再使用后向填充处理剩余缺失值
        combined_data = combined_data.bfill()
        
        # 删除仍有缺失值的列
        combined_data = combined_data.dropna(axis=1)
        
        logger.info(f"预处理后数据形状: {combined_data.shape}")
        logger.info(f"预处理后缺失值数量: {combined_data.isnull().sum().sum()}")
        
        self.processed_data = combined_data
        return combined_data
    
    def calculate_correlation(self):
        """计算品种之间的相关系数"""
        logger.info("开始计算相关性...")
        
        if self.processed_data is None:
            self.preprocess_data()
        
        # 计算相关系数矩阵
        self.correlation_matrix = self.processed_data.corr()
        
        logger.info("相关性计算完成")
        return self.correlation_matrix
    
    def analyze_correlation(self, threshold=0.7):
        """分析相关性，识别强相关关系
        
        参数：
        threshold: 相关性阈值，默认为0.7
        
        返回：
        strong_positive: 强正相关关系列表
        strong_negative: 强负相关关系列表
        """
        logger.info(f"开始分析相关性，阈值: {threshold}")
        
        if self.correlation_matrix is None:
            self.calculate_correlation()
        
        strong_positive = []
        strong_negative = []
        
        # 遍历相关系数矩阵，只考虑上三角部分
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                symbol1 = self.correlation_matrix.columns[i]
                symbol2 = self.correlation_matrix.columns[j]
                corr_value = self.correlation_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold:
                    if corr_value > 0:
                        strong_positive.append((symbol1, symbol2, corr_value))
                    else:
                        strong_negative.append((symbol1, symbol2, corr_value))
        
        # 按相关性强度排序
        strong_positive.sort(key=lambda x: x[2], reverse=True)
        strong_negative.sort(key=lambda x: x[2])
        
        logger.info(f"分析完成，发现 {len(strong_positive)} 对强正相关关系，{len(strong_negative)} 对强负相关关系")
        
        return strong_positive, strong_negative
    
    def visualize_correlation(self):
        """可视化相关系数矩阵"""
        logger.info("开始可视化相关系数矩阵...")
        
        if self.correlation_matrix is None:
            self.calculate_correlation()
        
        # 创建相关系数热力图
        plt.figure(figsize=(20, 18))
        sns.heatmap(self.correlation_matrix, annot=False, cmap='coolwarm', center=0, 
                   xticklabels=True, yticklabels=True, linewidths=0.5)
        plt.title('品种相关性热力图', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig('variety_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        logger.info("相关性热力图已保存到 variety_correlation_heatmap.png")
        
        # 创建相关性分布直方图
        plt.figure(figsize=(12, 6))
        sns.histplot(self.correlation_matrix.values.flatten(), bins=100, kde=True)
        plt.title('相关性系数分布', fontsize=16)
        plt.xlabel('相关系数', fontsize=12)
        plt.ylabel('频率', fontsize=12)
        plt.axvline(x=0.7, color='red', linestyle='--', label='强正相关阈值 (0.7)')
        plt.axvline(x=-0.7, color='blue', linestyle='--', label='强负相关阈值 (-0.7)')
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('correlation_distribution.png', dpi=300)
        logger.info("相关性分布直方图已保存到 correlation_distribution.png")
    
    def build_aw_beta_strategy(self, lookback_period=20, rebalance_period=5):
        """构建Alpha/Wealth (AW) Beta策略
        
        参数：
        lookback_period: 回看周期，用于计算收益率和波动率
        rebalance_period: 再平衡周期
        
        返回：
        strategy_returns: 策略收益率
        portfolio_weights: 投资组合权重
        """
        logger.info(f"开始构建AW Beta策略，回看周期: {lookback_period}，再平衡周期: {rebalance_period}")
        
        if self.processed_data is None:
            self.preprocess_data()
        
        # 计算每日收益率
        returns = self.processed_data.pct_change().dropna()
        
        # 计算品种的Beta值（相对市场组合）
        market_return = returns.mean(axis=1)
        beta_values = {}
        
        for symbol in returns.columns:
            # 计算品种与市场组合的协方差
            cov = returns[symbol].cov(market_return)
            # 计算市场组合的方差
            market_var = market_return.var()
            # 计算Beta值
            beta = cov / market_var if market_var != 0 else 0
            beta_values[symbol] = beta
        
        # 将Beta值转换为DataFrame
        beta_df = pd.Series(beta_values, name='beta')
        
        # 计算Alpha值（超额收益）
        alpha_values = {}
        for symbol in returns.columns:
            # 计算平均收益率
            avg_return = returns[symbol].mean()
            # 计算市场平均收益率
            avg_market_return = market_return.mean()
            # 计算Alpha值
            alpha = avg_return - beta_values[symbol] * avg_market_return
            alpha_values[symbol] = alpha
        
        # 将Alpha值转换为DataFrame
        alpha_df = pd.Series(alpha_values, name='alpha')
        
        # 合并Alpha和Beta
        aw_beta_df = pd.concat([alpha_df, beta_df], axis=1)
        
        # 计算每个品种的权重：Alpha / |Beta|
        aw_beta_df['weight'] = aw_beta_df['alpha'] / abs(aw_beta_df['beta']) if (aw_beta_df['beta'] != 0).all() else aw_beta_df['alpha']
        
        # 标准化权重，使其总和为1
        aw_beta_df['weight'] = aw_beta_df['weight'] / aw_beta_df['weight'].sum()
        
        # 风险均衡调整：使用逆波动率权重调整
        # 计算品种的波动率
        volatility = returns.std()
        # 计算逆波动率权重
        inv_vol_weights = 1 / volatility
        inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()
        
        # 结合AW Beta权重和逆波动率权重，实现风险均衡
        final_weights = (aw_beta_df['weight'] + inv_vol_weights) / 2
        final_weights = final_weights / final_weights.sum()
        
        # 计算策略收益率
        strategy_returns = returns.dot(final_weights)
        
        # 计算累计收益率
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # 输出结果
        logger.info("AW Beta策略构建完成")
        logger.info(f"策略平均日收益率: {strategy_returns.mean():.4f}")
        logger.info(f"策略年化收益率: {strategy_returns.mean() * 252:.4f}")
        logger.info(f"策略日波动率: {strategy_returns.std():.4f}")
        logger.info(f"策略夏普比率: {strategy_returns.mean() / strategy_returns.std() * np.sqrt(252):.4f}")
        
        return strategy_returns, final_weights, cumulative_returns, aw_beta_df
    
    def visualize_strategy_performance(self, strategy_returns, cumulative_returns):
        """可视化策略绩效"""
        logger.info("开始可视化策略绩效...")
        
        # 创建策略收益率和累计收益率图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
        
        # 绘制日收益率
        ax1.plot(strategy_returns.index, strategy_returns)
        ax1.set_title('AW Beta策略日收益率', fontsize=14)
        ax1.set_ylabel('收益率', fontsize=12)
        ax1.grid(True)
        
        # 绘制累计收益率
        ax2.plot(cumulative_returns.index, cumulative_returns)
        ax2.set_title('AW Beta策略累计收益率', fontsize=14)
        ax2.set_xlabel('日期', fontsize=12)
        ax2.set_ylabel('累计收益率', fontsize=12)
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('aw_beta_strategy_performance.png', dpi=300)
        logger.info("策略绩效图表已保存到 aw_beta_strategy_performance.png")
    
    def visualize_portfolio_weights(self, weights):
        """可视化投资组合权重"""
        logger.info("开始可视化投资组合权重...")
        
        # 将权重按降序排序
        sorted_weights = weights.sort_values(ascending=False)
        
        # 创建权重条形图
        plt.figure(figsize=(15, 10))
        bars = plt.bar(sorted_weights.index, sorted_weights.values)
        plt.title('AW Beta策略投资组合权重', fontsize=16)
        plt.xlabel('品种', fontsize=12)
        plt.ylabel('权重', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        
        # 添加权重数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', 
                    ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('portfolio_weights.png', dpi=300)
        logger.info("投资组合权重图表已保存到 portfolio_weights.png")
    
    def analyze_risk_contribution(self, weights):
        """分析投资组合的风险贡献度"""
        logger.info("开始分析风险贡献度...")
        
        if self.processed_data is None:
            self.preprocess_data()
        
        # 计算每日收益率
        returns = self.processed_data.pct_change().dropna()
        
        # 计算协方差矩阵
        cov_matrix = returns.cov()
        
        # 计算投资组合方差
        portfolio_var = weights.T.dot(cov_matrix).dot(weights)
        
        # 计算边际风险贡献
        marginal_risk_contribution = cov_matrix.dot(weights)
        
        # 计算风险贡献度
        risk_contribution = weights * marginal_risk_contribution / portfolio_var
        
        # 转换为Series以便可视化
        risk_contribution_series = pd.Series(risk_contribution, name='risk_contribution')
        
        # 按风险贡献度排序
        sorted_risk_contribution = risk_contribution_series.sort_values(ascending=False)
        
        # 计算风险集中度（前5个品种的风险贡献度之和）
        risk_concentration = sorted_risk_contribution.head(5).sum()
        
        logger.info(f"投资组合风险集中度（前5个品种）: {risk_concentration:.4f}")
        logger.info(f"最大风险贡献品种: {sorted_risk_contribution.idxmax()} ({sorted_risk_contribution.max():.4f})")
        logger.info(f"最小风险贡献品种: {sorted_risk_contribution.idxmin()} ({sorted_risk_contribution.min():.4f})")
        
        # 可视化风险贡献度
        plt.figure(figsize=(15, 10))
        bars = plt.bar(sorted_risk_contribution.index, sorted_risk_contribution.values)
        plt.title('投资组合风险贡献度', fontsize=16)
        plt.xlabel('品种', fontsize=12)
        plt.ylabel('风险贡献度', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        
        # 添加风险贡献度数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', 
                    ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('risk_contribution.png', dpi=300)
        logger.info("风险贡献度图表已保存到 risk_contribution.png")
        
        return risk_contribution_series
    
    def run_full_analysis(self):
        """运行完整的分析流程"""
        logger.info("开始运行完整的品种相关性分析...")
        
        # 1. 数据加载
        self.load_data()
        
        # 2. 数据预处理
        self.preprocess_data()
        
        # 3. 相关性分析
        self.calculate_correlation()
        strong_positive, strong_negative = self.analyze_correlation()
        
        # 输出强相关关系
        logger.info("\n===== 强正相关关系（相关性 ≥ 0.7）=====")
        for symbol1, symbol2, corr in strong_positive[:10]:  # 只显示前10对
            logger.info(f"{symbol1} 与 {symbol2}: {corr:.4f}")
        
        logger.info("\n===== 强负相关关系（相关性 ≤ -0.7）=====")
        for symbol1, symbol2, corr in strong_negative[:10]:  # 只显示前10对
            logger.info(f"{symbol1} 与 {symbol2}: {corr:.4f}")
        
        # 4. 可视化相关性
        self.visualize_correlation()
        
        # 5. 构建AW Beta策略
        strategy_returns, weights, cumulative_returns, aw_beta_df = self.build_aw_beta_strategy()
        
        # 6. 可视化策略绩效
        self.visualize_strategy_performance(strategy_returns, cumulative_returns)
        
        # 7. 可视化投资组合权重
        self.visualize_portfolio_weights(weights)
        
        # 8. 分析风险贡献度
        risk_contribution = self.analyze_risk_contribution(weights)
        
        logger.info("\n===== 分析完成 =====")
        logger.info(f"共加载 {len(self.all_data)} 个品种")
        logger.info(f"发现 {len(strong_positive)} 对强正相关关系")
        logger.info(f"发现 {len(strong_negative)} 对强负相关关系")
        logger.info(f"策略夏普比率: {strategy_returns.mean() / strategy_returns.std() * np.sqrt(252):.4f}")
        logger.info(f"风险集中度（前5个品种）: {risk_contribution.sort_values(ascending=False).head(5).sum():.4f}")
        
        return {
            'strong_positive': strong_positive,
            'strong_negative': strong_negative,
            'strategy_returns': strategy_returns,
            'cumulative_returns': cumulative_returns,
            'weights': weights,
            'risk_contribution': risk_contribution,
            'aw_beta_df': aw_beta_df
        }

if __name__ == "__main__":
    # 创建分析器实例
    analyzer = VarietyCorrelationAnalyzer()
    
    # 运行完整分析
    results = analyzer.run_full_analysis()
    
    # 保存结果
    logger.info("\n开始保存分析结果...")
    
    # 保存相关系数矩阵
    if analyzer.correlation_matrix is not None:
        analyzer.correlation_matrix.to_csv('variety_correlation_matrix.csv')
        logger.info("相关系数矩阵已保存到 variety_correlation_matrix.csv")
    
    # 保存AW Beta策略结果
    if 'aw_beta_df' in results:
        results['aw_beta_df'].to_csv('aw_beta_results.csv')
        logger.info("AW Beta策略结果已保存到 aw_beta_results.csv")
    
    # 保存投资组合权重
    if 'weights' in results:
        results['weights'].to_csv('portfolio_weights.csv')
        logger.info("投资组合权重已保存到 portfolio_weights.csv")
    
    # 保存风险贡献度
    if 'risk_contribution' in results:
        results['risk_contribution'].to_csv('risk_contribution.csv')
        logger.info("风险贡献度已保存到 risk_contribution.csv")
    
    logger.info("所有结果已保存，分析完成！")
