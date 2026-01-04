import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数
DATA_DIR = '../History_Data/hot_daily_market_data'
OUTPUT_DIR = 'target_position'
START_DATE = '2024-07-01'
CAPITAL = 10000000  # 1000万资金
RISK_FREE_RATE = 0.03  # 无风险利率
LOOKBACK_DAYS = 60  # 回溯天数
REBALANCE_FREQ = 1  # 再平衡频率，每日

# 品种列表
variety_list = ['i', 'rb', 'hc', 'jm', 'j', 'ss', 'ni', 'SF', 'SM', 'lc', 'sn', 'si', 'pb', 'cu', 'al', 'zn',
                'ao', 'SH', 'au', 'ag', 'OI', 'a', 'b', 'm', 'RM', 'p', 'y', 'CF', 'SR', 'c', 'cs', 'AP', 'PK',
                'jd', 'CJ', 'lh', 'sc', 'fu', 'lu', 'pg', 'bu', 'eg', 'TA', 'PF', 'PX', 'l', 'v', 'MA', 'pp',
                'eb', 'ru', 'nr', 'br', 'SA', 'FG', 'UR', 'sp', 'IF', 'IC', 'IH', 'IM', 'ec', 'T', 'TF']

class AWBetaStrategy:
    """AW Beta策略实现，包含动态再平衡、风险平价和品种选择"""
    
    def __init__(self, data_dir=DATA_DIR, output_dir=OUTPUT_DIR, start_date=START_DATE, capital=CAPITAL):
        """初始化策略
        
        参数：
        data_dir: 数据目录
        output_dir: 输出目录
        start_date: 开始日期
        capital: 总资金
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.start_date = pd.to_datetime(start_date)
        self.capital = capital
        
        self.all_data = None
        self.processed_data = None
        self.selected_varieties = None
        self.strategy_returns = []
        self.daily_positions = {}
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self):
        """加载所有品种的历史数据"""
        logger.info("开始加载历史数据...")
        
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
        """预处理数据，提取收盘价并对齐日期"""
        logger.info("开始预处理数据...")
        
        if self.all_data is None:
            self.load_data()
        
        # 提取所有品种的收盘价
        close_prices_dict = {}
        for symbol, data in self.all_data.items():
            close_prices_dict[symbol] = data['close']
        
        # 合并为一个DataFrame，对齐日期
        self.processed_data = pd.DataFrame(close_prices_dict)
        
        # 处理缺失值
        logger.info(f"原始数据形状: {self.processed_data.shape}")
        logger.info(f"缺失值数量: {self.processed_data.isnull().sum().sum()}")
        
        # 使用前向填充处理缺失值
        self.processed_data = self.processed_data.ffill()
        # 再使用后向填充处理剩余缺失值
        self.processed_data = self.processed_data.bfill()
        
        # 删除仍有缺失值的列
        self.processed_data = self.processed_data.dropna(axis=1)
        
        logger.info(f"预处理后数据形状: {self.processed_data.shape}")
        logger.info(f"预处理后缺失值数量: {self.processed_data.isnull().sum().sum()}")        
        
        return self.processed_data
    
    def select_varieties(self, corr_threshold=0.7, lookback_days=60, min_sharpe=0.5):
        """基于相关性和绩效筛选品种
        
        参数：
        corr_threshold: 相关性阈值，超过该阈值的品种将被排除
        lookback_days: 回溯天数，用于计算绩效
        min_sharpe: 最小夏普比率，低于该值的品种将被排除
        
        返回：
        selected_varieties: 筛选后的品种列表
        """
        logger.info(f"开始筛选品种，相关性阈值: {corr_threshold}，回溯天数: {lookback_days}，最小夏普比率: {min_sharpe}")
        
        if self.processed_data is None:
            self.preprocess_data()
        
        # 获取回溯期数据
        lookback_start = self.start_date - pd.Timedelta(days=lookback_days)
        lookback_data = self.processed_data[(self.processed_data.index >= lookback_start) & (self.processed_data.index < self.start_date)]
        
        if lookback_data.empty:
            logger.warning(f"回溯期数据不足，使用全部数据进行筛选")
            lookback_data = self.processed_data
        
        # 计算品种的绩效指标
        returns = lookback_data.pct_change().dropna()
        annual_returns = returns.mean() * 252
        volatility = returns.std()
        sharpe_ratios = (annual_returns - RISK_FREE_RATE) / volatility
        
        # 计算相关系数矩阵
        corr_matrix = lookback_data.corr()
        
        # 基于相关性和绩效筛选品种
        selected = []
        
        # 按夏普比率降序排序
        sorted_varieties = sharpe_ratios.sort_values(ascending=False)
        
        for variety in sorted_varieties.index:
            # 跳过夏普比率低于阈值的品种
            if sharpe_ratios[variety] < min_sharpe:
                continue
            
            # 检查与已选品种的相关性
            add_variety = True
            for selected_variety in selected:
                if abs(corr_matrix.loc[variety, selected_variety]) > corr_threshold:
                    add_variety = False
                    break
            
            if add_variety:
                selected.append(variety)
        
        # 如果没有选到足够的品种，放宽条件
        if len(selected) < 5:
            logger.warning(f"仅选到 {len(selected)} 个品种，放宽相关性阈值到 {corr_threshold + 0.1}")
            selected = []
            for variety in sorted_varieties.index:
                if sharpe_ratios[variety] < min_sharpe * 0.8:
                    continue
                
                add_variety = True
                for selected_variety in selected:
                    if abs(corr_matrix.loc[variety, selected_variety]) > corr_threshold + 0.1:
                        add_variety = False
                        break
                
                if add_variety:
                    selected.append(variety)
        
        self.selected_varieties = selected
        logger.info(f"品种筛选完成，共选中 {len(selected)} 个品种: {selected}")
        return selected
    
    def calculate_risk_parity_weights(self, returns_data, max_iterations=1000, tolerance=1e-6):
        """计算风险平价权重
        
        参数：
        returns_data: 收益率数据
        max_iterations: 最大迭代次数
        tolerance: 收敛容忍度
        
        返回：
        weights: 风险平价权重
        """
        logger.info("开始计算风险平价权重...")
        
        # 计算协方差矩阵
        cov_matrix = returns_data.cov()
        
        # 初始权重：等权
        n_assets = returns_data.shape[1]
        weights = np.ones(n_assets) / n_assets
        
        # 风险平价算法
        for i in range(max_iterations):
            # 计算投资组合方差
            port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # 计算边际风险贡献
            marginal_risk = np.dot(cov_matrix, weights) / np.sqrt(port_var)
            
            # 计算风险贡献
            risk_contribution = weights * marginal_risk
            
            # 计算风险贡献差异
            risk_diff = risk_contribution - np.mean(risk_contribution)
            
            # 检查收敛条件
            if np.max(np.abs(risk_diff)) < tolerance:
                logger.info(f"风险平价权重收敛，迭代次数: {i+1}")
                break
            
            # 调整权重
            weights = weights * (np.mean(risk_contribution) / risk_contribution)
            weights = weights / np.sum(weights)
        
        # 转换为Series
        weights_series = pd.Series(weights, index=returns_data.columns)
        logger.info("风险平价权重计算完成")
        return weights_series
    
    def calculate_aw_beta_weights(self, returns_data, lookback_days=20):
        """计算AW Beta权重
        
        参数：
        returns_data: 收益率数据
        lookback_days: 回看周期
        
        返回：
        aw_beta_weights: AW Beta权重
        """
        logger.info(f"开始计算AW Beta权重，回看周期: {lookback_days}")
        
        # 计算市场组合收益率
        market_return = returns_data.mean(axis=1)
        
        # 计算品种的Beta值
        beta_values = {}
        for symbol in returns_data.columns:
            cov = returns_data[symbol].cov(market_return)
            market_var = market_return.var()
            beta = cov / market_var if market_var != 0 else 0
            beta_values[symbol] = beta
        
        # 计算Alpha值
        alpha_values = {}
        for symbol in returns_data.columns:
            avg_return = returns_data[symbol].mean()
            avg_market_return = market_return.mean()
            alpha = avg_return - beta_values[symbol] * avg_market_return
            alpha_values[symbol] = alpha
        
        # 计算AW Beta权重：Alpha / |Beta|
        aw_beta_weights = {}
        for symbol in returns_data.columns:
            beta = beta_values[symbol]
            alpha = alpha_values[symbol]
            aw_beta_weights[symbol] = alpha / abs(beta) if beta != 0 else alpha
        
        # 转换为Series并标准化
        aw_beta_weights_series = pd.Series(aw_beta_weights)
        aw_beta_weights_series = aw_beta_weights_series / aw_beta_weights_series.sum()
        
        logger.info("AW Beta权重计算完成")
        return aw_beta_weights_series
    
    def calculate_strategy_metrics(self, returns):
        """计算策略评估指标
        
        参数：
        returns: 策略收益率序列
        
        返回：
        metrics: 策略评估指标字典
        """
        logger.info("开始计算策略评估指标...")
        
        # 年化收益率
        annual_return = returns.mean() * 252
        
        # 年化波动率
        annual_volatility = returns.std() * np.sqrt(252)
        
        # 夏普比率
        sharpe_ratio = (annual_return - RISK_FREE_RATE) / annual_volatility if annual_volatility != 0 else 0
        
        # 索提诺比率（仅考虑下行风险）
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - RISK_FREE_RATE) / downside_volatility if downside_volatility != 0 else 0
        
        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # 卡玛比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 胜率
        win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
        
        metrics = {
            '年化收益率': annual_return,
            '年化波动率': annual_volatility,
            '夏普比率': sharpe_ratio,
            '索提诺比率': sortino_ratio,
            '最大回撤': max_drawdown,
            '卡玛比率': calmar_ratio,
            '胜率': win_rate
        }
        
        logger.info("策略评估指标计算完成")
        return metrics
    
    def generate_daily_positions(self):
        """生成每日目标头寸"""
        logger.info(f"开始生成每日目标头寸，起始日期: {self.start_date}")
        
        if self.processed_data is None:
            self.preprocess_data()
        
        # 获取所有交易日期
        all_dates = self.processed_data.index[self.processed_data.index >= self.start_date]
        
        # 初始化策略收益率
        strategy_returns = []
        prev_cumulative_return = 1.0
        
        for date in all_dates:
            logger.info(f"\n处理日期: {date.strftime('%Y-%m-%d')}")
            
            # 只使用历史数据（不包括当前日期）
            historical_data = self.processed_data[self.processed_data.index < date]
            
            if historical_data.empty:
                logger.warning(f"日期 {date} 没有足够的历史数据，跳过")
                continue
            
            # 筛选品种（使用60天回溯数据）
            lookback_start = date - pd.Timedelta(days=60)
            lookback_data = historical_data[historical_data.index >= lookback_start]
            
            if len(lookback_data) < 20:
                logger.warning(f"日期 {date} 回溯数据不足，使用全部历史数据")
                lookback_data = historical_data
            
            # 计算收益率
            returns = lookback_data.pct_change().dropna()
            
            if returns.empty:
                logger.warning(f"日期 {date} 收益率数据不足，跳过")
                continue
            
            # 计算风险平价权重
            risk_parity_weights = self.calculate_risk_parity_weights(returns)
            
            # 计算AW Beta权重
            aw_beta_weights = self.calculate_aw_beta_weights(returns)
            
            # 结合风险平价和AW Beta权重（各占50%）
            final_weights = (risk_parity_weights + aw_beta_weights) / 2
            final_weights = final_weights / final_weights.sum()  # 确保权重和为1
            
            # 获取当前日期的收盘价
            current_prices = self.processed_data.loc[date]
            
            # 生成目标头寸
            target_positions = {}
            for symbol, weight in final_weights.items():
                # 计算每个品种的目标资金
                target_capital = self.capital * weight
                
                # 计算目标手数（假设每手为1个单位）
                price = current_prices[symbol]
                target_quantity = int(target_capital / price)
                
                target_positions[symbol] = {
                    'weight': weight,
                    'target_capital': target_capital,
                    'price': price,
                    'quantity': target_quantity
                }
            
            # 保存每日头寸
            self.daily_positions[date] = target_positions
            
            # 保存到文件
            positions_df = pd.DataFrame(target_positions).T
            positions_df.index.name = 'symbol'
            
            # 创建日期文件夹
            date_folder = os.path.join(self.output_dir, date.strftime('%Y-%m-%d'))
            os.makedirs(date_folder, exist_ok=True)
            
            # 保存头寸文件
            positions_file = os.path.join(date_folder, 'target_positions.csv')
            positions_df.to_csv(positions_file)
            logger.info(f"目标头寸已保存到 {positions_file}")
            
            # 计算策略收益率
            if len(strategy_returns) > 0:
                # 计算当日收益率
                current_cumulative_return = (1 + returns.mean()).cumprod().iloc[-1]
                daily_return = (current_cumulative_return / prev_cumulative_return) - 1
                strategy_returns.append(daily_return)
                prev_cumulative_return = current_cumulative_return
            else:
                # 第一天收益率为0
                strategy_returns.append(0.0)
                prev_cumulative_return = (1 + returns.mean()).cumprod().iloc[-1]
        
        # 转换策略收益率为Series
        self.strategy_returns = pd.Series(strategy_returns, index=all_dates[:len(strategy_returns)])
        
        logger.info(f"每日目标头寸生成完成，共生成 {len(self.daily_positions)} 天的头寸")
        return self.daily_positions
    
    def evaluate_strategy(self):
        """评估策略绩效"""
        logger.info("开始评估策略绩效")
        
        if len(self.strategy_returns) == 0:
            logger.warning("没有策略收益率数据，无法评估")
            return None
        
        # 计算策略评估指标
        metrics = self.calculate_strategy_metrics(self.strategy_returns)
        
        # 输出指标
        logger.info("\n===== 策略评估指标 =====")
        for metric_name, value in metrics.items():
            if metric_name in ['最大回撤']:
                logger.info(f"{metric_name}: {value:.4f} ({value*100:.2f}%)")
            elif metric_name in ['胜率']:
                logger.info(f"{metric_name}: {value:.4f} ({value*100:.2f}%)")
            else:
                logger.info(f"{metric_name}: {value:.4f}")
        
        # 可视化策略绩效
        self.visualize_strategy_performance()
        
        return metrics
    
    def visualize_strategy_performance(self):
        """可视化策略绩效"""
        logger.info("开始可视化策略绩效")
        
        if len(self.strategy_returns) == 0:
            logger.warning("没有策略收益率数据，无法可视化")
            return
        
        # 计算累计收益率
        cumulative_returns = (1 + self.strategy_returns).cumprod()
        
        # 创建图表
        plt.figure(figsize=(15, 12))
        
        # 子图1：策略收益率
        plt.subplot(3, 1, 1)
        plt.plot(self.strategy_returns.index, self.strategy_returns)
        plt.title('AW Beta策略每日收益率', fontsize=16)
        plt.ylabel('收益率', fontsize=12)
        plt.grid(True)
        
        # 子图2：累计收益率
        plt.subplot(3, 1, 2)
        plt.plot(cumulative_returns.index, cumulative_returns)
        plt.title('AW Beta策略累计收益率', fontsize=16)
        plt.ylabel('累计收益率', fontsize=12)
        plt.grid(True)
        
        # 子图3：最大回撤
        plt.subplot(3, 1, 3)
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        plt.plot(drawdown.index, drawdown)
        plt.title('AW Beta策略最大回撤', fontsize=16)
        plt.ylabel('回撤率', fontsize=12)
        plt.xlabel('日期', fontsize=12)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('strategy_performance.png', dpi=300)
        logger.info("策略绩效图表已保存到 strategy_performance.png")
    
    def run_strategy(self):
        """运行完整的AW Beta策略"""
        logger.info("开始运行AW Beta策略")
        
        # 1. 数据加载和预处理
        self.load_data()
        self.preprocess_data()
        
        # 2. 生成每日目标头寸
        self.generate_daily_positions()
        
        # 3. 策略评估
        metrics = self.evaluate_strategy()
        
        # 4. 保存策略结果
        self.save_strategy_results(metrics)
        
        logger.info("AW Beta策略运行完成")
        return metrics
    
    def save_strategy_results(self, metrics):
        """保存策略结果"""
        logger.info("开始保存策略结果")
        
        # 保存策略收益率
        if len(self.strategy_returns) > 0:
            self.strategy_returns.to_csv('strategy_returns.csv')
            logger.info("策略收益率已保存到 strategy_returns.csv")
        
        # 保存策略指标
        if metrics is not None:
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv('strategy_metrics.csv', index=False)
            logger.info("策略指标已保存到 strategy_metrics.csv")
        
        logger.info("策略结果保存完成")

if __name__ == "__main__":
    # 创建策略实例
    strategy = AWBetaStrategy()
    
    # 运行策略
    metrics = strategy.run_strategy()
    
    logger.info("\n===== AW Beta策略运行结束 =====")
