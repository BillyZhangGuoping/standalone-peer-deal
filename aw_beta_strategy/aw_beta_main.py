import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from position import get_contract_multiplier, calculate_position_size, calculate_portfolio_metrics
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数
DATA_DIR = 'History_Data/hot_daily_market_data'
OUTPUT_DIR = 'target_position'
START_DATE = '2024-07-01'
CAPITAL = 10000000  # 1000万资金
RISK_FREE_RATE = 0.03  # 无风险利率
LOOKBACK_PERIOD = 60  # 回看周期，从20延长到60天

# 品种列表和品类标签
VARIETY_CATEGORIES = {
    # 黑色系
    'i': '黑色系', 'rb': '黑色系', 'hc': '黑色系', 'jm': '黑色系', 'j': '黑色系', 'ss': '黑色系',
    # 有色系
    'ni': '有色系', 'sn': '有色系', 'si': '有色系', 'pb': '有色系', 'cu': '有色系', 'al': '有色系', 'zn': '有色系',
    # 农产品
    'SF': '农产品', 'SM': '农产品', 'a': '农产品', 'b': '农产品', 'm': '农产品', 'RM': '农产品', 'p': '农产品', 'y': '农产品',
    'CF': '农产品', 'SR': '农产品', 'c': '农产品', 'cs': '农产品', 'AP': '农产品', 'PK': '农产品', 'jd': '农产品', 'CJ': '农产品', 'lh': '农产品',
    # 能源化工
    'sc': '能源化工', 'fu': '能源化工', 'lu': '能源化工', 'pg': '能源化工', 'bu': '能源化工', 'eg': '能源化工', 'TA': '能源化工',
    'PF': '能源化工', 'PX': '能源化工', 'l': '能源化工', 'v': '能源化工', 'MA': '能源化工', 'pp': '能源化工', 'eb': '能源化工',
    'ru': '能源化工', 'nr': '能源化工', 'br': '能源化工', 'SA': '能源化工', 'FG': '能源化工', 'UR': '能源化工',
    # 贵金属
    'AU': '贵金属', 'AG': '贵金属',
    # 股指
    'IF': '股指', 'IC': '股指', 'IH': '股指', 'IM': '股指',
    # 外汇
    'ec': '外汇',
    # 利率
    'T': '利率', 'TF': '利率',
    # 其他
    'ao': '其他', 'SH': '其他', 'OI': '农产品', 'sp': '其他'
}

# 品种列表
variety_list = list(VARIETY_CATEGORIES.keys())

class AWBetaStrategySimple:
    """简化版AW Beta策略，恢复之前的高性能表现"""
    
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
        self.strategy_returns = None
        self.final_weights = None
        self.daily_positions = {}
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
    
    def winsorize_data(self, data, winsorize_ratio=0.01):
        """对数据进行去极值处理
        
        参数：
        data: 原始数据
        winsorize_ratio: 去极值比例，默认为1%
        
        返回：
        winsorized_data: 去极值后的数据
        """
        logger.info(f"开始对数据进行去极值处理，比例: {winsorize_ratio}")
        
        # 计算上下分位数
        lower_bound = data.quantile(winsorize_ratio)
        upper_bound = data.quantile(1 - winsorize_ratio)
        
        # 对每个品种进行去极值处理
        winsorized_data = data.copy()
        for column in winsorized_data.columns:
            winsorized_data[column] = winsorized_data[column].clip(lower=lower_bound[column], upper=upper_bound[column])
        
        logger.info("数据去极值处理完成")
        return winsorized_data
    
    def standardize_data(self, data):
        """对数据进行标准化处理
        
        参数：
        data: 原始数据
        
        返回：
        standardized_data: 标准化后的数据
        """
        logger.info("开始对数据进行标准化处理")
        
        # 计算每个品种的均值和标准差
        mean = data.mean()
        std = data.std()
        
        # 标准化处理
        standardized_data = (data - mean) / std
        
        logger.info("数据标准化处理完成")
        return standardized_data
    
    def neutralize_data(self, data):
        """对数据进行中性化处理（简化版，使用横截面均值中性化）
        
        参数：
        data: 原始数据
        
        返回：
        neutralized_data: 中性化后的数据
        """
        logger.info("开始对数据进行中性化处理")
        
        # 横截面均值中性化
        neutralized_data = data.sub(data.mean(axis=1), axis=0)
        
        logger.info("数据中性化处理完成")
        return neutralized_data
    
    def get_variety_category(self, symbol):
        """获取品种的品类标签
        
        参数：
        symbol: 品种代码
        
        返回：
        category: 品类标签
        """
        return VARIETY_CATEGORIES.get(symbol, '其他')
    
    def analyze_category_correlation(self, returns):
        """分析品类间的相关性
        
        参数：
        returns: 收益率数据
        
        返回：
        category_returns: 品类收益率
        category_corr_matrix: 品类相关系数矩阵
        """
        logger.info("开始分析品类间的相关性")
        
        # 为每个品种添加品类标签
        variety_categories = {symbol: self.get_variety_category(symbol) for symbol in returns.columns}
        
        # 计算每个品类的平均收益率
        category_returns = {}
        for category in set(variety_categories.values()):
            # 获取该品类的所有品种
            category_symbols = [symbol for symbol, cat in variety_categories.items() if cat == category]
            if category_symbols:
                category_returns[category] = returns[category_symbols].mean(axis=1)
        
        # 转换为DataFrame
        category_returns_df = pd.DataFrame(category_returns)
        
        # 计算品类相关系数矩阵
        category_corr_matrix = category_returns_df.corr()
        
        logger.info("品类相关性分析完成")
        return category_returns_df, category_corr_matrix
    
    def select_varieties(self, returns, top_n=30):
        """基于相关性和绩效选择品种
        
        参数：
        returns: 收益率数据
        top_n: 选择的品种数量
        
        返回：
        selected_varieties: 选择的品种列表
        """
        logger.info(f"开始基于相关性和绩效选择品种，选择数量: {top_n}")
        
        # 计算品种绩效（年化收益率）
        performance = returns.mean() * 252
        
        # 计算品种相关性矩阵
        corr_matrix = returns.corr()
        
        # 计算每个品种与其他品种的平均相关性
        avg_corr = corr_matrix.mean(axis=1)
        
        # 结合绩效和相关性进行评分
        # 绩效权重0.6，相关性权重0.4（相关性越低越好，所以取负值）
        scores = 0.6 * performance - 0.4 * avg_corr
        
        # 选择评分最高的top_n品种
        selected_varieties = scores.nlargest(top_n).index.tolist()
        
        logger.info(f"选择的品种数量: {len(selected_varieties)}")
        logger.info(f"选择的品种: {selected_varieties}")
        
        return selected_varieties
    
    def calculate_risk_parity_weights(self, returns, initial_weights=None, max_iterations=100, tolerance=1e-5):
        """计算风险平价权重
        
        参数：
        returns: 收益率数据
        initial_weights: 初始权重
        max_iterations: 最大迭代次数
        tolerance: 收敛 tolerance
        
        返回：
        risk_parity_weights: 风险平价权重
        """
        logger.info("开始计算风险平价权重")
        
        # 计算协方差矩阵
        cov_matrix = returns.cov()
        
        # 初始化权重
        if initial_weights is None:
            initial_weights = np.ones(len(returns.columns)) / len(returns.columns)
        weights = initial_weights.copy()
        
        # 迭代优化
        for i in range(max_iterations):
            # 计算投资组合方差
            port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # 计算每个品种的边际风险贡献
            mrc = np.dot(cov_matrix, weights) / np.sqrt(port_var) if port_var != 0 else np.zeros(len(weights))
            
            # 计算每个品种的风险贡献
            rc = weights * mrc
            
            # 处理可能的NaN或负值
            rc = np.maximum(rc, 1e-12)  # 确保风险贡献为正且非零
            
            # 计算风险贡献的标准差
            rc_std = rc.std()
            
            # 如果风险贡献足够均衡，收敛
            if rc_std < tolerance:
                logger.info(f"风险平价权重计算收敛，迭代次数: {i+1}")
                break
            
            # 调整权重：增加风险贡献小的品种权重，减少风险贡献大的品种权重
            weights *= np.sqrt(rc.mean() / rc)
            
            # 处理可能的NaN或无穷大
            weights = np.nan_to_num(weights, nan=1e-12, posinf=1e-12, neginf=1e-12)
            
            # 重新标准化权重
            weights = weights / weights.sum()
        
        if i == max_iterations - 1:
            logger.warning(f"风险平价权重计算未收敛，最大迭代次数: {max_iterations}")
        
        # 转换为Series
        risk_parity_weights = pd.Series(weights, index=returns.columns)
        
        # 确保没有NaN值，替换为等权重
        if risk_parity_weights.isnull().any():
            logger.warning("风险平价权重中存在NaN值，使用等权重替换")
            equal_weights = pd.Series(1/len(returns.columns), index=returns.columns)
            return equal_weights
        
        logger.info("风险平价权重计算完成")
        return risk_parity_weights
    
    def adjust_weights_for_hedging(self, weights, returns):
        """调整权重以实现品类对冲和动态Beta调整，使用合约价值(手数*合约乘数)进行风险分配
        
        参数：
        weights: 原始权重
        returns: 收益率数据
        
        返回：
        adjusted_weights: 调整后的权重
        """
        logger.info("开始调整权重以实现品类对冲")
        
        # 为每个品种添加品类标签
        variety_categories = {symbol: self.get_variety_category(symbol) for symbol in returns.columns}
        
        # 计算每个品类的权重占比
        category_weights = {}
        for category in set(variety_categories.values()):
            category_symbols = [symbol for symbol, cat in variety_categories.items() if cat == category]
            category_weights[category] = weights[category_symbols].sum() if category_symbols else 0
        
        # 计算品类收益率和相关性
        category_returns, category_corr = self.analyze_category_correlation(returns)
        
        # 计算动态Beta值用于动态对冲调整
        market_return = returns.mean(axis=1)
        dynamic_beta = {}
        for symbol in returns.columns:
            beta_estimates = self.kalman_filter_beta(returns[symbol], market_return)
            dynamic_beta[symbol] = beta_estimates.iloc[-1] if not beta_estimates.empty else 1.0
        
        # 输出原始品类权重分布
        logger.info("原始品类权重分布：")
        for category, weight in category_weights.items():
            logger.info(f"  {category}: {weight:.4f} ({weight*100:.2f}%)")
        
        # 调整权重：降低高度相关品类的集中度
        adjusted_weights = weights.copy()
        
        # 识别高集中度品类（权重占比超过30%）
        high_concentration_categories = [cat for cat, weight in category_weights.items() if abs(weight) > 0.3]
        
        if high_concentration_categories:
            logger.info(f"识别到高集中度品类：{high_concentration_categories}")
            
            # 寻找与高集中度品类负相关或低相关的品类
            for high_cat in high_concentration_categories:
                if high_cat not in category_corr.columns:
                    continue
                    
                # 获取与该品类相关性
                correlations = category_corr[high_cat]
                
                # 寻找低相关或负相关的品类
                hedge_categories = correlations[correlations < 0.3].index.tolist()
                
                if hedge_categories:
                    logger.info(f"  为 {high_cat} 寻找对冲品类：{hedge_categories}")
                    
                    # 降低高集中度品类的权重，增加对冲品类的权重
                    reduction_factor = 0.1  # 降低10%
                    reduction_amount = category_weights[high_cat] * reduction_factor
                    
                    # 计算每个对冲品类的权重增加量
                    hedge_categories = [cat for cat in hedge_categories if cat in category_weights]
                    if hedge_categories:
                        increase_per_hedge = reduction_amount / len(hedge_categories)
                        
                        # 降低高集中度品类的权重
                        for symbol, weight in adjusted_weights.items():
                            if variety_categories[symbol] == high_cat:
                                adjusted_weights[symbol] *= (1 - reduction_factor)
                        
                        # 增加对冲品类的权重
                        for hedge_cat in hedge_categories:
                            for symbol, weight in adjusted_weights.items():
                                if variety_categories[symbol] == hedge_cat:
                                    adjusted_weights[symbol] += increase_per_hedge / len([s for s, c in variety_categories.items() if c == hedge_cat])
        
        # 基于动态Beta值进行调整：Beta值较高的品种适当降低权重，Beta值较低的品种适当增加权重
        logger.info("基于动态Beta值调整权重")
        
        # 计算平均Beta值
        avg_beta = np.mean(list(dynamic_beta.values()))
        
        # 调整因子，控制Beta调整的幅度
        beta_adjustment_factor = 0.1
        
        for symbol, weight in adjusted_weights.items():
            beta = dynamic_beta[symbol]
            # 如果Beta值高于平均水平，降低权重；如果Beta值低于平均水平，增加权重
            if beta > avg_beta:
                # 高Beta品种降低权重
                adjusted_weights[symbol] *= (1 - beta_adjustment_factor * (beta / avg_beta - 1))
            elif beta < avg_beta:
                # 低Beta品种增加权重
                adjusted_weights[symbol] *= (1 + beta_adjustment_factor * (1 - beta / avg_beta))
        
        # 重新标准化权重，确保权重绝对值总和为1
        absolute_total = abs(adjusted_weights).sum()
        if absolute_total != 0:
            adjusted_weights = adjusted_weights / absolute_total
        else:
            # 如果总和为0，使用等权重
            adjusted_weights = pd.Series(1/len(returns.columns), index=returns.columns)
        
        # 计算调整后的品类权重分布
        adjusted_category_weights = {}
        for category in set(variety_categories.values()):
            category_symbols = [symbol for symbol, cat in variety_categories.items() if cat == category]
            adjusted_category_weights[category] = adjusted_weights[category_symbols].sum() if category_symbols else 0
        
        logger.info("调整后的品类权重分布：")
        for category, weight in adjusted_category_weights.items():
            logger.info(f"  {category}: {weight:.4f} ({weight*100:.2f}%)")
        
        logger.info("权重调整完成")
        return adjusted_weights
    
    def get_主力合约代码(self, base_symbol, date):
        """获取指定日期的主力合约代码
        
        参数：
        base_symbol: 基础品种代码，如 'A', 'RB'
        date: 日期，datetime类型
        
        返回：
        contract_symbol: 主力合约代码，如 'a2409.DCE', 'rb2410.SHFE'
        """
        # 检查是否加载了该品种的数据
        if base_symbol not in self.all_data:
            logger.warning(f"没有品种 {base_symbol} 的数据，无法获取主力合约代码")
            return f"{base_symbol}0000.UNK"
        
        # 获取该品种的历史数据
        variety_data = self.all_data[base_symbol]
        
        # 筛选该日期之前的数据
        past_data = variety_data[variety_data.index <= date]
        
        if past_data.empty:
            logger.warning(f"没有品种 {base_symbol} 在 {date.strftime('%Y-%m-%d')} 之前的数据，无法获取主力合约代码")
            return f"{base_symbol}0000.UNK"
        
        # 获取最近一条数据的合约代码
        latest_data = past_data.iloc[-1]
        contract_symbol = latest_data['symbol']
        
        return contract_symbol
    
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
    
    def build_aw_beta_strategy(self, end_date=None):
        """构建原始AW Beta策略
        
        参数：
        end_date: 截止日期，只使用该日期之前的数据计算权重
        
        返回：
        final_weights: 最终投资组合权重
        """
        if end_date:
            logger.info(f"开始构建AW Beta策略，截止日期: {end_date.strftime('%Y-%m-%d')}，回看周期: {LOOKBACK_PERIOD}")
        else:
            logger.info(f"开始构建AW Beta策略，回看周期: {LOOKBACK_PERIOD}")
        
        if self.processed_data is None:
            self.preprocess_data()
        
        # 获取截止日期之前的数据
        if end_date:
            data = self.processed_data[self.processed_data.index <= end_date]
        else:
            data = self.processed_data.copy()
        
        # 使用回看周期的数据
        if len(data) > LOOKBACK_PERIOD:
            data = data.tail(LOOKBACK_PERIOD)
        
        # 计算每日收益率
        returns = data.pct_change().dropna()
        
        # 如果收益率数据不足，返回等权重
        if len(returns) < 5:
            logger.warning(f"收益率数据不足，使用等权重: {len(returns)} 天")
            equal_weights = pd.Series(1/len(data.columns), index=data.columns)
            return equal_weights
        
        logger.info("因子数据处理：使用原始收益率数据")
        
        # 基于相关性和绩效选择品种
        selected_varieties = self.select_varieties(returns, top_n=30)
        
        # 只使用选择的品种
        returns = returns[selected_varieties]
        
        # 计算市场组合收益率
        market_return = returns.mean(axis=1)
        
        # 使用卡尔曼滤波估计动态Beta值
        beta_values = {}
        
        for symbol in returns.columns:
            # 计算动态Beta值
            beta_estimates = self.kalman_filter_beta(returns[symbol], market_return)
            # 使用最新的Beta估计值
            beta = beta_estimates.iloc[-1] if not beta_estimates.empty else 1.0
            beta_values[symbol] = beta
        
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
        
        # 将Alpha和Beta转换为DataFrame
        alpha_df = pd.Series(alpha_values, name='alpha')
        beta_df = pd.Series(beta_values, name='beta')
        aw_beta_df = pd.concat([alpha_df, beta_df], axis=1)
        
        # 计算每个品种的权重：Alpha / Beta（允许正负值，支持多空）
        aw_beta_df['weight'] = aw_beta_df['alpha'] / aw_beta_df['beta'] if (aw_beta_df['beta'] != 0).all() else aw_beta_df['alpha']
        
        # 标准化权重，考虑正负值，使其绝对值总和为1
        absolute_total = abs(aw_beta_df['weight']).sum()
        if absolute_total != 0:
            aw_beta_df['weight'] = aw_beta_df['weight'] / absolute_total
        else:
            # 如果总和为0，使用等权重
            aw_beta_df['weight'] = 1/len(aw_beta_df)
        
        # 限制单个品种的最大权重，避免过度集中
        max_weight = 0.1  # 单个品种最大权重10%
        aw_beta_df['weight'] = aw_beta_df['weight'].clip(lower=-max_weight, upper=max_weight)
        
        # 重新标准化，使用绝对值总和为1
        absolute_total = abs(aw_beta_df['weight']).sum()
        if absolute_total != 0:
            aw_beta_df['weight'] = aw_beta_df['weight'] / absolute_total
        else:
            aw_beta_df['weight'] = 1/len(aw_beta_df)
        
        # 风险平价优化：使用更复杂的风险平价方法
        # 以AW Beta权重作为初始权重
        initial_weights = aw_beta_df['weight'].values
        
        # 计算风险平价权重
        risk_parity_weights = self.calculate_risk_parity_weights(returns, initial_weights=initial_weights)
        
        # 结合AW Beta权重和风险平价权重
        final_weights = (aw_beta_df['weight'] + risk_parity_weights) / 2
        
        # 再次使用绝对值总和为1进行标准化
        absolute_total = abs(final_weights).sum()
        if absolute_total != 0:
            final_weights = final_weights / absolute_total
        else:
            final_weights = pd.Series(1/len(returns.columns), index=returns.columns)
        
        # 调整权重以实现品类对冲
        final_weights = self.adjust_weights_for_hedging(final_weights, returns)
        
        # 最后一次标准化，确保绝对值总和为1
        absolute_total = abs(final_weights).sum()
        if absolute_total != 0:
            final_weights = final_weights / absolute_total
        else:
            final_weights = pd.Series(1/len(returns.columns), index=returns.columns)
        
        # 计算风险集中度（前5个品种）
        sorted_weights = final_weights.sort_values(ascending=False)
        top_5_concentration = sorted_weights.head(5).sum()
        logger.info(f"风险集中度（前5个品种）: {top_5_concentration:.4f} ({top_5_concentration*100:.2f}%)")
        
        return final_weights
    
    def generate_daily_positions(self):
        """生成每日目标头寸"""
        logger.info(f"开始生成每日目标头寸，起始日期: {self.start_date}")
        
        if self.processed_data is None:
            self.preprocess_data()
        
        # 获取所有交易日期
        all_dates = self.processed_data.index[self.processed_data.index >= self.start_date]
        
        # 初始化策略收益率列表
        self.strategy_returns = []
        self.daily_weights = {}
        
        # 导入position模块中的函数
        try:
            import sys
            import os
            # 添加父目录到Python路径
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from position import get_contract_multiplier, calculate_position_size, calculate_portfolio_metrics
            use_position_functions = True
            logger.info("成功导入position模块函数")
        except ImportError as e:
            logger.warning(f"无法导入position模块函数: {e}，将使用自定义实现")
            use_position_functions = False
        
        for date in all_dates:
            logger.info(f"\n处理日期: {date.strftime('%Y-%m-%d')}")
            
            # 获取当前日期的收盘价
            current_prices = self.processed_data.loc[date]
            
            # 每日重新计算权重
            daily_weight = self.build_aw_beta_strategy(end_date=date)
            self.daily_weights[date] = daily_weight
            
            # 使用每日更新的权重生成目标头寸
            target_positions_list = []
            
            # 初始化风险分配字典
            risk_allocation = {}
            
            # 计算总风险分配比例
            total_risk = abs(daily_weight).sum()
            
            # 使用新的核心分配函数分配资金
            market_data = self.processed_data[self.processed_data.index <= date].tail(LOOKBACK_PERIOD)
            capital_allocation = self.allocate_capital_among_varieties(self.capital, market_data, daily_weight, current_prices)
            
            # 遍历所有选择的品种，确保每个品种都在输出中，包括未持仓的品种
            for symbol, weight in daily_weight.items():
                # 获取主力合约代码
                contract_symbol = self.get_主力合约代码(symbol, date)
                
                # 获取当前价格
                price = current_prices[symbol]
                
                # 初始化target_quantity为0，确保每个品种都在输出中
                target_quantity = 0
                risk_amount = 0
                stop_loss_price = 0
                
                # 确保价格是有效的数值
                if np.isnan(price) or price <= 0:
                    logger.warning(f"品种 {symbol} 的价格无效，使用零头寸")
                    valid_price = False
                    price = 0
                else:
                    valid_price = True
                
                # 获取合约乘数和保证金率
                try:
                    contract_multiplier, margin_rate = get_contract_multiplier(contract_symbol)
                except Exception as e:
                    contract_multiplier, margin_rate = 10, 0.1  # 默认值
                
                if valid_price:
                    # 使用新的核心分配函数结果
                    target_capital = capital_allocation.get(symbol, 0)
                    
                    # 计算止损价格（基于波动率，这里简化为价格的2%）
                    stop_loss_pct = 0.02  # 2%止损
                    stop_loss_price = price * (1 - stop_loss_pct) if target_capital > 0 else price * (1 + stop_loss_pct)
                    
                    # 计算每手价值（价格 * 合约乘数）
                    per_unit_value = price * contract_multiplier
                    
                    if per_unit_value > 0:
                        # 计算目标手数：手数 = 目标资金 / 每手价值
                        target_quantity = target_capital / per_unit_value
                        # 取整
                        target_quantity = int(target_quantity)
                    
                    # 计算风险金额（基于止损）
                    risk_per_unit = abs(price - stop_loss_price) * contract_multiplier
                    risk_amount = abs(target_quantity) * risk_per_unit
                
                # 计算持仓价值（手数 * 合约乘数 * 价格）
                position_value = abs(target_quantity) * price * contract_multiplier
                
                # 计算保证金占用
                margin_usage = position_value * margin_rate
                
                # 计算市值
                market_value = position_value if target_quantity > 0 else -position_value
                
                # 确定信号
                signal = 1 if target_quantity > 0 else -1 if target_quantity < 0 else 0
                
                # 构建头寸字典
                position_dict = {
                    'symbol': contract_symbol,
                    'current_price': price,
                    'stop_loss_price': stop_loss_price,
                    'contract_multiplier': contract_multiplier,
                    'position_size': target_quantity,  # 注意：这里改名为position_size，与另一个策略保持一致
                    'position_value': position_value,
                    'margin_usage': margin_usage,
                    'risk_amount': risk_amount,
                    'margin_rate': margin_rate,
                    'total_capital': self.capital,
                    'signal': signal,
                    'model_type': 'aw_beta_strategy',  # 固定为aw_beta_strategy
                    'market_value': market_value
                }
                
                target_positions_list.append(position_dict)
            
            # 保存每日头寸
            positions_df = pd.DataFrame(target_positions_list)
            
            # 只保留position_size不为0的品种
            positions_df = positions_df[positions_df['position_size'] != 0]
            
            # 如果使用position模块，计算投资组合指标
            if use_position_functions and not positions_df.empty:
                portfolio_metrics = calculate_portfolio_metrics(positions_df)
                logger.info(f"投资组合指标: {portfolio_metrics}")
            
            # 保存到文件
            positions_file = os.path.join(self.output_dir, f'target_positions_{date.strftime("%Y%m%d")}.csv')
            positions_df.to_csv(positions_file, index=False)
            logger.info(f"目标头寸已保存到 {positions_file}")
            
            # 计算当日策略收益率（如果是第一个日期，收益率为0）
            if len(self.strategy_returns) > 0:
                # 获取前一天的收盘价
                prev_date = all_dates[all_dates.get_loc(date) - 1]
                prev_prices = self.processed_data.loc[prev_date]
                
                # 计算每个品种的日收益率
                daily_returns = (current_prices - prev_prices) / prev_prices
                
                # 计算策略日收益率
                strategy_return = (daily_returns * daily_weight).sum()
                self.strategy_returns.append(strategy_return)
            else:
                self.strategy_returns.append(0)
        
        # 转换策略收益率为Series
        self.strategy_returns = pd.Series(self.strategy_returns, index=all_dates)
        
        logger.info(f"每日目标头寸生成完成，共生成 {len(all_dates)} 天的头寸")
        return self.daily_positions
    
    def calculate_max_drawdown(self, returns):
        """计算最大回撤
        
        参数：
        returns: 收益率序列
        
        返回：
        max_drawdown: 最大回撤值
        """
        # 计算累计收益率
        cumulative_returns = (1 + returns).cumprod()
        
        # 计算累计最大值
        cumulative_max = cumulative_returns.cummax()
        
        # 计算回撤
        drawdown = (cumulative_returns - cumulative_max) / cumulative_max
        
        # 计算最大回撤
        max_drawdown = drawdown.min()
        
        return max_drawdown
    
    def calculate_sortino_ratio(self, returns, risk_free_rate=RISK_FREE_RATE):
        """计算Sortino比率
        
        参数：
        returns: 收益率序列
        risk_free_rate: 无风险利率
        
        返回：
        sortino_ratio: Sortino比率
        """
        # 计算超额收益率
        excess_returns = returns - risk_free_rate / 252
        
        # 计算下行风险（仅考虑负收益率）
        downside_returns = excess_returns[excess_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        
        # 计算年化平均超额收益率
        annualized_excess_return = excess_returns.mean() * 252
        
        # 计算Sortino比率
        sortino_ratio = annualized_excess_return / downside_volatility if downside_volatility != 0 else 0
        
        return sortino_ratio
    
    def calculate_information_ratio(self, returns, benchmark_returns=None):
        """计算信息比率
        
        参数：
        returns: 策略收益率序列
        benchmark_returns: 基准收益率序列，默认为无风险利率
        
        返回：
        information_ratio: 信息比率
        """
        if benchmark_returns is None:
            # 如果没有基准收益率，使用无风险利率
            benchmark_returns = pd.Series(RISK_FREE_RATE / 252, index=returns.index)
        
        # 计算超额收益率
        excess_returns = returns - benchmark_returns
        
        # 计算年化平均超额收益率
        annualized_excess_return = excess_returns.mean() * 252
        
        # 计算跟踪误差（超额收益率的年化波动率）
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        # 计算信息比率
        information_ratio = annualized_excess_return / tracking_error if tracking_error != 0 else 0
        
        return information_ratio
        
    def calculate_volatility(self, variety, market_data):
        """计算品种波动率
        
        参数：
        variety: 品种代码
        market_data: 市场数据
        
        返回：
        volatility: 年化波动率
        """
        # 计算历史收益率
        returns = market_data[variety].pct_change().dropna()
        if len(returns) < 5:
            return 0.2  # 默认波动率
        
        # 计算年化波动率
        volatility = returns.std() * np.sqrt(252)
        return volatility
        
    def calculate_average_volatility(self, varieties, market_data):
        """计算品种池平均波动率
        
        参数：
        varieties: 品种列表
        market_data: 市场数据
        
        返回：
        avg_volatility: 平均年化波动率
        """
        volatilities = [self.calculate_volatility(v, market_data) for v in varieties]
        avg_volatility = np.mean(volatilities)
        return avg_volatility if avg_volatility > 0 else 0.2  # 默认平均波动率
        
    def calculate_liquidity_score(self, variety, market_data):
        """计算品种流动性得分（0-100）
        
        参数：
        variety: 品种代码
        market_data: 市场数据
        
        返回：
        liquidity_score: 流动性得分（0-100）
        """
        # 这里简化处理，实际应根据成交量、持仓量等数据计算
        # 假设所有品种流动性得分都为80
        return 80.0
        
    def calculate_dynamic_weights(self, varieties, market_data, strategy_signals):
        """计算动态权重
        
        参数：
        varieties: 品种列表
        market_data: 市场数据
        strategy_signals: 策略信号
        
        返回：
        weights: 动态权重
        """
        # 基于AW Beta和风险平价的混合权重
        # 这里简化处理，实际应使用更复杂的算法
        equal_weights = {v: 1/len(varieties) for v in varieties}
        return equal_weights
        
    def allocate_capital_among_varieties(self, total_capital, market_data, daily_weight, current_prices):
        """核心分配函数
        
        参数：
        total_capital: 总资金
        market_data: 市场数据
        daily_weight: 每日权重
        current_prices: 当前价格
        
        返回：
        allocation: 资金分配结果
        """
        # 步骤1：获取品种池
        varieties = list(daily_weight.keys())
        
        # 步骤2：计算各品种权重（使用已有的daily_weight）
        weights = daily_weight.to_dict()
        
        # 步骤3：分配资金
        allocation = {}
        strategy_signals = {v: weights[v] for v in varieties}  # 使用权重作为信号强度
        
        for variety in varieties:
            # 基础资金分配
            base_capital = total_capital * abs(weights[variety])
            
            # 信号强度调整
            signal_strength = abs(strategy_signals.get(variety, 0))
            if signal_strength > 0.3:  # 强信号
                adjustment_factor = 1.2
            elif signal_strength > 0.1:  # 中等信号
                adjustment_factor = 1.0
            else:  # 弱信号
                adjustment_factor = 0.7
                
            # 波动率调整
            volatility = self.calculate_volatility(variety, market_data)
            avg_volatility = self.calculate_average_volatility(varieties, market_data)
            vol_adjust = avg_volatility / max(volatility, 0.01)
            
            # 流动性调整
            liquidity = self.calculate_liquidity_score(variety, market_data)
            liquidity_adjust = liquidity / 100  # 假设流动性得分0-100
            
            # 最终分配资金（保留方向）
            final_capital = base_capital * adjustment_factor * vol_adjust * liquidity_adjust * (1 if weights[variety] > 0 else -1)
            allocation[variety] = final_capital
        
        # 步骤4：归一化
        total_allocated = sum(abs(v) for v in allocation.values())
        if total_allocated > 0:
            allocation = {k: v/total_allocated * total_capital for k, v in allocation.items()}
        
        return allocation
    
    def kalman_filter_beta(self, asset_returns, market_returns, initial_beta=1.0, q=0.001, r=0.01):
        """使用卡尔曼滤波估计动态Beta值
        
        参数：
        asset_returns: 资产收益率序列
        market_returns: 市场收益率序列
        initial_beta: 初始Beta值
        q: 状态噪声方差（Beta的变化率）
        r: 测量噪声方差（残差）
        
        返回：
        beta_estimates: 动态Beta估计值序列
        """
        # 初始化状态向量 [alpha, beta]
        x = np.array([[0.0], [initial_beta]])
        
        # 初始化状态协方差矩阵
        P = np.eye(2) * 0.1
        
        # 状态转移矩阵
        F = np.eye(2)
        
        # 测量矩阵 [1, market_return]
        H = np.zeros((1, 2))
        H[0, 0] = 1.0
        
        # 状态噪声协方差矩阵
        Q = np.eye(2)
        Q[0, 0] = q  # alpha的状态噪声方差
        Q[1, 1] = q  # beta的状态噪声方差
        
        # 测量噪声协方差矩阵
        R = np.array([[r]])
        
        # 存储Beta估计值
        beta_estimates = []
        
        # 执行卡尔曼滤波
        for i in range(len(asset_returns)):
            # 设置当前测量矩阵的market_return值
            H[0, 1] = market_returns.iloc[i]
            
            # 当前测量值
            z = np.array([[asset_returns.iloc[i]]])
            
            # 1. 状态预测
            x_pred = np.dot(F, x)
            P_pred = np.dot(np.dot(F, P), F.T) + Q
            
            # 2. 计算卡尔曼增益
            S = np.dot(np.dot(H, P_pred), H.T) + R
            K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(S))
            
            # 3. 状态更新
            x = x_pred + np.dot(K, (z - np.dot(H, x_pred)))
            
            # 4. 更新协方差矩阵
            P = np.dot((np.eye(2) - np.dot(K, H)), P_pred)
            
            # 存储Beta估计值
            beta_estimates.append(x[1, 0])
        
        # 转换为Series
        beta_estimates = pd.Series(beta_estimates, index=asset_returns.index)
        
        return beta_estimates
    
    def evaluate_strategy(self):
        """评估策略绩效"""
        logger.info("开始评估策略绩效")
        
        if self.strategy_returns is None or len(self.strategy_returns) == 0:
            logger.warning("没有策略收益率数据，无法评估")
            return None
        
        # 计算策略评估指标
        metrics = {
            '年化收益率': self.strategy_returns.mean() * 252,
            '年化波动率': self.strategy_returns.std() * np.sqrt(252),
            '夏普比率': self.strategy_returns.mean() / self.strategy_returns.std() * np.sqrt(252),
            '最大回撤': self.calculate_max_drawdown(self.strategy_returns),
            'Sortino比率': self.calculate_sortino_ratio(self.strategy_returns),
            '信息比率': self.calculate_information_ratio(self.strategy_returns),
            '日波动率': self.strategy_returns.std()
        }
        
        # 输出指标
        logger.info("\n===== 策略评估指标 =====")
        for metric_name, value in metrics.items():
            if metric_name in ['日波动率', '年化波动率']:
                logger.info(f"{metric_name}: {value:.4f} ({value*100:.2f}%)")
            elif metric_name == '最大回撤':
                logger.info(f"{metric_name}: {value:.4f} ({value*100:.2f}%)")
            else:
                logger.info(f"{metric_name}: {value:.4f}")
        
        # 计算最后一天的风险集中度
        if self.daily_weights:
            last_date = sorted(self.daily_weights.keys())[-1]
            last_weights = self.daily_weights[last_date]
            sorted_weights = last_weights.sort_values(ascending=False)
            top_5_concentration = sorted_weights.head(5).sum()
            logger.info(f"风险集中度（前5个品种）: {top_5_concentration:.4f} ({top_5_concentration*100:.2f}%)")
        
        return metrics
    
    def run_strategy(self):
        """运行完整的AW Beta策略"""
        logger.info("开始运行AW Beta策略")
        
        # 1. 数据加载和预处理
        self.load_data()
        self.preprocess_data()
        
        # 2. 构建策略
        self.build_aw_beta_strategy()
        
        # 3. 生成每日目标头寸
        self.generate_daily_positions()
        
        # 4. 策略评估
        metrics = self.evaluate_strategy()
        
        logger.info("AW Beta策略运行完成")
        return metrics

if __name__ == "__main__":
    # 创建策略实例
    strategy = AWBetaStrategySimple()
    
    # 运行策略
    metrics = strategy.run_strategy()
    
    logger.info("\n===== AW Beta策略运行结束 =====")
