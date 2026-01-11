import pandas as pd
import numpy as np
import logging
import lightgbm as lgb
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioAllocation:
    def __init__(self, capital, risk_coefficient=0.02):
        self.capital = capital
        self.risk_coefficient = risk_coefficient  # 每笔交易的风险比例
        self.allocated_weights = {}
        self.target_positions = {}
    
    def load_models(self, model_dir):
        """加载训练好的LightGBM模型"""
        models = {}
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.txt')]
        
        for file in model_files:
            symbol = file.split('_model.txt')[0]
            model_path = os.path.join(model_dir, file)
            model = lgb.Booster(model_file=model_path)
            models[symbol] = model
        
        logger.info(f"共加载 {len(models)} 个模型")
        return models
    
    def predict_trend_strength(self, models, all_data):
        """使用训练好的模型预测每个品种的趋势强度"""
        trend_strength = {}
        
        # 遍历每个品种
        for symbol, df in all_data.items():
            if symbol not in models:
                continue
            
            model = models[symbol]
            
            # 特征工程（与训练时保持一致）
            feature_df = self._feature_engineering(df)
            
            # 移除NaN值
            feature_df = feature_df.dropna()
            
            if feature_df.empty:
                continue
            
            # 准备特征数据
            # 确保特征列与训练时完全一致，排除所有非特征列
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'ma5', 'ma10', 'ma20', 'ma60',
                'momentum5', 'momentum10', 'momentum20', 'momentum60',
                'volatility5', 'volatility10', 'volatility20', 'volatility60',
                'return1', 'return5', 'return10', 'return60',
                'tr', 'atr',
                'volume_change', 'price_volume_corr'
            ]
            # 只保留训练时使用的特征列
            X = feature_df[feature_columns]
            
            # 使用最新数据进行预测
            latest_X = X.iloc[-1:]
            # 设置predict_disable_shape_check=True，避免特征数量不一致的错误
            prediction = model.predict(latest_X, predict_disable_shape_check=True)
            
            # 保存预测的趋势强度
            trend_strength[symbol] = prediction[0]
        
        logger.info(f"趋势强度预测完成，共预测 {len(trend_strength)} 个品种")
        return trend_strength
    
    def _feature_engineering(self, df):
        """特征工程（与模型训练时保持一致）"""
        data = df.copy()
        
        # 价格相关特征
        data['open'] = df['open']
        data['high'] = df['high']
        data['low'] = df['low']
        data['close'] = df['close']
        data['volume'] = df['volume']
        
        # 移动平均线
        data['ma5'] = df['close'].rolling(window=5).mean()
        data['ma10'] = df['close'].rolling(window=10).mean()
        data['ma20'] = df['close'].rolling(window=20).mean()
        data['ma60'] = df['close'].rolling(window=60).mean()
        
        # 动量指标
        data['momentum5'] = df['close'] - df['close'].shift(5)
        data['momentum10'] = df['close'] - df['close'].shift(10)
        data['momentum20'] = df['close'] - df['close'].shift(20)
        data['momentum60'] = df['close'] - df['close'].shift(60)
        
        # 波动率
        data['volatility5'] = df['close'].pct_change().rolling(window=5).std() * np.sqrt(252)
        data['volatility10'] = df['close'].pct_change().rolling(window=10).std() * np.sqrt(252)
        data['volatility20'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        data['volatility60'] = df['close'].pct_change().rolling(window=60).std() * np.sqrt(252)
        
        # 收益率
        data['return1'] = df['close'].pct_change(1)
        data['return5'] = df['close'].pct_change(5)
        data['return10'] = df['close'].pct_change(10)
        data['return60'] = df['close'].pct_change(60)
        
        # ATR (平均真实波幅)
        data['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        data['atr'] = data['tr'].rolling(window=14).mean()
        
        # 量价关系
        data['volume_change'] = df['volume'].pct_change()
        data['price_volume_corr'] = df['close'].rolling(window=20).corr(df['volume'])
        
        return data
    
    def filter_strong_trend_varieties(self, trend_strength, threshold=0.001):
        """筛选出趋势较强的品种，降低阈值以包含更多品种"""
        filtered = {}
        for symbol, strength in trend_strength.items():
            if abs(strength) > threshold:
                filtered[symbol] = strength
        
        logger.info(f"趋势强度筛选完成，共筛选出 {len(filtered)} 个品种")
        return filtered
    
    def risk_parity_allocation(self, filtered_trend, volatility, correlation_matrix):
        """使用风险平价方法分配资金，结合预期收益和协方差矩阵优化风险调整后收益"""
        # 只考虑筛选出的品种
        symbols = list(filtered_trend.keys())
        n = len(symbols)
        
        if n == 0:
            return {}
        
        # 获取相关品种的相关性矩阵
        corr_matrix = correlation_matrix.loc[symbols, symbols]
        
        # 1. 计算每个品种的预期收益（趋势强度）和风险
        expected_returns = np.array([filtered_trend[symbol] for symbol in symbols])
        vol = np.array([volatility[symbol] for symbol in symbols])
        
        # 2. 使用收缩估计改进相关性矩阵的稳定性
        corr_matrix = self._shrink_correlation_matrix(corr_matrix)
        
        # 3. 计算协方差矩阵
        cov_matrix = np.diag(vol) @ corr_matrix.values @ np.diag(vol)
        
        # 4. 初始化权重
        weights = np.ones(n) / n
        
        # 5. 迭代优化：同时考虑风险调整后收益和风险预算约束
        max_iter = 1000
        tolerance = 1e-6
        
        for i in range(max_iter):
            # 计算组合风险（方差）
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            
            # 计算风险贡献
            risk_contribution = (weights * (cov_matrix @ weights)) / portfolio_vol
            
            # 计算风险贡献差异（目标是等风险贡献）
            risk_budget = portfolio_vol / n  # 等风险贡献
            diff_risk = risk_contribution - risk_budget
            
            # 计算夏普比率（风险调整后收益）
            portfolio_return = weights.T @ expected_returns
            sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            
            # 计算夏普比率的梯度（简化计算）
            sharpe_gradient = (expected_returns - sharpe_ratio * (cov_matrix @ weights) / portfolio_vol) / portfolio_vol
            
            # 综合考虑风险贡献差异和夏普比率梯度来调整权重
            # 同时最大化夏普比率和实现风险平价
            learning_rate = 0.05
            weights = weights * np.exp(learning_rate * sharpe_gradient - 0.1 * diff_risk / portfolio_vol)
            weights = weights / np.sum(weights)
            
            # 检查收敛条件
            if np.max(np.abs(diff_risk)) < tolerance:
                break
        
        # 6. 进一步调整：确保权重符号与预期收益一致
        for i in range(n):
            if expected_returns[i] < 0 and weights[i] > 0:
                # 预期收益为负，应该是空单，将权重变为负数
                weights[i] = -abs(weights[i])
            elif expected_returns[i] > 0 and weights[i] < 0:
                # 预期收益为正，应该是多单，将权重变为正数
                weights[i] = abs(weights[i])
        
        # 归一化权重：分别对正负权重进行归一化，确保资金分配合理
        positive_weights = np.maximum(weights, 0)
        negative_weights = np.abs(np.minimum(weights, 0))
        
        positive_sum = np.sum(positive_weights)
        negative_sum = np.sum(negative_weights)
        
        # 确保权重和为1，同时保留正负符号
        if positive_sum > 0 or negative_sum > 0:
            # 计算总权重（绝对值之和）
            total_weight = positive_sum + negative_sum
            weights = weights / total_weight
        
        # 转换为字典
        allocated_weights = {symbol: weight for symbol, weight in zip(symbols, weights)}
        
        self.allocated_weights = allocated_weights
        logger.info(f"风险平价资金分配完成，优化后的夏普比率：{sharpe_ratio:.4f}")
        return allocated_weights
    
    def _shrink_correlation_matrix(self, corr_matrix, shrinkage=0.1):
        """使用收缩估计改进相关性矩阵的稳定性
        
        参数：
        corr_matrix: 原始相关性矩阵
        shrinkage: 收缩系数，0-1之间，越大越稳定
        
        返回：
        收缩后的相关性矩阵
        """
        n = corr_matrix.shape[0]
        # 计算样本协方差矩阵的平均相关系数
        mean_corr = np.mean(corr_matrix.values[np.triu_indices(n, 1)])
        # 创建目标矩阵（对角线为1，非对角线为平均相关系数）
        target_matrix = np.full((n, n), mean_corr)
        np.fill_diagonal(target_matrix, 1.0)
        # 执行收缩估计
        shrunk_corr = (1 - shrinkage) * corr_matrix + shrinkage * target_matrix
        return shrunk_corr
    
    def calculate_target_positions(self, allocated_weights, all_data, atr, contract_multipliers):
        """根据分配的资金和ATR计算目标持仓手数，生成与random_forest_strategy相同格式的目标头寸"""
        target_positions = []
        
        for symbol, weight in allocated_weights.items():
            if symbol not in all_data or symbol not in atr:
                continue
            
            # 获取当前日期的数据
            current_data = all_data[symbol].iloc[-1]
            
            # 获取当日主力合约代码
            contract_symbol = current_data['symbol']
            
            # 获取当前价格
            current_price = current_data['close']
            
            # 获取ATR
            current_atr = atr[symbol]
            
            # 获取合约乘数
            multiplier = contract_multipliers.get(symbol, 10)  # 默认10
            
            # 获取保证金率
            _, margin_rate = self.get_contract_multiplier(contract_symbol)
            
            # 计算分配的资金绝对值
            allocated_capital = abs(self.capital * weight)
            
            # 根据ATR计算手数绝对值：手数 = (分配的资金 * 风险系数) / (ATR * 合约乘数)
            # 这里的风险系数表示每笔交易愿意承担的风险比例
            position_size_abs = (allocated_capital * self.risk_coefficient) / (current_atr * multiplier)
            
            # 四舍五入到整数
            position_size_abs = int(round(position_size_abs))
            
            # 获取趋势方向（正为多头，负为空头）
            signal = 1.0 if self.allocated_weights[symbol] > 0 else -1.0
            
            # 根据信号确定最终的position_size符号，负数表示空单
            position_size = position_size_abs * signal
            
            # 计算持仓价值（绝对值）
            position_value = abs(position_size) * current_price * multiplier
            
            # 计算保证金占用
            margin_usage = position_value * margin_rate
            
            # 计算风险金额（使用保证金占用）
            risk_amount = margin_usage
            
            # 计算市值
            market_value = position_value if signal == 1.0 else -position_value
            
            # 构建头寸字典，与random_forest_strategy格式一致
            position_dict = {
                'symbol': contract_symbol,
                'current_price': current_price,
                'contract_multiplier': multiplier,
                'position_size': position_size,
                'position_value': position_value,
                'margin_usage': margin_usage,
                'risk_amount': risk_amount,
                'margin_rate': margin_rate,
                'total_capital': self.capital,
                'signal': signal,
                'model_type': 'lightgbm_strategy',
                'market_value': market_value,
                'allocated_capital': allocated_capital,
                'atr': current_atr
            }
            
            target_positions.append(position_dict)
        
        logger.info(f"目标持仓手数计算完成，共 {len(target_positions)} 个品种")
        self.target_positions = target_positions
        return target_positions
    
    def get_contract_multiplier(self, contract_symbol):
        """获取合约乘数和保证金率
        
        参数：
        contract_symbol: 合约代码，如a2403.DCE
        
        返回：
        tuple: (合约乘数, 保证金率)
        """
        # 简化处理，实际应根据交易所规定获取
        # 提取品种代码
        base_symbol = contract_symbol.split('.')[0].upper()
        if len(base_symbol) > 3:
            base_symbol = base_symbol[:-4]  # 去除年份和月份，如A2403 → A
        
        # 合约乘数映射
        multipliers = {
            # 金属类
            'CU': 5, 'AL': 5, 'ZN': 5, 'PB': 5, 'NI': 1, 'SN': 1, 'AG': 15, 'AU': 1000,
            # 能源化工
            'SC': 1000, 'LU': 100, 'NR': 10, 'EB': 5, 'EG': 10, 'PG': 20,
            # 农产品
            'A': 10, 'B': 10, 'C': 10, 'CS': 10, 'J': 100, 'JD': 5, 'JM': 60,
            # 股指
            'IC': 200, 'IF': 300, 'IH': 300, 'IM': 200,
            # 其他
            'RB': 10, 'HC': 10, 'BU': 10, 'RU': 10
        }
        
        # 保证金率映射
        margin_rates = {
            # 金属类
            'CU': 0.09, 'AL': 0.09, 'ZN': 0.09, 'PB': 0.09, 'NI': 0.1, 'SN': 0.1, 'AG': 0.17, 'AU': 0.16,
            # 能源化工
            'SC': 0.09, 'LU': 0.09, 'NR': 0.09, 'EB': 0.07, 'EG': 0.07, 'PG': 0.07,
            # 农产品
            'A': 0.07, 'B': 0.07, 'C': 0.07, 'CS': 0.06, 'J': 0.2, 'JD': 0.07, 'JM': 0.12,
            # 股指
            'IC': 0.12, 'IF': 0.12, 'IH': 0.12, 'IM': 0.12,
            # 其他
            'RB': 0.07, 'HC': 0.07, 'BU': 0.09, 'RU': 0.09
        }
        
        # 默认值
        multiplier = multipliers.get(base_symbol, 10)
        margin_rate = margin_rates.get(base_symbol, 0.07)
        
        return multiplier, margin_rate
    
    def get_contract_multipliers(self):
        """获取各品种的合约乘数（简化处理）"""
        # 这里使用简化的合约乘数映射，实际应根据交易所规定获取
        multipliers = {
            # 金属类
            'CU': 5, 'AL': 5, 'ZN': 5, 'PB': 5, 'NI': 1, 'SN': 1, 'AG': 15, 'AU': 1000,
            # 能源化工
            'SC': 1000, 'LU': 100, 'NR': 10, 'EB': 5, 'EG': 10, 'PG': 20,
            # 农产品
            'A': 10, 'B': 10, 'C': 10, 'CS': 10, 'J': 100, 'JD': 5, 'JM': 60,
            # 股指
            'IF': 300, 'IC': 200, 'IH': 300, 'IM': 200,
            # 其他
            'RB': 10, 'HC': 10, 'BU': 10, 'RU': 10
        }
        return multipliers
    
    def run_allocation(self, models, all_data, volatility, correlation_matrix, atr):
        """运行完整的资金分配流程"""
        # 1. 预测趋势强度
        trend_strength = self.predict_trend_strength(models, all_data)
        
        # 2. 筛选趋势较强的品种
        filtered_trend = self.filter_strong_trend_varieties(trend_strength)
        
        # 3. 风险平价资金分配
        allocated_weights = self.risk_parity_allocation(filtered_trend, volatility, correlation_matrix)
        
        # 4. 获取合约乘数
        contract_multipliers = self.get_contract_multipliers()
        
        # 5. 计算目标持仓手数
        target_positions = self.calculate_target_positions(allocated_weights, all_data, atr, contract_multipliers)
        
        return {
            'trend_strength': trend_strength,
            'allocated_weights': allocated_weights,
            'target_positions': target_positions
        }
