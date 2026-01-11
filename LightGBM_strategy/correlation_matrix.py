import pandas as pd
import numpy as np
import logging
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorrelationMatrix:
    def __init__(self):
        self.correlation_matrix = None
        self.volatility = {}
        self.atr = {}
    
    def calculate_daily_returns(self, all_data):
        """计算每个品种的日收益率"""
        returns = {}
        for symbol, df in all_data.items():
            # 计算日收益率
            df['daily_return'] = df['close'].pct_change()
            returns[symbol] = df['daily_return']
        
        # 将所有品种的收益率合并为一个DataFrame
        returns_df = pd.DataFrame(returns)
        return returns_df
    
    def calculate_rolling_correlation(self, returns_df, window=60):
        """计算滚动相关性矩阵，使用滚动窗口提高预测的准确性
        
        参数：
        returns_df: 各品种的日收益率DataFrame
        window: 滚动窗口大小，默认60个交易日
        
        返回：
        最新的滚动相关性矩阵
        """
        # 计算滚动相关性矩阵
        rolling_corr = returns_df.rolling(window=window).corr()
        # 获取最新的相关性矩阵
        latest_corr = rolling_corr.iloc[-len(returns_df.columns):]
        latest_corr = latest_corr.set_index(pd.Index(returns_df.columns))
        
        self.correlation_matrix = latest_corr
        logger.info(f"滚动相关性矩阵计算完成，窗口大小：{window}")
        return latest_corr
    
    def calculate_static_correlation(self, returns_df):
        """计算静态相关性矩阵（全样本）"""
        corr_matrix = returns_df.corr()
        self.correlation_matrix = corr_matrix
        logger.info("静态相关性矩阵计算完成")
        return corr_matrix
    
    def calculate_volatility(self, all_data, window=20):
        """计算每个品种的波动率"""
        for symbol, df in all_data.items():
            # 计算日收益率
            df['daily_return'] = df['close'].pct_change()
            # 计算滚动波动率（年化）
            df['volatility'] = df['daily_return'].rolling(window=window).std() * np.sqrt(252)
            # 保存最新波动率
            self.volatility[symbol] = df['volatility'].iloc[-1]
        
        logger.info(f"波动率计算完成，窗口大小：{window}")
        return self.volatility
    
    def calculate_atr(self, all_data, window=14):
        """计算每个品种的ATR（平均真实波幅）"""
        for symbol, df in all_data.items():
            # 计算真实波幅
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            # 计算ATR
            df['atr'] = df['tr'].rolling(window=window).mean()
            # 保存最新ATR
            self.atr[symbol] = df['atr'].iloc[-1]
        
        logger.info(f"ATR计算完成，窗口大小：{window}")
        return self.atr
    
    def save_results(self, output_dir):
        """保存计算结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存相关性矩阵
        if self.correlation_matrix is not None:
            corr_file = os.path.join(output_dir, 'correlation_matrix.csv')
            self.correlation_matrix.to_csv(corr_file)
            logger.info(f"相关性矩阵已保存到 {corr_file}")
        
        # 保存波动率
        if self.volatility:
            vol_file = os.path.join(output_dir, 'volatility.csv')
            pd.DataFrame.from_dict(self.volatility, orient='index', columns=['volatility']).to_csv(vol_file)
            logger.info(f"波动率已保存到 {vol_file}")
        
        # 保存ATR
        if self.atr:
            atr_file = os.path.join(output_dir, 'atr.csv')
            pd.DataFrame.from_dict(self.atr, orient='index', columns=['atr']).to_csv(atr_file)
            logger.info(f"ATR已保存到 {atr_file}")
    
    def calculate_all(self, all_data, window=60, use_rolling=True):
        """计算所有指标"""
        # 计算日收益率
        returns_df = self.calculate_daily_returns(all_data)
        
        # 计算相关性矩阵
        if use_rolling:
            self.calculate_rolling_correlation(returns_df, window=window)
        else:
            self.calculate_static_correlation(returns_df)
        
        # 计算波动率
        self.calculate_volatility(all_data, window=20)
        
        # 计算ATR
        self.calculate_atr(all_data, window=14)
        
        return {
            'correlation_matrix': self.correlation_matrix,
            'volatility': self.volatility,
            'atr': self.atr
        }
