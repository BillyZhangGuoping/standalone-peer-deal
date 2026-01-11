import pandas as pd
import numpy as np
import os
import logging
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTraining:
    def __init__(self, data_dir, lookahead=3):
        self.data_dir = data_dir
        self.lookahead = lookahead
        self.models = {}
        self.feature_importance = {}
    
    def load_data(self, variety_list):
        """加载多个品种的历史日线数据"""
        all_data = {}
        # 获取绝对路径
        abs_data_dir = os.path.abspath(self.data_dir)
        files = [f for f in os.listdir(abs_data_dir) if f.endswith('.csv')]
        
        for file in files:
            file_path = os.path.join(abs_data_dir, file)
            base_symbol = file.split('.')[0].upper()
            
            if base_symbol.lower() not in [v.lower() for v in variety_list]:
                continue
             
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # 确保symbol列存在，包含当日主力合约代码
            if 'symbol' not in df.columns:
                # 如果没有symbol列，使用当前代码生成
                df['symbol'] = df.index.map(lambda x: f"{base_symbol.lower()}{x.strftime('%y%m')}.{self._get_exchange(base_symbol)}")
            
            all_data[base_symbol] = df
        
        logger.info(f"共加载 {len(all_data)} 个品种的历史数据")
        return all_data
    
    def _get_exchange(self, base_symbol):
        """根据基础代码获取交易所"""
        # 简化处理，实际需要根据合约规则映射
        if base_symbol in ['A', 'B', 'C', 'CS', 'J', 'JD', 'JM', 'L', 'M', 'P', 'V', 'Y']:
            return 'DCE'
        elif base_symbol in ['CU', 'AL', 'ZN', 'PB', 'NI', 'SN', 'AG', 'AU', 'RB', 'HC', 'BU', 'RU', 'WR', 'SP']:
            return 'SHFE'
        elif base_symbol in ['CF', 'SR', 'OI', 'RM', 'MA', 'TA', 'ZC', 'FG', 'RS', 'RI', 'JR', 'LR', 'PM', 'WH', 'CY', 'AP', 'CJ', 'UR', 'SA', 'SF', 'SM', 'PF', 'PK', 'BO']:
            return 'CZCE'
        elif base_symbol in ['IC', 'IF', 'IH', 'IM', 'T', 'TF', 'TS']:
            return 'CFFEX'
        elif base_symbol in ['SC', 'LU', 'NR', 'EB', 'EG', 'PG', 'BC', 'BD', 'BR', 'LPG']:
            return 'INE'
        elif base_symbol in ['LC', 'SI', 'PT', 'PD', 'PS']:
            return 'GFEX'
        else:
            return 'DCE'
    
    def preprocess_data(self, all_data):
        """预处理数据：对齐日期，处理缺失值"""
        # 获取所有品种的公共日期范围
        all_dates = []
        for df in all_data.values():
            all_dates.extend(df.index.tolist())
        common_dates = sorted(list(set(all_dates)))
        
        processed_data = {}
        for symbol, df in all_data.items():
            # 对齐日期
            df = df.reindex(common_dates)
            # 处理缺失值
            df = df.fillna(method='ffill').fillna(method='bfill')
            processed_data[symbol] = df
        
        return processed_data
    
    def feature_engineering(self, df):
        """特征工程：计算技术指标和统计特征"""
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
        
        # 波动率
        data['volatility5'] = df['close'].pct_change().rolling(window=5).std() * np.sqrt(252)
        data['volatility10'] = df['close'].pct_change().rolling(window=10).std() * np.sqrt(252)
        data['volatility20'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # 收益率
        data['return1'] = df['close'].pct_change(1)
        data['return5'] = df['close'].pct_change(5)
        data['return10'] = df['close'].pct_change(10)
        
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
    
    def build_labels(self, df):
        """构建标签：未来5日趋势强度"""
        data = df.copy()
        # 计算未来5日的收益率
        data['future_return5'] = data['close'].pct_change(self.lookahead).shift(-self.lookahead)
        # 使用未来5日收益率作为趋势强度标签（连续值）
        return data
    
    def split_train_test(self, df):
        """划分训练集和测试集，固定数量：300个训练，60个评估，总共需要360个数据点"""
        # 移除NaN值
        df = df.dropna()
        
        # 确保至少有360个数据点
        if len(df) < 360:
            logger.warning(f"数据量不足360个，当前只有{len(df)}个数据点")
            return None, None, None, None
        
        # 分离特征和标签
        feature_columns = [col for col in df.columns if col not in ['future_return5', 'symbol']]
        X = df[feature_columns]
        y = df['future_return5']
        
        # 固定数量划分：300个用于训练，60个用于评估
        X_train = X.iloc[-360:-60]
        y_train = y.iloc[-360:-60]
        X_test = X.iloc[-60:]
        y_test = y.iloc[-60:]
        
        logger.info(f"训练集大小：{len(X_train)}，测试集大小：{len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """训练LightGBM回归模型"""
        # 创建LightGBM数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # 设置模型参数，降低学习率使模型训练更充分
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.01,  # 降低学习率
            'num_leaves': 63,  # 增加树的复杂度
            'max_depth': -1,  # 不限制树深度
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # 训练模型，暂时禁用早停，让模型充分训练
        callbacks = [
            # lgb.early_stopping(stopping_rounds=100),  # 暂时注释掉早停
            lgb.log_evaluation(period=100)
        ]
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,  # 增加训练轮数
            valid_sets=[test_data],
            callbacks=callbacks
        )
        
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """评估模型"""
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"模型评估结果：RMSE = {rmse:.6f}, R2 = {r2:.6f}")
        return {'rmse': rmse, 'r2': r2}
    
    def get_feature_importance(self, model, X_train):
        """获取特征重要性"""
        importance = model.feature_importance(importance_type='gain')
        feature_names = X_train.columns
        importance_dict = dict(zip(feature_names, importance))
        
        # 按重要性排序
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        return importance_dict
    
    def save_model(self, model, symbol, output_dir):
        """保存模型"""
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f'{symbol}_model.txt')
        model.save_model(model_path)
        logger.info(f"模型已保存到 {model_path}")
    
    def train_all_models(self, all_data, output_dir, end_date=None):
        """训练所有品种的模型，只使用截止日期之前的数据
        
        参数：
        all_data: 所有品种的历史数据
        output_dir: 模型保存目录
        end_date: 截止日期，只使用该日期之前的数据进行训练
        """ 
        for symbol, df in all_data.items():
            logger.info(f"开始训练 {symbol} 品种的模型")
            
            # 如果指定了截止日期，只使用截止日期之前的数据
            if end_date is not None:
                df = df[df.index <= end_date]
            
            # 特征工程：只使用历史数据计算指标，避免未来数据泄露
            feature_df = self.feature_engineering(df)
            
            # 构建标签：计算未来5日收益率作为趋势强度标签
            # 注意：这里使用shift(-lookahead)会导致标签包含未来数据
            # 但在split_train_test中，我们会移除包含NaN的行，确保只使用有效的历史数据进行训练
            labeled_df = self.build_labels(feature_df)
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = self.split_train_test(labeled_df)
            
            # 检查是否有足够的数据进行训练
            if X_train is None or len(X_train) < 300 or len(X_test) < 60:
                logger.warning(f"{symbol} 数据量不足360个，当前训练集{len(X_train) if X_train is not None else 0}个，测试集{len(X_test) if X_test is not None else 0}个，跳过训练")
                continue
            
            # 训练模型
            model = self.train_model(X_train, y_train, X_test, y_test)
            
            # 评估模型
            metrics = self.evaluate_model(model, X_test, y_test)
            
            # 获取特征重要性
            importance = self.get_feature_importance(model, X_train)
            self.feature_importance[symbol] = importance
            
            # 保存模型
            self.save_model(model, symbol, output_dir)
            self.models[symbol] = model
        
        # 保存特征重要性
        importance_file = os.path.join(output_dir, 'feature_importance.csv')
        pd.DataFrame(self.feature_importance).T.to_csv(importance_file)
        logger.info(f"特征重要性已保存到 {importance_file}")
        
        return self.models
