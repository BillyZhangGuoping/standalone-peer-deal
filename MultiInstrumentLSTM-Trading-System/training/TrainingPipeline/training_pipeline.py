import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os
import datetime

class TrainingPipeline:
    def __init__(self, data_collector, feature_engineer, sequence_builder, lstm_model, loss_function, optimizer, model_manager):
        """
        初始化训练流水线
        
        参数:
        - data_collector: 数据收集器实例
        - feature_engineer: 特征工程实例
        - sequence_builder: 序列构造器实例
        - lstm_model: LSTM模型实例
        - loss_function: 损失函数
        - optimizer: 优化器
        - model_manager: 模型管理器实例
        """
        self.data_collector = data_collector
        self.feature_engineer = feature_engineer
        self.sequence_builder = sequence_builder
        self.lstm_model = lstm_model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model_manager = model_manager
    
    def run_pipeline(self, epochs=100, batch_size=32, model_name='lstm_model'):
        """
        运行训练流水线
        
        参数:
        - epochs: 训练轮数
        - batch_size: 批量大小
        - model_name: 模型名称
        
        返回:
        - history: 训练历史
        - test_results: 测试结果
        """
        # 1. 数据收集
        print("Step 1: Collecting data...")
        all_data = self.data_collector.collect_all_varieties()
        valid_varieties = self.data_collector.get_valid_varieties(all_data)
        print(f"Found {len(valid_varieties)} valid varieties")
        
        # 2. 特征工程
        print("Step 2: Engineering features...")
        engineered_data = self.feature_engineer.engineer_all_varieties(all_data)
        
        # 3. 特征标准化
        print("Step 3: Standardizing features...")
        standardized_data, scalers = self.feature_engineer.standardize_features(engineered_data)
        
        # 4. 序列构造
        print("Step 4: Building sequences...")
        X, y, date_list = self.sequence_builder.build_sequences(standardized_data, all_data)
        print(f"Built {len(X)} sequences")
        
        # 5. 划分训练集、验证集和测试集
        print("Step 5: Splitting train/val/test sets...")
        datasets = self.sequence_builder.split_train_val_test(X, y, date_list)
        X_train, y_train, dates_train = datasets['train']
        X_val, y_val, dates_val = datasets['val']
        X_test, y_test, dates_test = datasets['test']
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # 6. 编译模型
        print("Step 6: Compiling model...")
        self.lstm_model.compile_model(self.loss_function, self.optimizer)
        
        # 7. 设置回调函数
        print("Step 7: Setting up callbacks...")
        callbacks = self._setup_callbacks(model_name)
        
        # 8. 训练模型
        print("Step 8: Training model...")
        history = self.lstm_model.train(
            X_train, y_train, X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        # 9. 模型评估
        print("Step 9: Evaluating model...")
        test_results = self._evaluate_model(X_test, y_test)
        
        # 10. 保存模型
        print("Step 10: Saving model...")
        metadata = {
            'variety_order': self.sequence_builder.variety_order,
            'scalers': scalers,
            'train_dates': dates_train,
            'val_dates': dates_val,
            'test_dates': dates_test,
            'model_config': {
                'hidden_units': self.lstm_model.hidden_units,
                'num_layers': self.lstm_model.num_layers,
                'dropout_rate': self.lstm_model.dropout_rate
            }
        }
        model_path = self.model_manager.save_model(self.lstm_model, model_name, metadata)
        print(f"Model saved to {model_path}")
        
        return history, test_results
    
    def _setup_callbacks(self, model_name):
        """
        设置回调函数
        
        参数:
        - model_name: 模型名称
        
        返回:
        - callbacks: 回调函数列表
        """
        callbacks = []
        
        # 早停
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # 模型检查点
        checkpoint_dir = os.path.join('checkpoints', model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        )
        callbacks.append(checkpoint)
        
        # TensorBoard
        log_dir = os.path.join('logs', model_name, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard)
        
        return callbacks
    
    def _evaluate_model(self, X_test, y_test):
        """
        评估模型
        
        参数:
        - X_test: 测试集输入
        - y_test: 测试集标签
        
        返回:
        - test_results: 测试结果
        """
        # 预测持仓权重
        y_pred = self.lstm_model.predict(X_test)
        
        # 计算测试集损失
        test_loss = self.loss_function(y_test, y_pred)
        
        # 计算投资组合收益
        portfolio_returns = np.sum(y_pred * y_test, axis=1)
        avg_portfolio_return = np.mean(portfolio_returns)
        std_portfolio_return = np.std(portfolio_returns)
        sharpe_ratio = avg_portfolio_return / (std_portfolio_return + 1e-8)
        
        # 计算最大回撤
        cumulative_returns = np.cumsum(portfolio_returns)
        max_drawdown = self._calculate_max_drawdown(cumulative_returns)
        
        test_results = {
            'test_loss': test_loss,
            'avg_portfolio_return': avg_portfolio_return,
            'std_portfolio_return': std_portfolio_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cumulative_returns': cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
        }
        
        print(f"Test Loss: {test_loss:.6f}")
        print(f"Average Portfolio Return: {avg_portfolio_return:.6f}")
        print(f"Portfolio Return Std: {std_portfolio_return:.6f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Max Drawdown: {max_drawdown:.6f}")
        print(f"Cumulative Returns: {cumulative_returns[-1]:.6f}")
        
        return test_results
    
    def _calculate_max_drawdown(self, cumulative_returns):
        """
        计算最大回撤
        
        参数:
        - cumulative_returns: 累积收益率
        
        返回:
        - max_drawdown: 最大回撤
        """
        if len(cumulative_returns) == 0:
            return 0
        
        peak = cumulative_returns[0]
        max_drawdown = 0
        
        for return_val in cumulative_returns:
            if return_val > peak:
                peak = return_val
            drawdown = (peak - return_val) / (peak + 1e-8)
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
