import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入各个模块
from data_processing.DataCollector.data_collector import DataCollector
from data_processing.FeatureEngineer.feature_engineer import FeatureEngineer
from data_processing.SequenceBuilder.sequence_builder import SequenceBuilder
from model.LSTM_Model.lstm_model import LSTMModel
from training.LossFunction.custom_loss import portfolio_return_loss
from training.Optimizer.optimizer_config import get_optimizer, get_default_optimizer_config
from training.TrainingPipeline.training_pipeline import TrainingPipeline
from model.ModelManager.model_manager import ModelManager
from backtest.PortfolioSimulator.portfolio_simulator import PortfolioSimulator
from backtest.RiskAnalyzer.risk_analyzer import RiskAnalyzer
from backtest.PerformanceMetrics.performance_metrics import PerformanceMetrics
from trading.SignalGenerator.signal_generator import SignalGenerator
from trading.PositionManager.position_manager import PositionManager
from trading.RiskController.risk_controller import RiskController

def main():
    """
    多品种LSTM交易系统主函数
    """
    print("Multi-Instrument LSTM Trading System")
    print("=" * 50)
    
    # 1. 配置参数
    config = {
        'data_dir': '../History_Data/hot_daily_market_data',
        'start_date': '2024-01-01',
        'sequence_length': 60,
        'prediction_horizon': 1,
        'hidden_units': 64,
        'num_layers': 2,
        'dropout_rate': 0.2,
        'epochs': 50,
        'batch_size': 32,
        'model_name': 'multi_instrument_lstm',
        'initial_capital': 1000000,
        'transaction_cost': 0.001,
        'max_leverage': 5.0,
        'max_position_percent': 0.15
    }
    
    print(f"Configuration: {config}")
    print("=" * 50)
    
    # 2. 初始化各个模块
    print("Initializing modules...")
    
    # 数据处理模块
    data_collector = DataCollector(data_dir=config['data_dir'], start_date=config['start_date'])
    feature_engineer = FeatureEngineer()
    sequence_builder = SequenceBuilder(sequence_length=config['sequence_length'], prediction_horizon=config['prediction_horizon'])
    
    # 模型管理模块
    model_manager = ModelManager()
    
    # 3. 训练模型
    print("\nTraining model...")
    
    # 创建自定义损失函数
    loss_function = portfolio_return_loss(l2_regularization=0.001)
    
    # 获取优化器
    optimizer_config = get_default_optimizer_config()
    optimizer = get_optimizer(optimizer_config)
    
    # 注意：这里需要先运行数据处理流程，获取输入形状和品种数量，然后再初始化LSTM模型
    # 所以我们需要先运行一部分数据处理流程
    
    # 3.1 数据收集
    print("Step 1: Collecting data...")
    all_data = data_collector.collect_all_varieties()
    
    # 只保留random_forest_main.py中指定的品种列表
    variety_list = ['i', 'rb', 'hc', 'jm', 'j', 'ss', 'ni', 'SF', 'SM', 'lc', 'sn', 'si', 'pb', 'cu', 'al', 'zn',
                    'ao', 'SH', 'au', 'ag', 'OI', 'a', 'b', 'm', 'RM', 'p', 'y', 'CF', 'SR', 'c', 'cs', 'AP', 'PK',
                    'jd', 'CJ', 'lh', 'sc', 'fu', 'lu', 'pg', 'bu', 'eg', 'TA', 'PF', 'PX', 'l', 'v', 'MA', 'pp',
                    'eb', 'ru', 'nr', 'br', 'SA', 'FG', 'UR', 'sp', 'IF', 'IC', 'IH', 'IM', 'ec','T', 'TF']
    
    # 转换品种列表为小写，以便匹配
    variety_list_lower = [v.lower() for v in variety_list]
    
    # 过滤all_data，只保留variety_list中的品种
    filtered_data = {}
    for variety, data in all_data.items():
        if variety.lower() in variety_list_lower:
            filtered_data[variety] = data
    
    print(f"Found {len(filtered_data)} valid varieties in the specified list")
    
    # 更新all_data为过滤后的数据
    all_data = filtered_data
    valid_varieties = list(all_data.keys())
    
    # 3.2 特征工程
    print("Step 2: Engineering features...")
    engineered_data = feature_engineer.engineer_all_varieties(all_data)
    
    # 3.3 特征标准化
    print("Step 3: Standardizing features...")
    standardized_data, scalers = feature_engineer.standardize_features(engineered_data)
    
    # 3.4 序列构造
    print("Step 4: Building sequences...")
    X, y, date_list = sequence_builder.build_sequences(standardized_data, all_data)
    print(f"Built {len(X)} sequences")
    
    # 检查是否生成了有效序列
    if len(X) == 0:
        print("Warning: No sequences were built. This may be due to insufficient data or no common time indices across varieties.")
        print("Trying with a smaller sample of varieties...")
        # 尝试使用前5个品种重新构建序列
        sample_varieties = list(standardized_data.keys())[:5]
        sample_standardized_data = {v: standardized_data[v] for v in sample_varieties}
        sample_all_data = {v: all_data[v] for v in sample_varieties}
        X, y, date_list = sequence_builder.build_sequences(sample_standardized_data, sample_all_data)
        print(f"Built {len(X)} sequences with {len(sample_varieties)} varieties")
        valid_varieties = sample_varieties
    
    # 如果仍然没有序列，退出程序
    if len(X) == 0:
        print("Error: Could not build any sequences. Please check your data.")
        return
    
    # 3.5 获取输入形状和品种数量
    input_shape = (X.shape[1], X.shape[2])
    num_varieties = len(valid_varieties)
    
    # 3.6 初始化LSTM模型
    print(f"Input shape: {input_shape}, Number of varieties: {num_varieties}")
    lstm_model = LSTMModel(
        input_shape=input_shape,
        num_varieties=num_varieties,
        hidden_units=config['hidden_units'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate']
    )
    
    # 3.7 创建训练流水线并运行
    training_pipeline = TrainingPipeline(
        data_collector=data_collector,
        feature_engineer=feature_engineer,
        sequence_builder=sequence_builder,
        lstm_model=lstm_model,
        loss_function=loss_function,
        optimizer=optimizer,
        model_manager=model_manager
    )
    
    # 3.8 按日期划分数据集：使用2025-1-1之前的数据进行训练，之后的数据进行测试
    print("Step 5: Splitting dataset by date...")
    # 转换日期列表为datetime对象
    date_list_dt = pd.to_datetime(date_list)
    # 定义2025年1月1日的日期
    split_date = pd.to_datetime('2025-01-01')
    
    # 划分训练集（2025-1-1之前）和测试集（2025年全年）
    train_mask = date_list_dt < split_date
    test_mask = date_list_dt >= split_date
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    dates_train = date_list_dt[train_mask].tolist()
    
    # 验证集使用训练集的最后一部分
    val_split = int(len(X_train) * 0.85)
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]
    dates_val = dates_train[val_split:]
    X_train = X_train[:val_split]
    y_train = y_train[:val_split]
    dates_train = dates_train[:val_split]
    
    X_test = X[test_mask]
    y_test = y[test_mask]
    dates_test = date_list_dt[test_mask].tolist()
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 打印日期范围，只有当列表不为空时
    if len(dates_train) > 0:
        print(f"Train date range: {dates_train[0]} to {dates_train[-1]}")
    else:
        print("Train date range: No training data available (all data is after 2025-01-01)")
    
    if len(dates_test) > 0:
        print(f"Test date range: {dates_test[0]} to {dates_test[-1]}")
    
    # 编译模型
    lstm_model.compile_model(loss_function, optimizer)
    
    # 训练模型，只有当有训练数据时
    if len(X_train) > 0:
        print("Training model...")
        history = lstm_model.train(
            X_train, y_train, X_val, y_val,
            epochs=config['epochs'],
            batch_size=config['batch_size']
        )
    else:
        print("No training data available, skipping model training...")
        # 使用随机初始化的模型进行预测
        print("Using randomly initialized model for predictions...")
    
    # 保存模型
    metadata = {
        'variety_order': sequence_builder.variety_order,
        'scalers': scalers,
        'train_dates': dates_train,
        'val_dates': dates_val,
        'test_dates': dates_test,
        'model_config': {
            'hidden_units': lstm_model.hidden_units,
            'num_layers': lstm_model.num_layers,
            'dropout_rate': lstm_model.dropout_rate
        }
    }
    model_path = model_manager.save_model(lstm_model, config['model_name'], metadata)
    print(f"Model saved to {model_path}")
    
    # 4. 回测
    print("\nBacktesting...")
    
    # 预测测试集
    y_pred = lstm_model.predict(X_test)
    
    # 创建组合模拟器
    portfolio_simulator = PortfolioSimulator(
        variety_order=sequence_builder.variety_order,
        initial_capital=config['initial_capital'],
        transaction_cost=config['transaction_cost']
    )
    
    # 运行回测
    backtest_results = portfolio_simulator.simulate(
        dates=dates_test,
        predictions=y_pred,
        returns=y_test
    )
    
    # 保存回测结果到CSV文件
    backtest_results.to_csv('backtest_results.csv')
    print(f"Backtest results saved to backtest_results.csv")
    
    # 提取并保存2025年的每日头寸
    # 使用日期索引过滤2025年的数据
    backtest_results_2025 = backtest_results[backtest_results.index.year == 2025]
    backtest_results_2025.to_csv('2025_daily_positions.csv')
    print(f"2025 daily positions saved to 2025_daily_positions.csv")
    
    # 5. 绩效评估
    print("\nPerformance Evaluation:")
    performance_metrics = PerformanceMetrics(backtest_results)
    performance_results = performance_metrics.run_performance_analysis()
    
    print(f"Annualized Return: {performance_results['annualized_return']:.4%}")
    print(f"Annualized Volatility: {performance_results['annualized_volatility']:.4%}")
    print(f"Sharpe Ratio: {performance_results['sharpe_ratio']:.4f}")
    print(f"Sortino Ratio: {performance_results['sortino_ratio']:.4f}")
    print(f"Max Drawdown: {performance_results['max_drawdown']:.4%}")
    print(f"Calmar Ratio: {performance_results['calmar_ratio']:.4f}")
    print(f"Win Rate: {performance_results['win_rate']:.4%}")
    print(f"Profit/Loss Ratio: {performance_results['profit_loss_ratio']:.4f}")
    
    # 6. 风险分析
    print("\nRisk Analysis:")
    risk_analyzer = RiskAnalyzer(backtest_results)
    risk_results = risk_analyzer.run_risk_analysis()
    
    print(f"95% VaR: {risk_results['var_95']:.4%}")
    print(f"95% CVaR: {risk_results['cvar_95']:.4%}")
    print(f"Max Drawdown: {risk_results['max_drawdown']:.4%}")
    print(f"Volatility: {risk_results['volatility'].mean():.4%}")
    print(f"Skewness: {risk_results['skewness']:.4f}")
    print(f"Kurtosis: {risk_results['kurtosis']:.4f}")
    
    # 7. 生成每日目标头寸文件
    print("\nGenerating Daily Target Positions...")
    
    # 创建信号生成器和仓位管理器
    signal_generator = SignalGenerator(variety_order=sequence_builder.variety_order)
    position_manager = PositionManager(
        variety_order=sequence_builder.variety_order,
        initial_capital=config['initial_capital'],
        max_leverage=config['max_leverage'],
        max_position_percent=config['max_position_percent']
    )
    
    # 获取品种信息
    variety_info = data_collector.varieties_info
    
    # 生成交易信号
    signals = signal_generator.generate_weight_signals(y_pred)
    
    # 生成当前时间戳，用于创建独立文件夹
    import os
    current_time = datetime.now()
    # 文件夹名称格式：YYMMDD_hhmm
    folder_name = current_time.strftime('%y%m%d_%H%M')
    target_position_dir = os.path.join('target_position', folder_name)
    if not os.path.exists(target_position_dir):
        os.makedirs(target_position_dir)
    
    # 生成每日目标头寸
    previous_positions = np.zeros(num_varieties)
    capital = config['initial_capital']
    
    # 为每个测试日生成目标头寸
    for i in range(len(dates_test)):
        date = dates_test[i]
        signal = signals[i]
        
        # 获取当日各品种的价格和合约乘数
        prices = []
        contract_multipliers = []
        margin_rates = []
        
        for variety in sequence_builder.variety_order:
            # 获取当前价格（使用实际数据中的收盘价）
            if variety in all_data and date in all_data[variety].index:
                price = all_data[variety].loc[date]['close']
            else:
                price = 100.0  # 默认价格
            prices.append(price)
            
            # 获取合约乘数
            if variety in variety_info.index:
                multiplier = variety_info.loc[variety]['multiplier']
            else:
                multiplier = 10.0  # 默认合约乘数
            contract_multipliers.append(multiplier)
            
            # 获取保证金率
            if variety in variety_info.index:
                margin_rate = variety_info.loc[variety]['margin_ratio']
            else:
                margin_rate = 0.1  # 默认保证金率
            margin_rates.append(margin_rate)
        
        prices = np.array(prices)
        contract_multipliers = np.array(contract_multipliers)
        margin_rates = np.array(margin_rates)
        
        # 更新仓位
        new_positions, position_changes = position_manager.update_positions(
            signal,
            prices,
            contract_multipliers
        )
        
        # 生成目标头寸列表
        target_positions = []
        for j in range(num_varieties):
            variety = sequence_builder.variety_order[j]
            current_price = prices[j]
            multiplier = contract_multipliers[j]
            margin_rate = margin_rates[j]
            position_size = new_positions[j]
            
            # 计算持仓价值和保证金占用
            position_value = abs(position_size) * current_price * multiplier
            margin_usage = position_value * margin_rate
            
            # 获取合约代码（使用数据中的实际合约代码，如a2505.DCE）
            # 从品种信息中获取交易所
            if variety in variety_info.index:
                exchange = variety_info.loc[variety]['exchange']
            else:
                # 默认使用DCE交易所
                exchange = 'DCE'
            # 生成合约代码：品种代码+年份月份+交易所（如a2505.DCE）
            contract_symbol = f"{variety.lower()}2505.{exchange}"
            
            # 创建头寸字典
            position_dict = {
                'symbol': contract_symbol,  # 完整合约代码
                'current_price': current_price,
                'contract_multiplier': multiplier,
                'position_size': int(position_size),  # 目标手数，取整数
                'position_value': position_value,
                'margin_usage': margin_usage,
                'margin_rate': margin_rate,
                'total_capital': capital,
                'signal': signal[j],
                'date': date.strftime('%Y-%m-%d')
            }
            target_positions.append(position_dict)
        
        # 保存每日目标头寸到CSV文件
        date_str = date.strftime('%Y%m%d')
        target_position_file = os.path.join(target_position_dir, f'target_positions_{date_str}.csv')
        
        # 将目标头寸转换为DataFrame并保存
        target_positions_df = pd.DataFrame(target_positions)
        target_positions_df.to_csv(target_position_file, index=False)
        
        # 更新前一天仓位
        previous_positions = new_positions.copy()
        
        # 更新资金（使用回测结果中的资金）
        if date in backtest_results.index:
            capital = backtest_results.loc[date]['capital']
            position_manager.update_capital(capital)
    
    print(f"Daily target positions saved to {target_position_dir} directory")
    
    # 8. 风险控制
    print("\nRisk Control:")
    risk_controller = RiskController(
        variety_order=sequence_builder.variety_order
    )
    
    # 检查风险
    risk_alerts, risk_adjustments = risk_controller.check_risk(
        portfolio_value=backtest_results['portfolio_value'].iloc[-1],
        daily_returns=backtest_results['daily_return'],
        positions=new_positions,
        prices=prices,
        contract_multipliers=contract_multipliers
    )
    
    if risk_alerts:
        print(f"Risk Alerts: {risk_alerts}")
        print(f"Risk Adjustments: {risk_adjustments}")
        
        # 根据风险调整仓位
        adjusted_positions = risk_controller.adjust_positions_for_risk(
            new_positions,
            risk_adjustments
        )
        print(f"Adjusted Positions: {adjusted_positions}")
    else:
        print("No risk alerts detected.")
    
    print("\n" + "=" * 50)
    print("Multi-Instrument LSTM Trading System completed successfully!")

if __name__ == "__main__":
    main()
