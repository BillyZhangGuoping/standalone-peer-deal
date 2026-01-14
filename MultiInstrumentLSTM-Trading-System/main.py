import os
import sys
import numpy as np
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
        'data_dir': 'History_Data',
        'start_date': '2015-01-01',
        'sequence_length': 60,
        'prediction_horizon': 1,
        'hidden_units': 64,
        'num_layers': 2,
        'dropout_rate': 0.2,
        'epochs': 100,
        'batch_size': 32,
        'model_name': 'multi_instrument_lstm',
        'initial_capital': 1000000,
        'transaction_cost': 0.001,
        'max_leverage': 3.0,
        'max_position_percent': 0.05
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
    valid_varieties = data_collector.get_valid_varieties(all_data)
    print(f"Found {len(valid_varieties)} valid varieties")
    
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
    
    # 注意：由于我们已经运行了数据处理流程，这里可以跳过重复的数据处理
    # 所以我们直接训练模型，而不是运行完整的流水线
    
    # 划分训练集、验证集和测试集
    datasets = sequence_builder.split_train_val_test(X, y, date_list)
    X_train, y_train, dates_train = datasets['train']
    X_val, y_val, dates_val = datasets['val']
    X_test, y_test, dates_test = datasets['test']
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 编译模型
    lstm_model.compile_model(loss_function, optimizer)
    
    # 训练模型
    print("Training model...")
    history = lstm_model.train(
        X_train, y_train, X_val, y_val,
        epochs=config['epochs'],
        batch_size=config['batch_size']
    )
    
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
    
    # 7. 交易信号生成
    print("\nGenerating trading signals...")
    signal_generator = SignalGenerator(variety_order=sequence_builder.variety_order)
    signals = signal_generator.generate_weight_signals(y_pred)
    
    # 8. 仓位管理
    print("\nPosition Management:")
    position_manager = PositionManager(
        variety_order=sequence_builder.variety_order,
        initial_capital=config['initial_capital'],
        max_leverage=config['max_leverage'],
        max_position_percent=config['max_position_percent']
    )
    
    # 简化处理：假设当前价格为100，合约乘数为10
    prices = np.ones(num_varieties) * 100
    contract_multipliers = np.ones(num_varieties) * 10
    
    # 更新仓位
    new_positions, position_changes = position_manager.update_positions(
        signals[-1],  # 使用最后一个预测信号
        prices,
        contract_multipliers
    )
    
    print(f"New Positions: {new_positions}")
    print(f"Position Changes: {position_changes}")
    
    # 9. 风险控制
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
