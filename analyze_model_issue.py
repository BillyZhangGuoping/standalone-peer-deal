import pandas as pd
import numpy as np
from models.random_forest import RandomForestModel
from random_forest_strategy.random_forest_main import preprocess_data
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_model_issue():
    """分析模型在上涨趋势中预测错误的原因"""
    logger.info("开始分析模型问题")
    
    # 1. 加载ag数据
    data_path = 'History_Data/hot_daily_market_data/AG.csv'
    logger.info(f"加载数据：{data_path}")
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # 2. 预处理数据
    logger.info("预处理数据")
    processed_data = preprocess_data(data)
    
    if processed_data.empty:
        logger.error("数据预处理失败")
        return
    
    # 3. 准备训练和测试数据
    # 使用2025年10月前的数据训练，10-12月的数据测试
    train_end_date = '2025-09-30'
    test_start_date = '2025-10-01'
    test_end_date = '2025-12-31'
    
    # 训练数据
    train_data = processed_data[processed_data.index <= train_end_date]
    # 测试数据（上涨趋势期）
    test_data = processed_data[(processed_data.index >= test_start_date) & (processed_data.index <= test_end_date)]
    
    logger.info(f"训练数据：{len(train_data)} 条记录")
    logger.info(f"测试数据：{len(test_data)} 条记录")
    
    # 4. 准备模型数据
    # 提取特征，排除symbol列
    feature_columns = [col for col in processed_data.columns if col != 'symbol']
    X_train = train_data[feature_columns].iloc[-300:]  # 最近300天训练
    
    # 计算真实标签：1表示上涨，-1表示下跌（3日趋势）
    logger.info("计算训练标签")
    train_with_label = train_data.copy()
    train_with_label['label'] = np.sign(train_with_label['close'].shift(-3) - train_with_label['close'])
    train_with_label = train_with_label.dropna(subset=['label'])
    
    if len(train_with_label) < 300:
        logger.warning("训练数据不足，使用全部可用数据")
        X_train = train_with_label[feature_columns]
        y_train = train_with_label['label']
    else:
        X_train = train_with_label[feature_columns].iloc[-300:]
        y_train = train_with_label['label'].iloc[-300:]
    
    logger.info(f"训练数据：{len(X_train)} 条，标签分布：{np.unique(y_train, return_counts=True)}")
    
    # 5. 训练模型
    logger.info("训练随机森林模型")
    model = RandomForestModel()
    model.train(X_train, y_train)
    
    # 6. 获取特征重要性
    logger.info("获取特征重要性")
    feature_importance = model.get_feature_importance()
    
    # 7. 分析上涨趋势期的预测
    logger.info("分析上涨趋势期的预测")
    test_features = test_data[feature_columns]
    
    # 使用模型的实际predict方法，获取正确的趋势强度
    trend_strength = model.predict(test_features)
    
    # 计算实际趋势
    test_with_actual = test_data.copy()
    test_with_actual['actual_trend'] = np.sign(test_with_actual['close'].shift(-3) - test_with_actual['close'])
    test_with_actual['predicted_signal'] = trend_strength
    
    # 8. 输出结果
    logger.info("\n=== 模型问题分析结果 ===")
    
    # 8.1 趋势统计
    print(f"\n2025年10-12月ag价格走势：")
    print(f"起始价：{test_data['close'].iloc[0]:.2f}")
    print(f"结束价：{test_data['close'].iloc[-1]:.2f}")
    print(f"涨幅：{((test_data['close'].iloc[-1] - test_data['close'].iloc[0]) / test_data['close'].iloc[0] * 100):.2f}%")
    
    # 8.2 预测结果统计
    print(f"\n预测结果统计：")
    print(f"平均预测信号：{trend_strength.mean():.4f}")
    print(f"预测信号>0的天数：{len(test_with_actual[test_with_actual['predicted_signal'] > 0])}")
    print(f"预测信号<0的天数：{len(test_with_actual[test_with_actual['predicted_signal'] < 0])}")
    
    # 8.3 特征重要性
    print(f"\n=== 特征重要性排名（前20）===")
    for i, (feature, importance) in enumerate(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20], 1):
        print(f"{i:2d}. {feature:20s}: {importance:.4f}")
    
    # 8.4 检查关键特征在上涨趋势中的表现
    print(f"\n=== 关键特征在上涨趋势中的平均值 ===")
    key_features = ['return_1', 'tr', 'price_ma5_diff', 'volume_ratio_10', 'trend_strength_ema', 'return_60']
    for feature in key_features:
        avg_value = test_data[feature].mean()
        print(f"{feature:20s}: {avg_value:.4f}")
    
    # 8.5 检查模型在强趋势期的表现
    strong_trend_days = test_data[test_data['is_strong_up_trend'] == 1]
    if not strong_trend_days.empty:
        strong_trend_features = strong_trend_days[feature_columns]
        strong_trend_pred = model.model.predict_proba(strong_trend_features)
        strong_trend_strength = strong_trend_pred[:, 1] - strong_trend_pred[:, 0]
        print(f"\n强上涨趋势期（{len(strong_trend_days)}天）的平均预测信号：{strong_trend_strength.mean():.4f}")
    
    # 9. 保存详细结果
    output_file = 'model_issue_analysis.csv'
    test_with_actual.to_csv(output_file, encoding='gbk')
    logger.info(f"详细分析结果已保存到：{output_file}")
    
    logger.info("模型问题分析完成")


if __name__ == "__main__":
    analyze_model_issue()
