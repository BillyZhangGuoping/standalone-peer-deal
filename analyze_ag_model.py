#!/usr/bin/env python3
"""
analyze_ag_model.py - 分析AG品种的随机森林模型表现
从2024-01-01开始分析AG品种的模型预测效果
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from random_forest_strategy.random_forest_main import preprocess_data, get_exchange
from models.random_forest import RandomForestModel

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_ag_model(start_date='2024-01-01', test_start_date=None, test_end_date=None):
    """
    分析AG品种的随机森林模型表现
    
    参数：
    start_date: 分析起始日期
    test_start_date: 测试集开始日期（可选）
    test_end_date: 测试集结束日期（可选）
    """
    logger.info(f"开始分析AG品种模型表现，起始日期: {start_date}")
    
    # 1. 加载AG数据
    data_path = 'History_Data/hot_daily_market_data/AG.csv'
    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在: {data_path}")
        return
    
    logger.info(f"加载AG数据: {data_path}")
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # 过滤起始日期之后的数据
    data = data[data.index >= start_date]
    
    if data.empty:
        logger.error(f"起始日期 {start_date} 之后没有数据")
        return
    
    logger.info(f"加载数据成功，共 {len(data)} 条记录")
    
    # 2. 预处理数据
    logger.info("预处理AG数据")
    processed_data = preprocess_data(data.copy())
    
    if processed_data.empty:
        logger.error("数据预处理失败")
        return
    
    logger.info(f"预处理后的数据: {len(processed_data)} 条记录")
    
    # 3. 准备训练和测试数据
    if test_start_date and test_end_date:
        # 使用指定日期范围作为测试集
        logger.info(f"使用指定日期范围作为测试集: {test_start_date} 到 {test_end_date}")
        
        # 转换字符串日期为datetime对象
        test_start_date = pd.to_datetime(test_start_date)
        test_end_date = pd.to_datetime(test_end_date)
        
        # 首先确保测试集日期在数据范围内
        if test_start_date < processed_data.index.min() or test_end_date > processed_data.index.max():
            logger.error(f"测试集日期范围超出数据范围: 数据范围 {processed_data.index.min().strftime('%Y-%m-%d')} 到 {processed_data.index.max().strftime('%Y-%m-%d')}")
            return
        
        # 分离测试集和训练集
        test_data = processed_data[(processed_data.index >= test_start_date) & (processed_data.index <= test_end_date)]
        train_data = processed_data[processed_data.index < test_start_date]
        
        logger.info(f"训练数据: {len(train_data)} 条")
        logger.info(f"测试数据: {len(test_data)} 条")
        
        # 确保训练数据足够
        if len(train_data) < 100:
            logger.error(f"训练数据不足，需要至少100条，实际 {len(train_data)} 条")
            return
    else:
        # 按照300/60的比例划分训练和测试数据
        train_size = 300
        test_size = 60
        
        if len(processed_data) < train_size + test_size:
            logger.warning(f"数据不足，使用全部数据进行分析 (需要 {train_size + test_size} 条，实际 {len(processed_data)} 条)")
            train_end = len(processed_data) - test_size
            if train_end < 100:
                logger.error("数据太少，无法进行有效分析")
                return
        else:
            train_end = train_size
        
        # 训练数据
        train_data = processed_data.iloc[:train_end]
        # 测试数据
        test_data = processed_data.iloc[train_end:train_end + test_size]
        
        logger.info(f"训练数据: {len(train_data)} 条")
        logger.info(f"测试数据: {len(test_data)} 条")
    
    logger.info(f"训练数据: {len(train_data)} 条")
    logger.info(f"测试数据: {len(test_data)} 条")
    
    # 4. 准备模型数据
    # 提取特征，排除symbol列
    feature_columns = [col for col in processed_data.columns if col != 'symbol']
    
    # 计算标签：1表示上涨，-1表示下跌，0表示横盘（使用1日趋势）
    logger.info("计算训练标签")
    train_with_label = train_data.copy()
    train_with_label['label'] = np.sign(train_with_label['close'].shift(-1) - train_with_label['close'])
    train_with_label = train_with_label.dropna(subset=['label'])
    
    test_with_label = test_data.copy()
    test_with_label['label'] = np.sign(test_with_label['close'].shift(-1) - test_with_label['close'])
    test_with_label = test_with_label.dropna(subset=['label'])
    
    logger.info(f"训练数据标签分布: {np.unique(train_with_label['label'], return_counts=True)}")
    logger.info(f"测试数据标签分布: {np.unique(test_with_label['label'], return_counts=True)}")
    
    # 5. 训练随机森林模型
    logger.info("训练随机森林模型")
    model = RandomForestModel()
    model.train(train_with_label[feature_columns], train_with_label['label'])
    
    # 6. 获取特征重要性
    logger.info("获取特征重要性")
    feature_importance = model.get_feature_importance()
    
    # 7. 模型预测
    logger.info("进行模型预测")
    train_pred = model.predict(train_with_label[feature_columns])
    test_pred = model.predict(test_with_label[feature_columns])
    
    # 8. 分析预测结果
    logger.info("分析预测结果")
    
    # 转换趋势强度为分类标签
    train_pred_label = np.sign(train_pred)
    test_pred_label = np.sign(test_pred)
    
    # 计算准确率
    train_accuracy = accuracy_score(train_with_label['label'], train_pred_label)
    test_accuracy = accuracy_score(test_with_label['label'], test_pred_label)
    
    # 9. 生成分析报告
    logger.info("生成分析报告")
    
    # 创建输出目录
    output_dir = 'analysis_reports'
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成报告文件名
    timestamp = datetime.now().strftime('%y%m%d_%H%M')
    report_file = os.path.join(output_dir, f'ag_model_analysis_{start_date}_{timestamp}.md')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# AG品种随机森林模型分析报告\n")
        f.write(f"\n## 报告基本信息\n")
        f.write(f"- 分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 分析起始日期: {start_date}\n")
        f.write(f"- 数据来源: {data_path}\n")
        f.write(f"- 模型类型: 随机森林\n")
        f.write(f"- 训练数据量: {len(train_with_label)} 条\n")
        f.write(f"- 测试数据量: {len(test_with_label)} 条\n")
        
        f.write(f"\n## 模型训练效果\n")
        f.write(f"### 准确率\n")
        f.write(f"- 训练集准确率: {train_accuracy:.4f}\n")
        f.write(f"- 测试集准确率: {test_accuracy:.4f}\n")
        
        f.write(f"\n### 分类报告\n")
        f.write(f"#### 训练集\n")
        f.write(f"```\n")
        f.write(classification_report(train_with_label['label'], train_pred_label))
        f.write(f"```\n")
        
        f.write(f"\n#### 测试集\n")
        f.write(f"```\n")
        f.write(classification_report(test_with_label['label'], test_pred_label))
        f.write(f"```\n")
        
        f.write(f"\n### 混淆矩阵\n")
        f.write(f"#### 训练集\n")
        f.write(f"```\n")
        cm_train = confusion_matrix(train_with_label['label'], train_pred_label)
        f.write(str(cm_train))
        f.write(f"```\n")
        
        f.write(f"\n#### 测试集\n")
        f.write(f"```\n")
        cm_test = confusion_matrix(test_with_label['label'], test_pred_label)
        f.write(str(cm_test))
        f.write(f"```\n")
        
        f.write(f"\n## 特征重要性\n")
        f.write(f"| 排名 | 特征名称 | 重要性 |\n")
        f.write(f"|------|----------|--------|\n")
        for i, (feature, importance) in enumerate(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20], 1):
            f.write(f"| {i:2d} | {feature:25s} | {importance:.6f} |\n")
        
        f.write(f"\n## 趋势强度分析\n")
        f.write(f"### 训练集趋势强度分布\n")
        train_strength_stats = pd.Series(train_pred).describe()
        f.write(f"```\n")
        f.write(str(train_strength_stats))
        f.write(f"```\n")
        
        f.write(f"\n### 测试集趋势强度分布\n")
        test_strength_stats = pd.Series(test_pred).describe()
        f.write(f"```\n")
        f.write(str(test_strength_stats))
        f.write(f"```\n")
        
        # 10. 保存详细预测结果
        logger.info("保存详细预测结果")
        
        # 训练集详细结果
        train_results = train_with_label.copy()
        train_results['predicted_strength'] = train_pred
        train_results['predicted_label'] = train_pred_label
        train_results['is_correct'] = (train_results['label'] == train_results['predicted_label']).astype(int)
        
        # 测试集详细结果
        test_results = test_with_label.copy()
        test_results['predicted_strength'] = test_pred
        test_results['predicted_label'] = test_pred_label
        test_results['is_correct'] = (test_results['label'] == test_results['predicted_label']).astype(int)
        
        # 保存结果到CSV
        train_results_file = os.path.join(output_dir, f'ag_train_results_{start_date}_{timestamp}.csv')
        test_results_file = os.path.join(output_dir, f'ag_test_results_{start_date}_{timestamp}.csv')
        
        train_results.to_csv(train_results_file, encoding='gbk')
        test_results.to_csv(test_results_file, encoding='gbk')
        
        logger.info(f"训练集详细结果已保存到: {train_results_file}")
        logger.info(f"测试集详细结果已保存到: {test_results_file}")
        
        f.write(f"\n## 详细结果文件\n")
        f.write(f"- [训练集详细结果]({train_results_file})\n")
        f.write(f"- [测试集详细结果]({test_results_file})\n")
        
        # 11. 添加趋势强度可视化建议
        f.write(f"\n## 可视化建议\n")
        f.write(f"1. **趋势强度时间序列图**: 绘制实际趋势与预测趋势强度的对比图\n")
        f.write(f"2. **特征重要性柱状图**: 展示前20个重要特征\n")
        f.write(f"3. **趋势强度分布直方图**: 分析预测强度的分布情况\n")
        f.write(f"4. **混淆矩阵热力图**: 直观展示模型预测的混淆情况\n")
        f.write(f"5. **价格与趋势强度散点图**: 分析价格与预测强度的关系\n")
    
    logger.info(f"AG品种模型分析报告已生成: {report_file}")
    logger.info("AG品种模型分析完成")

def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='分析AG品种的随机森林模型表现')
    parser.add_argument('--start-date', default='2024-01-01', help='分析起始日期，默认为2024-01-01')
    parser.add_argument('--test-start', help='测试集开始日期，例如2025-10-01')
    parser.add_argument('--test-end', help='测试集结束日期，例如2025-12-29')
    
    args = parser.parse_args()
    
    # 调用分析函数
    analyze_ag_model(
        start_date=args.start_date,
        test_start_date=args.test_start,
        test_end_date=args.test_end
    )

if __name__ == "__main__":
    main()
