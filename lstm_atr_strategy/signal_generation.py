import numpy as np
import pandas as pd


def generate_lstm_signals(predictions, confidence_threshold=0.6):
    """从LSTM预测结果生成初始信号"""
    signals = {}
    
    for symbol, pred in predictions.items():
        if pred is None:
            continue
        
        # 获取预测结果
        direction = pred['direction']
        prob = pred['probability']  # [up_prob, down_prob]
        
        # 计算信号强度：(做多概率 - 做空概率)
        signal_strength = prob[0] - prob[1]
        
        # 计算置信度：max(up_prob, down_prob)
        confidence = max(prob[0], prob[1])
        
        # 应用置信度过滤
        if confidence < confidence_threshold:
            signal = 0
        else:
            # 将信号映射到[-1, 1]区间
            signal = np.clip(signal_strength * 2, -1, 1)
        
        signals[symbol] = {
            'signal': signal,
            'confidence': confidence,
            'direction': direction,
            'probability': prob
        }
    
    return signals


def filter_signals(signals, min_signal_strength=0.2):
    """取消过滤信号，返回所有信号"""
    return signals


def normalize_signals(signals):
    """对信号进行截面标准化（Z-score）"""
    if not signals:
        return signals
    
    # 提取所有信号值
    signal_values = [info['signal'] for info in signals.values()]
    
    # 计算均值和标准差
    mean_signal = np.mean(signal_values)
    std_signal = np.std(signal_values)
    
    # 避免除以零
    if std_signal == 0:
        return signals
    
    # 标准化信号
    normalized_signals = {}
    for symbol, signal_info in signals.items():
        normalized_signal = (signal_info['signal'] - mean_signal) / std_signal
        # 再次限制在[-1, 1]区间
        normalized_signal = np.clip(normalized_signal, -1, 1)
        
        normalized_signals[symbol] = {
            **signal_info,
            'signal': normalized_signal,
            'original_signal': signal_info['signal']
        }
    
    return normalized_signals


def generate_combined_signals(model_manager, all_data, features, date, confidence_threshold=0.6):
    """生成所有品种的组合信号"""
    predictions = {}
    
    # 直接返回空字典，因为我们已经在position_calculator.py中处理了信号生成
    # 这个函数不再需要单独调用，而是在position_calculator.py中直接生成信号
    return predictions
