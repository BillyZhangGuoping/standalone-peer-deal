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
    """过滤信号，只保留强度足够的信号"""
    filtered_signals = {}
    
    for symbol, signal_info in signals.items():
        signal = signal_info['signal']
        if abs(signal) >= min_signal_strength:
            filtered_signals[symbol] = signal_info
    
    return filtered_signals


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
    
    # 第一步：获取每个品种的预测结果
    for base_symbol, data in all_data.items():
        # 检查该品种在该日期是否有数据
        if date not in data.index:
            continue
        
        # 获取该品种在该日期之前的数据
        past_data = data[data.index <= date]
        
        # 确保有足够数据
        if len(past_data) < 360:
            continue
        
        # 预测
        pred = model_manager.predict(base_symbol, past_data, features)
        if pred is not None:
            predictions[base_symbol] = pred
    
    # 第二步：生成初始信号
    initial_signals = generate_lstm_signals(predictions, confidence_threshold)
    
    # 第三步：过滤信号
    filtered_signals = filter_signals(initial_signals)
    
    # 第四步：截面标准化
    normalized_signals = normalize_signals(filtered_signals)
    
    return normalized_signals
