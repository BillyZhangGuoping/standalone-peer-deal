import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster

# 加载品种信息
def load_instrument_info():
    """加载品种基本信息"""
    df = pd.read_csv('Market_Inform/all_instruments_info.csv')
    return df

# 加载历史数据
def load_history_data():
    """加载所有品种的历史数据"""
    import os
    
    # 定义历史数据目录
    data_dir = 'History_Data/hot_daily_market_data'
    
    # 获取所有CSV文件
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # 加载所有品种的数据
    all_data = {}
    for file in files:
        file_path = os.path.join(data_dir, file)
        base_symbol = file.split('.')[0].upper()
        
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            if 'close' in df.columns:
                all_data[base_symbol] = df['close']
        except Exception as e:
            print(f"加载{file}失败: {e}")
    
    return pd.DataFrame(all_data)

# 计算品种相关性
def calculate_correlation(data):
    """计算品种之间的相关性"""
    # 计算收益率，但不删除NaN值
    returns = data.pct_change(fill_method=None)
    
    # 计算相关性矩阵，自动只考虑有共同数据点的品种对
    correlation_matrix = returns.corr()
    
    # 处理非有限值（NaN, inf, -inf）
    correlation_matrix = correlation_matrix.fillna(0.0)  # 用0填充NaN
    correlation_matrix = correlation_matrix.replace([np.inf, -np.inf], 0.0)  # 用0替换inf和-inf
    
    return correlation_matrix

# 基于相关性进行聚类
def cluster_instruments(correlation_matrix, threshold=0.5):
    """基于相关性矩阵进行层次聚类"""
    from scipy.spatial.distance import squareform
    
    # 将相关性转换为距离（1 - 相关性）
    distance_matrix = 1 - correlation_matrix
    
    # 将方阵转换为压缩距离矩阵
    condensed_distance = squareform(distance_matrix.values)
    
    # 执行层次聚类
    linked = linkage(condensed_distance, method='average')
    
    # 根据阈值确定聚类标签，相关性大于0.5即距离小于0.5
    clusters = fcluster(linked, 1 - threshold, criterion='distance')
    
    # 创建聚类结果字典
    cluster_dict = {}
    for i, symbol in enumerate(correlation_matrix.index):
        cluster_id = f"cluster_{clusters[i]}"
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(symbol)
    
    # 过滤掉只有一个品种的聚类
    filtered_clusters = {}
    for cluster_id, symbols in cluster_dict.items():
        if len(symbols) > 1:
            filtered_clusters[cluster_id] = symbols
    
    return filtered_clusters

# 计算每个聚类的权重
def calculate_cluster_weights(cluster_dict, instrument_info, data):
    """计算每个聚类中品种的权重"""
    cluster_weights = {}
    
    for cluster_id, symbols in cluster_dict.items():
        # 过滤出该聚类中的品种信息
        # 使用symbol列来匹配，因为它包含了简化的合约代码
        cluster_info = instrument_info[instrument_info['symbol'].isin(symbols)]
        
        # 如果没有找到品种信息，使用默认权重
        if cluster_info.empty:
            # 计算每个品种的平均成交量
            avg_volume = data[symbols].count()  # 使用非空数据点数量作为活跃度指标
            total_volume = avg_volume.sum()
            
            # 计算权重
            weights = avg_volume / total_volume
            
            # 创建权重字典
            weight_dict = {}
            for symbol in symbols:
                weight_dict[symbol] = float(weights[symbol])
            
            cluster_weights[cluster_id] = weight_dict
        else:
            # 计算每个品种的权重，这里我们使用持仓量(position)作为权重因子
            cluster_info['weight_factor'] = cluster_info['position']
            total_weight = cluster_info['weight_factor'].sum()
            
            # 计算权重
            cluster_info['weight'] = cluster_info['weight_factor'] / total_weight
            
            # 创建权重字典
            weight_dict = {}
            for _, row in cluster_info.iterrows():
                weight_dict[row['symbol']] = float(row['weight'])
            
            # 对于聚类中但没有信息的品种，设置默认权重
            for symbol in symbols:
                if symbol not in weight_dict:
                    weight_dict[symbol] = 0.0
            
            # 重新归一化权重
            total = sum(weight_dict.values())
            if total > 0:
                for symbol in weight_dict:
                    weight_dict[symbol] /= total
            
            cluster_weights[cluster_id] = weight_dict
    
    return cluster_weights

# 计算聚类之间的相关性
def calculate_cluster_correlation(cluster_dict, correlation_matrix):
    """计算聚类之间的相关性"""
    cluster_correlation = {}
    
    # 获取所有聚类ID
    cluster_ids = list(cluster_dict.keys())
    
    # 计算每对聚类之间的相关性
    for i in range(len(cluster_ids)):
        for j in range(i+1, len(cluster_ids)):
            cluster1 = cluster_ids[i]
            cluster2 = cluster_ids[j]
            
            # 获取两个聚类中的品种
            symbols1 = cluster_dict[cluster1]
            symbols2 = cluster_dict[cluster2]
            
            # 计算两个聚类之间的平均相关性
            avg_corr = correlation_matrix.loc[symbols1, symbols2].mean().mean()
            
            # 保存聚类对的相关性
            cluster_correlation[f"{cluster1}_{cluster2}"] = float(avg_corr)
    
    return cluster_correlation

# 主函数
def main():
    # 1. 加载品种信息
    print("正在加载品种信息...")
    instrument_info = load_instrument_info()
    
    # 2. 加载历史数据
    print("正在加载历史数据...")
    history_data = load_history_data()
    
    # 3. 计算相关性矩阵
    print("正在计算品种相关性...")
    correlation_matrix = calculate_correlation(history_data)
    
    # 4. 基于相关性进行聚类
    print("正在进行品种聚类...")
    clusters = cluster_instruments(correlation_matrix)
    
    # 5. 计算每个聚类中品种的权重
    print("正在计算品种权重...")
    cluster_weights = calculate_cluster_weights(clusters, instrument_info, history_data)
    
    # 6. 计算聚类之间的相关性
    print("正在计算聚类之间的相关性...")
    cluster_correlations = calculate_cluster_correlation(clusters, correlation_matrix)
    
    # 7. 为每个品种生成相关性列表
    print("正在生成每个品种的相关性列表...")
    instrument_correlations = {}
    for symbol in correlation_matrix.index:
        # 获取该品种与其他所有品种的相关性
        corr_series = correlation_matrix[symbol]
        # 转换为字典，排除自己
        corr_dict = {}
        for other_symbol, corr_value in corr_series.items():
            if symbol != other_symbol:
                corr_dict[other_symbol] = float(corr_value)
        # 按相关性从高到低排序
        sorted_corr = dict(sorted(corr_dict.items(), key=lambda x: abs(x[1]), reverse=True))
        instrument_correlations[symbol] = sorted_corr
    
    # 8. 构建最终结果
    result = {
        "instrument_correlations": instrument_correlations,  # 每个品种与其他品种的相关性
        "clusters": clusters,  # 聚类结果，每个类包含的品种
        "cluster_correlations": cluster_correlations,  # 类与类之间的相关性
        "cluster_weights": cluster_weights  # 每个类中品种的权重
    }
    
    # 9. 保存结果到JSON文件
    print("正在保存结果...")
    with open('Market_Inform/correlation_analysis_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print("分析完成！结果已保存到Market_Inform/correlation_analysis_result.json")
    
    # 10. 打印聚类结果
    print("\n聚类结果：")
    for cluster_id, symbols in clusters.items():
        print(f"{cluster_id}: {', '.join(symbols)}")
    
    # 11. 打印聚类之间的相关性
    print("\n聚类之间的相关性：")
    for cluster_pair, corr in cluster_correlations.items():
        print(f"{cluster_pair}: {corr:.4f}")

if __name__ == "__main__":
    main()