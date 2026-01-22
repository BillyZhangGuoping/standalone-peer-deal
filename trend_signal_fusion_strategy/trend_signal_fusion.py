# -*- coding: utf-8 -*-
"""
独立趋势信号融合模块：
输入：每日收盘后64品种复权价格数据
输出：次日各品种趋势方向（-1/0/1）+ 信号强度（0~1）
处理逻辑：
1. 计算统计信号（价格回归斜率t统计量）
2. 加载/模拟ML信号（用户替换为真实ML信号逻辑）
3. 标准化+融合（静态加权/投票法可选）
4. 自动跳过数据不足的品种
"""
import numpy as np
import pandas as pd
from scipy import stats
# import numba as nb  # 可选加速，无需则注释@nb.jit

# ===================== 1. 核心配置（用户可根据回测调优） =====================
# 分板块窗口期配置
WINDOW_CONFIG = {
    "黑色": 10, "能化": 10, "有色": 10, "农产品": 15,
    "贵金属": 20, "股指": 5, "利率": 20, "建材": 10
}
# t统计量显著性阈值（过滤弱趋势）
T_THRESH = 1.645
# 标准化后弱信号阈值（融合后过滤）
FUSION_THRESH = 0.1
# 期货板块划分（64品种）
PLATES = {
    "黑色": ['i', 'rb', 'hc', 'jm', 'j', 'ss', 'ni'],
    "能化": ['sc', 'fu', 'lu', 'pg', 'bu', 'eg', 'TA', 'PF', 'PX', 'l', 'v', 'MA', 'pp', 'eb', 'ru', 'nr', 'br'],
    "农产品": ['SF', 'SM', 'lc', 'OI', 'a', 'b', 'm', 'RM', 'p', 'y', 'CF', 'SR', 'c', 'cs', 'AP', 'PK', 'jd', 'CJ', 'lh'],
    "贵金属": ['ao', 'SH', 'au', 'ag'],
    "有色": ['sn', 'si', 'pb', 'cu', 'al', 'zn'],
    "股指": ['IF', 'IC', 'IH', 'IM', 'ec'],
    "利率": ['T', 'TF'],
    "建材": ['SA', 'FG', 'UR', 'sp']
}
# 融合配置
FUSION_CONFIG = {
    "fusion_method": "weighted",  # 可选：weighted（加权）/ vote（投票）
    "stat_weight": 0.4,  # 统计信号权重（仅weighted生效）
    "ml_weight": 0.6     # ML信号权重（仅weighted生效）
}

# ===================== 2. 工具函数（基础映射/数据检查） =====================
def build_variety_mapping(variety_list):
    """构建品种→板块/窗口期映射"""
    variety2plate = {}
    variety2window = {}
    for plate, vars in PLATES.items():
        window = WINDOW_CONFIG[plate]
        for var in vars:
            if var in variety_list:
                variety2plate[var] = plate
                variety2window[var] = window
    # 未匹配的品种默认配置
    for var in variety_list:
        if var not in variety2plate:
            variety2plate[var] = "其他"
            variety2window[var] = 10
    return variety2plate, variety2window

def check_data_sufficient(price_series, window):
    """检查数据是否足够（至少>=窗口期）"""
    if len(price_series) < window:
        return False
    # 检查是否有过多空值
    if pd.isna(price_series).sum() / len(price_series) > 0.1:
        return False
    return True

# ===================== 3. 统计信号计算（斜率t统计量） =====================
# @nb.jit(nopython=True)
def linear_reg_tstat(y):
    """计算价格序列的斜率和t统计量"""
    n = len(y)
    x = np.arange(n)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # 计算斜率
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    if denominator == 0:
        return 0.0, 0.0
    slope = numerator / denominator
    
    # 计算t统计量
    y_hat = slope * x + (y_mean - slope * x_mean)
    res = y - y_hat
    se = np.sqrt(np.sum(res ** 2) / (n - 2) / denominator) if n > 2 else 0.0
    t_stat = slope / se if se != 0 else 0.0
    return slope, t_stat

def calculate_stat_signal(close_data, variety_list):
    """计算统计信号（趋势方向+强度）"""
    _, variety2window = build_variety_mapping(variety_list)
    stat_dir = []
    stat_strength = []
    
    for var in variety_list:
        window = variety2window[var]
        price_series = close_data[var].dropna().values
        
        # 数据不足则跳过（无趋势）
        if not check_data_sufficient(price_series, window):
            stat_dir.append(0)
            stat_strength.append(0.0)
            continue
        
        # 计算斜率和t统计量
        slope, t_stat = linear_reg_tstat(price_series[-window:])
        
        # 过滤弱趋势
        if abs(t_stat) < T_THRESH:
            stat_dir.append(0)
            stat_strength.append(0.0)
        else:
            stat_dir.append(np.sign(slope).astype(int))
            stat_strength.append(abs(t_stat))
    
    return pd.DataFrame({
        "品种": variety_list,
        "stat_dir": stat_dir,
        "stat_strength": stat_strength
    })

# ===================== 4. ML信号加载（用户替换为真实逻辑） =====================
def load_ml_signal(variety_list):
    """
    加载ML趋势信号（用户需替换为真实ML信号加载逻辑）
    输出格式：品种、ml_dir（-1/0/1）、ml_strength（0~10）
    """
    # 模拟ML信号（用户删除此段，替换为真实加载逻辑）
    np.random.seed(42)
    ml_dir = np.random.choice([-1, 0, 1], size=len(variety_list))
    ml_strength = np.random.uniform(0, 10, size=len(variety_list))
    
    # 真实场景示例：从文件/数据库加载
    # ml_signal_df = pd.read_csv("ml_trend_signal.csv", encoding="utf-8-sig")
    # ml_signal_df = ml_signal_df[ml_signal_df["品种"].isin(variety_list)]
    
    return pd.DataFrame({
        "品种": variety_list,
        "ml_dir": ml_dir,
        "ml_strength": ml_strength
    })

# ===================== 5. 信号标准化+融合 =====================
def standardize_signal(dir_arr, strength_arr):
    """单信号标准化：方向×归一化强度 → [-1,1]"""
    dir_arr = np.array(dir_arr)
    strength_arr = np.array(strength_arr)
    
    # 避免除零
    str_max = strength_arr.max()
    str_min = strength_arr.min()
    if str_max - str_min == 0:
        return np.zeros_like(dir_arr)
    
    # 归一化强度+方向
    norm_str = (strength_arr - str_min) / (str_max - str_min)
    F = dir_arr * norm_str
    return np.clip(F, -1, 1)

def fuse_signals(stat_signal_df, ml_signal_df):
    """信号融合（加权/投票二选一）"""
    # 标准化两个信号
    stat_F = standardize_signal(
        stat_signal_df["stat_dir"].values,
        stat_signal_df["stat_strength"].values
    )
    ml_F = standardize_signal(
        ml_signal_df["ml_dir"].values,
        ml_signal_df["ml_strength"].values
    )
    
    # 融合逻辑
    fusion_method = FUSION_CONFIG["fusion_method"]
    if fusion_method == "weighted":
        # 静态加权融合
        w_stat = FUSION_CONFIG["stat_weight"]
        w_ml = FUSION_CONFIG["ml_weight"]
        fusion_F = w_stat * stat_F + w_ml * ml_F
    elif fusion_method == "vote":
        # 投票法融合（方向一致才保留）
        fusion_F = np.zeros_like(stat_F)
        consistent_idx = (stat_F * ml_F) > 0
        fusion_F[consistent_idx] = (stat_F[consistent_idx] + ml_F[consistent_idx]) / 2
    else:
        raise ValueError(f"不支持的融合方式：{fusion_method}")
    
    # 融合后过滤弱信号
    fusion_F[abs(fusion_F) < FUSION_THRESH] = 0
    fusion_F = np.clip(fusion_F, -1, 1)
    
    # 转换为最终趋势方向+强度
    final_dir = np.sign(fusion_F).astype(int)
    final_strength = abs(fusion_F)  # 强度归一化到0~1
    
    # 组装结果
    result_df = stat_signal_df[["品种"]].copy()
    result_df["趋势方向"] = final_dir
    result_df["信号强度"] = final_strength.round(4)
    result_df["融合方式"] = fusion_method
    result_df["数据状态"] = ["充足" if s > 0 else "不足/无趋势" for s in final_strength]
    
    return result_df

# ===================== 6. 主函数（每日调用入口） =====================
def generate_daily_trend_signal(close_data):
    """
    每日收盘后调用：生成次日64品种趋势信号
    :param close_data: pd.DataFrame → 索引=日期，列=品种，值=复权收盘价（需包含至少最大窗口期数据）
    :return: pd.DataFrame → 品种、趋势方向、信号强度、融合方式、数据状态
    """
    # 1. 基础检查
    if close_data.empty:
        raise ValueError("收盘价数据为空！")
    variety_list = close_data.columns.tolist()
    
    # 2. 计算统计信号
    print("=== 计算统计趋势信号 ===")
    stat_signal_df = calculate_stat_signal(close_data, variety_list)
    
    # 3. 加载ML信号
    print("=== 加载ML趋势信号 ===")
    ml_signal_df = load_ml_signal(variety_list)
    
    # 4. 信号融合
    print("=== 融合趋势信号 ===")
    final_signal_df = fuse_signals(stat_signal_df, ml_signal_df)
    
    # 5. 输出提示
    insufficient_count = (final_signal_df["数据状态"] == "不足/无趋势").sum()
    print(f"=== 信号生成完成：有效信号{len(final_signal_df)-insufficient_count}个，数据不足{insufficient_count}个 ===")
    
    return final_signal_df

# ===================== 7. 测试代码（验证功能） =====================
if __name__ == "__main__":
    # 从History_Data/hot_daily_market_data目录加载正式历史数据
    import os
    
    # 定义数据目录
    DATA_DIR = "../History_Data/hot_daily_market_data"
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    # 初始化一个空的DataFrame用于存储所有品种的收盘价
    close_data = pd.DataFrame()
    
    # 遍历所有CSV文件，提取收盘价数据
    for file in csv_files:
        # 提取品种代码（文件名的前部分，如A.csv -> A）
        variety = file.split('.')[0].upper()
        
        # 读取文件
        file_path = os.path.join(DATA_DIR, file)
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # 提取收盘价，并添加到close_data中
        if 'close' in df.columns:
            # 使用品种名称作为列名，收盘价作为值
            close_series = df['close']
            close_series.name = variety
            
            # 合并到主DataFrame
            if close_data.empty:
                close_data = pd.DataFrame(close_series)
            else:
                close_data = pd.merge(close_data, pd.DataFrame(close_series), left_index=True, right_index=True, how='outer')
    
    # 确保索引是日期类型
    close_data.index = pd.to_datetime(close_data.index)
    
    # 按日期排序
    close_data = close_data.sort_index()
    
    # 过滤出2024年1月1日至2025年1月2日的数据
    start_date = '2024-01-01'
    end_date = '2025-01-02'
    close_data = close_data.loc[start_date:end_date]
    
    # 打印数据基本信息
    print(f"=== 加载的数据信息 ===")
    print(f"数据日期范围: {close_data.index.min()} 至 {close_data.index.max()}")
    print(f"包含品种数量: {len(close_data.columns)}")
    print(f"数据形状: {close_data.shape}")
    
    # 生成2025-1-2日的次日趋势信号（即2025-1-3日的信号）
    trend_signal_df = generate_daily_trend_signal(close_data)
    
    # 打印结果
    print("\n=== 2025-1-3日64品种趋势信号（按品种排序）===")
    print(trend_signal_df.sort_values(by="品种"))
    
    # 保存结果
    trend_signal_df.to_csv("2025-01-03_趋势信号.csv", index=False, encoding="utf-8-sig")
    print("\n=== 结果已保存至：2025-01-03_趋势信号.csv ===")