import numpy as np
import pandas as pd
from scipy import stats
# import numba as nb  # 可选，加速计算，无需可注释

# ===================== 趋势信号配置（期货64品种适配） =====================
# 分板块窗口期配置（可根据回测调优）
WINDOW_CONFIG = {
    "黑色": 10, "能化": 10, "有色": 10, "农产品": 15,
    "贵金属": 20, "股指": 5, "利率": 20, "建材": 10
}
# t统计量显著性阈值（90%置信度，过滤弱趋势，可调至1.96（95%））
T_THRESH = 1.645
# 期货板块划分（复用你原有框架的板块）
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
# 建立品种→窗口期的映射
def build_variety2window(variety_list):
    variety2window = {}
    for plate, vars in PLATES.items():
        window = WINDOW_CONFIG[plate]
        for var in vars:
            if var in variety_list:
                variety2window[var] = window
    # 未匹配的品种用默认窗口10
    for var in variety_list:
        if var not in variety2window:
            variety2window[var] = 10
    return variety2window

# ===================== 核心计算函数（斜率t统计量） =====================
# @nb.jit(nopython=True)  # Numba加速，无需则删除此行
def linear_reg_tstat(y):
    """
    单序列线性回归，计算斜率的t统计量
    :param y: 价格序列（一维numpy数组，长度=窗口期）
    :return: slope（斜率）, t_stat（斜率的t统计量）
    """
    n = len(y)
    x = np.arange(n)  # 自变量：时间序列[0,1,2,...,n-1]
    # 计算均值
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    # 计算斜率β和截距α
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    if denominator == 0:
        return 0.0, 0.0
    slope = numerator / denominator
    # 计算残差
    y_hat = slope * x + (y_mean - slope * x_mean)
    res = y - y_hat
    # 计算斜率的标准误
    se = np.sqrt(np.sum(res ** 2) / (n - 2) / denominator)
    if se == 0:
        return slope, 0.0
    # 计算t统计量
    t_stat = slope / se
    return slope, t_stat

def cal_trend_signal(close_data, variety2window, t_thresh=T_THRESH):
    """
    批量计算64个品种的趋势信号（trend_dir, trend_strength）
    :param close_data: 收盘价DataFrame（index=日期，columns=品种，复权后）
    :return: trend_signal_df（品种, trend_dir, trend_strength）
    """
    trend_dir = []
    trend_strength = []
    variety_list = close_data.columns.tolist()
    for var in variety_list:
        window = variety2window[var]
        price_series = close_data[var].values[-window:]  # 取最新窗口期价格
        if len(price_series) < window:
            # 数据不足，无趋势
            trend_dir.append(0)
            trend_strength.append(0.0)
            continue
        # 计算斜率和t统计量
        slope, t_stat = linear_reg_tstat(price_series)
        # 过滤弱趋势
        if abs(t_stat) < t_thresh:
            trend_dir.append(0)
            trend_strength.append(0.0)
        else:
            trend_dir.append(np.sign(slope).astype(int))
            trend_strength.append(abs(t_stat))
    # 构建结果
    trend_signal_df = pd.DataFrame({
        "品种": variety_list,
        "trend_dir": trend_dir,
        "trend_strength": trend_strength
    })
    return trend_signal_df

# ===================== 回测/实盘每日调用逻辑（对接原有框架） =====================
def daily_trend_signal(close_data_daily, variety_list, variety2window=None):
    """
    每日增量计算趋势信号
    :param close_data_daily: 截至当日的复权收盘价DataFrame（需包含至少最大窗口期数据）
    :return: 当日趋势信号df
    """
    if variety2window is None:
        variety2window = build_variety2window(variety_list)
    # 计算当日趋势信号
    signal_df = cal_trend_signal(close_data_daily, variety2window)
    return signal_df

# ===================== 测试代码（验证效果） =====================
if __name__ == "__main__":
    # 模拟64品种复权收盘价数据（替换为你的真实数据）
    np.random.seed(42)
    variety_list = list(build_variety2window({}).keys())
    date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    close_data = pd.DataFrame(
        np.random.randn(len(date_range), len(variety_list)).cumsum(axis=0) + 100,
        columns=variety_list,
        index=date_range
    )
    # 构建品种→窗口期映射
    variety2window = build_variety2window(variety_list)
    # 取最后一日数据计算趋势信号
    signal_df = daily_trend_signal(close_data, variety_list, variety2window)
    # 打印结果
    print("=== 价格回归斜率t统计量趋势信号 ===")
    print(signal_df[signal_df["trend_dir"] != 0].head(10))  # 打印有趋势的品种