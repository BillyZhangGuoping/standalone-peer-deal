# ===================== 1. 导入核心库 =====================
import time
import os
import sys
import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import variation
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# 将项目根目录添加到Python路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 导入获取合约乘数和保证金率的函数
from utility.instrument_utils import get_contract_multiplier

# ===================== 2. 全局配置（根据你的实盘调整） =====================
# 资金与风控
PRINCIPAL = 3000000  # 总本金（元）
MARGIN_RATIO_LIMIT = 0.70  # 保证金上限占比（70%）
TARGET_MARGIN_RATIO = 0.65  # 目标保证金占比（65%，留5%缓冲）
WEIGHT_LOWER = -0.8  # 单品种权重下限（空仓最大权重）- 放宽限制
WEIGHT_UPPER = 0.8  # 单品种权重上限（多仓最大权重）- 放宽限制
COV_DAYS = 60  # 协方差计算周期（60日）
# 趋势信号参数
ALPHA = 0.5  # 趋势增强系数（0.3~0.7可调）
STRENGTH_THRESH = 2.0  # 趋势强度过滤阈值
# Nelder-Mead优化参数（调整）
MAX_ITER = 30000  # 进一步提高迭代上限
MAX_FEV = 60000   # 进一步提高函数调用上限
FATOL = 1e-4      # 适当放宽收敛阈值，提高收敛概率
XATOL = 1e-4
# 惩罚项系数（当前简化目标函数未使用，但保留配置）
LAMBDA1 = 1e5   # 降低权重和惩罚系数
LAMBDA2 = 1e5   # 降低上下限惩罚系数
LAMBDA3 = 1e5   # 大幅降低保证金惩罚系数（核心调整）
LAMBDA4 = 1e4   # 降低板块分散惩罚系数

# ===================== 3. 核心工具函数（风险计算/信号处理/评估） =====================
def marginal_risk_contribution(weights, cov_matrix, asset_margins):
    """计算边际风险贡献（MRC）"""
    weights = np.array(weights)
    cov_matrix = np.array(cov_matrix)
    # 组合波动率（考虑保证金杠杆）
    vol = np.sqrt(weights.T @ cov_matrix @ weights)
    # 边际风险贡献
    mrc = (cov_matrix @ weights) / vol
    # 结合保证金杠杆调整
    mrc = mrc * asset_margins
    return mrc

def risk_contribution(weights, cov_matrix, asset_margins):
    """计算各品种风险贡献（RC）"""
    weights = np.array(weights)
    mrc = marginal_risk_contribution(weights, cov_matrix, asset_margins)
    rc = weights * mrc
    return rc

def calculate_plate_risks(weights, cov_matrix, asset_margins, variety_list):
    """计算板块风险贡献与分散度"""
    # 期货板块划分（适配64个主流品种）
    plates = {
        "黑色": ['i', 'rb', 'hc', 'jm', 'j', 'ss', 'ni'],
        "能化": ['sc', 'fu', 'lu', 'pg', 'bu', 'eg', 'TA', 'PF', 'PX', 'l', 'v', 'MA', 'pp', 'eb', 'ru', 'nr', 'br'],
        "农产品": ['SF', 'SM', 'lc', 'OI', 'a', 'b', 'm', 'RM', 'p', 'y', 'CF', 'SR', 'c', 'cs', 'AP', 'PK', 'jd', 'CJ', 'lh'],
        "贵金属": ['ao', 'SH', 'au', 'ag'],
        "有色": ['sn', 'si', 'pb', 'cu', 'al', 'zn'],
        "股指": ['IF', 'IC', 'IH', 'IM', 'ec'],
        "利率": ['T', 'TF'],
        "建材": ['SA', 'FG', 'UR', 'sp']
    }
    # 将cov_matrix转换为NumPy数组，以便支持多维索引
    cov_matrix_np = np.array(cov_matrix)
    
    # 筛选有效品种
    plate_rc = []
    for plate, vars in plates.items():
        plate_idx = [variety_list.index(v) for v in vars if v in variety_list]
        if len(plate_idx) == 0:
            continue
        plate_weights = weights[plate_idx]
        # 使用NumPy数组进行索引
        plate_cov = cov_matrix_np[plate_idx, :][:, plate_idx]
        plate_margins = asset_margins[plate_idx]
        # 板块风险贡献
        plate_mrc = marginal_risk_contribution(plate_weights, plate_cov, plate_margins)
        plate_rc.append(np.sum(plate_weights * plate_mrc))
    # 板块风险分散度（变异系数CV）
    plate_cv = variation(plate_rc) if len(plate_rc) > 1 else 0
    return plate_rc, plate_cv

def standardize_signal(trend_dir, trend_strength, strength_thresh=2.0):
    """标准化趋势信号→趋势因子F（-1~1）"""
    trend_dir = np.array(trend_dir)
    trend_strength = np.array(trend_strength)
    # 过滤弱趋势
    trend_strength[trend_strength < strength_thresh] = 0
    trend_dir[trend_strength < strength_thresh] = 0
    # 最大最小归一化
    s_max = trend_strength.max()
    s_min = trend_strength.min()
    if s_max - s_min == 0:
        return np.zeros_like(trend_dir)
    norm_strength = (trend_strength - s_min) / (s_max - s_min)
    # 生成趋势因子
    F = trend_dir * norm_strength
    return np.clip(F, -1.0, 1.0)

def calculate_target_lots(W_target, variety_list, asset_margins, asset_multipliers, close_data, principal):
    """资金权重→期货目标手数（整数）"""
    target_lots = np.zeros(len(variety_list), dtype=int)
    for i, var in enumerate(variety_list):
        w = W_target[i]
        if w == 0:
            continue
        # 品种最新价
        p = close_data[var].iloc[-1]
        # 合约乘数+保证金比例
        m = asset_multipliers[i]
        r = asset_margins[i]
        # 目标手数（向下取整，避免保证金不足）
        lot = (w * principal) / (p * m * r)
        target_lots[i] = np.floor(lot)
        # 过滤小于1手的仓位
        if abs(target_lots[i]) < 1:
            target_lots[i] = 0
    return target_lots

def evaluate_algorithm(result, weights, cov_matrix, asset_margins, variety_list, principal, run_time):
    """完整算法评估（收敛/风控/风险/效率）"""
    # 1. 收敛性
    converge_metrics = {
        "收敛状态": result.success,
        "迭代次数": result.nit,
        "目标函数值": result.fun.round(8),
        "函数调用次数": result.nfev
    }
    # 2. 约束满足度
    sum_weight = np.sum(weights)
    # 单品种超限
    upper_violate = np.sum(weights > WEIGHT_UPPER)
    lower_violate = np.sum(weights < WEIGHT_LOWER)
    violate_rate = (upper_violate + lower_violate) / len(weights)
    # 保证金占比
    nominal_capital = weights * principal
    margin_usage = np.abs(nominal_capital) * asset_margins
    total_margin = np.sum(margin_usage)
    margin_ratio = total_margin / principal
    margin_deviation = abs(margin_ratio - TARGET_MARGIN_RATIO)
    constraint_metrics = {
        "权重和偏差": abs(sum_weight - 1).round(6),
        "单品种权重超限率(%)": (violate_rate * 100).round(2),
        "保证金占比(%)": (margin_ratio * 100).round(2),
        "保证金偏离目标(%)": (margin_deviation * 100).round(2)
    }
    # 3. 风险分配
    rc = risk_contribution(weights, cov_matrix, asset_margins)
    rc_abs = np.abs(rc)
    risk_metrics = {
        "风险贡献均值": np.mean(rc_abs).round(8),
        "风险贡献标准差": np.std(rc_abs).round(8),
        "风险贡献CV": variation(rc_abs).round(4),
        "最大/最小风险贡献比": (np.max(rc_abs)/np.min(rc_abs) if np.min(rc_abs)>0 else np.inf).round(2)
    }
    # 板块风险
    _, plate_cv = calculate_plate_risks(weights, cov_matrix, asset_margins, variety_list)
    plate_metrics = {"板块风险分散度CV": plate_cv.round(4)}
    # 4. 效率
    process = psutil.Process()
    peak_mem = process.memory_info().rss / 1024 / 1024 / 1024
    efficiency_metrics = {
        "单次运行耗时(秒)": round(run_time, 2),
        "峰值内存(GB)": round(peak_mem, 2)
    }
    # 合并
    all_metrics = {**converge_metrics, **constraint_metrics, **risk_metrics, **plate_metrics, **efficiency_metrics}
    return all_metrics

# ===================== 4. 简化目标函数（仅风险贡献方差+轻量级约束） =====================
def risk_parity_objective_with_penalty(weights, cov_matrix, asset_margins, variety_list):
    """简化目标函数：优化风险贡献方差+轻量级约束，帮助算法收敛"""
    weights = np.array(weights)
    cov_matrix = np.array(cov_matrix)
    asset_margins = np.array(asset_margins)
    
    # 核心目标：风险贡献方差最小
    rc = risk_contribution(weights, cov_matrix, asset_margins)
    rc_abs = np.abs(rc)
    core_objective = np.var(rc_abs)  # 只优化风险贡献方差
    
    # 添加轻量级约束惩罚，帮助算法收敛
    # 1. 权重和约束（轻微惩罚）
    sum_weight_penalty = 1e-3 * (np.sum(weights) - 1)**2
    
    # 2. 单品种权重边界约束（轻微惩罚）
    boundary_penalty = 1e-3 * (np.sum(np.maximum(0, weights - WEIGHT_UPPER)**2) + 
                              np.sum(np.maximum(0, WEIGHT_LOWER - weights)**2))
    
    # 总目标函数
    total_objective = core_objective + sum_weight_penalty + boundary_penalty
    
    return total_objective

# ===================== 5. HRP兜底（强化：保证金约束） =====================
def hrp_portfolio(returns, variety_list, asset_margins):
    """HRP兜底+保证金约束修正"""
    # 相关性→距离矩阵
    corr = returns.corr()
    dist = np.sqrt(0.5 * (1 - corr))
    # 层次聚类
    link = linkage(pdist(dist), method='ward')
    clusters = fcluster(link, t=8, criterion='maxclust')
    # 分层权重
    cluster_unique = np.unique(clusters)
    cluster_weights = np.ones(len(cluster_unique)) / len(cluster_unique)
    hrp_weights = np.zeros(len(variety_list))
    for i, cluster in enumerate(cluster_unique):
        idx = np.where(clusters == cluster)[0]
        hrp_weights[idx] = cluster_weights[i] / len(idx)
    # 保证金约束修正（调整至目标占比）
    nominal_capital = hrp_weights * PRINCIPAL
    margin_usage = np.abs(nominal_capital) * asset_margins
    total_margin = np.sum(margin_usage)
    target_margin = PRINCIPAL * TARGET_MARGIN_RATIO
    shrink_ratio = target_margin / total_margin if total_margin > 0 else 1.0
    hrp_weights = hrp_weights * shrink_ratio
    # 归一化+硬约束
    hrp_weights = hrp_weights / np.sum(hrp_weights)
    hrp_weights = np.clip(hrp_weights, WEIGHT_LOWER, WEIGHT_UPPER)
    return hrp_weights

# ===================== 6. 主优化函数（更新：初始权重+参数+协方差处理+多算法尝试） =====================
def optimize_portfolio(cov_matrix, asset_margins, variety_list, returns):
    """主优化流程：优化初始权重+调整参数+协方差矩阵处理+多算法尝试"""
    # 1. 协方差矩阵稳定性处理（添加小的对角线扰动，避免奇异矩阵）
    cov_matrix_np = np.array(cov_matrix)
    # 添加小的对角线扰动
    cov_matrix_stable = cov_matrix_np + np.eye(len(cov_matrix_np)) * 1e-6
    
    # 2. 初始权重：改进的基于波动率的权重分配
    n_assets = len(variety_list)
    # 计算各品种的年化波动率
    returns_recent = returns.tail(COV_DAYS)  # 使用最近COV_DAYS天的收益
    volatility = returns_recent.std() * np.sqrt(252)  # 年化波动率
    # 波动率越低，权重越高
    inverse_volatility = 1 / volatility
    init_weights = inverse_volatility.values  # 转换为numpy数组
    # 归一化
    init_weights = init_weights / np.sum(init_weights)
    
    # 3. 目标函数封装（简化版，仅保留核心目标）
    def simplified_obj_fun(weights):
        """简化目标函数：仅优化风险贡献方差，移除所有约束惩罚"""
        weights = np.array(weights)
        rc = risk_contribution(weights, cov_matrix_stable, asset_margins)
        rc_abs = np.abs(rc)
        return np.var(rc_abs)  # 只优化风险贡献方差
    
    # 4. 多算法优化尝试
    start_time = time.time()
    # 为L-BFGS-B算法设置边界约束
    bounds = [(WEIGHT_LOWER, WEIGHT_UPPER) for _ in range(n_assets)]
    
    algorithms = [
        {
            'name': 'Nelder-Mead',
            'method': 'Nelder-Mead',
            'options': {
                'disp': False,
                'maxiter': MAX_ITER,
                'maxfev': MAX_FEV,
                'fatol': 1e-4,
                'xatol': 1e-4,
                'adaptive': True,
                'initial_simplex': None  # 使用默认初始单纯形
            }
        },
        {
            'name': 'L-BFGS-B',
            'method': 'L-BFGS-B',
            'options': {
                'disp': False,
                'maxiter': MAX_ITER,
                'ftol': 1e-4,
                'gtol': 1e-4
            },
            'bounds': bounds
        }
    ]
    
    # 尝试不同算法
    result = None
    for algo in algorithms:
        print(f"尝试算法: {algo['name']}")
        # 准备参数
        minimize_kwargs = {
            'fun': simplified_obj_fun,
            'x0': init_weights,
            'method': algo['method'],
            'options': algo['options']
        }
        # 如果算法有bounds参数，添加它
        if 'bounds' in algo:
            minimize_kwargs['bounds'] = algo['bounds']
        
        result = minimize(**minimize_kwargs)
        if result.success:
            print(f"=== {algo['name']}优化成功 ===")
            break
    
    run_time = time.time() - start_time
    
    # 5. 权重修正/兜底（HRP传入asset_margins）
    if result and result.success:
        optimal_weights = result.x / np.sum(result.x)
        # 最终保证金修正
        nominal_capital = optimal_weights * PRINCIPAL
        margin_usage = np.abs(nominal_capital) * asset_margins
        total_margin = np.sum(margin_usage)
        if total_margin > PRINCIPAL * MARGIN_RATIO_LIMIT:
            optimal_weights = optimal_weights * (PRINCIPAL * MARGIN_RATIO_LIMIT / total_margin)
            optimal_weights = optimal_weights / np.sum(optimal_weights)
        # 硬约束修正
        optimal_weights = np.clip(optimal_weights, WEIGHT_LOWER, WEIGHT_UPPER)
        optimal_weights = optimal_weights / np.sum(optimal_weights)
    else:
        print("=== 所有优化算法失败，启用HRP兜底 ===")
        optimal_weights = hrp_portfolio(returns, variety_list, asset_margins)  # 传入asset_margins
    
    # 6. 直接调整保证金占比至目标水平
    # 计算当前保证金占比
    nominal_capital = optimal_weights * PRINCIPAL
    margin_usage = np.abs(nominal_capital) * asset_margins
    total_margin = np.sum(margin_usage)
    current_margin_ratio = total_margin / PRINCIPAL
    print(f"调整前保证金占比: {current_margin_ratio:.2%}")
    
    # 计算目标保证金金额
    target_margin = PRINCIPAL * TARGET_MARGIN_RATIO
    
    # 计算调整系数
    if total_margin > 0:
        adjust_factor = target_margin / total_margin
    else:
        adjust_factor = 1.0
    
    print(f"调整系数: {adjust_factor:.2f}")
    
    # 直接调整权重
    adjusted_weights = optimal_weights * adjust_factor
    
    # 计算调整后的保证金占比
    nominal_capital = adjusted_weights * PRINCIPAL
    margin_usage = np.abs(nominal_capital) * asset_margins
    total_margin = np.sum(margin_usage)
    new_margin_ratio = total_margin / PRINCIPAL
    print(f"调整后保证金占比: {new_margin_ratio:.2%}")
    
    # 检查并修正超出上下限的权重
    # 找出超出范围的权重
    upper_exceed = adjusted_weights > WEIGHT_UPPER
    lower_exceed = adjusted_weights < WEIGHT_LOWER
    
    # 计算超出的比例
    exceed_ratios = np.ones_like(adjusted_weights)
    exceed_ratios[upper_exceed] = WEIGHT_UPPER / adjusted_weights[upper_exceed]
    exceed_ratios[lower_exceed] = WEIGHT_LOWER / adjusted_weights[lower_exceed]
    
    # 如果有超出范围的权重，计算整体缩放比例
    if np.any(upper_exceed) or np.any(lower_exceed):
        scale_factor = np.min(exceed_ratios) * 0.95  # 保留5%的安全边际
        adjusted_weights = adjusted_weights * scale_factor
        print(f"权重超出范围，应用缩放因子: {scale_factor:.2f}")
        
        # 重新计算保证金占比
        nominal_capital = adjusted_weights * PRINCIPAL
        margin_usage = np.abs(nominal_capital) * asset_margins
        total_margin = np.sum(margin_usage)
        new_margin_ratio = total_margin / PRINCIPAL
        print(f"缩放后保证金占比: {new_margin_ratio:.2%}")
    
    # 使用调整后的权重作为最优权重
    optimal_weights = adjusted_weights
    
    # 7. 最终硬约束修正
    optimal_weights = np.clip(optimal_weights, WEIGHT_LOWER, WEIGHT_UPPER)
    print(f"最终权重和: {np.sum(optimal_weights):.2f}")
    
    # 8. 评估
    metrics = evaluate_algorithm(result, optimal_weights, cov_matrix_stable, asset_margins, variety_list, PRINCIPAL, run_time)
    return optimal_weights, metrics

# ===================== 7. 趋势信号融合+持仓生成 =====================
def generate_target_positions(optimal_weights, variety_list, asset_margins, asset_multipliers, close_data, trend_signal_df):
    """融合趋势信号，生成次日目标持仓"""
    # 1. 标准化趋势信号
    F = standardize_signal(
        trend_dir=trend_signal_df["trend_dir"].values,
        trend_strength=trend_signal_df["trend_strength"].values,
        strength_thresh=STRENGTH_THRESH
    )
    # 2. 趋势融合权重
    W_fusion = optimal_weights * (1 + ALPHA * F)
    W_fusion = W_fusion / np.sum(W_fusion)  # 归一化
    # 3. 硬约束修正
    W_fusion = np.clip(W_fusion, WEIGHT_LOWER, WEIGHT_UPPER)
    # 保证金约束（超限则缩容）
    nominal_capital = W_fusion * PRINCIPAL
    margin_usage = np.abs(nominal_capital) * asset_margins
    total_margin = np.sum(margin_usage)
    if total_margin > PRINCIPAL * MARGIN_RATIO_LIMIT:
        shrink_ratio = (PRINCIPAL * MARGIN_RATIO_LIMIT) / total_margin
        W_fusion = W_fusion * shrink_ratio
        W_fusion = W_fusion / np.sum(W_fusion)
    # 4. 计算目标手数
    target_lots = calculate_target_lots(
        W_target=W_fusion,
        variety_list=variety_list,
        asset_margins=asset_margins,
        asset_multipliers=asset_multipliers,
        close_data=close_data,
        principal=PRINCIPAL
    )
    # 5. 生成持仓指令
    position_df = pd.DataFrame({
        "品种": variety_list,
        "风险平价权重": optimal_weights.round(6),
        "趋势因子F": F.round(4),
        "融合权重": W_fusion.round(6),
        "目标手数": target_lots
    })
    # 模拟当前持仓（替换为你的实盘持仓）
    position_df["当前持仓"] = 0
    position_df["仓差"] = position_df["目标手数"] - position_df["当前持仓"]
    # 交易指令
    def gen_order(row):
        if row["仓差"] > 0:
            return f"开多{row['仓差']}手"
        elif row["仓差"] < 0:
            return f"开空{abs(row['仓差'])}手"
        else:
            return "持仓不变"
    position_df["交易指令"] = position_df.apply(gen_order, axis=1)
    return position_df

# ===================== 8. 真实数据加载 =====================
def load_real_data():
    """加载真实数据，从History_Data/hot_daily_market_data目录读取"""
    # 64个期货品种列表（主流品种）
    variety_list = [
        # 黑色
        'i', 'rb', 'hc', 'jm', 'j', 'ss', 'ni',
        # 能化
        'sc', 'fu', 'lu', 'pg', 'bu', 'eg', 'TA', 'PF', 'PX', 'l', 'v', 'MA', 'pp', 'eb', 'ru', 'nr', 'br',
        # 农产品
        'SF', 'SM', 'lc', 'OI', 'a', 'b', 'm', 'RM', 'p', 'y', 'CF', 'SR', 'c', 'cs', 'AP', 'PK', 'jd', 'CJ', 'lh',
        # 贵金属
        'ao', 'SH', 'au', 'ag',
        # 有色
        'sn', 'si', 'pb', 'cu', 'al', 'zn',
        # 股指
        'IF', 'IC', 'IH', 'IM', 'ec',
        # 利率
        'T', 'TF',
        # 建材
        'SA', 'FG', 'UR', 'sp'
    ]
    
    # 定义数据目录
    DATA_DIR = os.path.join(parent_dir, "History_Data/hot_daily_market_data")
    
    # 读取真实收盘价数据
    close_data = pd.DataFrame()
    
    for variety in variety_list:
        # 构造文件路径，文件名是品种名的大写.csv
        file_name = f"{variety.upper()}.csv"
        file_path = os.path.join(DATA_DIR, file_name)
        
        try:
            # 读取CSV文件，将第一列作为日期索引
            df = pd.read_csv(file_path, index_col=0)
            # 将索引转换为日期类型
            df.index = pd.to_datetime(df.index)
            # 提取收盘价列，并重命名为品种名
            close_series = df['close'].rename(variety)
            # 将该品种的收盘价数据合并到close_data中
            close_data = pd.concat([close_data, close_series], axis=1)
        except Exception as e:
            print(f"读取品种 {variety} 时出错: {e}")
    
    # 删除含有缺失值的行
    close_data = close_data.dropna()
    
    # 计算对数收益率
    returns = np.log(close_data / close_data.shift(1)).dropna()
    
    # 计算协方差矩阵（使用最近COV_DAYS天的数据）
    recent_returns = returns.tail(COV_DAYS)
    cov_matrix = recent_returns.cov() * 252  # 年化
    
    # 获取合约乘数和保证金率
    asset_margins = []
    asset_multipliers = []
    
    for sec in variety_list:
        # 使用get_contract_multiplier函数获取合约乘数和保证金率
        multiplier, margin_rate = get_contract_multiplier(sec)
        asset_multipliers.append(multiplier)
        asset_margins.append(margin_rate)
    
    asset_margins = np.array(asset_margins)
    asset_multipliers = np.array(asset_multipliers)
    
    # 生成模拟机器学习趋势信号（方向：-1/0/1，强度：0-10）
    # 注意：这里应该替换为真实的机器学习信号
    trend_dir = np.random.choice([-1, 0, 1], size=len(variety_list))
    trend_strength = np.random.uniform(0, 10, size=len(variety_list))
    trend_signal_df = pd.DataFrame({
        "品种": variety_list,
        "trend_dir": trend_dir,
        "trend_strength": trend_strength
    })
    
    return variety_list, close_data, returns, cov_matrix, asset_margins, asset_multipliers, trend_signal_df

# 添加命令行参数处理
import argparse

# ===================== 8. 真实数据加载 =====================
def load_real_data(target_date=None):
    """加载真实数据，从History_Data/hot_daily_market_data目录读取"""
    # 64个期货品种列表（主流品种）
    variety_list = [
        # 黑色
        'i', 'rb', 'hc', 'jm', 'j', 'ss', 'ni',
        # 能化
        'sc', 'fu', 'lu', 'pg', 'bu', 'eg', 'TA', 'PF', 'PX', 'l', 'v', 'MA', 'pp', 'eb', 'ru', 'nr', 'br',
        # 农产品
        'SF', 'SM', 'lc', 'OI', 'a', 'b', 'm', 'RM', 'p', 'y', 'CF', 'SR', 'c', 'cs', 'AP', 'PK', 'jd', 'CJ', 'lh',
        # 贵金属
        'ao', 'SH', 'au', 'ag',
        # 有色
        'sn', 'si', 'pb', 'cu', 'al', 'zn',
        # 股指
        'IF', 'IC', 'IH', 'IM', 'ec',
        # 利率
        'T', 'TF',
        # 建材
        'SA', 'FG', 'UR', 'sp'
    ]
    
    # 定义数据目录
    DATA_DIR = os.path.join(parent_dir, "History_Data/hot_daily_market_data")
    
    # 读取真实收盘价数据和主力合约信息
    close_data = pd.DataFrame()
    主力合约_map = {}
    
    for variety in variety_list:
        # 构造文件路径，文件名是品种名的大写.csv
        file_name = f"{variety.upper()}.csv"
        file_path = os.path.join(DATA_DIR, file_name)
        
        try:
            # 读取CSV文件，将第一列作为日期索引
            df = pd.read_csv(file_path, index_col=0)
            # 将索引转换为日期类型
            df.index = pd.to_datetime(df.index)
            
            # 提取收盘价列，并重命名为品种名
            close_series = df['close'].rename(variety)
            # 将该品种的收盘价数据合并到close_data中
            close_data = pd.concat([close_data, close_series], axis=1)
            
            # 提取主力合约信息
            # 保存每个品种的symbol列数据
            主力合约_map[variety] = df['symbol']
        except Exception as e:
            print(f"读取品种 {variety} 时出错: {e}")
    
    # 删除含有缺失值的行
    close_data = close_data.dropna()
    
    # 如果指定了目标日期，只保留该日期之前的数据
    if target_date:
        target_date = pd.to_datetime(target_date)
        close_data = close_data[close_data.index <= target_date]
    
    # 计算对数收益率
    returns = np.log(close_data / close_data.shift(1)).dropna()
    
    # 计算协方差矩阵（使用最近COV_DAYS天的数据）
    recent_returns = returns.tail(COV_DAYS)
    cov_matrix = recent_returns.cov() * 252  # 年化
    
    # 获取合约乘数和保证金率
    asset_margins = []
    asset_multipliers = []
    
    for sec in variety_list:
        # 使用get_contract_multiplier函数获取合约乘数和保证金率
        multiplier, margin_rate = get_contract_multiplier(sec)
        asset_multipliers.append(multiplier)
        asset_margins.append(margin_rate)
    
    asset_margins = np.array(asset_margins)
    asset_multipliers = np.array(asset_multipliers)
    
    # 生成模拟机器学习趋势信号（方向：-1/0/1，强度：0-10）
    # 注意：这里应该替换为真实的机器学习信号
    trend_dir = np.random.choice([-1, 0, 1], size=len(variety_list))
    trend_strength = np.random.uniform(0, 10, size=len(variety_list))
    trend_signal_df = pd.DataFrame({
        "品种": variety_list,
        "trend_dir": trend_dir,
        "trend_strength": trend_strength
    })
    
    return variety_list, close_data, returns, cov_matrix, asset_margins, asset_multipliers, trend_signal_df, 主力合约_map

# ===================== 9. 生成目标格式持仓文件 =====================
def generate_target_format_file(position_df, variety_list, asset_margins, asset_multipliers, close_data, target_date,主力合约_map, output_dir=None):
    """生成指定格式的目标持仓文件
    
    Args:
        position_df: 生成的持仓DataFrame
        variety_list: 品种列表
        asset_margins: 保证金率列表
        asset_multipliers: 合约乘数列表
        close_data: 收盘价数据
        target_date: 目标日期
        主力合约_map: 主力合约映射关系，键为品种名，值为包含symbol列的Series
        output_dir: 输出目录，如果为None则使用默认目录结构
    
    Returns:
        str: 生成的文件路径
    """
    # 格式化目标日期
    target_dt = pd.to_datetime(target_date)
    
    # 如果没有指定输出目录，使用默认的risk_parity_target_position目录结构
    if output_dir is None:
        # 定义主目标目录
        main_target_dir = os.path.join(current_dir, "risk_parity_target_position")
        os.makedirs(main_target_dir, exist_ok=True)
        
        # 生成YYMMDD_hhmm格式的子目录名
        current_time = pd.Timestamp.now().strftime("%y%m%d_%H%M")
        sub_dir = os.path.join(main_target_dir, current_time)
        os.makedirs(sub_dir, exist_ok=True)
        
        # 格式化目标日期为YYYYMMDD格式
        target_date_str = target_dt.strftime("%Y%m%d")
        
        # 构造输出文件名
        output_file = os.path.join(sub_dir, f"target_positions_{target_date_str}.csv")
    else:
        # 如果指定了输出目录，直接使用该目录
        os.makedirs(output_dir, exist_ok=True)
        target_date_str = target_dt.strftime("%Y%m%d")
        output_file = os.path.join(output_dir, f"target_positions_{target_date_str}.csv")
    
    # 创建目标格式的DataFrame
    target_df = pd.DataFrame(columns=[
        'symbol', 'current_price', 'contract_multiplier', 'position_size', 
        'position_value', 'margin_usage', 'risk_amount', 'margin_rate', 
        'total_capital', 'signal', 'model_type', 'market_value', 
        'allocated_capital', 'atr'
    ])
    
    # 遍历每个品种，生成目标格式数据
    for i, variety in enumerate(variety_list):
        # 获取当前品种的信息
        variety_row = position_df[position_df['品种'] == variety].iloc[0]
        
        # 获取当前价格
        current_price = close_data[variety].iloc[-1]
        
        # 获取合约乘数和保证金率
        multiplier = asset_multipliers[i]
        margin_rate = asset_margins[i]
        
        # 计算持仓大小（目标手数）
        position_size = variety_row['目标手数']
        
        # 计算持仓价值
        position_value = abs(position_size) * current_price * multiplier
        
        # 计算保证金使用
        margin_usage = position_value * margin_rate
        
        # 风险金额（这里简化为保证金使用）
        risk_amount = margin_usage
        
        # 总资金（固定值）
        total_capital = PRINCIPAL
        
        # 信号（根据趋势因子F）
        signal = variety_row['趋势因子F']
        
        # 模型类型
        model_type = 'random_forest_strategy'
        
        # 市值
        market_value = position_size * current_price * multiplier
        
        # 分配资金（这里简化为保证金使用）
        allocated_capital = margin_usage
        
        # ATR（这里简化为0，实际应从数据中获取）
        atr = 0.0
        
        # 获取真实的主力合约名称
        symbol = ""
        if variety in 主力合约_map:
            # 获取该品种在目标日期的主力合约
            variety_主力合约 = 主力合约_map[variety]
            # 找到目标日期或最接近的前一个交易日的主力合约
            try:
                # 找到目标日期或之前的最新数据
                symbol = variety_主力合约[variety_主力合约.index <= target_dt].iloc[-1]
            except IndexError:
                # 如果没有找到数据，使用默认格式
                year = target_dt.year % 100
                month = target_dt.month
                # 确定交易所后缀
                if variety in ['i', 'rb', 'hc', 'jm', 'j', 'ss', 'ni', 'sc', 'fu', 'lu', 'pg', 'bu', 'eg', 'l', 'v', 'MA', 'pp', 'eb', 'ru', 'nr', 'br', 'a', 'b', 'm', 'RM', 'p', 'y', 'c', 'cs', 'jd', 'lh']:
                    exchange = 'DCE'
                elif variety in ['TA', 'PF', 'PX', 'SF', 'SM', 'lc', 'OI', 'CF', 'SR', 'AP', 'PK', 'CJ', 'SA', 'FG', 'UR']:
                    exchange = 'CZCE'
                elif variety in ['au', 'ag', 'al', 'cu', 'zn', 'pb', 'sn', 'si']:
                    exchange = 'SHFE'
                elif variety in ['IF', 'IC', 'IH', 'IM', 'ec', 'T', 'TF']:
                    exchange = 'CFFEX'
                elif variety in ['sp']:
                    exchange = 'SHFE'
                else:
                    exchange = 'INE'
                
                # 构建主力合约代码
                symbol = f"{variety}{year}{month:02d}.{exchange}"
        else:
            # 如果没有主力合约信息，使用默认格式
            year = target_dt.year % 100
            month = target_dt.month
            # 确定交易所后缀
            if variety in ['i', 'rb', 'hc', 'jm', 'j', 'ss', 'ni', 'sc', 'fu', 'lu', 'pg', 'bu', 'eg', 'l', 'v', 'MA', 'pp', 'eb', 'ru', 'nr', 'br', 'a', 'b', 'm', 'RM', 'p', 'y', 'c', 'cs', 'jd', 'lh']:
                exchange = 'DCE'
            elif variety in ['TA', 'PF', 'PX', 'SF', 'SM', 'lc', 'OI', 'CF', 'SR', 'AP', 'PK', 'CJ', 'SA', 'FG', 'UR']:
                exchange = 'CZCE'
            elif variety in ['au', 'ag', 'al', 'cu', 'zn', 'pb', 'sn', 'si']:
                exchange = 'SHFE'
            elif variety in ['IF', 'IC', 'IH', 'IM', 'ec', 'T', 'TF']:
                exchange = 'CFFEX'
            elif variety in ['sp']:
                exchange = 'SHFE'
            else:
                exchange = 'INE'
            
            # 构建主力合约代码
            symbol = f"{variety}{year}{month:02d}.{exchange}"
        
        # 添加到目标DataFrame
        target_df.loc[i] = [
            symbol, current_price, multiplier, position_size, 
            position_value, margin_usage, risk_amount, margin_rate, 
            total_capital, signal, model_type, market_value, 
            allocated_capital, atr
        ]
    
    # 保存到CSV文件
    target_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n=== 目标格式持仓文件已生成：{output_file} ===")
    return output_file

# 导入趋势信号生成模块
import sys
import os
# 将当前目录添加到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from risk_sample_trend import build_variety2window, daily_trend_signal

# ===================== 10. 生成日期范围内每日目标头寸 =====================
def generate_daily_target_positions(start_date, end_date):
    """生成指定日期范围内每日的次日目标头寸
    
    Args:
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD
    """
    # 转换为日期类型
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # 首先加载所有数据，获取交易日列表和主力合约信息
    # 使用None作为target_date，加载所有可用数据
    variety_list, all_close_data, all_returns, _, _, _, _, all_主力合约_map = load_real_data(None)
    
    # 获取所有交易日的列表
    all_trading_days = all_close_data.index
    
    # 过滤出指定日期范围内的交易日
    trading_days = all_trading_days[(all_trading_days >= start_dt) & (all_trading_days <= end_dt)]
    
    if len(trading_days) == 0:
        print(f"=== {start_date}到{end_date}之间没有交易日 ===")
        return
    
    # 定义统一的输出目录
    main_target_dir = os.path.join(current_dir, "risk_parity_target_position")
    os.makedirs(main_target_dir, exist_ok=True)
    
    # 生成YYMMDD_hhmm格式的目录名
    current_time = pd.Timestamp.now().strftime("%y%m%d_%H%M")
    output_dir = os.path.join(main_target_dir, current_time)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== 开始生成{start_date}到{end_date}每日目标头寸 ===")
    print(f"=== 输出目录：{output_dir} ===")
    print(f"=== 共{len(trading_days)}个交易日 ===")
    
    # 预计算初始协方差矩阵和相关数据
    print("\n=== 预计算初始协方差矩阵 ===")
    
    # 找到第一个交易日之前的COV_DAYS个交易日
    first_trading_day = trading_days[0]
    pre_trading_days = all_trading_days[all_trading_days < first_trading_day]
    
    if len(pre_trading_days) < COV_DAYS:
        print(f"=== 第一个交易日 {first_trading_day.strftime('%Y-%m-%d')} 之前的交易日不足{COV_DAYS}天，无法预计算协方差矩阵 ===")
        return
    
    # 获取初始的COV_DAYS天收益率数据
    initial_start_date = pre_trading_days[-COV_DAYS]
    # 确保只获取COV_DAYS天的数据
    initial_returns = all_returns.loc[initial_start_date:first_trading_day].tail(COV_DAYS)
    
    # 确保有足够的数据
    if len(initial_returns) < COV_DAYS:
        print(f"=== 初始收益率数据不足{COV_DAYS}天，无法预计算协方差矩阵 ===")
        return
    
    # 计算初始协方差矩阵（年化）
    initial_cov_matrix = initial_returns.cov() * 252
    
    # 获取合约乘数和保证金率
    asset_margins = []
    asset_multipliers = []
    
    for sec in variety_list:
        multiplier, margin_rate = get_contract_multiplier(sec)
        asset_multipliers.append(multiplier)
        asset_margins.append(margin_rate)
    
    asset_margins = np.array(asset_margins)
    asset_multipliers = np.array(asset_multipliers)
    
    # 构建品种→窗口期映射
    variety2window = build_variety2window(variety_list)
    
    # 遍历每个交易日，生成目标头寸
    for i, target_date in enumerate(trading_days):
        target_date_str = target_date.strftime("%Y-%m-%d")
        print(f"\n=== 处理交易日：{target_date_str} ({i+1}/{len(trading_days)}) ===")
        
        try:
            # 获取当前交易日的收盘价数据
            close_data = all_close_data.loc[:target_date]
            
            # 如果是第一个交易日，使用预计算的协方差矩阵和收益率数据
            if i == 0:
                cov_matrix = initial_cov_matrix
                window_returns = initial_returns
            else:
                # 增量更新协方差矩阵
                # 获取前一个交易日
                prev_trading_day = trading_days[i-1]
                # 获取新的收益率数据（当前交易日的收益率）
                new_return = all_returns.loc[target_date]
                # 更新协方差矩阵
                cov_matrix = update_covariance_incremental(
                    old_cov=cov_matrix,
                    old_returns=window_returns,
                    new_return=new_return,
                    window_size=COV_DAYS
                )
                # 更新滚动窗口收益率数据：移除最旧的，添加新的
                window_returns = pd.concat([window_returns.iloc[1:], pd.DataFrame([new_return], index=[target_date])])
                # 确保window_returns仍然恰好包含COV_DAYS行数据
                window_returns = window_returns.tail(COV_DAYS)
            
            print(f"协方差矩阵维度：{cov_matrix.shape}")
            
            # 使用新的趋势信号生成逻辑
            trend_signal_df = daily_trend_signal(close_data, variety_list, variety2window)
            print(f"趋势信号生成完成：{len(trend_signal_df)}个品种")
            
            # 获取主力合约信息
            主力合约_map = {}
            for variety in variety_list:
                if variety in all_主力合约_map:
                    主力合约_map[variety] = all_主力合约_map[variety]
            
            # 2. 运行优化
            optimal_weights, metrics = optimize_portfolio(cov_matrix, asset_margins, variety_list, all_returns)
            
            # 3. 生成次日目标持仓
            position_df = generate_target_positions(
                optimal_weights=optimal_weights,
                variety_list=variety_list,
                asset_margins=asset_margins,
                asset_multipliers=asset_multipliers,
                close_data=close_data,
                trend_signal_df=trend_signal_df
            )
            
            # 4. 生成目标格式持仓文件
            generate_target_format_file(position_df, variety_list, asset_margins, asset_multipliers, close_data, target_date_str, 主力合约_map, output_dir)
            
        except Exception as e:
            print(f"处理交易日 {target_date_str} 时出错：{e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n=== {start_date}到{end_date}每日目标头寸生成完成 ===")
    print(f"=== 所有文件已保存至：{output_dir} ===")

def update_covariance_incremental(old_cov, old_returns, new_return, window_size):
    """增量更新协方差矩阵
    
    Args:
        old_cov: 旧协方差矩阵（年化）
        old_returns: 旧收益率数据（未年化），包含window_size行
        new_return: 新收益率数据（未年化），一行数据
        window_size: 滚动窗口大小
        
    Returns:
        new_cov: 更新后的协方差矩阵（年化）
    """
    # 确保old_returns包含足够的数据
    if len(old_returns) != window_size:
        raise ValueError(f"old_returns必须包含{window_size}行数据")
    
    # 将年化协方差转换为日度协方差
    old_cov_daily = old_cov / 252
    
    # 移除最旧的收益率数据
    oldest_return = old_returns.iloc[0]
    remaining_returns = old_returns.iloc[1:]
    
    # 计算旧均值
    old_mean = old_returns.mean()
    
    # 计算新均值（移除最旧数据，添加新数据）
    new_mean = (old_mean * window_size - oldest_return + new_return) / window_size
    
    # 计算协方差更新
    # 旧协方差矩阵公式：cov = (X^T X) / (n-1) - (n/(n-1)) * μ μ^T
    n = window_size
    
    # 移除最旧数据的贡献
    X_old = old_returns.values.T
    X_old_squared = X_old @ X_old.T
    X_old_squared_updated = X_old_squared - np.outer(oldest_return, oldest_return)
    
    # 添加新数据的贡献
    X_new_squared = X_old_squared_updated + np.outer(new_return, new_return)
    
    # 计算新的协方差矩阵（日度）
    new_cov_daily = (X_new_squared / (n-1)) - (n/(n-1)) * np.outer(new_mean, new_mean)
    
    # 转换为年化协方差
    new_cov = new_cov_daily * 252
    
    return new_cov

# ===================== 11. 主运行流程（一键执行） =====================
if __name__ == "__main__":
    # 添加命令行参数处理
    parser = argparse.ArgumentParser(description='生成指定日期或日期范围的次日目标头寸')
    parser.add_argument('--date', type=str, default=None, help='目标日期，格式：YYYY-MM-DD')
    parser.add_argument('--start-date', type=str, default=None, help='开始日期，格式：YYYY-MM-DD，用于生成日期范围')
    parser.add_argument('--end-date', type=str, default=None, help='结束日期，格式：YYYY-MM-DD，用于生成日期范围')
    args = parser.parse_args()
    
    # 检查参数
    if args.start_date and args.end_date:
        # 生成日期范围内每日目标头寸
        generate_daily_target_positions(args.start_date, args.end_date)
    elif args.date:
        # 生成单个日期的目标头寸
        # 1. 加载真实数据
        print("=== 加载真实数据 ===")
        variety_list, close_data, returns, cov_matrix, asset_margins, asset_multipliers, trend_signal_df, 主力合约_map = load_real_data(args.date)
        print(f"数据加载完成：{len(variety_list)}个品种，{len(returns)}日收益率")
        
        # 2. 运行优化
        print("\n=== 运行优化 ===")
        optimal_weights, metrics = optimize_portfolio(cov_matrix, asset_margins, variety_list, returns)
        
        # 3. 打印评估报告
        print("\n" + "="*80)
        print("=== 算法评估报告 ===")
        print("="*80)
        for k, v in metrics.items():
            print(f"{k:25}: {v}")
        
        # 4. 生成次日目标持仓
        print("\n=== 生成次日目标持仓 ===")
        position_df = generate_target_positions(
            optimal_weights=optimal_weights,
            variety_list=variety_list,
            asset_margins=asset_margins,
            asset_multipliers=asset_multipliers,
            close_data=close_data,
            trend_signal_df=trend_signal_df
        )
        
        # 5. 输出持仓指令（仅显示有交易的品种）
        print("\n" + "="*80)
        print("=== 次日目标持仓指令（有交易） ===")
        print("="*80)
        trade_df = position_df[position_df["仓差"] != 0][["品种", "趋势因子F", "融合权重", "当前持仓", "目标手数", "仓差", "交易指令"]]
        print(trade_df.head(20))  # 显示前20条
        
        # 6. 保存结果
        output_dir = f"./output_{args.date}" if args.date else "./output"
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 评估报告
        pd.DataFrame([metrics]).to_csv(f"{output_dir}/算法评估报告.csv", index=False, encoding='utf-8-sig')
        # 持仓指令
        position_df.to_csv(f"{output_dir}/次日目标持仓指令.csv", index=False, encoding='utf-8-sig')
        print(f"\n=== 结果已保存至：{output_dir} ===")
        print(f"\n=== 生成的是 {close_data.index[-1].date()} 的次日目标头寸 ===")
        
        # 7. 生成目标格式持仓文件
        generate_target_format_file(position_df, variety_list, asset_margins, asset_multipliers, close_data, args.date, 主力合约_map)
    else:
        print("请指定日期或日期范围")
        print("使用方法：")
        print("  生成单个日期：python risk_sample.py --date YYYY-MM-DD")
        print("  生成日期范围：python risk_sample.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD")