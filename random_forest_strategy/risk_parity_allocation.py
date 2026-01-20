# ===================== 1. 导入核心库 =====================
import time
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import variation
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入获取合约乘数和保证金率的函数
from utility.instrument_utils import get_contract_multiplier

# ===================== 2. 全局配置（根据实盘调整） =====================
# 资金与风控
PRINCIPAL = 3000000  # 总本金（元）
MARGIN_RATIO_LIMIT = 0.70  # 保证金上限占比（70%）
TARGET_MARGIN_RATIO = 0.65  # 目标保证金占比（65%）
WEIGHT_LOWER = -0.5  # 单品种权重下限
WEIGHT_UPPER = 0.5  # 单品种权重上限
COV_DAYS = 60  # 协方差计算周期
# 趋势信号参数
ALPHA = 0.5  # 趋势增强系数（0.3~0.7可调）
STRENGTH_THRESH = 2.0  # 趋势强度过滤阈值
# L-BFGS-B优化参数（高维适配）
MAX_ITER_LBFGS = 5000  # 迭代上限（远低于Nelder-Mead，足够收敛）
GTOL = 1e-5  # 梯度收敛阈值
# 仅保留权重和约束的轻量惩罚项（系数大幅降低）
LAMBDA_SUM = 1e4  # 权重和=1惩罚系数（从1e6→1e4）

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

# ===================== 4. L-BFGS-B目标函数（极简版） =====================
def risk_parity_core_objective(weights, cov_matrix, asset_margins):
    """极简目标函数：仅优化风险贡献均衡+权重和约束"""
    weights = np.array(weights)
    cov_matrix = np.array(cov_matrix)
    asset_margins = np.array(asset_margins)
    
    # 1. 核心目标：风险贡献方差最小（无权重，直接优化）
    rc = risk_contribution(weights, cov_matrix, asset_margins)
    rc_abs = np.abs(rc)
    core_obj = np.var(rc_abs)
    
    # 2. 轻量惩罚项：权重和=1（系数仅1e4，避免函数值过大）
    sum_w = np.sum(weights)
    penalty_sum = LAMBDA_SUM * (sum_w - 1) ** 2
    
    # 总目标函数（值可控，无极端值）
    total_obj = core_obj + penalty_sum
    return total_obj

# ===================== 5. 保证金显式修正函数（核心） =====================
def adjust_margin_ratio(weights, asset_margins, principal, target_ratio, max_ratio):
    """优化后显式修正权重至目标保证金占比（替代惩罚项）"""
    weights = np.array(weights)
    # 计算当前保证金占用
    nominal_cap = weights * principal
    margin_usage = np.abs(nominal_cap) * asset_margins
    total_margin = np.sum(margin_usage)
    target_margin = principal * target_ratio
    max_margin = principal * max_ratio
    
    # 缩放权重至目标占比（避免超限）
    if total_margin == 0:
        # 极端情况：权重全0，按等权分配至目标占比
        weights = np.ones_like(weights) / len(weights)
        total_margin = np.sum(np.abs(weights * principal) * asset_margins)
    
    # 缩放系数（不超过上限）
    scale_ratio = min(target_margin / total_margin, max_margin / total_margin)
    weights_scaled = weights * scale_ratio
    
    # 归一化+硬约束
    weights_scaled = weights_scaled / np.sum(weights_scaled)
    weights_scaled = np.clip(weights_scaled, WEIGHT_LOWER, WEIGHT_UPPER)
    
    # 最终校验：确保不超限
    final_margin = np.sum(np.abs(weights_scaled * principal) * asset_margins)
    if final_margin > max_margin:
        weights_scaled = weights_scaled * (max_margin / final_margin)
        weights_scaled = weights_scaled / np.sum(weights_scaled)
    
    return weights_scaled

# ===================== 6. HRP兜底（保留，仅作为最终兜底） =====================
def hrp_portfolio(returns, variety_list, asset_margins):
    """HRP兜底+保证金约束"""
    try:
        corr = returns.corr()
        dist = np.sqrt(0.5 * (1 - corr))
        link = linkage(pdist(dist), method='ward')
        clusters = fcluster(link, t=8, criterion='maxclust')
        cluster_unique = np.unique(clusters)
        cluster_weights = np.ones(len(cluster_unique)) / len(cluster_unique)
        hrp_weights = np.zeros(len(variety_list))
        for i, cluster in enumerate(cluster_unique):
            idx = np.where(clusters == cluster)[0]
            hrp_weights[idx] = cluster_weights[i] / len(idx)
        return hrp_weights
    except Exception as e:
        print(f"警告：HRP计算失败: {e}，使用等权重分配")
        return np.ones(len(variety_list)) / len(variety_list)

# ===================== 7. 主优化函数（L-BFGS-B+分阶段修正） =====================
def optimize_portfolio(cov_matrix, asset_margins, variety_list, returns, principal):
    """主优化：L-BFGS-B+分阶段修正（风险均衡→保证金→板块）"""
    # 1. 协方差矩阵稳定性处理（添加小的对角线扰动，避免奇异矩阵）
    cov_matrix_np = np.array(cov_matrix)
    # 添加小的对角线扰动
    cov_matrix_stable = cov_matrix_np + np.eye(len(cov_matrix_np)) * 1e-6
    
    n_assets = len(variety_list)
    # 初始权重：等权（简单且稳定）
    init_weights = np.ones(n_assets) / n_assets
    
    # ---------------------- 阶段1：L-BFGS-B优化核心目标（风险均衡+权重和） ----------------------
    # 目标函数封装
    def obj_fun(weights):
        return risk_parity_core_objective(weights, cov_matrix_stable, asset_margins)
    # L-BFGS-B优化（显式上下限约束）
    start_time = time.time()
    result = minimize(
        fun=obj_fun,
        x0=init_weights,
        method='L-BFGS-B',  # 核心更换为L-BFGS-B
        bounds=[(WEIGHT_LOWER, WEIGHT_UPPER)] * n_assets,  # 显式上下限约束
        options={
            'disp': False,
            'maxiter': MAX_ITER_LBFGS,
            'gtol': GTOL,
            'maxcor': 20  # 内存占用与收敛速度的平衡
        }
    )
    run_time = time.time() - start_time
    
    # ---------------------- 阶段2：修正权重至目标保证金占比（核心） ----------------------
    if result.success:
        optimal_weights = result.x / np.sum(result.x)  # 强制权重和=1
    else:
        # L-BFGS-B也收敛失败时，用HRP兜底
        print("=== L-BFGS-B收敛失败，启用HRP兜底 ===")
        optimal_weights = hrp_portfolio(returns, variety_list, asset_margins)  # 传入asset_margins
    
    # 显式修正保证金占比（替代惩罚项）
    optimal_weights = adjust_margin_ratio(
        weights=optimal_weights,
        asset_margins=asset_margins,
        principal=principal,
        target_ratio=TARGET_MARGIN_RATIO,
        max_ratio=MARGIN_RATIO_LIMIT
    )
    
    return optimal_weights

# ===================== 7. 趋势信号融合+持仓生成 =====================
def generate_target_positions(optimal_weights, variety_list, asset_margins, asset_multipliers, close_data, trend_signal_df, principal):
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
    nominal_capital = W_fusion * principal
    margin_usage = np.abs(nominal_capital) * asset_margins
    total_margin = np.sum(margin_usage)
    if total_margin > principal * MARGIN_RATIO_LIMIT:
        shrink_ratio = (principal * MARGIN_RATIO_LIMIT) / total_margin
        W_fusion = W_fusion * shrink_ratio
        W_fusion = W_fusion / np.sum(W_fusion)
    # 4. 计算目标手数
    target_lots = np.zeros(len(variety_list), dtype=int)
    for i, var in enumerate(variety_list):
        w = W_fusion[i]
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
    return W_fusion, target_lots

# ===================== 8. 资金分配策略函数 =====================
def risk_parity_allocation(capital, varieties_data, date, all_data=None):
    """风险平价资金分配策略"""
    print(f"使用风险平价分配策略，处理日期: {date}")
    
    # 1. 初步数据检查：排除价格数据不足的品种
    valid_symbols = list(varieties_data.keys())
    if not valid_symbols:
        print("没有品种有交易信号，无法进行风险平价分配")
        return {}, {}
    
    # 2. 构建价格数据DataFrame
    close_data = pd.DataFrame()
    valid_symbols_with_data = []
    
    for base_symbol in valid_symbols:
        try:
            # 从all_data获取完整的历史数据
            if all_data:
                # 处理大小写问题，尝试不同的大小写组合
                possible_symbols = [base_symbol, base_symbol.upper(), base_symbol.lower()]
                found_symbol = None
                for sym in possible_symbols:
                    if sym in all_data:
                        found_symbol = sym
                        break
                
                if found_symbol:
                    # 使用完整的历史数据
                    df = all_data[found_symbol]
                    # 筛选数据到指定日期
                    df = df[df.index <= date]
                    
                    # 检查数据长度
                    if len(df) < 60:
                        print(f"警告：{base_symbol}历史数据不足60条，排除该品种")
                        continue
                    
                    # 获取收盘价
                    close_series = df['close'].rename(base_symbol)
                    close_data = pd.concat([close_data, close_series], axis=1)
                    valid_symbols_with_data.append(base_symbol)
                else:
                    # 使用varieties_data中的价格数据（作为备选）
                    data = varieties_data[base_symbol]
                    prices = data['prices']
                    if len(prices) < 60:
                        print(f"警告：{base_symbol}价格数据不足60条，排除该品种")
                        continue
                    
                    # 创建日期索引
                    dates = pd.date_range(end=date, periods=len(prices), freq='D')
                    price_series = pd.Series(prices, index=dates, name=base_symbol)
                    close_data = pd.concat([close_data, price_series], axis=1)
                    valid_symbols_with_data.append(base_symbol)
            else:
                # 使用varieties_data中的价格数据（作为备选）
                data = varieties_data[base_symbol]
                prices = data['prices']
                if len(prices) < 60:
                    print(f"警告：{base_symbol}价格数据不足60条，排除该品种")
                    continue
                
                # 创建日期索引
                dates = pd.date_range(end=date, periods=len(prices), freq='D')
                price_series = pd.Series(prices, index=dates, name=base_symbol)
                close_data = pd.concat([close_data, price_series], axis=1)
                valid_symbols_with_data.append(base_symbol)
            
        except Exception as e:
            print(f"警告：处理{base_symbol}数据出错: {e}，排除该品种")
            continue
    
    if not valid_symbols_with_data:
        print("所有品种数据不足，无法进行风险平价分配")
        return {}, {}
    
    # 更新valid_symbols
    valid_symbols = valid_symbols_with_data
    
    # 3. 计算收益率和协方差矩阵
    returns = np.log(close_data / close_data.shift(1)).dropna()
    
    # 4. 进一步数据验证和清洗
    if returns.empty:
        print("警告：计算收益率时出现空数据，无法进行风险平价分配")
        return {}, {}
    
    recent_returns = returns.tail(COV_DAYS)
    
    if len(recent_returns) < 10:  # 需要至少10天的收益率数据
        print("警告：近期收益率数据不足，无法进行风险平价分配")
        return {}, {}
    
    # 5. 计算协方差矩阵，处理可能出现的问题
    try:
        # 添加调试信息
        print(f"调试：recent_returns类型: {type(recent_returns)}")
        print(f"调试：recent_returns形状: {recent_returns.shape}")
        print(f"调试：recent_returns内容: {recent_returns.head()}")
        
        # 计算协方差矩阵
        cov_matrix = recent_returns.cov() * 252  # 年化
        
        # 添加调试信息
        print(f"调试：cov_matrix类型: {type(cov_matrix)}")
        print(f"调试：cov_matrix形状: {cov_matrix.shape if hasattr(cov_matrix, 'shape') else '标量'}")
        
        # 检查协方差矩阵是否包含NaN或无穷大值
        has_invalid_values = False
        
        # 使用try-except处理不同类型的cov_matrix
        try:
            # 检查是否为DataFrame或Series
            if isinstance(cov_matrix, (pd.DataFrame, pd.Series)):
                # 使用values属性转换为numpy数组
                cov_array = cov_matrix.values
                has_invalid_values = np.isnan(cov_array).any() or np.isinf(cov_array).any()
            else:
                # 标量情况
                has_invalid_values = np.isnan(cov_matrix) or np.isinf(cov_matrix)
        except Exception as inner_e:
            print(f"调试：检查协方差矩阵时出错: {inner_e}")
            has_invalid_values = True
        
        if has_invalid_values:
            print("警告：协方差矩阵包含无效值，无法进行风险平价分配")
            return {}, {}
    except Exception as e:
        print(f"警告：计算协方差矩阵时出错: {e}，无法进行风险平价分配")
        import traceback
        traceback.print_exc()
        # 尝试使用简单的等权重分配作为备选
        print("尝试使用等权重分配作为备选")
        allocation_dict = {base_symbol: capital / len(valid_symbols) for base_symbol in valid_symbols}
        risk_units = {base_symbol: capital / len(valid_symbols) for base_symbol in valid_symbols}
        return allocation_dict, risk_units
    
    # 4. 获取合约乘数和保证金率
    asset_margins = []
    asset_multipliers = []
    
    for base_symbol in valid_symbols:
        # 使用get_contract_multiplier函数获取合约乘数和保证金率
        data = varieties_data[base_symbol]
        contract_symbol = data['contract_symbol']
        multiplier, margin_rate = get_contract_multiplier(contract_symbol)
        asset_multipliers.append(multiplier)
        asset_margins.append(margin_rate)
    
    asset_margins = np.array(asset_margins)
    asset_multipliers = np.array(asset_multipliers)
    
    # 5. 生成趋势信号DataFrame
    trend_dir = []
    trend_strength = []
    
    for base_symbol in valid_symbols:
        data = varieties_data[base_symbol]
        # 使用品种的信号作为趋势方向
        trend_dir.append(data['signal'])
        # 使用趋势强度
        trend_strength.append(data['trend_strength'] * 10)  # 缩放为0-10范围
    
    trend_signal_df = pd.DataFrame({
        'trend_dir': trend_dir,
        'trend_strength': trend_strength
    })
    
    # 6. 运行风险平价优化
    optimal_weights = optimize_portfolio(cov_matrix, asset_margins, valid_symbols, recent_returns, capital)
    
    # 7. 生成目标权重和手数
    W_fusion, target_lots = generate_target_positions(
        optimal_weights=optimal_weights,
        variety_list=valid_symbols,
        asset_margins=asset_margins,
        asset_multipliers=asset_multipliers,
        close_data=close_data,
        trend_signal_df=trend_signal_df,
        principal=capital
    )
    
    # 8. 计算分配资金
    allocation_dict = {}
    risk_units = {}
    
    for i, base_symbol in enumerate(valid_symbols):
        w = W_fusion[i]
        lot = target_lots[i]
        
        if lot != 0:
            # 计算分配资金
            allocated_capital = w * capital
            allocation_dict[base_symbol] = allocated_capital
            # 计算风险单位（简化处理，使用分配资金作为风险单位）
            risk_units[base_symbol] = allocated_capital
    
    return allocation_dict, risk_units
