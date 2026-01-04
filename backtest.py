import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
from functions import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_sortino_ratio,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_information_ratio,
    ensure_directory_exists
)

# 配置参数
TARGET_POSITION_DIR = 'random_forest_strategy/target_position'
PRIMARY_DATA_DIR = 'History_Data/hot_daily_market_data'
SECONDARY_DATA_DIR = 'History_Data/secondary_daily_market_data'
OVER_DATA_DIR = 'History_Data/over_daily_market_data'
VALIDATION_RESULT_DIR = 'validation_result'
START_DATE = '2024-07-01'


def load_all_target_positions(target_dir):
    """加载所有目标仓位文件"""
    # 使用正确的路径格式
    pattern = os.path.join(target_dir, 'target_positions_*.csv')
    files = glob.glob(pattern)
    # 按日期排序
    files.sort()
    
    target_positions = {}
    for file in files:
        try:
            # 从文件名提取日期，格式为YYYYMMDD
            date_str = os.path.basename(file).split('_')[-1].split('.')[0]
            date = datetime.strptime(date_str, '%Y%m%d')
            df = pd.read_csv(file)
            target_positions[date] = df
            print(f"  成功加载文件: {file}")
        except PermissionError as e:
            print(f"  警告：无法访问文件 {file}，权限被拒绝: {e}")
            continue
        except Exception as e:
            print(f"  警告：加载文件 {file} 失败: {e}")
            continue
    
    return target_positions


def load_all_history_data(primary_dir, secondary_dir):
    """加载所有历史数据，包括主力、次主力和备用合约数据"""
    all_data = []
    
    # 加载主力合约数据
    print(f"  正在加载主力合约数据: {primary_dir}")
    primary_files = [f for f in os.listdir(primary_dir) if f.endswith('.csv')]
    for file in primary_files:
        file_path = os.path.join(primary_dir, file)
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        all_data.append(df)
    
    # 加载次主力合约数据
    print(f"  正在加载次主力合约数据: {secondary_dir}")
    secondary_files = [f for f in os.listdir(secondary_dir) if f.endswith('.csv')]
    for file in secondary_files:
        file_path = os.path.join(secondary_dir, file)
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        all_data.append(df)
    
    # 加载备用合约数据（主力合约切换后的数据）
    if os.path.exists(OVER_DATA_DIR):
        print(f"  正在加载备用合约数据: {OVER_DATA_DIR}")
        over_files = [f for f in os.listdir(OVER_DATA_DIR) if f.endswith('.csv')]
        for file in over_files:
            file_path = os.path.join(OVER_DATA_DIR, file)
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            all_data.append(df)
    else:
        print(f"  备用合约数据目录 {OVER_DATA_DIR} 不存在，跳过加载")
    
    # 合并所有数据
    combined_df = pd.concat(all_data)
    return combined_df


def backtest():
    """回测主函数"""
    # 加载数据
    print("正在加载目标仓位数据...")
    target_positions = load_all_target_positions(TARGET_POSITION_DIR)
    
    print("正在加载历史数据...")
    # 加载主力和次主力合约数据
    history_data = load_all_history_data(PRIMARY_DATA_DIR, SECONDARY_DATA_DIR)
    
    # 确保验证结果目录存在
    ensure_directory_exists(VALIDATION_RESULT_DIR)
    
    # 按日期排序
    sorted_dates = sorted(target_positions.keys())
    
    # 初始化回测结果
    daily_returns = []
    actual_trading_days = []  # 保存实际交易日期（目标头寸生成日的下一个交易日）
    processed_dates = []  # 保存实际处理的目标头寸生成日期
    current_positions = {}  # 保存当前持仓，键为symbol，值为持仓数量
    initial_capital = 10000000
    current_capital = initial_capital
    equity_curve = [initial_capital]
    
    # 记录每日交易明细
    daily_trades = []
    
    print(f"开始回测，共{len(sorted_dates)}个交易日...")
    
    # 处理每个日期
    for i, date in enumerate(sorted_dates):
        print(f"正在处理日期: {date.strftime('%Y-%m-%d')}...")
        
        # 获取当天的目标仓位
        target_df = target_positions[date]
        
        # 计算次日日期
        next_date = date + timedelta(days=1)
        
        # 过滤出下一个交易日的数据
        next_trading_day_data = history_data[history_data.index > date].sort_index()
        
        if next_trading_day_data.empty:
            print(f"  警告：没有{date}之后的交易数据，跳过")
            continue
        
        next_trading_day = next_trading_day_data.index[0]
        
        # 获取次日的完整数据（开盘价和收盘价）
        next_day_data = next_trading_day_data.loc[next_trading_day:next_trading_day]
        
        # 提取基础品种（去除月份）
        def get_base_symbol(symbol):
            """从合约代码中提取基础品种，如a2409.DCE -> a"""
            # 处理格式如a2409.DCE
            if '.' in symbol:
                return symbol.split('.')[0][0] if symbol.split('.')[0][1].isdigit() else symbol.split('.')[0][:2]
            return symbol
        
        # 按基础品种分组
        base_symbol_dict = {}
        for _, row in target_df.iterrows():
            symbol = row['symbol']
            base_symbol = get_base_symbol(symbol)
            if base_symbol not in base_symbol_dict:
                base_symbol_dict[base_symbol] = []
            base_symbol_dict[base_symbol].append(symbol)
        
        # 处理每个基础品种的合约切换
        for base_symbol, symbols in base_symbol_dict.items():
            # 查找当前持仓中该基础品种的所有合约
            current_base_positions = {sym: qty for sym, qty in current_positions.items() if get_base_symbol(sym) == base_symbol}
            
            # 获取目标持仓中该基础品种的合约
            target_symbol = symbols[0]  # 假设每个基础品种只有一个目标合约
            target_qty = target_df[target_df['symbol'] == target_symbol]['position_size'].iloc[0]
            
            # 如果当前持仓中有该基础品种的其他合约，需要平仓
            for current_symbol, current_qty in current_base_positions.items():
                if current_symbol != target_symbol:
                    print(f"  检测到合约切换：{base_symbol} 从 {current_symbol} 切换到 {target_symbol}")
                    
                    # 平仓原有合约
                    next_day_symbol_data = next_day_data[next_day_data['symbol'] == current_symbol]
                    if next_day_symbol_data.empty:
                        # 尝试从完整的history_data中查找该合约在next_trading_day的数据
                        print(f"  警告：主力合约数据中没有{current_symbol}在{next_trading_day}的数据，尝试从完整历史数据中查找")
                        full_symbol_data = history_data[(history_data.index == next_trading_day) & (history_data['symbol'] == current_symbol)]
                        if not full_symbol_data.empty:
                            next_day_symbol_data = full_symbol_data
                            print(f"  从完整历史数据中找到了{current_symbol}在{next_trading_day}的数据")
                        else:
                            # 查找该合约在历史数据中最近一天的收盘价
                            print(f"  警告：从完整历史数据中也没有找到{current_symbol}在{next_trading_day}的数据")
                            contract_data = history_data[history_data['symbol'] == current_symbol]
                            if not contract_data.empty:
                                # 获取最近一天的收盘价
                                latest_data = contract_data.sort_index().iloc[-1]
                                latest_close = latest_data['close']
                                latest_date = latest_data.name.strftime('%Y-%m-%d')
                                print(f"  使用最近一天{latest_date}的收盘价{latest_close}作为平仓价格")
                                
                                # 计算合约乘数
                                multiplier, _ = get_contract_multiplier(current_symbol)
                                
                                # 平仓盈亏（使用最近收盘价作为平仓价格，没有交易成本）
                                close_pnl = -current_qty * (latest_close - latest_close) * multiplier
                                
                                # 记录交易
                                daily_trades.append({
                                    'date': date.strftime('%Y-%m-%d'),
                                    'symbol': current_symbol,
                                    'prev_position': current_qty,
                                    'prev_close': latest_close,  # 使用最近收盘价作为持仓成本
                                    'current_position': 0,
                                    'current_close': latest_close,
                                    'position_change': -current_qty,
                                    'trade_price': latest_close,
                                    'settlement_price': latest_close,
                                    'trade_pnl': close_pnl,
                                    'hold_pnl': 0.0,
                                    'total_pnl': close_pnl,
                                    'contract_multiplier': multiplier,
                                    'transaction_type': '合约切换平仓（使用最近收盘价）'
                                })
                                
                                # 从当前持仓中移除该合约
                                del current_positions[current_symbol]
                            else:
                                print(f"  警告：找不到{current_symbol}的任何历史数据，无法平仓")
                            continue
                    
                    next_open = next_day_symbol_data['open'].iloc[0]
                    next_close = next_day_symbol_data['close'].iloc[0]
                    
                    # 计算合约乘数
                    multiplier, _ = get_contract_multiplier(current_symbol)
                    
                    # 平仓盈亏
                    close_pnl = -current_qty * (next_close - next_open) * multiplier
                    
                    # 记录交易
                    daily_trades.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'symbol': current_symbol,
                        'prev_position': current_qty,
                        'prev_close': next_open,  # 使用开盘价作为持仓成本
                        'current_position': 0,
                        'current_close': next_close,
                        'position_change': -current_qty,
                        'trade_price': next_open,
                        'settlement_price': next_close,
                        'trade_pnl': close_pnl,
                        'hold_pnl': 0.0,
                        'total_pnl': close_pnl,
                        'contract_multiplier': multiplier,
                        'transaction_type': '合约切换平仓'
                    })
                    
                    # 从当前持仓中移除该合约
                    del current_positions[current_symbol]
        
        # 创建一个当日所有交易品种的集合
        all_symbols = set(current_positions.keys()) | set(target_df['symbol'].tolist())
        
        # 处理每个品种
        for symbol in all_symbols:
            # 昨日持仓数量
            prev_position = current_positions.get(symbol, 0)
            
            # 昨日close价格
            prev_close = 0.0
            prev_data = history_data[(history_data.index == date) & (history_data['symbol'] == symbol)]
            if not prev_data.empty:
                prev_close = prev_data['close'].iloc[0]
            
            # 当日目标仓位
            target_row = target_df[target_df['symbol'] == symbol]
            target_position = target_row['position_size'].iloc[0] if not target_row.empty else 0
            
            # 当日持仓数量（目标仓位）
            current_position = target_position
            
            # 变动手数
            position_change = current_position - prev_position
            
            # 获取次日的开盘价和收盘价
            next_day_symbol_data = next_day_data[next_day_data['symbol'] == symbol]
            if next_day_symbol_data.empty:
                # 尝试从完整的history_data中查找该合约在next_trading_day的数据
                print(f"  警告：没有{symbol}在{next_trading_day}的数据，尝试从完整历史数据中查找")
                full_symbol_data = history_data[(history_data.index == next_trading_day) & (history_data['symbol'] == symbol)]
                if not full_symbol_data.empty:
                    next_day_symbol_data = full_symbol_data
                    print(f"  从完整历史数据中找到了{symbol}在{next_trading_day}的数据")
                else:
                    # 查找该合约在历史数据中最近一天的收盘价
                    print(f"  警告：从完整历史数据中也没有找到{symbol}在{next_trading_day}的数据")
                    contract_data = history_data[history_data['symbol'] == symbol]
                    if not contract_data.empty:
                        # 获取最近一天的收盘价
                        latest_data = contract_data.sort_index().iloc[-1]
                        latest_close = latest_data['close']
                        latest_date = latest_data.name.strftime('%Y-%m-%d')
                        print(f"  使用最近一天{latest_date}的收盘价{latest_close}作为交易价格")
                        
                        # 创建模拟的next_day_symbol_data
                        next_day_symbol_data = pd.DataFrame({
                            'symbol': [symbol],
                            'open': [latest_close],
                            'close': [latest_close],
                            'high': [latest_close],
                            'low': [latest_close],
                            'volume': [0],
                            'money': [0],
                            'open_interest': [0]
                        }, index=[next_trading_day])
                    else:
                        print(f"  警告：找不到{symbol}的任何历史数据，跳过")
                        continue
            
            next_open = next_day_symbol_data['open'].iloc[0]
            next_close = next_day_symbol_data['close'].iloc[0]
            
            # 计算合约乘数
            multiplier, _ = get_contract_multiplier(symbol)
            
            # 使用开盘价计算交易盈亏（开仓/平仓）
            trade_pnl = 0.0
            if position_change != 0:
                # 计算交易盈亏
                trade_pnl = position_change * (next_close - next_open) * multiplier
            
            # 使用收盘价计算持仓盈亏
            hold_pnl = 0.0
            if prev_position != 0:
                hold_pnl = prev_position * (next_close - prev_close) * multiplier
            
            # 总盈亏
            total_pnl = trade_pnl + hold_pnl
            
            # 更新当前仓位
            if target_position != 0:
                current_positions[symbol] = target_position
            elif symbol in current_positions:
                del current_positions[symbol]
            
            # 记录交易明细
            daily_trades.append({
                'date': date.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'prev_position': prev_position,
                'prev_close': prev_close,
                'current_position': current_position,
                'current_close': next_close,
                'position_change': position_change,
                'trade_price': next_open,
                'settlement_price': next_close,
                'trade_pnl': trade_pnl,
                'hold_pnl': hold_pnl,
                'total_pnl': total_pnl,
                'contract_multiplier': multiplier,
                'transaction_type': '正常交易'
            })
        
        # 计算当日总盈亏
        day_pnl = sum(trade['total_pnl'] for trade in daily_trades if trade['date'] == date.strftime('%Y-%m-%d'))
        
        # 更新资金
        current_capital += day_pnl
        equity_curve.append(current_capital)
        
        # 计算日收益率
        daily_return = day_pnl / equity_curve[-2] if equity_curve[-2] != 0 else 0
        daily_returns.append(daily_return)
        actual_trading_days.append(next_trading_day)  # 保存实际交易日期
        processed_dates.append(date)  # 保存实际处理的目标头寸生成日期
        
        print(f"  当日盈亏: {day_pnl:.2f}, 当日收益率: {daily_return:.4f}, 总资产: {current_capital:.2f}")
    
    # 计算回测指标
    print("\n正在计算回测指标...")
    
    # 转换为Series
    daily_returns_series = pd.Series(daily_returns)
    
    # 计算指标
    sharpe_ratio = calculate_sharpe_ratio(daily_returns_series)
    max_drawdown = calculate_max_drawdown(daily_returns_series)
    sortino_ratio = calculate_sortino_ratio(daily_returns_series)
    win_rate = calculate_win_rate(daily_returns_series)
    profit_factor = calculate_profit_factor(daily_returns_series)
    
    # 计算信息比率
    information_ratio = calculate_information_ratio(daily_returns_series)
    
    # 计算年化收益率
    annualized_return = daily_returns_series.mean() * 252
    
    # 计算年化波动率
    annualized_volatility = daily_returns_series.std() * np.sqrt(252)
    
    # 计算总收益率
    total_return = (current_capital - initial_capital) / initial_capital
    
    # 输出指标
    print(f"回测指标:")
    print(f"  总收益率: {total_return:.4f} ({total_return*100:.2f}%)")
    print(f"  年化收益率: {annualized_return:.4f} ({annualized_return*100:.2f}%)")
    print(f"  年化波动率: {annualized_volatility:.4f} ({annualized_volatility*100:.2f}%)")
    print(f"  夏普比率: {sharpe_ratio:.4f}")
    print(f"  最大回撤: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
    print(f"  索提诺比率: {sortino_ratio:.4f}")
    print(f"  信息比率: {information_ratio:.4f}")
    print(f"  胜率: {win_rate:.4f} ({win_rate*100:.2f}%)")
    print(f"  盈利因子: {profit_factor:.4f}")
    
    # 保存回测结果
    print("\n正在保存回测结果...")
    
    # 保存每日收益率
    # 使用实际处理的日期列表，确保所有数组长度一致
    valid_length = len(daily_returns)
    returns_df = pd.DataFrame({
        'date': [date.strftime('%Y-%m-%d') for date in processed_dates],  # 实际处理的目标头寸生成日期
        'actual_trading_date': actual_trading_days,  # 实际交易日期（下一个交易日）
        'daily_return': daily_returns,
        'equity': equity_curve[1:]
    })
    
    returns_file = os.path.join(VALIDATION_RESULT_DIR, 'daily_returns.csv')
    returns_df.to_csv(returns_file, index=False)
    print(f"  每日收益已保存到: {returns_file}")
    
    # 保存回测指标
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'sortino_ratio': sortino_ratio,
        'information_ratio': information_ratio,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'initial_capital': initial_capital,
        'final_capital': current_capital,
        'trading_days': len(daily_returns)
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_file = os.path.join(VALIDATION_RESULT_DIR, 'backtest_metrics.csv')
    metrics_df.to_csv(metrics_file, index=False)
    print(f"  回测指标已保存到: {metrics_file}")
    
    # 保存每日交易明细
    trades_df = pd.DataFrame(daily_trades)
    trades_file = os.path.join(VALIDATION_RESULT_DIR, 'daily_trades.csv')
    trades_df.to_csv(trades_file, index=False)
    print(f"  每日交易明细已保存到: {trades_file}")
    
    print("\n回测完成！")


# 从position.py导入get_contract_multiplier函数
from position import get_contract_multiplier


if __name__ == "__main__":
    backtest()
