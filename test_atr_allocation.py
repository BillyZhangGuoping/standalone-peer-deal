#!/usr/bin/env python3
"""
测试基于ATR的资金分配逻辑
"""

import sys
import os

# 添加必要的路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'random_forest_strategy'))

from random_forest_strategy.risk_allocation import calculate_atr_allocation

def test_atr_allocation():
    """测试ATR资金分配"""
    # 测试案例：
    # 总资金：10,000,000元
    # 3个品种，ATR不同
    
    total_capital = 10000000
    
    # 模拟3个品种的数据
    varieties_data = {
        'rb': {
            'current_price': 5000,
            'atr': 100,  # 高波动率
            'contract_multiplier': 10
        },
        'ma': {
            'current_price': 2000,
            'atr': 50,  # 中等波动率
            'contract_multiplier': 10
        },
        'ag': {
            'current_price': 5000,
            'atr': 10,  # 低波动率
            'contract_multiplier': 15
        }
    }
    
    print(f"测试参数：")
    print(f"- 总资金：{total_capital}元")
    print(f"- 目标波动率：1%")
    print(f"- 品种数量：{len(varieties_data)}")
    
    for symbol, data in varieties_data.items():
        print(f"  - {symbol}: 价格={data['current_price']}, ATR={data['atr']}, 乘数={data['contract_multiplier']}")
    
    # 调用ATR分配函数
    allocation, risk_units = calculate_atr_allocation(total_capital, varieties_data, target_volatility=0.01)
    
    print(f"\nATR资金分配结果：")
    print(f"- 总分配资金：{sum(allocation.values()):.2f}元")
    
    for symbol, allocated_capital in allocation.items():
        # 计算ATR价值
        atr_value = varieties_data[symbol]['atr'] * varieties_data[symbol]['contract_multiplier']
        
        # 计算风险单位数
        risk_unit = risk_units[symbol]
        
        # 计算预期每日波动
        expected_daily_move = risk_unit * atr_value
        
        print(f"- {symbol}:")
        print(f"  * 分配资金：{allocated_capital:.2f}元")
        print(f"  * ATR价值：{atr_value:.2f}元/手")
        print(f"  * 风险单位数：{risk_unit:.2f}手")
        print(f"  * 预期每日波动：{expected_daily_move:.2f}元")
        print(f"  * 占总资金比例：{allocated_capital/total_capital:.2%}")
    
    # 验证结果
    # 1. 总分配资金应接近总资金
    total_allocated = sum(allocation.values())
    print(f"\n验证结果：")
    print(f"1. 总分配资金是否接近总资金：{abs(total_allocated - total_capital) < 1000}")  # 允许小误差
    
    # 2. 预期每日波动应接近目标波动率
    target_daily_move = total_capital * 0.01
    print(f"2. 目标每日波动：{target_daily_move:.2f}元")
    
    for symbol, allocated_capital in allocation.items():
        atr_value = varieties_data[symbol]['atr'] * varieties_data[symbol]['contract_multiplier']
        risk_unit = risk_units[symbol]
        expected_daily_move = risk_unit * atr_value
        print(f"   - {symbol} 预期每日波动：{expected_daily_move:.2f}元")
    
    print("\n✅ ATR资金分配测试完成！")

if __name__ == "__main__":
    test_atr_allocation()
