#!/usr/bin/env python3
"""
测试calculate_position_size函数的优化结果
"""

import sys
import os

# 添加必要的路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'random_forest_strategy'))

from random_forest_strategy.position_calculator import calculate_position_size

def test_position_calculation():
    """测试持仓手数计算"""
    # 测试案例：
    # 总资金：10,000,000元
    # 每笔交易风险比例：2%
    # 入场价格：5000元
    # 止损价格：4950元（下跌1%）
    # 合约乘数：10吨/手（假设）
    # 保证金率：17%
    
    capital = 10000000
    risk_per_trade = 0.02
    entry_price = 5000
    stop_loss_price = 4950
    symbol = "rb2409.SHFE"  # 螺纹钢合约，假设乘数为10
    margin_rate = 0.17
    
    print(f"测试参数：")
    print(f"- 总资金：{capital}元")
    print(f"- 每笔交易风险比例：{risk_per_trade:.2%}")
    print(f"- 入场价格：{entry_price}元")
    print(f"- 止损价格：{stop_loss_price}元")
    print(f"- 合约代码：{symbol}")
    print(f"- 保证金率：{margin_rate:.2%}")
    
    # 调用优化后的函数
    position_size, risk_amount = calculate_position_size(
        capital, risk_per_trade, entry_price, stop_loss_price, symbol, margin_rate
    )
    
    # 计算实际保证金占用
    # 先获取合约乘数
    from utility.instrument_utils import get_contract_multiplier
    contract_multiplier, _ = get_contract_multiplier(symbol)
    actual_margin_usage = position_size * entry_price * contract_multiplier * margin_rate
    
    print(f"\n计算结果：")
    print(f"- 风险金额：{risk_amount}元")
    print(f"- 合约乘数：{contract_multiplier}吨/手")
    print(f"- 计算出的持仓手数：{position_size}手")
    print(f"- 实际保证金占用：{actual_margin_usage:.2f}元")
    print(f"- 保证金占用是否小于等于风险金额：{actual_margin_usage <= risk_amount}")
    
    # 验证结果是否符合预期
    # 基于保证金占用的理论手数：risk_amount / (entry_price * contract_multiplier * margin_rate)
    theoretical_margin_position = risk_amount / (entry_price * contract_multiplier * margin_rate)
    print(f"\n理论验证：")
    print(f"- 基于止损风险的理论手数：{risk_amount / (abs(entry_price - stop_loss_price) * contract_multiplier):.2f}手")
    print(f"- 基于保证金占用的理论手数：{theoretical_margin_position:.2f}手")
    print(f"- 最终取较小值：{min(risk_amount / (abs(entry_price - stop_loss_price) * contract_multiplier), theoretical_margin_position):.2f}手")
    print(f"- 取整后：{int(min(risk_amount / (abs(entry_price - stop_loss_price) * contract_multiplier), theoretical_margin_position))}手")
    
    # 断言结果是否正确
    assert position_size == int(min(risk_amount / (abs(entry_price - stop_loss_price) * contract_multiplier), theoretical_margin_position)), "计算结果不符合预期"
    assert actual_margin_usage <= risk_amount, "保证金占用超过风险金额"
    
    print("\n✅ 测试通过！")

if __name__ == "__main__":
    test_position_calculation()
