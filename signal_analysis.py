# 分析信号反转逻辑对持仓的影响
import numpy as np

# 模拟模型预测结果
model_predictions = [1, -1, 1, -1]  # 模型预测：1表示上涨，-1表示下跌

print("信号反转逻辑分析：")
print("-" * 50)

for i, pred in enumerate(model_predictions):
    # 原始预测结果
    original_signal = pred
    
    # 反转后的信号
    reversed_signal = -pred
    
    # 计算目标手数（假设分配资金计算出的手数为10）
    allocated_quantity = 10
    target_quantity_original = allocated_quantity * original_signal
    target_quantity_reversed = allocated_quantity * reversed_signal
    
    # 计算市值
    current_price = 1000
    contract_multiplier = 10
    position_value = allocated_quantity * current_price * contract_multiplier
    market_value_original = position_value if original_signal == 1 else -position_value
    market_value_reversed = position_value if reversed_signal == 1 else -position_value
    
    print(f"\n情况 {i+1}：")
    print(f"  模型预测：{original_signal} (1=上涨, -1=下跌)")
    print(f"  反转后信号：{reversed_signal}")
    print(f"  原始目标手数：{target_quantity_original} (正数=多单, 负数=空单)")
    print(f"  反转后目标手数：{target_quantity_reversed} (正数=多单, 负数=空单)")
    print(f"  原始市值：{market_value_original} (正数=多单市值, 负数=空单市值)")
    print(f"  反转后市值：{market_value_reversed} (正数=多单市值, 负数=空单市值)")

print("\n" + "-" * 50)
print("结论：")
print("1. 信号反转只会改变持仓方向，不会产生'负空单'这种异常情况")
print("2. 空单本身就是负数，反转后会变成正数（多单）")
print("3. 多单本身就是正数，反转后会变成负数（空单）")
print("4. 市值计算也会根据信号方向正确调整")
print("5. 反转信号的目的是纠正模型预测的方向错误，使信号与实际行情一致")
