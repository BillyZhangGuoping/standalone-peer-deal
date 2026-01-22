# -*- coding: utf-8 -*-
"""
资金分配方法的具体实现：
包含各种资金分配方法的具体实现，都继承自BaseAllocationMethod接口
"""
import sys
import os

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interfaces import BaseAllocationMethod

# 导入风险平价分配模块
from risk_parity_allocation import risk_parity_allocation


class RiskParityAllocation(BaseAllocationMethod):
    """风险平价资金分配方法的实现
    
    封装了risk_parity_allocation.py的功能
    """
    
    def allocate(self, capital, varieties_data, date, all_data=None):
        """根据风险平价方法分配资金
        
        参数：
        capital: 总资金
        varieties_data: 品种数据字典
        date: 指定日期，格式为datetime对象
        all_data: 所有品种的历史数据，格式为：
            {品种代码: pd.DataFrame(历史数据)}
        
        返回：
        allocation_dict: 资金分配字典，格式为 {品种代码: 分配资金}
        risk_units: 风险单位字典，格式为 {品种代码: 风险单位}
        """
        try:
            # 调用risk_parity_allocation.py的risk_parity_allocation函数分配资金
            allocation_dict, risk_units = risk_parity_allocation(
                capital=capital,
                varieties_data=varieties_data,
                date=date,
                all_data=all_data
            )
            
            return allocation_dict, risk_units
        except Exception as e:
            raise Exception(f"风险平价分配方法失败: {e}")
