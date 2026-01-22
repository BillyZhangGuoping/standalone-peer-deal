# -*- coding: utf-8 -*-
"""
标准化接口定义模块：
定义趋势信号生成和资金分配的标准接口
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseTrendModel(ABC):
    """趋势信号生成的基础接口
    
    所有趋势信号模型都需要继承此类并实现抽象方法
    """
    
    def __init__(self, params=None):
        """初始化趋势模型
        
        参数：
        params: 模型参数
        """
        self.params = params or {}
    
    @abstractmethod
    def generate_signals(self, data, date, variety_list):
        """生成指定日期的趋势信号
        
        参数：
        data: 历史数据，格式为pd.DataFrame，索引为日期，列为品种，值为收盘价
        date: 指定日期，格式为datetime对象
        variety_list: 品种列表
        
        返回：
        signals: 趋势信号，格式为pd.DataFrame，包含列：
            - 品种: 品种代码
            - 趋势方向: 趋势方向，取值为-1（下跌）、0（无趋势）、1（上涨）
            - 信号强度: 信号强度，取值范围为0到1
        """
        pass
    
    def __str__(self):
        """返回模型名称"""
        return self.__class__.__name__


class BaseAllocationMethod(ABC):
    """资金分配方法的基础接口
    
    所有资金分配方法都需要继承此类并实现抽象方法
    """
    
    def __init__(self, params=None):
        """初始化资金分配方法
        
        参数：
        params: 分配方法参数
        """
        self.params = params or {}
    
    @abstractmethod
    def allocate(self, capital, varieties_data, date, all_data=None):
        """根据品种数据和日期分配资金
        
        参数：
        capital: 总资金
        varieties_data: 品种数据字典，格式为：
            {品种代码: {
                'current_price': 当前价格,
                'contract_multiplier': 合约乘数,
                'margin_rate': 保证金率,
                'contract_symbol': 合约代码,
                'signal': 趋势方向,
                'trend_strength': 信号强度,
                'prices': 价格历史数据
            }}
        date: 指定日期，格式为datetime对象
        all_data: 所有品种的历史数据，格式为：
            {品种代码: pd.DataFrame(历史数据)}
        
        返回：
        allocation_dict: 资金分配字典，格式为 {品种代码: 分配资金}
        risk_units: 风险单位字典，格式为 {品种代码: 风险单位}
        """
        pass
    
    def __str__(self):
        """返回分配方法名称"""
        return self.__class__.__name__


class SignalFusion(BaseTrendModel):
    """信号融合模型，用于融合多种趋势信号
    
    可以将不同趋势模型的信号按照权重融合
    """
    
    def __init__(self, models=None, params=None):
        """初始化信号融合模型
        
        参数：
        models: 趋势模型列表，格式为：
            [{"model": BaseTrendModel实例, "weight": 权重}]
        params: 融合参数
        """
        super().__init__(params)
        self.models = models or []
    
    def add_model(self, model, weight=1.0):
        """添加趋势模型
        
        参数：
        model: BaseTrendModel实例
        weight: 模型权重
        """
        self.models.append({"model": model, "weight": weight})
    
    def generate_signals(self, data, date, variety_list):
        """生成融合后的趋势信号
        
        参数：
        data: 历史数据，格式为pd.DataFrame，索引为日期，列为品种，值为收盘价
        date: 指定日期，格式为datetime对象
        variety_list: 品种列表
        
        返回：
        signals: 融合后的趋势信号，格式为pd.DataFrame
        """
        if not self.models:
            raise ValueError("信号融合模型中没有添加任何趋势模型")
        
        # 生成所有模型的信号
        all_signals = []
        total_weight = 0.0
        
        for model_info in self.models:
            model = model_info["model"]
            weight = model_info["weight"]
            
            try:
                signals = model.generate_signals(data, date, variety_list)
                signals["weight"] = weight
                all_signals.append(signals)
                total_weight += weight
            except Exception as e:
                print(f"模型{model}生成信号失败: {e}")
                continue
        
        if not all_signals:
            raise ValueError("所有趋势模型生成信号失败")
        
        # 归一化权重
        for signals in all_signals:
            signals["weight"] = signals["weight"] / total_weight
        
        # 融合信号
        fusion_df = pd.DataFrame({"品种": variety_list})
        fusion_df["趋势方向"] = 0
        fusion_df["信号强度"] = 0.0
        
        for signals in all_signals:
            # 确保信号数据包含所有品种
            signals = pd.merge(fusion_df[["品种"]], signals, on="品种", how="left")
            signals["趋势方向"] = signals["趋势方向"].fillna(0)
            signals["信号强度"] = signals["信号强度"].fillna(0.0)
            
            # 加权融合
            fusion_df["趋势方向"] += signals["趋势方向"] * signals["weight"]
            fusion_df["信号强度"] += signals["信号强度"] * signals["weight"]
        
        # 处理融合结果
        fusion_df["趋势方向"] = fusion_df["趋势方向"].apply(lambda x: int(round(x)) if abs(x) >= 0.5 else 0)
        fusion_df["信号强度"] = fusion_df["信号强度"].clip(0.0, 1.0)
        
        return fusion_df
    
    def __str__(self):
        """返回模型名称和包含的模型"""
        model_names = [str(model_info["model"]) for model_info in self.models]
        return f"SignalFusion({', '.join(model_names)})"
