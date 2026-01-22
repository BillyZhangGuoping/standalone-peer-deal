# -*- coding: utf-8 -*-
"""
配置管理模块：
负责加载、解析和管理策略配置文件
"""
import json
import os
import sys

class ConfigManager:
    """配置管理器，用于加载和管理策略配置"""
    
    def __init__(self, config_path=None):
        """初始化配置管理器
        
        参数：
        config_path: 配置文件路径，如果为None则使用默认路径
        """
        if config_path is None:
            # 默认配置文件路径
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'config.json'
            )
        
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            raise Exception(f"加载配置文件失败: {e}")
    
    def get_general_config(self):
        """获取通用配置"""
        return self.config.get('general', {})
    
    def get_variety_list(self):
        """获取品种列表"""
        return self.config.get('variety_list', [])
    
    def get_trend_model_config(self):
        """获取趋势模型配置"""
        return self.config.get('trend_model', {})
    
    def get_allocation_config(self):
        """获取资金分配配置"""
        return self.config.get('allocation_method', {})
    
    def save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise Exception(f"保存配置文件失败: {e}")
    
    def update_config(self, key_path, value):
        """更新配置
        
        参数：
        key_path: 配置键路径，如 "general.capital"
        value: 新的配置值
        """
        keys = key_path.split('.')
        config = self.config
        
        for i, key in enumerate(keys[:-1]):
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        self.save_config()

# 全局配置管理器实例
config_manager = None

def get_config():
    """获取全局配置管理器实例"""
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager()
    return config_manager
