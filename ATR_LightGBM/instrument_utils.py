"""
instrument_utils.py - 合约工具类
提供获取合约乘数、保证金率等通用功能
"""

import pandas as pd
import os
import logging

# 全局缓存，存储合约乘数和保证金率信息
_instrument_info_cache = None

def get_contract_multiplier(symbol):
    """获取合约乘数和保证金率
    
    参数：
    symbol: 合约代码
    
    返回：
    multiplier: 合约乘数
    margin_ratio: 保证金率
    """
    global _instrument_info_cache
    
    logger = logging.getLogger(__name__)
    
    # 如果缓存为空，读取合约信息
    if _instrument_info_cache is None:
        # 构建合约信息文件路径
        # 从Market_Inform目录读取
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(parent_dir, 'Market_Inform', 'all_instruments_info.csv')
        
        # 如果文件存在，从文件中读取并缓存
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # 构建合约信息字典，使用小写sec_id作为键，确保大小写不敏感
            _instrument_info_cache = {}
            for _, row in df.iterrows():
                # 使用小写sec_id作为键，确保大小写不敏感
                sec_id_lower = row['sec_id'].lower()
                _instrument_info_cache[sec_id_lower] = (row['multiplier'], row['margin_ratio'])
            logger.info(f"成功加载合约信息，共 {len(_instrument_info_cache)} 个合约")
        else:
            # 如果文件不存在，初始化空缓存
            _instrument_info_cache = {}
            logger.error(f"合约信息文件不存在: {csv_path}")
    
    # 提取合约代码的基础部分（如IF, IC, IH, A, B等）
    # 处理不同交易所的合约代码格式：
    # - DCE/SHFE: 如 "a2409.DCE" 或 "fu2505.SHFE"（4位数字）
    # - CZCE: 如 "sa505.CZCE"（3位数字）
    # - CFFEX: 如 "IF2409.CFFEX" 或 "IC"
    
    # 先移除交易所后缀，如 ".CZCE"
    symbol_without_exchange = symbol.split('.')[0] if '.' in symbol else symbol
    
    # 查找第一个数字的位置
    first_digit_index = None
    for i, char in enumerate(symbol_without_exchange):
        if char.isdigit():
            first_digit_index = i
            break
    
    if first_digit_index is not None:
        # 提取数字前的字母作为基础品种代码
        base_symbol = symbol_without_exchange[:first_digit_index]
    else:
        # 没有数字，使用完整符号作为基础品种代码
        base_symbol = symbol_without_exchange
    
    # 转换为小写，确保大小写不敏感
    base_symbol_lower = base_symbol.lower()
    
    # 从缓存中获取合约乘数和保证金率，不存在则返回默认值
    if base_symbol_lower in _instrument_info_cache:
        multiplier, margin_ratio = _instrument_info_cache[base_symbol_lower]
        logger.debug(f"合约 {symbol} 的基础代码 {base_symbol} 匹配到 {base_symbol_lower}，乘数: {multiplier}, 保证金率: {margin_ratio}")
        return multiplier, margin_ratio
    else:
        logger.warning(f"合约 {symbol} 的基础代码 {base_symbol}（小写: {base_symbol_lower}）未找到，使用默认值")
        return 10, 0.1  # 默认乘数为10，保证金率为10%
