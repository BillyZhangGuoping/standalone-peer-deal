import sys
import os
import ctypes
import inspect

# 添加当前目录到路径
sys.path.append('.')

def analyze_pyd_file(file_path):
    """分析.pyd文件的结构和导出函数"""
    print(f"\n=== 分析文件: {file_path} ===")
    
    try:
        # 尝试直接导入模块
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"尝试导入模块: {module_name}")
        
        # 动态导入模块
        module = __import__(module_name)
        
        # 打印模块基本信息
        print(f"模块类型: {type(module)}")
        print(f"模块文档: {module.__doc__}")
        
        # 打印模块内容
        print(f"\n模块内容: {dir(module)}")
        
        # 分析导出的函数和类
        print("\n导出的函数和类:")
        for item in dir(module):
            if not item.startswith('_'):
                attr = getattr(module, item)
                if callable(attr):
                    print(f"  {item}: callable")
                    # 尝试获取函数签名
                    try:
                        sig = inspect.signature(attr)
                        print(f"    签名: {sig}")
                    except Exception as e:
                        print(f"    无法获取签名: {e}")
                else:
                    print(f"  {item}: {type(attr).__name__}")
                    print(f"    值: {attr}")
                    
    except Exception as e:
        print(f"导入模块时出错: {e}")
        
        # 尝试使用ctypes加载
        try:
            print("\n尝试使用ctypes加载...")
            dll = ctypes.CDLL(file_path)
            print("使用ctypes加载成功")
            # 注意：无法直接获取Python扩展模块的导出函数
            # 因为它们使用了特殊的命名约定
            print("提示: 无法直接通过ctypes获取Python扩展的导出函数")
        except Exception as e2:
            print(f"ctypes加载失败: {e2}")

# 分析所有.pyd文件
pyd_files = [
    "calc_funcs.pyd",
    "check.pyd",
    "data_process.pyd",
    "functions.pyd",
    "long_short_signals.pyd",
    "mom.pyd",
    "rules.pyd",
    "stock_timing.pyd",
    "utils.pyd",
    "volatility.pyd",
    "daily/position.pyd",
    "cs_source/stock_timing.pyd",
    "cs_source/utils.pyd"
]

for pyd_file in pyd_files:
    if os.path.exists(pyd_file):
        analyze_pyd_file(pyd_file)
    else:
        print(f"\n文件不存在: {pyd_file}")

print("\n=== 分析完成 ===")