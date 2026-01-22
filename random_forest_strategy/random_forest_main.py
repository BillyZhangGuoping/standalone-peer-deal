# -*- coding: utf-8 -*-
"""
随机森林策略入口文件：
调用main_process.py中的main函数，实现模块化拆分后的策略运行
"""
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入主处理模块
from main_process import main

if __name__ == "__main__":
    main()