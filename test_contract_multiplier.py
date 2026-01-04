#!/usr/bin/env python3
"""
Test script to debug the get_contract_multiplier function
"""
import sys
import os

# Add the current directory to the path so we can import the position module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from position import get_contract_multiplier

# Test some sample symbols
test_symbols = [
    "IC2603.CFFEX",
    "IF2603.CFFEX", 
    "IH2603.CFFEX",
    "IM2603.CFFEX",
    "ag2602.SHFE",
    "au2602.SHFE",
    "cu2602.SHFE",
    "rb2602.SHFE",
    "j2605.DCE",
    "m2605.DCE",
    "TA605.CZCE",
    "CF605.CZCE"
]

print("Testing get_contract_multiplier function:")
print("=" * 60)

for symbol in test_symbols:
    multiplier, margin_rate = get_contract_multiplier(symbol)
    print(f"Symbol: {symbol}")
    print(f"  Multiplier: {multiplier}")
    print(f"  Margin Rate: {margin_rate}")
    print("-" * 60)