import sys
import os

sys.path.append('.')

# List of modules to inspect
modules = [
    'calc_funcs',
    'check',
    'data_process',
    'functions',
    'long_short_signals',
    'mom',
    'rules',
    'stock_timing',
    'utils',
    'volatility',
    'daily.position',
    'daily.position_55_7525',
    'cs_source.stock_timing',
    'cs_source.utils'
]

for module_name in modules:
    try:
        print(f"\n=== Inspecting module: {module_name} ===")
        module = __import__(module_name)
        # Handle submodules
        if '.' in module_name:
            components = module_name.split('.')
            for comp in components[1:]:
                module = getattr(module, comp)
        
        # Print module contents
        print(f"Module contents: {dir(module)}")
        
        # Try to find functions and classes
        print("\nFunctions and classes:")
        for item in dir(module):
            if not item.startswith('_'):
                attr = getattr(module, item)
                if callable(attr):
                    print(f"  {item}: callable")
                else:
                    print(f"  {item}: {type(attr).__name__}")
                    
    except Exception as e:
        print(f"Error importing {module_name}: {e}")

print("\n=== Inspection complete ===")