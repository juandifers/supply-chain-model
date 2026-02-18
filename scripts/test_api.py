# scripts/peek_supplysim_api.py
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
import inspect    
from TGB.modules import synthetic_data
    
print("Module:", synthetic_data.__file__)
for name, obj in vars(synthetic_data).items():
    if callable(obj) and not name.startswith("_"):
        sig = ""
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            pass
        print(f"{name}{sig}")
