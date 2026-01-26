#!/usr/bin/env python3
"""Test script to check calibrator imports."""

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_ROOT))

print("Testing imports...")

try:
    import numpy as np
    print("✅ numpy")
except Exception as e:
    print(f"❌ numpy: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print("✅ pandas")
except Exception as e:
    print(f"❌ pandas: {e}")
    sys.exit(1)

try:
    import joblib
    print("✅ joblib")
except Exception as e:
    print(f"❌ joblib: {e}")
    sys.exit(1)

try:
    from scipy.optimize import minimize
    from scipy.special import expit
    print("✅ scipy")
except Exception as e:
    print(f"❌ scipy: {e}")
    sys.exit(1)

try:
    from gx1.xgb.calibration import PlattScaler, IsotonicScaler, QuantileClipper
    print("✅ gx1.xgb.calibration")
except Exception as e:
    print(f"❌ gx1.xgb.calibration: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All imports OK!")
