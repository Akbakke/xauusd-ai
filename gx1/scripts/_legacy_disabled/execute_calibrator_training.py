#!/usr/bin/env python3
"""
Execute calibrator training directly via Python import (bypasses shell issues).
"""

import os
import sys
from pathlib import Path

# Set working directory
workspace_root = Path(__file__).resolve().parent
os.chdir(str(workspace_root))

# Add to path
sys.path.insert(0, str(workspace_root))

# Set environment
os.environ["PYTHONPATH"] = str(workspace_root) + (":" + os.environ.get("PYTHONPATH", "") if os.environ.get("PYTHONPATH") else "")

print(f"Workspace: {workspace_root}")
print(f"Python: {sys.executable}")
print(f"Python version: {sys.version}")
print()

# Check and install dependencies if needed
print("Checking dependencies...")
missing = []

try:
    import numpy
    print(f"✅ numpy {numpy.__version__}")
except ImportError:
    missing.append("numpy")

try:
    import pandas
    print(f"✅ pandas {pandas.__version__}")
except ImportError:
    missing.append("pandas")

try:
    import scipy
    print(f"✅ scipy {scipy.__version__}")
except ImportError:
    missing.append("scipy")

try:
    import sklearn
    print(f"✅ sklearn {sklearn.__version__}")
except ImportError:
    missing.append("scikit-learn")

try:
    import joblib
    print(f"✅ joblib")
except ImportError:
    missing.append("joblib")

try:
    import pyarrow
    print(f"✅ pyarrow {pyarrow.__version__}")
except ImportError:
    missing.append("pyarrow")

if missing:
    print(f"\n⚠️  Missing packages: {missing}")
    print("Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U"] + missing)
    print("✅ Installed")
    print()

# Test gx1 import
print("Testing gx1 imports...")
try:
    from gx1.xgb.calibration import PlattScaler, IsotonicScaler, QuantileClipper
    print("✅ gx1.xgb.calibration")
except Exception as e:
    print(f"❌ gx1.xgb.calibration: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("Running calibrator training...")
print("=" * 60)
print()

# Import and run
from gx1.scripts.train_xgb_calibrator_multiyear import main

# Set argv
sys.argv = [
    "train_xgb_calibrator_multiyear.py",
    "--years", "2020", "2021", "2022", "2023", "2024", "2025",
    "--calibrator-type", "platt",
    "--n-samples-per-year", "50000",
]

# Run
try:
    exit_code = main()
    sys.exit(exit_code)
except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    sys.exit(130)
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
