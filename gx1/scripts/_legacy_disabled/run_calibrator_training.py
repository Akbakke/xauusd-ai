#!/usr/bin/env python3
"""
Wrapper script to run calibrator training with proper environment setup.
"""

import os
import sys
import subprocess
from pathlib import Path

# Get workspace root
workspace_root = Path(__file__).resolve().parent

# Change to workspace root
os.chdir(str(workspace_root))

# Add to Python path
sys.path.insert(0, str(workspace_root))

# Find Python interpreter
python_exe = sys.executable

print(f"Workspace: {workspace_root}")
print(f"Python: {python_exe}")
print(f"Python version: {sys.version}")
print()

# Check dependencies
print("Checking dependencies...")
try:
    import numpy
    print(f"✅ numpy {numpy.__version__}")
except ImportError as e:
    print(f"❌ numpy: {e}")
    sys.exit(1)

try:
    import pandas
    print(f"✅ pandas {pandas.__version__}")
except ImportError as e:
    print(f"❌ pandas: {e}")
    sys.exit(1)

try:
    import scipy
    print(f"✅ scipy {scipy.__version__}")
except ImportError as e:
    print(f"❌ scipy: {e}")
    print("Installing scipy...")
    subprocess.check_call([python_exe, "-m", "pip", "install", "scipy"])
    import scipy
    print(f"✅ scipy {scipy.__version__} (installed)")

try:
    import sklearn
    print(f"✅ sklearn {sklearn.__version__}")
except ImportError as e:
    print(f"❌ sklearn: {e}")
    print("Installing scikit-learn...")
    subprocess.check_call([python_exe, "-m", "pip", "install", "scikit-learn"])
    import sklearn
    print(f"✅ sklearn {sklearn.__version__} (installed)")

try:
    import joblib
    print(f"✅ joblib")
except ImportError as e:
    print(f"❌ joblib: {e}")
    print("Installing joblib...")
    subprocess.check_call([python_exe, "-m", "pip", "install", "joblib"])
    import joblib
    print(f"✅ joblib (installed)")

try:
    import pyarrow
    print(f"✅ pyarrow {pyarrow.__version__}")
except ImportError as e:
    print(f"❌ pyarrow: {e}")
    print("Installing pyarrow...")
    subprocess.check_call([python_exe, "-m", "pip", "install", "pyarrow"])
    import pyarrow
    print(f"✅ pyarrow {pyarrow.__version__} (installed)")

print()
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

# Import and run main
from gx1.scripts.train_xgb_calibrator_multiyear import main

# Set sys.argv
sys.argv = [
    "train_xgb_calibrator_multiyear.py",
    "--years", "2020", "2021", "2022", "2023", "2024", "2025",
    "--calibrator-type", "platt",
    "--n-samples-per-year", "50000",
]

# Run main
sys.exit(main())
