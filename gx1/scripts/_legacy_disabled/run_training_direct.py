#!/usr/bin/env python3
"""
Direct runner that bypasses shell issues.
"""

import os
import sys
import subprocess
from pathlib import Path

# Get workspace root
workspace_root = Path(__file__).resolve().parent
os.chdir(str(workspace_root))

# Find Python
python_exe = sys.executable

print(f"Python: {python_exe}")
print(f"Working dir: {os.getcwd()}")
print()

# Build command
script_path = workspace_root / "gx1" / "scripts" / "train_xgb_calibrator_multiyear.py"

cmd = [
    python_exe,
    str(script_path),
    "--years", "2020", "2021", "2022", "2023", "2024", "2025",
    "--calibrator-type", "platt",
    "--n-samples-per-year", "50000",
]

print(f"Running: {' '.join(cmd)}")
print()

# Run with proper environment
env = os.environ.copy()
env["PYTHONPATH"] = str(workspace_root) + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

try:
    result = subprocess.run(
        cmd,
        cwd=str(workspace_root),
        env=env,
        check=False,
        capture_output=False,  # Show output in real-time
    )
    sys.exit(result.returncode)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
