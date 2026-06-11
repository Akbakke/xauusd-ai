#!/usr/bin/env python3
"""Execute find_prebuilt_files.py via subprocess."""
import subprocess
import sys
from pathlib import Path

workspace = Path(__file__).resolve().parent
script_path = workspace / "run_find_prebuilt.py"

result = subprocess.run(
    [sys.executable, str(script_path)],
    cwd=str(workspace),
    capture_output=True,
    text=True,
)

print(result.stdout)
if result.stderr:
    print("STDERR:", file=sys.stderr)
    print(result.stderr, file=sys.stderr)

sys.exit(result.returncode)
