#!/usr/bin/env python3
"""Run --list-prebuilt directly."""
import sys
import os
from pathlib import Path

workspace_root = Path(__file__).resolve().parent
os.chdir(str(workspace_root))
sys.path.insert(0, str(workspace_root))

from gx1.scripts.train_xgb_calibrator_multiyear import main

sys.argv = [
    "train_xgb_calibrator_multiyear.py",
    "--years", "2020", "2021", "2022", "2023", "2024", "2025",
    "--list-prebuilt",
]

sys.exit(main())
