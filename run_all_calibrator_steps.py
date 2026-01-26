#!/usr/bin/env python3
"""
Run all calibrator steps: list prebuilt, train, then A/B test.
"""

import sys
import os
import subprocess
from pathlib import Path

workspace_root = Path(__file__).resolve().parent
os.chdir(str(workspace_root))
sys.path.insert(0, str(workspace_root))

print("=" * 60)
print("STEP 1: List Prebuilt Mapping")
print("=" * 60)
print()

# Step 1: List prebuilt
sys.argv = [
    "train_xgb_calibrator_multiyear.py",
    "--years", "2020", "2021", "2022", "2023", "2024", "2025",
    "--list-prebuilt",
]

from gx1.scripts.train_xgb_calibrator_multiyear import main as train_main

exit_code = train_main()
if exit_code != 0:
    print(f"\n❌ STEP 1 FAILED: exit code {exit_code}")
    sys.exit(exit_code)

print()
print("=" * 60)
print("STEP 2: Train Calibrator")
print("=" * 60)
print()

# Step 2: Train
sys.argv = [
    "train_xgb_calibrator_multiyear.py",
    "--years", "2020", "2021", "2022", "2023", "2024", "2025",
    "--calibrator-type", "platt",
    "--n-samples-per-year", "50000",
]

exit_code = train_main()
if exit_code != 0:
    print(f"\n❌ STEP 2 FAILED: exit code {exit_code}")
    sys.exit(exit_code)

print()
print("=" * 60)
print("STEP 3: Check Generated Files")
print("=" * 60)
print()

# Find generated files
from gx1.scripts.train_xgb_calibrator_multiyear import resolve_gx1_data_dir
gx1_data = resolve_gx1_data_dir()
calibrators_dir = gx1_data / "models" / "calibrators"

if not calibrators_dir.exists():
    print(f"❌ Calibrators directory not found: {calibrators_dir}")
    sys.exit(1)

# Find latest files
calibrator_files = sorted(calibrators_dir.glob("xgb_calibrator_platt_*.pkl"), reverse=True)
clipper_files = sorted(calibrators_dir.glob("xgb_clipper_*.pkl"), reverse=True)
metadata_files = sorted(calibrators_dir.glob("calibration_metadata_*.json"), reverse=True)

if not calibrator_files:
    print("❌ No calibrator files found")
    sys.exit(1)
if not clipper_files:
    print("❌ No clipper files found")
    sys.exit(1)
if not metadata_files:
    print("❌ No metadata files found")
    sys.exit(1)

calibrator_path = calibrator_files[0]
clipper_path = clipper_files[0]
metadata_path = metadata_files[0]

print(f"✅ Calibrator: {calibrator_path.name}")
print(f"   Size: {calibrator_path.stat().st_size / 1024:.1f} KB")
print(f"✅ Clipper: {clipper_path.name}")
print(f"   Size: {clipper_path.stat().st_size / 1024:.1f} KB")
print(f"✅ Metadata: {metadata_path.name}")

# Read and verify metadata
import json
with open(metadata_path) as f:
    metadata = json.load(f)

print()
print("Metadata verification:")
print(f"  Years included: {metadata.get('years_included', [])}")
print(f"  n_samples_per_year: {metadata.get('n_samples_per_year', 0)}")
print(f"  Outputs calibrated: {metadata.get('outputs_calibrated', [])}")
if metadata.get('clipper_bounds'):
    print(f"  Clipper bounds:")
    for channel, bounds in metadata['clipper_bounds'].items():
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            print(f"    {channel}: [{bounds[0]:.4f}, {bounds[1]:.4f}]")

print()
print("=" * 60)
print("STEP 4: Ready for A/B Test")
print("=" * 60)
print()
print("Run this command:")
print()
print(f"python3 gx1/scripts/run_multiyear_2020_2025_xgb_repair_ab.py \\")
print(f"  --years 2020 2021 2022 2023 2024 2025 \\")
print(f"  --calibrator-path \"{calibrator_path}\" \\")
print(f"  --clipper-path \"{clipper_path}\"")
print()
