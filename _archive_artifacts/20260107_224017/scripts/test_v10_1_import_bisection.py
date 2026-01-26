#!/usr/bin/env python3
"""
Import bisection test to find which import "poisons" the process.

Run with: python3 scripts/test_v10_1_import_bisection.py
"""

import faulthandler
import sys
from pathlib import Path

faulthandler.enable()
faulthandler.dump_traceback_later(20, repeat=False)

print("=" * 80)
print("IMPORT BISECTION TEST")
print("=" * 80)
print()

# Simulate the import order that oanda_demo_runner uses
# Based on oanda_demo_runner.py imports

steps = [
    ("STEP 1: Basic imports", lambda: __import__('json') and __import__('logging') and __import__('os')),
    ("STEP 2: numpy", lambda: __import__('numpy')),
    ("STEP 3: pandas", lambda: __import__('pandas')),
    ("STEP 4: torch", lambda: __import__('torch')),
    ("STEP 5: torch.nn", lambda: __import__('torch.nn')),
    ("STEP 6: yaml", lambda: __import__('yaml')),
    ("STEP 7: joblib", lambda: __import__('joblib')),
    ("STEP 8: gx1.features.runtime_v9", lambda: __import__('gx1.features.runtime_v9')),
    ("STEP 9: gx1.execution.live_features", lambda: __import__('gx1.execution.live_features')),
    ("STEP 10: gx1.execution.broker_client", lambda: __import__('gx1.execution.broker_client')),
    ("STEP 11: gx1.execution.entry_manager", lambda: __import__('gx1.execution.entry_manager')),
    ("STEP 12: gx1.execution.exit_manager", lambda: __import__('gx1.execution.exit_manager')),
    ("STEP 13: gx1.models.entry_v10.entry_v10_hybrid_transformer", lambda: __import__('gx1.models.entry_v10.entry_v10_hybrid_transformer')),
    ("STEP 14: gx1.models.entry_v10.entry_v10_bundle", lambda: __import__('gx1.models.entry_v10.entry_v10_bundle')),
    ("STEP 15: gx1.execution.oanda_demo_runner", lambda: __import__('gx1.execution.oanda_demo_runner')),
]

for step_name, import_func in steps:
    print(f"{step_name}...")
    sys.stdout.flush()
    try:
        import_func()
        print(f"  ✅ OK")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        faulthandler.cancel_dump_traceback_later()
        sys.exit(1)
    print()

print("=" * 80)
print("ALL IMPORTS SUCCESSFUL")
print("=" * 80)
print()

print("Now testing EntryV10HybridTransformer creation...")
import torch
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    torch.backends.mps.enabled = False

from gx1.models.entry_v10.entry_v10_hybrid_transformer import EntryV10HybridTransformer

print("Creating EntryV10HybridTransformer...")
sys.stdout.flush()
model = EntryV10HybridTransformer(
    seq_input_dim=16,
    snap_input_dim=88,
    max_seq_len=90,
    variant="v10_1",
    enable_auxiliary_heads=True,
)
print("✅ EntryV10HybridTransformer created successfully")

faulthandler.cancel_dump_traceback_later()
print("✅ ALL TESTS PASSED")

