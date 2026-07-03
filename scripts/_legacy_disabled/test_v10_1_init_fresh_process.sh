#!/usr/bin/env bash
# Test 1 (fresh process version): Ren init + én inferens
# Runs Python in a completely fresh subprocess to avoid any cached modules

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

POLICY_THRESHOLD018="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_0_18.yaml"
LOG_FILE="/tmp/test_v10_1_init_fresh.log"

echo "=================================================================================="
echo "TEST 1 (Fresh Process): V10.1 Init + One Inference"
echo "=================================================================================="
echo ""

export GX1_LOGLEVEL=INFO
export GX1_FORCE_TORCH_DEVICE=cpu

# Create Python script that runs in fresh process
cat > /tmp/test_v10_1_init_script.py << 'PYSCRIPT'
import sys
from pathlib import Path
from gx1.execution.oanda_demo_runner import GX1DemoRunner

policy_path = Path("gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_0_18.yaml")
output_dir = Path("data/temp/test_v10_1_init_fresh")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CREATING GX1DemoRunner (fresh process)...")
print("=" * 80)

try:
    runner = GX1DemoRunner(
        policy_path,
        dry_run_override=True,
        replay_mode=True,
        fast_replay=True,
        output_dir=output_dir,
    )
    
    print("=" * 80)
    print("✅ GX1DemoRunner created successfully!")
    print(f"entry_v10_enabled: {runner.entry_v10_enabled}")
    print(f"entry_v10_bundle: {runner.entry_v10_bundle is not None}")
    if runner.entry_v10_bundle:
        print(f"transformer_model: {runner.entry_v10_bundle.transformer_model is not None}")
        print(f"device: {runner.entry_v10_bundle.device}")
    print("=" * 80)
    
    if not runner.entry_v10_enabled:
        print("❌ FAIL: entry_v10_enabled=False")
        sys.exit(1)
    
    if runner.entry_v10_bundle is None:
        print("❌ FAIL: entry_v10_bundle is None")
        sys.exit(1)
    
    print("✅ TEST 1 PASSED")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYSCRIPT

# Run in fresh Python process (no cached modules)
echo "Running in fresh Python process..."
cd "$PROJECT_ROOT"
timeout 60 python3 -B /tmp/test_v10_1_init_script.py 2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

echo ""
echo "=================================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TEST 1 PASSED"
else
    echo "❌ TEST 1 FAILED - Exit code: $EXIT_CODE"
fi
echo "=================================================================================="
echo ""
echo "Log file: $LOG_FILE"

exit $EXIT_CODE

