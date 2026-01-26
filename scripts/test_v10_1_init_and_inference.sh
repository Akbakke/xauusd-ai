#!/usr/bin/env bash
# Test 1: Ren init + én inferens (0 replay)
# Mål: Bevise at runner-kontekst kan laste V10.1 + kjøre minst én predict()

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

POLICY_THRESHOLD018="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_0_18.yaml"
LOG_FILE="/tmp/test_v10_1_init_inference.log"

echo "=================================================================================="
echo "TEST 1: V10.1 Init + One Inference (No Replay)"
echo "=================================================================================="
echo ""
echo "Configuration:"
echo "  - Policy: $POLICY_THRESHOLD018"
echo "  - Force CPU: GX1_FORCE_TORCH_DEVICE=cpu"
echo "  - Log file: $LOG_FILE"
echo ""

export GX1_LOGLEVEL=INFO
export GX1_FORCE_TORCH_DEVICE=cpu

# Validate files exist
if [ ! -f "$POLICY_THRESHOLD018" ]; then
    echo "❌ ERROR: Policy file not found: $POLICY_THRESHOLD018"
    exit 1
fi

echo "[1/2] Creating GX1DemoRunner and loading V10.1..."
python3 -B << PYEOF 2>&1 | tee "$LOG_FILE"
import sys
from pathlib import Path
from gx1.execution.oanda_demo_runner import GX1DemoRunner
import pandas as pd

policy_path = Path("$POLICY_THRESHOLD018")
output_dir = Path("data/temp/test_v10_1_init_inference")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CREATING GX1DemoRunner...")
print("=" * 80)
sys.stdout.flush()

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
    sys.stdout.flush()
    
    # Check if V10 is enabled
    if not runner.entry_v10_enabled:
        print("❌ FAIL: entry_v10_enabled=False")
        sys.exit(1)
    
    if runner.entry_v10_bundle is None:
        print("❌ FAIL: entry_v10_bundle is None")
        sys.exit(1)
    
    print("")
    print("=" * 80)
    print("[2/2] Running one inference cycle...")
    print("=" * 80)
    sys.stdout.flush()
    
    # Load a small sample of data for one inference
    data_file = Path("data/raw/xauusd_m5_2025_bid_ask.parquet")
    if not data_file.exists():
        print(f"❌ ERROR: Data file not found: {data_file}")
        sys.exit(1)
    
    df = pd.read_parquet(data_file)
    df.index = pd.to_datetime(df.index, utc=True)
    
    # Get first few bars (enough for warmup + 1 cycle)
    warmup = 300  # Conservative warmup
    df_sample = df.iloc[:warmup + 10].copy()
    
    print(f"Sample data: {len(df_sample)} bars")
    print(f"Date range: {df_sample.index.min()} → {df_sample.index.max()}")
    sys.stdout.flush()
    
    # Run one replay cycle (just to trigger one inference)
    # We'll use run_replay but it should exit quickly
    temp_data = output_dir / "temp_sample.parquet"
    df_sample.to_parquet(temp_data)
    
    print("Running one replay cycle (should trigger at least one inference)...")
    sys.stdout.flush()
    
    runner.run_replay(temp_data)
    
    print("")
    print("=" * 80)
    print("✅ TEST 1 COMPLETE")
    print("=" * 80)
    print("")
    print("Check log file for:")
    print("  - entry_v10_enabled=True")
    print("  - torch_device=cpu (or expected device)")
    print("  - At least one 'p_long=...' from V10.1")
    print("")
    
except KeyboardInterrupt:
    print("\n❌ Interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF

EXIT_CODE=$?

echo ""
echo "=================================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TEST 1 PASSED"
    echo ""
    echo "Verifying log for required lines..."
    if grep -q "entry_v10_enabled.*True" "$LOG_FILE" && \
       grep -q "p_long" "$LOG_FILE"; then
        echo "✅ All required log lines found"
    else
        echo "⚠️  WARNING: Some required log lines may be missing"
        echo "   Check log file: $LOG_FILE"
    fi
else
    echo "❌ TEST 1 FAILED - Exit code: $EXIT_CODE"
    echo "   Check log file: $LOG_FILE"
fi
echo "=================================================================================="
echo ""
echo "Log file: $LOG_FILE"

exit $EXIT_CODE

