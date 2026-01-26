#!/usr/bin/env bash
# Minimal reproduction script to debug Transformer loading hang in runner context
#
# Runs GX1DemoRunner init with minimal data (1-2 days) to see if Transformer loading hangs.
# Uses fail-fast debugging (faulthandler) to get stack trace if it hangs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

POLICY_THRESHOLD018="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_0_18.yaml"
DATA_FILE="data/raw/xauusd_m5_2025_bid_ask.parquet"

# Minimal test data (2 days UTC) - just enough to trigger model loading
START_TS="2025-01-01T00:00:00Z"
END_TS="2025-01-03T00:00:00Z"  # 2 days

LOG_FILE="/tmp/repro_transformer_hang.log"

echo "=================================================================================="
echo "REPRO: Transformer Loading Hang in Runner Context"
echo "=================================================================================="
echo ""
echo "Configuration:"
echo "  - Policy: $POLICY_THRESHOLD018"
echo "  - Period: 2 days (${START_TS} → ${END_TS})"
echo "  - Workers: 1 (single worker, no parallel chunks)"
echo "  - Log level: DEBUG"
echo "  - Log file: $LOG_FILE"
echo ""
echo "This script will:"
echo "  1. Load GX1DemoRunner (triggers Transformer loading)"
echo "  2. Exit immediately after model loading (no full replay)"
echo "  3. Use faulthandler to dump stack trace if it hangs >20s"
echo ""

# Set log level
export GX1_LOGLEVEL=DEBUG
# Force CPU to avoid MPS issues
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Validate files exist
if [ ! -f "$POLICY_THRESHOLD018" ]; then
    echo "❌ ERROR: Policy file not found: $POLICY_THRESHOLD018"
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "❌ ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Prepare minimal test data
echo "[1/2] Preparing minimal test data (2 days, EU/OVERLAP/US only)..."
TEST_DATA_DIR="data/temp/repro_transformer_hang_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DATA_DIR"

python3 -c "
import pandas as pd
from pathlib import Path
import sys

data_file = sys.argv[1]
start_ts = sys.argv[2]
end_ts = sys.argv[3]
output_file = sys.argv[4]

from gx1.execution.live_features import infer_session_tag

df = pd.read_parquet(data_file)
df.index = pd.to_datetime(df.index, utc=True)

df = df[(df.index >= start_ts) & (df.index < end_ts)].copy()

# Filter to SNIPER sessions
sessions = ['EU', 'OVERLAP', 'US']
mask = df.index.map(lambda ts: infer_session_tag(ts) in sessions)
df = df[mask]

print(f'✅ Test data: {len(df):,} rows')
print(f'   Date range: {df.index.min()} → {df.index.max()}')

df.to_parquet(output_file)
print(f'✅ Saved to: {output_file}')
" "$DATA_FILE" "$START_TS" "$END_TS" "$TEST_DATA_DIR/test_data.parquet" 2>&1 | tee -a "$LOG_FILE"

TEST_DATA="$TEST_DATA_DIR/test_data.parquet"

# Run minimal init test (just load runner, exit after model loading)
echo ""
echo "[2/2] Loading GX1DemoRunner (this triggers Transformer loading)..."
echo "   If it hangs >20s, faulthandler will dump stack trace to stderr/log"
echo ""

OUTPUT_DIR="$TEST_DATA_DIR/output"
mkdir -p "$OUTPUT_DIR"

python3 << PYEOF 2>&1 | tee -a "$LOG_FILE"
import sys
import faulthandler
from pathlib import Path

# Enable faulthandler for this script too
faulthandler.enable()

from gx1.execution.oanda_demo_runner import GX1DemoRunner

policy_path = Path("$POLICY_THRESHOLD018")
data_file = Path("$TEST_DATA")
output_dir = Path("$OUTPUT_DIR")

print("=" * 80)
print("CREATING GX1DemoRunner - This will trigger Transformer loading...")
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
    print("=" * 80)
    print("")
    print("Transformer loading completed. Exiting (no full replay).")
    sys.stdout.flush()
    
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
    echo "✅ REPRO COMPLETE - No hang detected"
else
    echo "❌ REPRO FAILED - Exit code: $EXIT_CODE"
fi
echo "=================================================================================="
echo ""
echo "Log file: $LOG_FILE"
echo ""
echo "If it hung, check log file for faulthandler stack trace (should appear after ~20s)"
echo ""

exit $EXIT_CODE

