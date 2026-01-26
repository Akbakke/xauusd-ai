#!/usr/bin/env bash
# Test 2: Mini replay (1 week) with GX1_FORCE_TORCH_DEVICE=cpu, 1 worker
# Mål: Bevise at alt plumbing rundt runner + dataset + journaling + perf fungerer

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

POLICY_THRESHOLD018="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_0_18.yaml"
DATA_FILE="data/raw/xauusd_m5_2025_bid_ask.parquet"

# 1 week test (UTC)
START_TS="2025-01-01T00:00:00Z"
END_TS="2025-01-08T00:00:00Z"  # 1 week

LOG_FILE="/tmp/test_v10_1_mini_replay.log"

echo "=================================================================================="
echo "TEST 2: Mini Replay (1 week) - V10.1 with Performance Tracking"
echo "=================================================================================="
echo ""
echo "Configuration:"
echo "  - Policy: $POLICY_THRESHOLD018"
echo "  - Period: 1 week (${START_TS} → ${END_TS})"
echo "  - Workers: 1 (single worker, no parallel chunks)"
echo "  - Force CPU: GX1_FORCE_TORCH_DEVICE=cpu"
echo "  - Log level: INFO"
echo "  - Log file: $LOG_FILE"
echo ""

export GX1_LOGLEVEL=INFO
export GX1_FORCE_TORCH_DEVICE=cpu

# Validate files exist
if [ ! -f "$POLICY_THRESHOLD018" ]; then
    echo "❌ ERROR: Policy file not found: $POLICY_THRESHOLD018"
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "❌ ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Prepare test data
echo "[1/3] Preparing test data (1 week, EU/OVERLAP/US only)..."
TEST_DATA_DIR="data/temp/test_v10_1_mini_replay_$(date +%Y%m%d_%H%M%S)"
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

# Run replay (single worker, no parallel chunks)
echo ""
echo "[2/3] Running mini replay with performance timing (single worker)..."
echo "   This should take ~5-15 minutes..."
echo ""

OUTPUT_DIR="$TEST_DATA_DIR/output"
mkdir -p "$OUTPUT_DIR"

python3 << PYEOF 2>&1 | tee -a "$LOG_FILE"
import sys
from pathlib import Path
from gx1.execution.oanda_demo_runner import GX1DemoRunner

policy_path = Path("$POLICY_THRESHOLD018")
data_file = Path("$TEST_DATA")
output_dir = Path("$OUTPUT_DIR")

print("=" * 80)
print("RUNNING MINI REPLAY (1 week, 1 worker)")
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
    
    print(f"entry_v10_enabled: {runner.entry_v10_enabled}")
    if runner.entry_v10_bundle:
        print(f"device: {runner.entry_v10_bundle.device}")
    sys.stdout.flush()
    
    if not runner.entry_v10_enabled:
        print("❌ FAIL: entry_v10_enabled=False")
        sys.exit(1)
    
    print("Starting replay...")
    sys.stdout.flush()
    
    runner.run_replay(data_file)
    
    print("")
    print("=" * 80)
    print("✅ Replay complete!")
    print("=" * 80)
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

# Check for performance summary
echo ""
echo "[3/3] Verifying output..."
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    # Check for performance summary
    if grep -q "\[REPLAY_PERF_SUMMARY\]" "$LOG_FILE"; then
        echo "✅ Found [REPLAY_PERF_SUMMARY] in log"
        echo ""
        echo "Performance Summary:"
        grep "\[REPLAY_PERF_SUMMARY\]" "$LOG_FILE" | tail -1
        echo ""
    else
        echo "⚠️  WARNING: No [REPLAY_PERF_SUMMARY] found in log"
    fi
    
    # Check for trade journal
    TRADE_JOURNAL="$OUTPUT_DIR/trade_journal"
    if [ -d "$TRADE_JOURNAL" ]; then
        echo "✅ Trade journal directory exists: $TRADE_JOURNAL"
        
        # Check for merged parquet or JSON trades
        if [ -f "$TRADE_JOURNAL/merged_trade_journal.parquet" ]; then
            echo "✅ Found merged_trade_journal.parquet"
        elif [ -d "$TRADE_JOURNAL/trades" ]; then
            TRADE_COUNT=$(find "$TRADE_JOURNAL/trades" -name "trade_*.json" 2>/dev/null | wc -l | tr -d ' ')
            echo "✅ Found $TRADE_COUNT trade JSON files"
        fi
    else
        echo "⚠️  WARNING: Trade journal directory not found"
    fi
    
    # Check for required log lines
    echo ""
    echo "Verifying required log lines..."
    if grep -q "entry_v10_enabled.*True" "$LOG_FILE"; then
        echo "✅ entry_v10_enabled=True found"
    else
        echo "❌ entry_v10_enabled=True NOT found"
    fi
    
    if grep -q "p_long" "$LOG_FILE"; then
        echo "✅ p_long found in log (V10.1 inference working)"
    else
        echo "⚠️  WARNING: p_long not found in log"
    fi
fi

echo ""
echo "=================================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TEST 2 PASSED"
else
    echo "❌ TEST 2 FAILED - Exit code: $EXIT_CODE"
fi
echo "=================================================================================="
echo ""
echo "Log file: $LOG_FILE"
echo "Output dir: $OUTPUT_DIR"
echo ""

exit $EXIT_CODE

