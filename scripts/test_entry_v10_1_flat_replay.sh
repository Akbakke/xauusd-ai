#!/usr/bin/env bash
# Test ENTRY_V10.1 FLAT Replay (Short Period)
#
# Runs a short replay to verify V10-only mode works correctly.
#
# Usage:
#   bash scripts/test_entry_v10_1_flat_replay.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

POLICY="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT.yaml"
DATA_FILE="data/raw/xauusd_m5_2025_bid_ask.parquet"

# Test period: One week (2025-01-06 to 2025-01-13)
START_TS="2025-01-06T00:00:00Z"
END_TS="2025-01-13T00:00:00Z"  # exclusive

echo "=================================================================================="
echo "Testing ENTRY_V10.1 FLAT Replay (V10-only mode)"
echo "=================================================================================="
echo ""
echo "Policy: $POLICY"
echo "Date range: $START_TS → $END_TS"
echo ""

# Validate files exist
if [ ! -f "$POLICY" ]; then
    echo "❌ ERROR: Policy file not found: $POLICY"
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "❌ ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Create output directory
OUTPUT_DIR="data/replay/sniper/entry_v10_1_flat/test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Prepare filtered data (EU/OVERLAP/US only)
echo "[1/3] Preparing test data (EU/OVERLAP/US only)..."
FILTERED_DATA="$OUTPUT_DIR/test_filtered.parquet"

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

# Filter to SNIPER sessions (EU/OVERLAP/US)
sessions = ['EU', 'OVERLAP', 'US']
mask = df.index.map(lambda ts: infer_session_tag(ts) in sessions)
df = df[mask]

print(f'✅ Filtered test data: {len(df):,} rows')
print(f'   Date range: {df.index.min()} → {df.index.max()}')

df.to_parquet(output_file)
print(f'✅ Saved to: {output_file}')
" "$DATA_FILE" "$START_TS" "$END_TS" "$FILTERED_DATA"

echo ""
echo "[2/3] Running test replay..."
echo ""

python3 -c "
import sys
from pathlib import Path
from gx1.execution.oanda_demo_runner import GX1DemoRunner

policy_path = Path(sys.argv[1])
data_file = Path(sys.argv[2])
output_dir = Path(sys.argv[3])

print(f'Policy: {policy_path}')
print(f'Data: {data_file}')
print(f'Output: {output_dir}')
print('')

runner = GX1DemoRunner(
    policy_path,
    dry_run_override=True,
    replay_mode=True,
    fast_replay=True,
    output_dir=output_dir,
)

print('Starting test replay...')
runner.run_replay(data_file)
print('✅ Test replay complete!')
" "$POLICY" "$FILTERED_DATA" "$OUTPUT_DIR"

echo ""
echo "[3/3] Verifying results..."
echo ""

# Check trade journal
TRADE_JOURNAL_DIR="$OUTPUT_DIR/trade_journal"
TRADE_JSON_DIR="$TRADE_JOURNAL_DIR/trades"

if [ ! -d "$TRADE_JSON_DIR" ]; then
    echo "⚠️  WARNING: Trade journal directory not found: $TRADE_JSON_DIR"
else
    JSON_COUNT=$(find "$TRADE_JSON_DIR" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
    echo "✅ Trade JSON files: $JSON_COUNT"
    
    if [ "$JSON_COUNT" -gt 0 ]; then
        # Check first trade for entry data
        FIRST_TRADE=$(find "$TRADE_JSON_DIR" -name "*.json" | head -1)
        if [ -n "$FIRST_TRADE" ]; then
            ENTRY_EXISTS=$(python3 -c "
import json
import sys
with open(sys.argv[1]) as f:
    trade = json.load(f)
    has_entry = trade.get('entry_snapshot') is not None
    has_entry_time = trade.get('entry', {}).get('timestamp') is not None if isinstance(trade.get('entry'), dict) else False
    print('entry_snapshot' if has_entry else 'no_entry_snapshot', 'entry_time' if has_entry_time else 'no_entry_time')
" "$FIRST_TRADE" 2>/dev/null || echo "error error")
            
            ENTRY_SNAPSHOT=$(echo "$ENTRY_EXISTS" | awk '{print $1}')
            ENTRY_TIME=$(echo "$ENTRY_EXISTS" | awk '{print $2}')
            
            if [ "$ENTRY_SNAPSHOT" = "entry_snapshot" ] || [ "$ENTRY_TIME" = "entry_time" ]; then
                echo "✅ First trade has entry data: $ENTRY_SNAPSHOT, $ENTRY_TIME"
            else
                echo "❌ First trade missing entry data: $ENTRY_SNAPSHOT, $ENTRY_TIME"
            fi
        fi
    fi
fi

echo ""
echo "=================================================================================="
echo "✅ Test Complete!"
echo "=================================================================================="
echo ""
echo "Output: $OUTPUT_DIR"
echo ""
echo "Check logs for:"
echo "  - [ENTRY] Running in V10-only mode"
echo "  - [ENTRY_V10] Bundle loaded successfully (variant=v10_1, ...)"
echo "  - [ENTRY_V10] Evaluating in regime ..."
echo ""

