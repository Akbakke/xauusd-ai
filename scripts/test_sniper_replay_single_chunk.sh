#!/bin/bash
# SNIPER Single Chunk Test
#
# Tests SNIPER replay on a small subset of Q1 data (1 chunk) to verify everything works.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

POLICY="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml"
DATA_FILE="data/raw/xauusd_m5_2025_bid_ask.parquet"

# Q1 2025 (UTC) - just first week for quick test
START_TS="2025-01-01T00:00:00Z"
END_TS="2025-01-08T00:00:00Z"  # First week only

echo "=================================================================================="
echo "SNIPER Single Chunk Test (First Week of Q1)"
echo "=================================================================================="
echo ""
echo "Policy: $POLICY"
echo "Date Range: $START_TS → $END_TS"
echo ""

# Prepare filtered data (EU/OVERLAP/US only)
echo "[1/3] Preparing test data (EU/OVERLAP/US only)..."
TEMP_DATA_DIR="data/temp/sniper_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEMP_DATA_DIR"

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

# Filter to Sniper sessions
sessions = ['EU', 'OVERLAP', 'US']
mask = df.index.map(lambda ts: infer_session_tag(ts) in sessions)
df = df[mask]

print(f'Filtered test data: {len(df):,} rows')
print(f'Date range: {df.index.min()} → {df.index.max()}')

df.to_parquet(output_file)
print(f'✅ Saved to: {output_file}')
" "$DATA_FILE" "$START_TS" "$END_TS" "$TEMP_DATA_DIR/test_data.parquet"

TEST_DATA="$TEMP_DATA_DIR/test_data.parquet"

# Run replay
echo ""
echo "[2/3] Running SNIPER replay..."
RUN_TAG="SNIPER_TEST_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="gx1/wf_runs/$RUN_TAG"
mkdir -p "$OUTPUT_DIR"

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
)

print('Starting replay...')
runner.run_replay(data_file)
print('✅ Replay complete!')
" "$POLICY" "$TEST_DATA" "$OUTPUT_DIR"

# Check results
echo ""
echo "[3/3] Checking results..."
if [ -d "$OUTPUT_DIR/trade_journal/trades" ]; then
    N_TRADES=$(find "$OUTPUT_DIR/trade_journal/trades" -name "*.json" | wc -l | tr -d ' ')
    echo "✅ Trade journal found: $N_TRADES trades"
    
    if [ "$N_TRADES" -gt 0 ]; then
        echo ""
        echo "Sample trades:"
        find "$OUTPUT_DIR/trade_journal/trades" -name "*.json" | head -3 | while read f; do
            echo "  $(basename $f)"
        done
    fi
else
    echo "⚠️  No trade journal found"
fi

if [ -f "$OUTPUT_DIR/trade_journal/trade_journal_index.csv" ]; then
    echo "✅ Trade journal index found"
    echo ""
    echo "First few trades:"
    head -5 "$OUTPUT_DIR/trade_journal/trade_journal_index.csv" | column -t -s,
fi

echo ""
echo "=================================================================================="
echo "✅ SNIPER Test Complete!"
echo "=================================================================================="
echo ""
echo "Run Tag: $RUN_TAG"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Check trade journal: ls -lah $OUTPUT_DIR/trade_journal/trades/"
echo "  2. View index: cat $OUTPUT_DIR/trade_journal/trade_journal_index.csv"
echo "  3. If successful, run full parallel replay: ./scripts/run_sniper_parallel_replay_q1.sh"
echo ""

