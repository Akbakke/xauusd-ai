#!/usr/bin/env bash
# ENTRY_V10.1 FLAT Smoketest Replay
#
# Runs a short replay (2025-03-01 to 2025-03-07) to verify V10-only entry works correctly.
# Uses the same parallel replay mechanism as FULLYEAR runner.
#
# Usage:
#   bash scripts/run_sniper_entry_v10_1_flat_smoketest.sh [--n-workers N]

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
test "$(pwd)" = "$ROOT" || { echo "❌ FAIL: Not in repo-root (pwd=$(pwd), root=$ROOT)"; exit 1; }
echo "[RUN_CTX] root=$ROOT"
echo "[RUN_CTX] head=$(git rev-parse --short HEAD)"
echo "[RUN_CTX] whoami=$(whoami) host=$(hostname)"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Parse arguments
N_WORKERS=2  # Default 2 workers for smoketest (small dataset)
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-workers)
            N_WORKERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

POLICY="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT.yaml"
DATA_FILE="${GX1_CANONICAL_TAPE_ROOT:-/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL}/year=2025/part-000.parquet"

# Smoketest period: 2 months (January + February 2025)
START_TS="2025-01-01T00:00:00Z"
END_TS="2025-03-01T00:00:00Z"  # exclusive

OUTPUT_DIR="data/replay/tests/v10_1_smoketest_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "=================================================================================="
echo "ENTRY_V10.1 FLAT Smoketest Replay"
echo "=================================================================================="
echo ""
echo "Policy: $POLICY"
echo "Period: $START_TS → $END_TS"
echo "Workers: $N_WORKERS"
echo "Output: $OUTPUT_DIR"
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

echo "[1/4] Preparing filtered data (EU/OVERLAP/US only)..."
FILTERED_DATA="$OUTPUT_DIR/filtered_data.parquet"

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
echo "[2/4] Splitting into $N_WORKERS chunks..."
CHUNK_DIR="$OUTPUT_DIR/chunks"
mkdir -p "$CHUNK_DIR"

python3 -c "
import pandas as pd
from pathlib import Path
import sys

data_file = sys.argv[1]
n_workers = int(sys.argv[2])
chunk_dir = Path(sys.argv[3])

df = pd.read_parquet(data_file)
chunk_size = (len(df) + n_workers - 1) // n_workers

chunk_files = []
for i in range(n_workers):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(df))
    
    if start_idx >= len(df):
        break
    
    chunk_df = df.iloc[start_idx:end_idx]
    chunk_file = chunk_dir / f'chunk_{i}.parquet'
    chunk_df.to_parquet(chunk_file)
    
    print(f'  Chunk {i}: {len(chunk_df):,} bars ({chunk_df.index.min()} → {chunk_df.index.max()})')
    chunk_files.append(chunk_file)

print(f'✅ Created {len(chunk_files)} chunks')
" "$FILTERED_DATA" "$N_WORKERS" "$CHUNK_DIR"

echo ""
echo "[3/4] Running replay with $N_WORKERS parallel workers..."
PARALLEL_DIR="$OUTPUT_DIR/parallel_chunks"
mkdir -p "$PARALLEL_DIR"

run_chunk() {
    local chunk_id=$1
    local chunk_file="$CHUNK_DIR/chunk_${chunk_id}.parquet"
    local chunk_output_dir="$PARALLEL_DIR/chunk_${chunk_id}"
    
    mkdir -p "$chunk_output_dir"
    
    echo "[CHUNK $chunk_id/$N_WORKERS] Starting..."
    
    # Set thread environment variables for this worker
    export GX1_CHUNK_ID="$chunk_id"
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export GX1_XGB_THREADS=1
    
    python3 -c "
import sys
from pathlib import Path
from gx1.execution.oanda_demo_runner import GX1DemoRunner

chunk_id = int(sys.argv[1])
policy_path = Path(sys.argv[2])
chunk_file = Path(sys.argv[3])
output_dir = Path(sys.argv[4])

runner = GX1DemoRunner(
    policy_path,
    dry_run_override=True,
    replay_mode=True,
    fast_replay=True,
    output_dir=output_dir,
)

runner.run_replay(str(chunk_file))
print(f'[CHUNK {chunk_id}] ✅ Complete')
" "$chunk_id" "$POLICY" "$chunk_file" "$chunk_output_dir" 2>&1 | tee "$PARALLEL_DIR/chunk_${chunk_id}.log"
    
    echo "[CHUNK $chunk_id/$N_WORKERS] ✅ Done"
}

# Run chunks in parallel
for i in $(seq 0 $((N_WORKERS - 1))); do
    run_chunk $i &
done

# Wait for all jobs to complete
wait

echo ""
echo "✅ All chunks completed!"

echo ""
echo "[4/4] Verifying trade journal..."
echo ""

# Check if trade journals exist
TRADE_JOURNAL_DIRS=$(find "$PARALLEL_DIR" -type d -name "trade_journal" | head -1)
if [ -z "$TRADE_JOURNAL_DIRS" ]; then
    echo "❌ ERROR: No trade journals found in $PARALLEL_DIR"
    exit 1
fi

# Check first trade journal
FIRST_JOURNAL=$(find "$PARALLEL_DIR" -type d -name "trade_journal" | head -1)
FIRST_TRADE_JSON=$(find "$FIRST_JOURNAL" -name "*.json" | head -1)

if [ -z "$FIRST_TRADE_JSON" ]; then
    echo "⚠️  WARNING: No trades found in trade journal"
    echo "   This might be expected if the period had no entry signals"
    exit 0
fi

echo "✅ Found trade journal: $FIRST_JOURNAL"
echo "   First trade: $(basename $FIRST_TRADE_JSON)"
echo ""

# Verify entry data in first trade
python3 -c "
import json
from pathlib import Path
import sys

trade_file = Path('$FIRST_TRADE_JSON')
with open(trade_file, 'r') as f:
    trade = json.load(f)

print('📋 Verifying first trade:')
print(f'  Trade ID: {trade.get(\"trade_id\", \"N/A\")}')
print(f'  Entry time: {\"✅\" if trade.get(\"entry_time\") else \"❌\"} {trade.get(\"entry_time\", \"MANGER\")}')
print(f'  Entry snapshot: {\"✅\" if trade.get(\"entry_snapshot\") else \"❌\"} {\"finnes\" if trade.get(\"entry_snapshot\") else \"MANGER\"}')

entry = trade.get('entry', {})
print(f'  entry.p_long_v10_1: {\"✅\" if entry.get(\"p_long_v10_1\") is not None else \"❌\"} {entry.get(\"p_long_v10_1\", \"MANGER\")}')
print(f'  Exit summary: {\"✅\" if trade.get(\"exit_summary\") else \"❌\"}')

if not trade.get('entry_time'):
    print('')
    print('❌ PROBLEM: Ingen entry_time!')
    sys.exit(1)
if not trade.get('entry_snapshot'):
    print('')
    print('❌ PROBLEM: Ingen entry_snapshot!')
    sys.exit(1)
if entry.get('p_long_v10_1') is None:
    print('')
    print('❌ PROBLEM: Ingen entry.p_long_v10_1!')
    sys.exit(1)

print('')
print('✅ All entry data er til stede!')
print('✅ V10-only entry fungerer korrekt!')
"

echo ""
echo "=================================================================================="
echo "✅ Smoketest Complete!"
echo "=================================================================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Trade journals: $PARALLEL_DIR/*/trade_journal/"
echo ""

