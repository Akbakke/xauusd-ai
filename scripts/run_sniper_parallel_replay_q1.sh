#!/bin/bash
# SNIPER Parallel Replay for Q1 2025 (EU/OVERLAP/US)
#
# Splits Q1 data into 7 chunks and runs parallel replay, then merges trade journals.
#
# Usage:
#   ./scripts/run_sniper_parallel_replay_q1.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

N_WORKERS=7
POLICY="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml"
DATA_FILE="data/raw/xauusd_m5_2025_bid_ask.parquet"

# Q1 2025 (UTC)
START_TS="2025-01-01T00:00:00Z"
END_TS="2025-04-01T00:00:00Z"  # exclusive

echo "=================================================================================="
echo "SNIPER Parallel Replay Q1 2025"
echo "=================================================================================="
echo ""
echo "Policy: $POLICY"
echo "Workers: $N_WORKERS"
echo "Date Range: $START_TS → $END_TS"
echo ""

# Step 1: Prepare filtered Q1 data (EU/OVERLAP/US only)
echo "[1/4] Preparing Q1 data (EU/OVERLAP/US only)..."
TEMP_DATA_DIR="data/temp/sniper_q1_$(date +%Y%m%d_%H%M%S)"
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

print(f'Filtered Q1 data: {len(df):,} rows')
print(f'Date range: {df.index.min()} → {df.index.max()}')

df.to_parquet(output_file)
print(f'✅ Saved to: {output_file}')
" "$DATA_FILE" "$START_TS" "$END_TS" "$TEMP_DATA_DIR/q1_filtered.parquet"

FILTERED_DATA="$TEMP_DATA_DIR/q1_filtered.parquet"

# Step 2: Split into chunks
echo ""
echo "[2/4] Splitting into $N_WORKERS chunks..."
CHUNK_DIR="$TEMP_DATA_DIR/chunks"
mkdir -p "$CHUNK_DIR"

python3 -c "
import pandas as pd
from pathlib import Path
import sys

input_file = sys.argv[1]
n_chunks = int(sys.argv[2])
chunk_dir = Path(sys.argv[3])

df = pd.read_parquet(input_file)
total_bars = len(df)
chunk_size = total_bars // n_chunks
remainder = total_bars % n_chunks

print(f'Total bars: {total_bars:,}')
print(f'Chunk size: ~{chunk_size:,} bars each')

chunk_files = []
start_idx = 0

for i in range(n_chunks):
    current_chunk_size = chunk_size + (1 if i < remainder else 0)
    end_idx = start_idx + current_chunk_size
    
    chunk_df = df.iloc[start_idx:end_idx].copy()
    chunk_file = chunk_dir / f'chunk_{i}.parquet'
    chunk_df.to_parquet(chunk_file)
    
    print(f'  Chunk {i}: {len(chunk_df):,} bars ({chunk_df.index.min()} → {chunk_df.index.max()})')
    chunk_files.append(chunk_file)
    
    start_idx = end_idx

print(f'✅ Created {len(chunk_files)} chunks')
" "$FILTERED_DATA" "$N_WORKERS" "$CHUNK_DIR"

# Step 3: Run parallel replays
echo ""
echo "[3/4] Running $N_WORKERS parallel replays..."
RUN_TAG="SNIPER_OBS_Q1_2025_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="gx1/wf_runs/$RUN_TAG"
mkdir -p "$OUTPUT_DIR/parallel_chunks"

# Function to run a single chunk
run_chunk() {
    local chunk_id=$1
    local chunk_file="$CHUNK_DIR/chunk_${chunk_id}.parquet"
    local chunk_output_dir="$OUTPUT_DIR/parallel_chunks/chunk_${chunk_id}"
    
    mkdir -p "$chunk_output_dir"
    
    echo "[CHUNK $chunk_id/$N_WORKERS] Starting..."
    
    # Set environment variables for this chunk
    export GX1_CHUNK_ID="$chunk_id"
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export GX1_XGB_THREADS=1
    
    # Run replay
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
)

runner.run_replay(chunk_file)
print(f'[CHUNK {chunk_id}] ✅ Complete')
" "$chunk_id" "$POLICY" "$chunk_file" "$chunk_output_dir" 2>&1 | tee "$OUTPUT_DIR/parallel_chunks/chunk_${chunk_id}.log"
    
    echo "[CHUNK $chunk_id/$N_WORKERS] ✅ Done"
}

# Run chunks in parallel using background jobs
for i in $(seq 0 $((N_WORKERS - 1))); do
    run_chunk $i &
done

# Wait for all jobs to complete
wait

echo ""
echo "✅ All chunks completed!"

# Step 4: Merge trade journals (using dedicated merge script)
echo ""
echo "[4/4] Merging trade journals..."

python3 gx1/scripts/merge_trade_journals.py "$OUTPUT_DIR"

echo ""
echo "=================================================================================="
echo "✅ SNIPER Q1 Parallel Replay Complete!"
echo "=================================================================================="
echo ""
echo "Run Tag: $RUN_TAG"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Trade Journal: $OUTPUT_DIR/trade_journal/"
echo "Chunk Logs: $OUTPUT_DIR/parallel_chunks/"
echo ""
echo "Next steps:"
echo "  1. Check trade journal: ls -lah $OUTPUT_DIR/trade_journal/trades/"
echo "  2. View summary: cat $OUTPUT_DIR/trade_journal/trade_journal_index.csv"
echo "  3. Verify FARM lock: ./scripts/check_farm_lock.sh"
echo ""
