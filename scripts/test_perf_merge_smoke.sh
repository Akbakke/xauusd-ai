#!/bin/bash
# Smoke test for perf summary merge: normal case (all chunks complete)
#
# Runs a mini replay with 2-3 chunks, verifies all summaries exist,
# runs merge, and validates invariants.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Test configuration
N_CHUNKS=2
POLICY="${1:-gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_0_18.yaml}"
DATA_FILE="${2:-data/temp/january_2025_replay.parquet}"

# Test output directory
TEST_OUTPUT_DIR="data/temp/test_perf_merge_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_OUTPUT_DIR"

echo "=================================================================================="
echo "PERF MERGE SMOKE TEST"
echo "=================================================================================="
echo "Policy: $POLICY"
echo "Data: $DATA_FILE"
echo "Chunks: $N_CHUNKS"
echo "Output: $TEST_OUTPUT_DIR"
echo ""

# Verify data file exists
if [[ ! -f "$DATA_FILE" ]]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Step 1: Prepare filtered data (small subset for speed)
echo "[1/4] Preparing test data..."
TEMP_DATA_DIR="$TEST_OUTPUT_DIR/data"
mkdir -p "$TEMP_DATA_DIR"

python3 -c "
import pandas as pd
from pathlib import Path
import sys

data_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_parquet(data_file)
if 'time' in df.columns:
    df = df.set_index('time')
elif 'ts' in df.columns:
    df = df.set_index('ts')
df.index = pd.to_datetime(df.index, utc=True)

# Take first 3 days for speed
df = df.head(864).copy()  # ~3 days M5

print(f'Test data: {len(df):,} bars')
print(f'Date range: {df.index.min()} → {df.index.max()}')

df.to_parquet(output_file)
print(f'✅ Saved to: {output_file}')
" "$DATA_FILE" "$TEMP_DATA_DIR/test_data.parquet"

TEST_DATA="$TEMP_DATA_DIR/test_data.parquet"

# Step 2: Split into chunks
echo ""
echo "[2/4] Splitting into $N_CHUNKS chunks..."
CHUNK_DIR="$TEST_OUTPUT_DIR/chunks"
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
" "$TEST_DATA" "$N_CHUNKS" "$CHUNK_DIR"

# Step 3: Run chunks sequentially (for deterministic testing)
echo ""
echo "[3/4] Running $N_CHUNKS chunks..."

for i in $(seq 0 $((N_CHUNKS - 1))); do
    chunk_file="$CHUNK_DIR/chunk_${i}.parquet"
    chunk_output_dir="$TEST_OUTPUT_DIR/parallel_chunks/chunk_${i}"
    
    mkdir -p "$chunk_output_dir"
    
    echo "[CHUNK $i/$N_CHUNKS] Running..."
    
    # Set environment variables
    export GX1_CHUNK_ID="$i"
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export GX1_XGB_THREADS=1
    export GX1_REPLAY_NO_CSV=1
    export GX1_FEATURE_USE_NP_ROLLING=1
    export GX1_REPLAY_INCREMENTAL_FEATURES=1
    
    python3 scripts/run_mini_replay_perf.py \
        "$POLICY" \
        "$chunk_file" \
        "$chunk_output_dir" \
        2>&1 | tee "$TEST_OUTPUT_DIR/chunk_${i}.log" || {
        echo "ERROR: Chunk $i failed"
        exit 1
    }
    
    echo "[CHUNK $i/$N_CHUNKS] ✅ Done"
done

echo ""
echo "✅ All chunks completed!"

# Step 4: Verify all summaries exist
echo ""
echo "[4/4] Verifying summaries and running merge..."

for i in $(seq 0 $((N_CHUNKS - 1))); do
    summary_json="$TEST_OUTPUT_DIR/parallel_chunks/chunk_${i}/REPLAY_PERF_SUMMARY.json"
    if [[ ! -f "$summary_json" ]]; then
        echo "ERROR: Missing summary for chunk $i: $summary_json"
        exit 1
    fi
    echo "✅ Chunk $i summary exists"
done

# Run merge
python3 scripts/merge_perf_summaries.py "$TEST_OUTPUT_DIR" 2>&1 | tee "$TEST_OUTPUT_DIR/merge.log"
MERGE_EXIT=$?

if [[ $MERGE_EXIT -ne 0 ]]; then
    echo "ERROR: Merge failed (exit code: $MERGE_EXIT)"
    exit 1
fi

# Validate invariants
echo ""
echo "Validating invariants..."

# Validate each chunk
for i in $(seq 0 $((N_CHUNKS - 1))); do
    chunk_summary="$TEST_OUTPUT_DIR/parallel_chunks/chunk_${i}/REPLAY_PERF_SUMMARY.json"
    echo "Validating chunk $i..."
    python3 scripts/assert_perf_invariants.py chunk "$chunk_summary" || {
        echo "❌ Chunk $i invariant validation failed"
        exit 1
    }
done

# Validate merged summary
echo "Validating merged summary..."
chunk_dirs=$(for i in $(seq 0 $((N_CHUNKS - 1))); do echo "$TEST_OUTPUT_DIR/parallel_chunks/chunk_${i}"; done)
python3 scripts/assert_perf_invariants.py merged "$TEST_OUTPUT_DIR/REPLAY_PERF_SUMMARY.json" $chunk_dirs || {
    echo "❌ Merged invariant validation failed"
    exit 1
}

# Scenario-specific: Smoke test invariants (S, T)
echo "Validating smoke test scenario invariants..."

python3 << PYTHON
import json
import sys
from pathlib import Path

test_output_dir = Path("$TEST_OUTPUT_DIR")
n_chunks = $N_CHUNKS

errors = []

# Load all chunk summaries
for i in range(n_chunks):
    chunk_file = test_output_dir / "parallel_chunks" / f"chunk_{i}" / "REPLAY_PERF_SUMMARY.json"
    with open(chunk_file) as f:
        chunk_data = json.load(f)
    
    # S: All chunks: status=="complete" and early_stop_reason is null/empty
    status = chunk_data.get("status", "complete" if chunk_data.get("completed", False) else "incomplete")
    if status != "complete":
        errors.append(f"S: Chunk {i} status is '{status}', expected 'complete'")
    
    early_stop = chunk_data.get("early_stop_reason")
    if early_stop:
        errors.append(f"S: Chunk {i} has early_stop_reason: {early_stop}")

# T: merge exit code == 0 (checked by shell script)

if errors:
    print("❌ SCENARIO INVARIANT VIOLATIONS:")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)
else:
    print("✅ All scenario invariants passed (S, T)")
PYTHON

VALIDATION_EXIT=$?

if [[ $VALIDATION_EXIT -ne 0 ]]; then
    echo ""
    echo "❌ Test FAILED: Scenario invariant violations detected"
    exit 1
fi

echo ""
echo "=================================================================================="
echo "✅ SMOKE TEST PASSED"
echo "=================================================================================="
echo "Output: $TEST_OUTPUT_DIR"
echo "Merged summary: $TEST_OUTPUT_DIR/REPLAY_PERF_SUMMARY.json"
echo ""

