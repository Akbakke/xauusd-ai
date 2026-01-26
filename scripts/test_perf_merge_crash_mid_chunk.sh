#!/bin/bash
# Test for perf summary merge: crash mid-chunk case
#
# Runs smoke test with crash injection for one chunk, verifies summary is written
# in finally block, and verifies merge marks chunk as incomplete.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Test configuration
N_CHUNKS=2
CRASH_CHUNK_ID=1  # Crash chunk 1 (0-indexed)
CRASH_AFTER_BARS=50  # Crash after 50 bars processed
POLICY="${1:-gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_0_18.yaml}"
DATA_FILE="${2:-data/temp/january_2025_replay.parquet}"

# Test output directory
TEST_OUTPUT_DIR="data/temp/test_perf_merge_crash_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_OUTPUT_DIR"

echo "=================================================================================="
echo "PERF MERGE CRASH MID-CHUNK TEST"
echo "=================================================================================="
echo "Policy: $POLICY"
echo "Data: $DATA_FILE"
echo "Chunks: $N_CHUNKS"
echo "Crash chunk: $CRASH_CHUNK_ID (after $CRASH_AFTER_BARS bars)"
echo "Output: $TEST_OUTPUT_DIR"
echo ""

# Verify data file exists
if [[ ! -f "$DATA_FILE" ]]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Step 1: Prepare test data
echo "[1/5] Preparing test data..."
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
echo "[2/5] Splitting into $N_CHUNKS chunks..."
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

chunk_files = []
start_idx = 0

for i in range(n_chunks):
    current_chunk_size = chunk_size + (1 if i < remainder else 0)
    end_idx = start_idx + current_chunk_size
    
    chunk_df = df.iloc[start_idx:end_idx].copy()
    chunk_file = chunk_dir / f'chunk_{i}.parquet'
    chunk_df.to_parquet(chunk_file)
    chunk_files.append(chunk_file)
    start_idx = end_idx

print(f'✅ Created {len(chunk_files)} chunks')
" "$TEST_DATA" "$N_CHUNKS" "$CHUNK_DIR"

# Step 3: Run chunks with crash injection for chunk 1
echo ""
echo "[3/5] Running chunks (chunk $CRASH_CHUNK_ID will crash after $CRASH_AFTER_BARS bars)..."

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
    
    # Set crash injection for target chunk
    if [[ $i -eq $CRASH_CHUNK_ID ]]; then
        export GX1_TEST_INDUCE_CRASH_CHUNK_ID="$CRASH_CHUNK_ID"
        export GX1_TEST_CRASH_AFTER_N_BARS="$CRASH_AFTER_BARS"
        echo "  → Crash injection ACTIVE (will crash after $CRASH_AFTER_BARS bars)"
    else
        unset GX1_TEST_INDUCE_CRASH_CHUNK_ID
        unset GX1_TEST_CRASH_AFTER_N_BARS
    fi
    
    set +e  # Allow chunk to fail
    python3 scripts/run_mini_replay_perf.py \
        "$POLICY" \
        "$chunk_file" \
        "$chunk_output_dir" \
        2>&1 | tee "$TEST_OUTPUT_DIR/chunk_${i}.log"
    CHUNK_EXIT=$?
    set -e
    
    if [[ $i -eq $CRASH_CHUNK_ID ]]; then
        if [[ $CHUNK_EXIT -eq 0 ]]; then
            echo "❌ ERROR: Crash chunk should have failed but returned exit code 0"
            exit 1
        fi
        echo "[CHUNK $i/$N_CHUNKS] ✅ Failed as expected (exit code: $CHUNK_EXIT)"
    else
        if [[ $CHUNK_EXIT -ne 0 ]]; then
            echo "❌ ERROR: Non-crash chunk $i failed (exit code: $CHUNK_EXIT)"
            exit 1
        fi
        echo "[CHUNK $i/$N_CHUNKS] ✅ Completed successfully"
    fi
done

echo ""
echo "✅ All chunks processed (chunk $CRASH_CHUNK_ID crashed as expected)"

# Step 4: Verify crashed chunk summary exists
echo ""
echo "[4/5] Verifying summaries..."

for i in $(seq 0 $((N_CHUNKS - 1))); do
    summary_json="$TEST_OUTPUT_DIR/parallel_chunks/chunk_${i}/REPLAY_PERF_SUMMARY.json"
    if [[ ! -f "$summary_json" ]]; then
        echo "❌ ERROR: Missing summary for chunk $i: $summary_json"
        echo "   (Summary should be written in finally block even on crash)"
        exit 1
    fi
    echo "✅ Chunk $i summary exists"
done

# Verify crashed chunk is marked as incomplete
echo ""
echo "Verifying crashed chunk status..."

# W: The crashing chunk: chunk summary FILE MUST EXIST (finally), but status=="incomplete"
crashed_summary_file="$TEST_OUTPUT_DIR/parallel_chunks/chunk_${CRASH_CHUNK_ID}/REPLAY_PERF_SUMMARY.json"
if [[ ! -f "$crashed_summary_file" ]]; then
    echo "❌ TEST FAILED (W): Crashed chunk summary file does not exist: $crashed_summary_file"
    exit 1
fi
echo "✅ Crashed chunk summary file exists (W)"

python3 << PYTHON
import json
from pathlib import Path
import sys

test_output_dir = Path("$TEST_OUTPUT_DIR")
crash_chunk_id = $CRASH_CHUNK_ID

crashed_summary_file = test_output_dir / "parallel_chunks" / f"chunk_{crash_chunk_id}" / "REPLAY_PERF_SUMMARY.json"
with open(crashed_summary_file) as f:
    crashed_summary = json.load(f)

status = crashed_summary.get("status", "complete" if crashed_summary.get("completed", False) else "incomplete")
early_stop_reason = crashed_summary.get("early_stop_reason")
bars_processed = crashed_summary.get("bars_processed", 0)
bars_total = crashed_summary.get("bars_total", 0)

# W: status must be "incomplete"
if status != "incomplete":
    print(f"❌ TEST FAILED (W): Crashed chunk status is '{status}', expected 'incomplete'")
    sys.exit(1)

# X: early_stop_reason must contain "TEST_INDUCED_CRASH"
if not early_stop_reason or "TEST_INDUCED_CRASH" not in early_stop_reason:
    print(f"❌ TEST FAILED (X): early_stop_reason doesn't mention TEST_INDUCED_CRASH")
    print(f"   Got: {early_stop_reason}")
    sys.exit(1)

if bars_processed >= bars_total:
    print(f"❌ ERROR: bars_processed ({bars_processed}) >= bars_total ({bars_total})")
    print(f"   (Crashed chunk should have processed fewer bars)")
    sys.exit(1)

print(f"✅ Crashed chunk correctly marked:")
print(f"   status: {status} (W)")
print(f"   early_stop_reason: {early_stop_reason} (X)")
print(f"   bars_processed: {bars_processed}/{bars_total}")
PYTHON

CRASH_VALIDATION_EXIT=$?
if [[ $CRASH_VALIDATION_EXIT -ne 0 ]]; then
    exit 1
fi

# Step 5: Run merge and verify it fails (Y)
echo ""
echo "[5/5] Running merge (should fail due to incomplete chunk)..."

set +e  # Allow merge to fail
python3 scripts/merge_perf_summaries.py "$TEST_OUTPUT_DIR" 2>&1 | tee "$TEST_OUTPUT_DIR/merge.log"
MERGE_EXIT=$?
set -e

# Y: merge exit code != 0
if [[ $MERGE_EXIT -eq 0 ]]; then
    echo "❌ TEST FAILED (Y): Merge should have failed but returned exit code 0"
    echo "   (Merge must fail when any chunk has status != 'complete')"
    cat "$TEST_OUTPUT_DIR/merge.log"
    exit 1
fi

# Z: merge output must list early_stop_reasons and identify chunk_id as incomplete
if ! grep -q "Incomplete chunks" "$TEST_OUTPUT_DIR/merge.log"; then
    echo "❌ TEST FAILED (Z): Error message doesn't mention 'Incomplete chunks'"
    echo "Merge output:"
    cat "$TEST_OUTPUT_DIR/merge.log"
    exit 1
fi

if ! grep -q "chunk_${CRASH_CHUNK_ID}" "$TEST_OUTPUT_DIR/merge.log"; then
    echo "❌ TEST FAILED (Z): Error message doesn't mention chunk_${CRASH_CHUNK_ID}"
    echo "Merge output:"
    cat "$TEST_OUTPUT_DIR/merge.log"
    exit 1
fi

if ! grep -q "TEST_INDUCED_CRASH" "$TEST_OUTPUT_DIR/merge.log"; then
    echo "❌ TEST FAILED (Z): Error message doesn't mention TEST_INDUCED_CRASH"
    echo "Merge output:"
    cat "$TEST_OUTPUT_DIR/merge.log"
    exit 1
fi

echo "✅ Merge correctly failed with exit code $MERGE_EXIT (Y)"
echo "✅ Error message identifies incomplete chunk and reason (Z)"
echo "Error message:"
grep "Incomplete chunks" "$TEST_OUTPUT_DIR/merge.log" | head -1

echo ""
echo "=================================================================================="
echo "✅ CRASH MID-CHUNK TEST PASSED"
echo "=================================================================================="
echo "Output: $TEST_OUTPUT_DIR"
echo "Merged summary: $TEST_OUTPUT_DIR/REPLAY_PERF_SUMMARY.json"
echo ""

