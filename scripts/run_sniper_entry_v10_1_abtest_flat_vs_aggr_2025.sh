#!/usr/bin/env bash
# SNIPER ENTRY_V10.1 A/B Test: FLAT vs AGGR FULLYEAR 2025
#
# Runs FULLYEAR replay for both FLAT and AGGR variants with parallel workers and generates comparison report.
#
# Usage:
#   bash scripts/run_sniper_entry_v10_1_abtest_flat_vs_aggr_2025.sh [--n-workers N]

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
N_WORKERS=7  # Default 7 workers
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

POLICY_FLAT="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT.yaml"
POLICY_AGGR="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_AGGR.yaml"
DATA_FILE="data/raw/xauusd_m5_2025_bid_ask.parquet"

# FULLYEAR 2025 (UTC)
START_TS="2025-01-01T00:00:00Z"
END_TS="2026-01-01T00:00:00Z"  # exclusive

echo "=================================================================================="
echo "SNIPER ENTRY_V10.1 A/B Test: FLAT vs AGGR FULLYEAR 2025"
echo "=================================================================================="
echo ""
echo "Configuration:"
echo "  - FLAT Policy: $POLICY_FLAT"
echo "  - AGGR Policy: $POLICY_AGGR"
echo "  - Entry: ENTRY_V10.1 HYBRID (variant=v10_1, seq_len=90)"
echo "  - Data source: Same as P4.1 replays"
echo "  - Date range: $START_TS → $END_TS"
echo "  - Workers: $N_WORKERS (parallel replay)"
echo ""

# Validate configs exist
if [ ! -f "$POLICY_FLAT" ]; then
    echo "❌ ERROR: FLAT policy file not found: $POLICY_FLAT"
    exit 1
fi

if [ ! -f "$POLICY_AGGR" ]; then
    echo "❌ ERROR: AGGR policy file not found: $POLICY_AGGR"
    exit 1
fi

# Validate data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Create output directories
OUTPUT_BASE="data/replay/sniper/entry_v10_1"
mkdir -p "$OUTPUT_BASE"
RUN_TAG="ABTEST_FLAT_VS_AGGR_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR_FLAT="$OUTPUT_BASE/flat/2025/$RUN_TAG"
OUTPUT_DIR_AGGR="$OUTPUT_BASE/aggr/2025/$RUN_TAG"
mkdir -p "$OUTPUT_DIR_FLAT"
mkdir -p "$OUTPUT_DIR_AGGR"

echo "Output directories:"
echo "  - FLAT: $OUTPUT_DIR_FLAT"
echo "  - AGGR: $OUTPUT_DIR_AGGR"
echo ""

# Prepare filtered data (EU/OVERLAP/US only, same as P4.1)
echo "[1/5] Preparing FULLYEAR 2025 data (EU/OVERLAP/US only)..."
FILTERED_DATA="$OUTPUT_BASE/fullyear_2025_filtered.parquet"

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

print(f'✅ Filtered FULLYEAR 2025 data: {len(df):,} rows')
print(f'   Date range: {df.index.min()} → {df.index.max()}')

df.to_parquet(output_file)
print(f'✅ Saved to: {output_file}')
" "$DATA_FILE" "$START_TS" "$END_TS" "$FILTERED_DATA"

echo ""
echo "[2/5] Running FLAT replay with $N_WORKERS parallel workers..."
echo "   This may take a while for FULLYEAR data..."
echo ""

# Step 2a: Split into chunks
CHUNK_DIR_FLAT="$OUTPUT_DIR_FLAT/chunks"
mkdir -p "$CHUNK_DIR_FLAT"

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

print(f'✅ Created {len(chunk_files)} chunks for FLAT')
" "$FILTERED_DATA" "$N_WORKERS" "$CHUNK_DIR_FLAT"

# Step 2b: Run parallel replays for FLAT
mkdir -p "$OUTPUT_DIR_FLAT/parallel_chunks"

run_chunk() {
    local chunk_id=$1
    local chunk_file="$CHUNK_DIR_FLAT/chunk_${chunk_id}.parquet"
    local chunk_output_dir="$OUTPUT_DIR_FLAT/parallel_chunks/chunk_${chunk_id}"
    
    mkdir -p "$chunk_output_dir"
    
    echo "[FLAT CHUNK $chunk_id/$N_WORKERS] Starting..."
    
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

runner.run_replay(chunk_file)
print(f'[FLAT CHUNK {chunk_id}] ✅ Complete')
" "$chunk_id" "$POLICY_FLAT" "$chunk_file" "$chunk_output_dir" 2>&1 | tee "$OUTPUT_DIR_FLAT/parallel_chunks/chunk_${chunk_id}.log"
    
    echo "[FLAT CHUNK $chunk_id/$N_WORKERS] ✅ Done"
}

# Run chunks in parallel
for i in $(seq 0 $((N_WORKERS - 1))); do
    run_chunk $i &
done

# Wait for all jobs to complete
wait

echo ""
echo "✅ All FLAT chunks completed!"

# Step 2c: Merge FLAT trade journals
echo ""
echo "Merging FLAT trade journals..."
python3 gx1/scripts/merge_trade_journals.py "$OUTPUT_DIR_FLAT"

echo ""
echo "[3/5] Running AGGR replay with $N_WORKERS parallel workers..."
echo "   This may take a while for FULLYEAR data..."
echo ""

# Step 3a: Split into chunks
CHUNK_DIR_AGGR="$OUTPUT_DIR_AGGR/chunks"
mkdir -p "$CHUNK_DIR_AGGR"

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

print(f'✅ Created {len(chunk_files)} chunks for AGGR')
" "$FILTERED_DATA" "$N_WORKERS" "$CHUNK_DIR_AGGR"

# Step 3b: Run parallel replays for AGGR
mkdir -p "$OUTPUT_DIR_AGGR/parallel_chunks"

run_chunk_aggr() {
    local chunk_id=$1
    local chunk_file="$CHUNK_DIR_AGGR/chunk_${chunk_id}.parquet"
    local chunk_output_dir="$OUTPUT_DIR_AGGR/parallel_chunks/chunk_${chunk_id}"
    
    mkdir -p "$chunk_output_dir"
    
    echo "[AGGR CHUNK $chunk_id/$N_WORKERS] Starting..."
    
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

runner.run_replay(chunk_file)
print(f'[AGGR CHUNK {chunk_id}] ✅ Complete')
" "$chunk_id" "$POLICY_AGGR" "$chunk_file" "$chunk_output_dir" 2>&1 | tee "$OUTPUT_DIR_AGGR/parallel_chunks/chunk_${chunk_id}.log"
    
    echo "[AGGR CHUNK $chunk_id/$N_WORKERS] ✅ Done"
}

# Run chunks in parallel
for i in $(seq 0 $((N_WORKERS - 1))); do
    run_chunk_aggr $i &
done

# Wait for all jobs to complete
wait

echo ""
echo "✅ All AGGR chunks completed!"

# Step 3c: Merge AGGR trade journals
echo ""
echo "Merging AGGR trade journals..."
python3 gx1/scripts/merge_trade_journals.py "$OUTPUT_DIR_AGGR"

echo ""
echo "[4/5] Verifying trade journals..."
echo ""

# Check FLAT trade journal
TRADE_JOURNAL_FLAT="$OUTPUT_DIR_FLAT/trade_journal"
TRADE_JOURNAL_INDEX_FLAT="$TRADE_JOURNAL_FLAT/trade_journal_index.csv"

if [ ! -f "$TRADE_JOURNAL_INDEX_FLAT" ]; then
    echo "⚠️  WARNING: FLAT trade_journal_index.csv not found"
else
    TRADE_COUNT_FLAT=$(tail -n +2 "$TRADE_JOURNAL_INDEX_FLAT" 2>/dev/null | wc -l | tr -d ' ' || echo "0")
    echo "✅ FLAT trade journal: $TRADE_COUNT_FLAT trades"
fi

# Check AGGR trade journal
TRADE_JOURNAL_AGGR="$OUTPUT_DIR_AGGR/trade_journal"
TRADE_JOURNAL_INDEX_AGGR="$TRADE_JOURNAL_AGGR/trade_journal_index.csv"

if [ ! -f "$TRADE_JOURNAL_INDEX_AGGR" ]; then
    echo "⚠️  WARNING: AGGR trade_journal_index.csv not found"
else
    TRADE_COUNT_AGGR=$(tail -n +2 "$TRADE_JOURNAL_INDEX_AGGR" 2>/dev/null | wc -l | tr -d ' ' || echo "0")
    echo "✅ AGGR trade journal: $TRADE_COUNT_AGGR trades"
fi

echo ""
echo "[5/5] Running comparison analysis..."
echo ""

# Run comparison analysis
python3 -m gx1.tools.analysis.compare_entry_v10_1_flat_vs_aggr \
    --flat-dir "$TRADE_JOURNAL_FLAT" \
    --aggr-dir "$TRADE_JOURNAL_AGGR" \
    --output-report "reports/rl/entry_v10/ENTRY_V10_1_FLAT_VS_AGGR_2025.md"

echo ""
echo "=================================================================================="
echo "✅ SNIPER ENTRY_V10.1 A/B Test Complete!"
echo "=================================================================================="
echo ""
echo "Run Tag: $RUN_TAG"
echo ""
echo "FLAT Output: $OUTPUT_DIR_FLAT"
echo "AGGR Output: $OUTPUT_DIR_AGGR"
echo ""
echo "Comparison Report: reports/rl/entry_v10/ENTRY_V10_1_FLAT_VS_AGGR_2025.md"
echo ""
echo "Next steps:"
echo "  1. Review comparison report: cat reports/rl/entry_v10/ENTRY_V10_1_FLAT_VS_AGGR_2025.md"
echo "  2. Check trade journals:"
echo "     - FLAT: ls -lah $TRADE_JOURNAL_FLAT/trades/"
echo "     - AGGR: ls -lah $TRADE_JOURNAL_AGGR/trades/"
echo ""

