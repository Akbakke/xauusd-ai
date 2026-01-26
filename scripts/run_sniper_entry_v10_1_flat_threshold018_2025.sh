#!/usr/bin/env bash
# SNIPER ENTRY_V10.1 FLAT THRESHOLD 0.18 FULLYEAR 2025 Replay (FIXED)
#
# Runs FULLYEAR 2025 replay with ENTRY_V10.1 in FLAT mode with threshold=0.18.
# This is the threshold-tested baseline (best balance of trade count, win rate, tail risk).
#
# NOTE: This is a FIXED version that ensures proper trade journal logging:
#   - entry_snapshot and feature_context are properly logged
#   - CSV columns are correctly formatted (no misalignment)
#   - pnl_bps values are numeric (not strings)
#
# Usage:
#   bash scripts/run_sniper_entry_v10_1_flat_threshold018_2025.sh [--n-workers N]

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

POLICY_THRESHOLD018="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_0_18.yaml"
DATA_FILE="data/raw/xauusd_m5_2025_bid_ask.parquet"

# FULLYEAR 2025 (UTC)
START_TS="2025-01-01T00:00:00Z"
END_TS="2026-01-01T00:00:00Z"  # exclusive

echo "=================================================================================="
echo "SNIPER ENTRY_V10.1 FLAT THRESHOLD 0.18 FULLYEAR 2025 Replay (FIXED)"
echo "=================================================================================="
echo ""
echo "Configuration:"
echo "  - Policy: $POLICY_THRESHOLD018"
echo "  - Entry: ENTRY_V10.1 HYBRID (variant=v10_1, seq_len=90)"
echo "  - Mode: FLAT with threshold=0.18 (threshold-tested baseline)"
echo "  - Threshold: min_prob_long=0.18, p_side_min.long=0.18"
echo "  - Data source: Same as P4.1 replays"
echo "  - Date range: $START_TS → $END_TS"
echo "  - Workers: $N_WORKERS (parallel replay)"
echo ""
echo "Note: This is a FIXED version with proper trade journal logging."
echo ""

# Validate config exists
if [ ! -f "$POLICY_THRESHOLD018" ]; then
    echo "❌ ERROR: THRESHOLD 0.18 policy file not found: $POLICY_THRESHOLD018"
    exit 1
fi

# Validate data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Create output directories
OUTPUT_BASE="data/replay/sniper/entry_v10_1_flat_threshold0_18/2025"
mkdir -p "$OUTPUT_BASE"
RUN_TAG="FLAT_THRESHOLD_0_18_FIXED_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$OUTPUT_BASE/$RUN_TAG"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Prepare filtered data (EU/OVERLAP/US only, same as P4.1)
echo "[1/4] Preparing FULLYEAR 2025 data (EU/OVERLAP/US only)..."
FILTERED_DATA="$OUTPUT_BASE/fullyear_2025_filtered.parquet"

# Check if filtered data already exists from a previous run in the same base directory
if [ -f "$FILTERED_DATA" ]; then
    echo "✅ Using existing filtered data: $FILTERED_DATA"
else
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
fi

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
echo "   This may take a while for FULLYEAR data..."
echo ""

# Run chunks in parallel
PARALLEL_CHUNKS_DIR="$OUTPUT_DIR/parallel_chunks"
mkdir -p "$PARALLEL_CHUNKS_DIR"
mkdir -p "$OUTPUT_DIR/logs"

run_chunk() {
    local chunk_id=$1
    local policy_path=$2
    local chunk_file="$CHUNK_DIR/chunk_${chunk_id}.parquet"
    local chunk_output_dir="$PARALLEL_CHUNKS_DIR/chunk_${chunk_id}"
    
    mkdir -p "$chunk_output_dir"
    
    echo "[CHUNK $chunk_id/$N_WORKERS] Starting..."
    
    # Set thread environment variables for this worker (CRITICAL: must be set before any imports)
    # This ensures each worker uses only 1 thread, so n_jobs=7 means exactly 7 workers
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
import os

policy_path = Path(sys.argv[1])
chunk_file = Path(sys.argv[2])
output_dir = Path(sys.argv[3])

runner = GX1DemoRunner(
    policy_path,
    dry_run_override=True,
    replay_mode=True,
    fast_replay=True,
    output_dir=output_dir,
)

runner.run_replay(chunk_file)
print(f'[CHUNK {chunk_id}] ✅ Complete')
" "$policy_path" "$chunk_file" "$chunk_output_dir" > "$OUTPUT_DIR/logs/chunk_${chunk_id}.log" 2>&1
    
    echo "[CHUNK $chunk_id/$N_WORKERS] ✅ Done"
}

# Run chunks in parallel
for i in $(seq 0 $((N_WORKERS - 1))); do
    run_chunk $i "$POLICY_THRESHOLD018" &
done

# Wait for all jobs to complete
wait

echo ""
echo "✅ All chunks completed!"

# Step 4: Merge trade journals
echo ""
echo "[4/4] Merging trade journals..."
MERGED_TRADE_JOURNAL_DIR="$OUTPUT_DIR/trade_journal"
python3 gx1/scripts/merge_trade_journals.py "$PARALLEL_CHUNKS_DIR" --output-dir "$MERGED_TRADE_JOURNAL_DIR"

MERGED_TRADE_JOURNAL_INDEX="$MERGED_TRADE_JOURNAL_DIR/trade_journal_index.csv"
MERGED_TRADE_JOURNAL_PARQUET="$MERGED_TRADE_JOURNAL_DIR/merged_trade_journal.parquet"

if [ -f "$MERGED_TRADE_JOURNAL_INDEX" ]; then
    TRADE_COUNT=$(tail -n +2 "$MERGED_TRADE_JOURNAL_INDEX" 2>/dev/null | wc -l | tr -d ' ' || echo "0")
    echo "✅ Merged trade journal found: $TRADE_COUNT trades"
    echo "   Path: $MERGED_TRADE_JOURNAL_INDEX"
    
    # Convert to parquet for easier analysis
    python3 -c "
import pandas as pd
from pathlib import Path
import sys

csv_path = Path(sys.argv[1])
parquet_path = Path(sys.argv[2])

df = pd.read_csv(csv_path)
df.to_parquet(parquet_path)
print(f'✅ Converted merged trade journal to parquet: {parquet_path}')
" "$MERGED_TRADE_JOURNAL_INDEX" "$MERGED_TRADE_JOURNAL_PARQUET"
    
    # Verify a sample trade JSON has entry_snapshot and feature_context
    echo ""
    echo "Verifying trade journal structure..."
    python3 -c "
import json
from pathlib import Path
import sys

trades_dir = Path(sys.argv[1]) / 'trades'
sample_file = list(trades_dir.glob('*.json'))[0] if trades_dir.exists() else None

if sample_file:
    with open(sample_file, 'r') as f:
        trade = json.load(f)
    
    has_entry_snapshot = trade.get('entry_snapshot') is not None
    has_feature_context = trade.get('feature_context') is not None
    has_exit_summary = trade.get('exit_summary') is not None
    
    print(f'Sample trade: {sample_file.name}')
    print(f'  entry_snapshot: {\"✅\" if has_entry_snapshot else \"❌\"} ({type(trade.get(\"entry_snapshot\")).__name__})')
    print(f'  feature_context: {\"✅\" if has_feature_context else \"❌\"} ({type(trade.get(\"feature_context\")).__name__})')
    print(f'  exit_summary: {\"✅\" if has_exit_summary else \"❌\"} ({type(trade.get(\"exit_summary\")).__name__})')
    
    if not has_entry_snapshot or not has_feature_context:
        print('')
        print('⚠️  WARNING: Trade journal structure incomplete!')
        sys.exit(1)
    else:
        print('')
        print('✅ Trade journal structure verified!')
else:
    print('⚠️  No trade JSON files found for verification')
" "$MERGED_TRADE_JOURNAL_DIR"
    
else
    echo "❌ ERROR: Merged trade journal index not found: $MERGED_TRADE_JOURNAL_INDEX"
    exit 1
fi

echo ""
echo "=================================================================================="
echo "✅ SNIPER ENTRY_V10.1 FLAT THRESHOLD 0.18 FULLYEAR 2025 Replay Complete!"
echo "=================================================================================="
echo ""
echo "Run Tag: $RUN_TAG"
echo "Output: $OUTPUT_DIR"
echo "Merged Trade Journal (CSV): $MERGED_TRADE_JOURNAL_INDEX"
echo "Merged Trade Journal (Parquet): $MERGED_TRADE_JOURNAL_PARQUET"
echo ""
echo "Next steps:"
echo "  1. Run risk profile analysis:"
echo "     python -m gx1.tools.analysis.analyze_entry_v10_1_threshold_018_risk_profile \\"
echo "       --trade-journal-dir $MERGED_TRADE_JOURNAL_DIR \\"
echo "       --output-report reports/rl/entry_v10/ENTRY_V10_1_THRESHOLD_0_18_RISK_PROFILE_2025.md \\"
echo "       --output-json data/entry_v10/entry_v10_1_threshold_0_18_risk_hotspots_2025.json"
echo ""

