#!/usr/bin/env bash
# SNIPER ENTRY_V10.1 FLAT UNGATED FULLYEAR 2025 Replay
#
# Runs FULLYEAR 2025 replay with ENTRY_V10.1 in FLAT UNGATED mode (no p_long threshold filtering).
# This generates ALL V10.1 signals for threshold discovery and label-quality analysis.
#
# Usage:
#   bash scripts/run_sniper_entry_v10_1_flat_ungated_fullyear_2025.sh [--n-workers N]

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

POLICY_UNGATED="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT_UNGATED.yaml"
DATA_FILE="data/raw/xauusd_m5_2025_bid_ask.parquet"

# FULLYEAR 2025 (UTC)
START_TS="2025-01-01T00:00:00Z"
END_TS="2026-01-01T00:00:00Z"  # exclusive

echo "=================================================================================="
echo "SNIPER ENTRY_V10.1 FLAT UNGATED FULLYEAR 2025 Replay"
echo "=================================================================================="
echo ""
echo "Configuration:"
echo "  - Policy: $POLICY_UNGATED"
echo "  - Entry: ENTRY_V10.1 HYBRID (variant=v10_1, seq_len=90)"
echo "  - Mode: FLAT UNGATED (no p_long threshold filtering)"
echo "  - Purpose: Threshold discovery and label-quality analysis"
echo "  - Data source: Same as P4.1 replays"
echo "  - Date range: $START_TS → $END_TS"
echo "  - Workers: $N_WORKERS (parallel replay)"
echo ""

# Validate config exists
if [ ! -f "$POLICY_UNGATED" ]; then
    echo "❌ ERROR: UNGATED policy file not found: $POLICY_UNGATED"
    exit 1
fi

# Validate data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Create output directories
OUTPUT_BASE="data/replay/sniper/entry_v10_1_flat_ungated/2025"
mkdir -p "$OUTPUT_BASE"
RUN_TAG="ABTEST_FLAT_UNGATED_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$OUTPUT_BASE/$RUN_TAG"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Prepare filtered data (EU/OVERLAP/US only, same as P4.1)
echo "[1/4] Preparing FULLYEAR 2025 data (EU/OVERLAP/US only)..."
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
echo "[2/4] Running UNGATED replay with $N_WORKERS parallel workers..."
echo "   This may take a while for FULLYEAR data..."
echo ""

# Step 2a: Split into chunks
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

# Step 2b: Run parallel replays
mkdir -p "$OUTPUT_DIR/parallel_chunks"

run_chunk() {
    local chunk_id=$1
    local chunk_file="$CHUNK_DIR/chunk_${chunk_id}.parquet"
    local chunk_output_dir="$OUTPUT_DIR/parallel_chunks/chunk_${chunk_id}"
    
    mkdir -p "$chunk_output_dir"
    
    echo "[CHUNK $chunk_id/$N_WORKERS] Starting..."
    
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
print(f'[CHUNK {chunk_id}] ✅ Complete')
" "$chunk_id" "$POLICY_UNGATED" "$chunk_file" "$chunk_output_dir" 2>&1 | tee "$OUTPUT_DIR/parallel_chunks/chunk_${chunk_id}.log"
    
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

# Step 2c: Merge trade journals
echo ""
echo "[3/4] Merging trade journals..."
python3 gx1/scripts/merge_trade_journals.py "$OUTPUT_DIR"

MERGED_TRADE_JOURNAL="$OUTPUT_DIR/trade_journal"
MERGED_TRADE_JOURNAL_INDEX="$MERGED_TRADE_JOURNAL/trade_journal_index.csv"

# Verify trade journal
echo ""
echo "[4/4] Verifying trade journal..."

if [ ! -f "$MERGED_TRADE_JOURNAL_INDEX" ]; then
    echo "⚠️  WARNING: trade_journal_index.csv not found"
    TRADE_COUNT=0
else
    TRADE_COUNT=$(tail -n +2 "$MERGED_TRADE_JOURNAL_INDEX" 2>/dev/null | wc -l | tr -d ' ' || echo "0")
    echo "✅ Trade journal: $TRADE_COUNT trades"
fi

# Create merged parquet for analysis
if [ -f "$MERGED_TRADE_JOURNAL_INDEX" ] && [ "$TRADE_COUNT" -gt 0 ]; then
    echo ""
    echo "Creating merged trade journal parquet for analysis..."
    
    MERGED_PARQUET="$OUTPUT_DIR/merged_trade_journal.parquet"
    
    python3 -c "
import pandas as pd
from pathlib import Path
import json
import sys

trade_journal_dir = Path(sys.argv[1])
output_parquet = Path(sys.argv[2])

# Load trade journal index
index_csv = trade_journal_dir / 'trade_journal_index.csv'
if not index_csv.exists():
    print(f'ERROR: trade_journal_index.csv not found: {index_csv}')
    sys.exit(1)

df = pd.read_csv(index_csv)

# Load full trade data from JSON files
trades_dir = trade_journal_dir / 'trades'
if trades_dir.exists():
    trades_data = []
    for json_file in sorted(trades_dir.glob('*.json')):
        with open(json_file, 'r') as f:
            trade_data = json.load(f)
            # Flatten nested dicts for easier analysis
            flat_trade = {}
            for key, value in trade_data.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flat_trade[f'{key}.{subkey}'] = subvalue
                else:
                    flat_trade[key] = value
            trades_data.append(flat_trade)
    
    if trades_data:
        df_trades = pd.DataFrame(trades_data)
        # Merge with index on trade_id
        if 'trade_id' in df.columns and 'trade_id' in df_trades.columns:
            df = df.merge(df_trades, on='trade_id', how='left', suffixes=('', '_json'))
        else:
            df = pd.concat([df, df_trades], axis=1)

df.to_parquet(output_parquet)
print(f'✅ Saved merged trade journal to: {output_parquet}')
print(f'   Total trades: {len(df):,}')
" "$MERGED_TRADE_JOURNAL" "$MERGED_PARQUET"
fi

echo ""
echo "=================================================================================="
echo "✅ SNIPER ENTRY_V10.1 FLAT UNGATED FULLYEAR 2025 Replay Complete!"
echo "=================================================================================="
echo ""
echo "Run Tag: $RUN_TAG"
echo ""
echo "Output Directory: $OUTPUT_DIR"
echo "Trade Journal: $MERGED_TRADE_JOURNAL"
if [ -f "$MERGED_PARQUET" ]; then
    echo "Merged Parquet: $MERGED_PARQUET"
fi
echo ""
echo "Total Trades: $TRADE_COUNT"
echo ""
echo "Next steps:"
echo "  1. Run label-quality analysis:"
echo "     python -m gx1.tools.debug.analyze_entry_v10_label_quality \\"
echo "         --variant v10_1 \\"
echo "         --trade-journal $MERGED_TRADE_JOURNAL \\"
echo "         --output-json data/entry_v10/entry_v10_1_label_quality_2025_ungated.json \\"
echo "         --output-report reports/rl/entry_v10/ENTRY_V10_1_LABEL_QUALITY_2025_UNGATED.md"
echo ""
echo "  2. Run p_long distribution analysis:"
echo "     python -m gx1.tools.analysis.describe_entry_v10_1_p_long_distribution \\"
echo "         --trade-journal $MERGED_TRADE_JOURNAL \\"
echo "         --label-quality-json data/entry_v10/entry_v10_1_label_quality_2025_ungated.json \\"
echo "         --output-report reports/rl/entry_v10/ENTRY_V10_1_P_LONG_DISTRIBUTION_2025_UNGATED.md"
echo ""

