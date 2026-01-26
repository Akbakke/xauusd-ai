#!/usr/bin/env bash
# SNIPER ENTRY_V10.1 FLAT THRESHOLD 0.30 FULLYEAR 2025 Replay
#
# Runs FULLYEAR 2025 replay with ENTRY_V10.1 in FLAT mode with threshold=0.30.
# This is a threshold-tested baseline based on UNGATED analysis.
#
# Usage:
#   bash scripts/run_sniper_entry_v10_1_flat_threshold030_2025.sh [--n-workers N]

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

POLICY_THRESHOLD030="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_030.yaml"
DATA_FILE="data/raw/xauusd_m5_2025_bid_ask.parquet"

# FULLYEAR 2025 (UTC)
START_TS="2025-01-01T00:00:00Z"
END_TS="2026-01-01T00:00:00Z"  # exclusive

echo "=================================================================================="
echo "SNIPER ENTRY_V10.1 FLAT THRESHOLD 0.30 FULLYEAR 2025 Replay"
echo "=================================================================================="
echo ""
echo "Configuration:"
echo "  - Policy: $POLICY_THRESHOLD030"
echo "  - Entry: ENTRY_V10.1 HYBRID (variant=v10_1, seq_len=90)"
echo "  - Mode: FLAT with threshold=0.30 (threshold-tested baseline)"
echo "  - Threshold: min_prob_long=0.30, p_side_min.long=0.30"
echo "  - Data source: Same as P4.1 replays"
echo "  - Date range: $START_TS → $END_TS"
echo "  - Workers: $N_WORKERS (parallel replay)"
echo ""

# Validate config exists
if [ ! -f "$POLICY_THRESHOLD030" ]; then
    echo "❌ ERROR: THRESHOLD 0.30 policy file not found: $POLICY_THRESHOLD030"
    exit 1
fi

# Validate data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Create output directories
OUTPUT_BASE="data/replay/sniper/entry_v10_1_flat_threshold030/2025"
mkdir -p "$OUTPUT_BASE"
RUN_TAG="FLAT_THRESHOLD_030_$(date +%Y%m%d_%H%M%S)"
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
echo "[2/4] Running THRESHOLD 0.30 replay with $N_WORKERS parallel workers..."
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
" "$chunk_id" "$POLICY_THRESHOLD030" "$chunk_file" "$chunk_output_dir" 2>&1 | tee "$OUTPUT_DIR/parallel_chunks/chunk_${chunk_id}.log"
    
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
python3 gx1/scripts/merge_trade_journals.py "$OUTPUT_DIR" || {
    echo "⚠️  Merge script failed, but continuing..."
}

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

# Generate summary report
echo ""
echo "Generating summary report..."

SUMMARY_REPORT="reports/rl/entry_v10/ENTRY_V10_1_FLAT_030_FULLYEAR.md"

python3 << PYEOF
import pandas as pd
from pathlib import Path
from datetime import datetime

output_dir = Path('$OUTPUT_DIR')
trade_journal_index = output_dir / 'trade_journal' / 'trade_journal_index.csv'

lines = [
    '# ENTRY_V10.1 FLAT THRESHOLD 0.30 FULLYEAR 2025 Replay Results',
    '',
    f'**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    '',
    '**Configuration:**',
    '- Policy: GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_030',
    '- Entry: ENTRY_V10.1 HYBRID (variant=v10_1, seq_len=90)',
    '- Threshold: min_prob_long=0.30, p_side_min.long=0.30',
    '- Sizing: FLAT (baseline sizing, no aggressive overlays)',
    '- Exit: ExitCritic V1 + RULE5/RULE6A',
    '',
    '## Results Summary',
    '',
]

if trade_journal_index.exists():
    df = pd.read_csv(trade_journal_index, on_bad_lines='skip', engine='python')
    
    # Filter trades with entry_time and pnl_bps
    df = df[df['entry_time'].notna() & (df['entry_time'] != '')].copy()
    
    if len(df) > 0:
        # Try to parse dates
        try:
            df['entry_time_parsed'] = pd.to_datetime(df['entry_time'], errors='coerce', utc=True)
            valid_dates = df[df['entry_time_parsed'].notna()]
            if len(valid_dates) > 0:
                lines.append(f'- **Total Trades:** {len(valid_dates):,}')
                lines.append(f'- **Date Range:** {valid_dates["entry_time_parsed"].min()} → {valid_dates["entry_time_parsed"].max()}')
        except:
            lines.append(f'- **Total Trades:** {len(df):,}')
        
        # PnL statistics
        if 'pnl_bps' in df.columns:
            df['pnl_bps_numeric'] = pd.to_numeric(df['pnl_bps'], errors='coerce')
            pnl = df['pnl_bps_numeric'].dropna()
            if len(pnl) > 0:
                lines.extend([
                    '',
                    '## PnL Statistics',
                    '',
                    f'- **Mean PnL:** {pnl.mean():.2f} bps',
                    f'- **Median PnL:** {pnl.median():.2f} bps',
                    f'- **Std PnL:** {pnl.std():.2f} bps',
                    f'- **Min PnL:** {pnl.min():.2f} bps',
                    f'- **Max PnL:** {pnl.max():.2f} bps',
                    f'- **Win Rate:** {(pnl > 0).mean()*100:.1f}%',
                    f'- **Total PnL:** {pnl.sum():.2f} bps',
                ])
        
        # Sessions
        if 'session' in df.columns:
            session_counts = df['session'].value_counts()
            lines.extend([
                '',
                '## Sessions',
                '',
            ])
            for session, count in session_counts.items():
                lines.append(f'- **{session}:** {count:,} trades ({count/len(df)*100:.1f}%)')
        
        # Exit reasons
        if 'exit_reason' in df.columns:
            exit_counts = df['exit_reason'].value_counts().head(5)
            lines.extend([
                '',
                '## Exit Reasons (Top 5)',
                '',
            ])
            for reason, count in exit_counts.items():
                lines.append(f'- **{reason}:** {count:,} trades')
    else:
        lines.append('- **Total Trades:** 0 (no trades with entry_time found)')
else:
    lines.append('- **Status:** Trade journal not found or merge failed')

lines.extend([
    '',
    '## Output',
    '',
    f'- **Output Directory:** `$OUTPUT_DIR`',
    f'- **Trade Journal:** `$OUTPUT_DIR/trade_journal`',
    '',
    '## Notes',
    '',
    '- This is a threshold-tested baseline (threshold=0.30 based on UNGATED analysis)',
    '- Compare results with UNGATED replay to assess threshold impact',
    '- Review PnL statistics and trade count vs UNGATED variant',
    '',
])

# Write report
summary_path = Path('$SUMMARY_REPORT')
summary_path.parent.mkdir(parents=True, exist_ok=True)
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f'✅ Summary report saved to: {summary_path}')
PYEOF

echo ""
echo "=================================================================================="
echo "✅ SNIPER ENTRY_V10.1 FLAT THRESHOLD 0.30 FULLYEAR 2025 Replay Complete!"
echo "=================================================================================="
echo ""
echo "Run Tag: $RUN_TAG"
echo ""
echo "Output Directory: $OUTPUT_DIR"
echo "Trade Journal: $MERGED_TRADE_JOURNAL"
echo "Summary Report: $SUMMARY_REPORT"
echo ""
echo "Total Trades: $TRADE_COUNT"
echo ""
echo "Next steps:"
echo "  1. Review summary report: cat $SUMMARY_REPORT"
echo "  2. Compare with UNGATED results to assess threshold impact"
echo "  3. Review trade count and PnL statistics"
echo ""

