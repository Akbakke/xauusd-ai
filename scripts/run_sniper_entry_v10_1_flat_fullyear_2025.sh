#!/usr/bin/env bash
# SNIPER ENTRY_V10.1 FLAT FULLYEAR 2025 Replay
#
# Runs FULLYEAR replay (Q1-Q4 2025) with ENTRY_V10.1 FLAT config.
# Uses same data source as P4.1 replays.
#
# Usage:
#   bash scripts/run_sniper_entry_v10_1_flat_fullyear_2025.sh

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

POLICY="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT.yaml"
DATA_FILE="data/raw/xauusd_m5_2025_bid_ask.parquet"

# FULLYEAR 2025 (UTC)
START_TS="2025-01-01T00:00:00Z"
END_TS="2026-01-01T00:00:00Z"  # exclusive

echo "=================================================================================="
echo "SNIPER ENTRY_V10.1 FLAT FULLYEAR 2025 Replay"
echo "=================================================================================="
echo ""
echo "Configuration:"
echo "  - Policy: $POLICY"
echo "  - Entry: ENTRY_V10.1 HYBRID (variant=v10_1, seq_len=90)"
echo "  - Sizing: FLAT (baseline sizing, same as P4.1)"
echo "  - Exit: ExitCritic V1 + RULE5/RULE6A (same as P4.1)"
echo "  - Data source: Same as P4.1 replays"
echo "  - Date range: $START_TS → $END_TS"
echo ""

# Validate config exists
if [ ! -f "$POLICY" ]; then
    echo "❌ ERROR: Policy file not found: $POLICY"
    exit 1
fi

# Validate data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ ERROR: Data file not found: $DATA_FILE"
    echo "   Expected: $DATA_FILE"
    exit 1
fi

# Create output directory
OUTPUT_BASE="data/replay/sniper/entry_v10_1_flat/2025"
mkdir -p "$OUTPUT_BASE"
RUN_TAG="ENTRY_V10_1_FLAT_FULLYEAR_2025_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$OUTPUT_BASE/$RUN_TAG"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Prepare filtered data (EU/OVERLAP/US only, same as P4.1)
echo "[1/3] Preparing FULLYEAR 2025 data (EU/OVERLAP/US only)..."
FILTERED_DATA="$OUTPUT_DIR/fullyear_2025_filtered.parquet"

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
echo "[2/3] Running FULLYEAR replay..."
echo "   This may take a while for FULLYEAR data..."
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

print('Starting FULLYEAR 2025 replay...')
runner.run_replay(data_file)
print('✅ Replay complete!')
" "$POLICY" "$FILTERED_DATA" "$OUTPUT_DIR"

echo ""
echo "[3/3] Verifying trade journal..."
echo ""

# Check trade journal
TRADE_JOURNAL_DIR="$OUTPUT_DIR/trade_journal"
TRADE_JOURNAL_INDEX="$TRADE_JOURNAL_DIR/trade_journal_index.csv"
TRADE_JSON_DIR="$TRADE_JOURNAL_DIR/trades"

if [ ! -f "$TRADE_JOURNAL_INDEX" ]; then
    echo "⚠️  WARNING: trade_journal_index.csv not found"
else
    TRADE_COUNT=$(tail -n +2 "$TRADE_JOURNAL_INDEX" 2>/dev/null | wc -l | tr -d ' ' || echo "0")
    echo "✅ Trade journal index found: $TRADE_COUNT trades"
fi

if [ ! -d "$TRADE_JSON_DIR" ]; then
    echo "⚠️  WARNING: trade_journal/trades/ directory not found"
else
    JSON_COUNT=$(find "$TRADE_JSON_DIR" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
    echo "✅ Trade JSON files found: $JSON_COUNT files"
fi

# Verify p_long_v10_1 in trades (quick check)
if [ -f "$TRADE_JOURNAL_INDEX" ] && [ "$(head -1 "$TRADE_JOURNAL_INDEX" | grep -c 'p_long' || echo '0')" -gt 0 ]; then
    echo "✅ Trade journal contains p_long fields"
else
    echo "⚠️  WARNING: Trade journal may not contain p_long fields"
fi

echo ""
echo "=================================================================================="
echo "✅ SNIPER ENTRY_V10.1 FLAT FULLYEAR 2025 Replay Complete!"
echo "=================================================================================="
echo ""
echo "Run Tag: $RUN_TAG"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Trade Journal: $TRADE_JOURNAL_DIR/"
echo ""
echo "Next steps:"
echo "  1. Run label quality analysis:"
echo "     python -m gx1.tools.debug.analyze_entry_v10_label_quality \\"
echo "       --v10-dataset data/entry_v10/entry_v10_1_dataset_seq90.parquet \\"
echo "       --v10-model models/entry_v10/entry_v10_1_transformer.pt \\"
echo "       --v10-meta models/entry_v10/entry_v10_1_transformer_meta.json \\"
echo "       --trade-journal $TRADE_JOURNAL_DIR \\"
echo "       --output-report reports/rl/entry_v10/ENTRY_V10_1_LABEL_QUALITY_2025.md"
echo ""
echo "  2. Check trade journal: ls -lah $TRADE_JOURNAL_DIR/trades/"
echo "  3. View summary: cat $TRADE_JOURNAL_INDEX"
echo ""

