#!/usr/bin/env bash
# Quick performance profile for ENTRY_V10.1 FLAT threshold 0.18
#
# Runs a small time period (1 month) with 1 worker to get performance metrics.
# This is MUCH faster than FULLYEAR and sufficient to identify bottlenecks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

POLICY_THRESHOLD018="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_0_18.yaml"
DATA_FILE="data/raw/xauusd_m5_2025_bid_ask.parquet"

# 1 month test (UTC) - enough data to get meaningful perf metrics
START_TS="2025-01-01T00:00:00Z"
END_TS="2025-02-01T00:00:00Z"  # 1 month

LOG_FILE="/tmp/profile_v10_1_threshold018_quick.log"

echo "=================================================================================="
echo "QUICK PERFORMANCE PROFILE - ENTRY_V10.1 FLAT THRESHOLD 0.18"
echo "=================================================================================="
echo ""
echo "Configuration:"
echo "  - Policy: $POLICY_THRESHOLD018"
echo "  - Entry: ENTRY_V10.1 HYBRID (variant=v10_1, seq_len=90)"
echo "  - Threshold: min_prob_long=0.18"
echo "  - Period: 1 month (${START_TS} → ${END_TS})"
echo "  - Workers: 1 (single worker, no parallel chunks)"
echo "  - Log level: INFO"
echo "  - Log file: $LOG_FILE"
echo ""

# Set log level
export GX1_LOGLEVEL=INFO
# Force CPU device for replay (ops safety - avoids MPS hangs)
export GX1_FORCE_TORCH_DEVICE=cpu

# Validate files exist
if [ ! -f "$POLICY_THRESHOLD018" ]; then
    echo "❌ ERROR: Policy file not found: $POLICY_THRESHOLD018"
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "❌ ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Prepare test data
echo "[1/3] Preparing test data (1 month, EU/OVERLAP/US only)..."
TEST_DATA_DIR="data/temp/perf_quick_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DATA_DIR"

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

# Filter to SNIPER sessions
sessions = ['EU', 'OVERLAP', 'US']
mask = df.index.map(lambda ts: infer_session_tag(ts) in sessions)
df = df[mask]

print(f'✅ Test data: {len(df):,} rows')
print(f'   Date range: {df.index.min()} → {df.index.max()}')

df.to_parquet(output_file)
print(f'✅ Saved to: {output_file}')
" "$DATA_FILE" "$START_TS" "$END_TS" "$TEST_DATA_DIR/test_data.parquet" 2>&1 | tee -a "$LOG_FILE"

TEST_DATA="$TEST_DATA_DIR/test_data.parquet"

# Run replay (single worker, no parallel chunks)
echo ""
echo "[2/3] Running replay with performance timing (single worker)..."
echo "   This should take ~10-30 minutes..."
echo ""

OUTPUT_DIR="$TEST_DATA_DIR/output"
mkdir -p "$OUTPUT_DIR"

# Use run_mini_replay_perf.py for fail-fast stack dumps and perf summary
python3 scripts/run_mini_replay_perf.py "$POLICY_THRESHOLD018" "$TEST_DATA" "$OUTPUT_DIR" "$LOG_FILE" 2>&1 | tee -a "$LOG_FILE"

# Check for performance summary
echo ""
echo "[3/3] Checking for performance summary..."
echo ""

if grep -q "\[REPLAY_PERF_SUMMARY\]" "$LOG_FILE"; then
    echo "✅ Found [REPLAY_PERF_SUMMARY] in log"
    echo ""
    echo "Performance Summary:"
    grep "\[REPLAY_PERF_SUMMARY\]" "$LOG_FILE" | tail -1
    echo ""
else
    echo "⚠️  WARNING: No [REPLAY_PERF_SUMMARY] found in log"
    echo "   Check log file: $LOG_FILE"
    exit 1
fi

echo ""
echo "=================================================================================="
echo "✅ QUICK PERFORMANCE PROFILE COMPLETE"
echo "=================================================================================="
echo ""
echo "Log file: $LOG_FILE"
echo "Output dir: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Generate performance report:"
echo "     python -m gx1.tools.analysis.summarize_replay_perf_quick_v10_1_threshold018 \\"
echo "       --log-file $LOG_FILE"
echo "  2. Review report:"
echo "     cat reports/rl/entry_v10/ENTRY_V10_1_REPLAY_PERF_QUICK_THRESHOLD_0_18.md"
echo ""

