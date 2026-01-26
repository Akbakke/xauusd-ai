#!/bin/bash
# scripts/run_replay_1m_perf.sh
# 1-month replay milestone for scaling verification.
# Uses thread limits wrapper for stability and NP_ROLLING toggle.

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
test "$(pwd)" = "$ROOT" || { echo "❌ FAIL: Not in repo-root (pwd=$(pwd), root=$ROOT)"; exit 1; }
echo "[RUN_CTX] root=$ROOT"
echo "[RUN_CTX] head=$(git rev-parse --short HEAD)"
echo "[RUN_CTX] whoami=$(whoami) host=$(hostname)"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default: 1 month from start date
START_DATE="${1:-2025-01-01T00:00:00Z}"
END_DATE="${2:-2025-02-01T00:00:00Z}"

# Policy and data file (can be overridden)
POLICY="${3:-gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_0_18.yaml}"
DATA_FILE="${4:-data/temp/january_2025_replay.parquet}"

# Output directory
OUTPUT_DIR="${5:-data/temp/replay_1m_$(date +%Y%m%d_%H%M%S)}"

echo "=================================================================================="
echo "1-MONTH REPLAY MILESTONE (Scaling Verification)"
echo "=================================================================================="
echo "Start: $START_DATE"
echo "End: $END_DATE"
echo "Policy: $POLICY"
echo "Data file: $DATA_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "=================================================================================="
echo ""

# Use thread limits wrapper
export GX1_REPLAY_NO_CSV=1
export GX1_FEATURE_USE_NP_ROLLING=1
export GX1_REPLAY_INCREMENTAL_FEATURES=1

# Run with date filtering
cd "$PROJECT_ROOT"
"$SCRIPT_DIR/run_replay_with_thread_limits.sh" \
    scripts/run_mini_replay_perf.py \
    "$POLICY" \
    "$DATA_FILE" \
    "$OUTPUT_DIR" \
    --start "$START_DATE" \
    --end "$END_DATE"

echo ""
echo "=================================================================================="
echo "1-month replay complete."
echo "Performance summary: $OUTPUT_DIR/REPLAY_PERF_SUMMARY.md"
echo "Performance JSON: $OUTPUT_DIR/REPLAY_PERF_SUMMARY.json"
echo "=================================================================================="

