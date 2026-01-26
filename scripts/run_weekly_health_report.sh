#!/bin/bash
# Weekly Health Report runner for SNIPER

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
test "$(pwd)" = "$ROOT" || { echo "❌ FAIL: Not in repo-root (pwd=$(pwd), root=$ROOT)"; exit 1; }
echo "[RUN_CTX] root=$ROOT"
echo "[RUN_CTX] head=$(git rev-parse --short HEAD)"
echo "[RUN_CTX] whoami=$(whoami) host=$(hostname)"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <journal_dir> <start_date> <end_date> [output_csv] [baseline_regime_json] [archive_dir]"
    echo ""
    echo "Arguments:"
    echo "  journal_dir        Root directory containing trade journals (parallel_chunks or trade_journal)"
    echo "  start_date         Start date (YYYY-MM-DD)"
    echo "  end_date           End date (YYYY-MM-DD)"
    echo "  output_csv         Optional: Output CSV path"
    echo "  baseline_regime_json Optional: Baseline regime distribution JSON"
    echo "  archive_dir        Optional: Archive directory (reports/weekly/YYYY-MM-DD/)"
    echo ""
    echo "Example:"
    echo "  $0 gx1/wf_runs/SNIPER_OBS_Q4_2025_baseline_20251221_135105 2025-10-01 2025-10-07"
    echo ""
    echo "With archiving:"
    echo "  $0 gx1/wf_runs/SNIPER_OBS_Q4_2025_baseline_20251221_135105 2025-10-01 2025-10-07 \\"
    echo "    reports/weekly_health.csv \\"
    echo "    docs/ops/baseline_regime_dist.json \\"
    echo "    reports/weekly/2025-10-07"
    exit 1
fi

JOURNAL_DIR="$1"
START_DATE="$2"
END_DATE="$3"
OUTPUT_CSV="${4:-}"
BASELINE_REGIME="${5:-}"
ARCHIVE_DIR="${6:-}"

# Get baseline commit hash and policy name
BASELINE_COMMIT=""
POLICY_NAME=""
if [ -f "docs/ops/BASELINE.md" ]; then
    BASELINE_COMMIT=$(grep "Commit Hash" docs/ops/BASELINE.md | head -1 | sed 's/.*`\([^`]*\)`.*/\1/')
    POLICY_NAME=$(grep "File" docs/ops/BASELINE.md | head -1 | sed 's|.*/\([^/]*\)\.yaml.*|\1|')
fi

# Validate journal directory
if [ ! -d "$JOURNAL_DIR" ]; then
    echo "ERROR: Journal directory not found: $JOURNAL_DIR" >&2
    exit 1
fi

# Validate dates
if ! date -d "$START_DATE" >/dev/null 2>&1 && ! date -j -f "%Y-%m-%d" "$START_DATE" >/dev/null 2>&1; then
    echo "ERROR: Invalid start date format: $START_DATE (expected YYYY-MM-DD)" >&2
    exit 1
fi

if ! date -d "$END_DATE" >/dev/null 2>&1 && ! date -j -f "%Y-%m-%d" "$END_DATE" >/dev/null 2>&1; then
    echo "ERROR: Invalid end date format: $END_DATE (expected YYYY-MM-DD)" >&2
    exit 1
fi

# Build command
CMD="cd \"$PROJECT_ROOT\" && PYTHONPATH=\"$PROJECT_ROOT\" python gx1/sniper/analysis/weekly_health_report.py"
CMD="$CMD --journal-root \"$JOURNAL_DIR\""
CMD="$CMD --start-date \"$START_DATE\""
CMD="$CMD --end-date \"$END_DATE\""

if [ -n "$OUTPUT_CSV" ]; then
    CMD="$CMD --output-csv \"$OUTPUT_CSV\""
fi

if [ -n "$BASELINE_REGIME" ]; then
    if [ ! -f "$BASELINE_REGIME" ]; then
        echo "ERROR: Baseline regime JSON not found: $BASELINE_REGIME" >&2
        exit 1
    fi
    CMD="$CMD --baseline-regime-dist \"$BASELINE_REGIME\""
fi

if [ -n "$ARCHIVE_DIR" ]; then
    CMD="$CMD --archive-dir \"$ARCHIVE_DIR\""
    if [ -n "$BASELINE_COMMIT" ]; then
        CMD="$CMD --baseline-commit \"$BASELINE_COMMIT\""
    fi
    if [ -n "$POLICY_NAME" ]; then
        CMD="$CMD --policy-name \"$POLICY_NAME\""
    fi
fi

# Run
eval "$CMD"

