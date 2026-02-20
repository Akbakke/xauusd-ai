#!/bin/bash
# Script to run XGB → Transformer Truth Report
# Usage:
#   ./run_truth_report.sh <date_from> <date_to> <output_dir_for_reports>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATE_FROM="${1:-2025-01-01}"
DATE_TO="${2:-2025-01-08}"
REPORTS_OUTPUT_DIR="${3:-${GX1_DATA:-/tmp/gx1_data}/reports/truth}"

echo "=========================================="
echo "XGB → Transformer Truth Report Generator"
echo "=========================================="
echo "Date range: $DATE_FROM to $DATE_TO"
echo "Reports output: $REPORTS_OUTPUT_DIR"
echo ""

cd "$WORKSPACE_ROOT"

# Run truth report generator
/home/andre2/venvs/gx1/bin/python gx1/scripts/build_xgb_transformer_truth_report.py \
    --date-from "$DATE_FROM" \
    --date-to "$DATE_TO" \
    --mode PREBUILT \
    --output-dir "$REPORTS_OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Report generation complete"
echo "Check: $REPORTS_OUTPUT_DIR"
echo "=========================================="
