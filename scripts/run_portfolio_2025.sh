#!/bin/bash
# Portfolio 2025: Combine FARM + SNIPER full-year replays

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

# Defaults
FARM_RUN_DIR=""
SNIPER_RUN_DIR=""
MAX_OPEN_TRADES=1
OUTPUT_DIR="reports/portfolio/2025"

# Parse arguments
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
    --farm-run-dir DIR      FARM full-year run directory (required)
    --sniper-run-dir DIR    SNIPER full-year run directory (required)
    --max-open-trades N     Maximum concurrent open trades (default: 1)
    --output-dir DIR        Output directory (default: reports/portfolio/2025)
    --help                  Show this help message

Example:
    $0 \\
        --farm-run-dir gx1/wf_runs/PORTFOLIO_2025_FARM_FULLYEAR_20251221_120000 \\
        --sniper-run-dir gx1/wf_runs/PORTFOLIO_2025_SNIPER_FULLYEAR_20251221_120000 \\
        --max-open-trades 1 \\
        --output-dir reports/portfolio/2025
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --farm-run-dir)
            FARM_RUN_DIR="$2"
            shift 2
            ;;
        --sniper-run-dir)
            SNIPER_RUN_DIR="$2"
            shift 2
            ;;
        --max-open-trades)
            MAX_OPEN_TRADES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$FARM_RUN_DIR" ]]; then
    echo "ERROR: --farm-run-dir is required" >&2
    usage
fi

if [[ -z "$SNIPER_RUN_DIR" ]]; then
    echo "ERROR: --sniper-run-dir is required" >&2
    usage
fi

# Check directories exist
if [[ ! -d "$FARM_RUN_DIR" ]]; then
    echo "ERROR: FARM run directory not found: $FARM_RUN_DIR" >&2
    exit 1
fi

if [[ ! -d "$SNIPER_RUN_DIR" ]]; then
    echo "ERROR: SNIPER run directory not found: $SNIPER_RUN_DIR" >&2
    exit 1
fi

# Run combination script
echo "=" | tr -d '\n'
echo "Portfolio 2025: Combining FARM + SNIPER"
echo "=" | tr -d '\n'
echo ""
echo "FARM run dir: $FARM_RUN_DIR"
echo "SNIPER run dir: $SNIPER_RUN_DIR"
echo "Max open trades: $MAX_OPEN_TRADES"
echo "Output dir: $OUTPUT_DIR"
echo ""

PYTHONPATH="$PROJECT_ROOT" python3 gx1/portfolio/combine_farm_sniper_2025.py \
    --farm-run-dir "$FARM_RUN_DIR" \
    --sniper-run-dir "$SNIPER_RUN_DIR" \
    --max-open-trades "$MAX_OPEN_TRADES" \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "=" | tr -d '\n'
    echo "Portfolio combination complete!"
    echo "=" | tr -d '\n'
    echo ""
    echo "Reports written to: $OUTPUT_DIR"
    echo "  - portfolio_trades.jsonl"
    echo "  - portfolio_metrics.csv"
    echo "  - summary.txt"
    echo ""
else
    echo "ERROR: Portfolio combination failed (exit code: $EXIT_CODE)" >&2
    exit $EXIT_CODE
fi

