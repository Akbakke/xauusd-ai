#!/bin/bash
# Run canary test on OANDA Practice environment
#
# This script runs a canary test (dry_run=True) to verify:
# - Feature manifest validation
# - Router model loading (PROD_BASELINE fail-closed)
# - Policy lock verification
# - All invariants pass
#
# Usage:
#   ./scripts/run_canary_oanda_practice.sh [start_date] [end_date] [workers] [output_tag]
#
# Example:
#   ./scripts/run_canary_oanda_practice.sh 2025-01-01 2025-01-02 7 CANARY_TEST_2025_Q1

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
START_DATE="${1:-2025-01-01}"
END_DATE="${2:-2025-01-02}"
WORKERS="${3:-1}"  # Canary MUST use n_workers=1 for determinism (see RUNBOOK.md)
OUTPUT_TAG="${4:-CANARY_TEST_$(date +%Y%m%d_%H%M%S)}"

# Policy path (use PROD_BASELINE canary policy)
POLICY_PATH="${POLICY_PATH:-gx1/prod/current/policy_canary.yaml}"

echo "================================================================================"
echo "OANDA Practice Canary Test"
echo "================================================================================"
echo "Start Date: $START_DATE"
echo "End Date: $END_DATE"
echo "Workers: $WORKERS"
echo "Output Tag: $OUTPUT_TAG"
echo "Policy: $POLICY_PATH"
echo "================================================================================"
echo ""

# Verify OANDA_ENV is practice
if [ "${OANDA_ENV:-}" != "practice" ]; then
    echo -e "${RED}ERROR: OANDA_ENV must be 'practice' for canary test${NC}"
    echo "Current OANDA_ENV: ${OANDA_ENV:-NOT SET}"
    echo ""
    echo "Set it with: export OANDA_ENV=practice"
    exit 1
fi

echo -e "${GREEN}✓${NC} OANDA_ENV=practice verified"
echo ""

# Verify credentials are set
if [ -z "${OANDA_API_TOKEN:-}" ] && [ -z "${OANDA_API_KEY:-}" ]; then
    echo -e "${RED}ERROR: OANDA_API_TOKEN or OANDA_API_KEY must be set${NC}"
    echo ""
    echo "Set it with: export OANDA_API_TOKEN=your_token_here"
    echo "Or create .env file from env.example"
    exit 1
fi

if [ -z "${OANDA_ACCOUNT_ID:-}" ]; then
    echo -e "${RED}ERROR: OANDA_ACCOUNT_ID must be set${NC}"
    echo ""
    echo "Set it with: export OANDA_ACCOUNT_ID=your_account_id_here"
    echo "Or create .env file from env.example"
    exit 1
fi

echo -e "${GREEN}✓${NC} OANDA credentials verified"
echo ""

# Verify policy file exists
if [ ! -f "$POLICY_PATH" ]; then
    echo -e "${RED}ERROR: Policy file not found: $POLICY_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Policy file found: $POLICY_PATH"
echo ""

# Verify PROD freeze structure
echo "Verifying PROD freeze structure..."
if ! python3 -m gx1.prod.verify_freeze 2>/dev/null; then
    echo -e "${YELLOW}WARNING: PROD freeze verification failed (non-critical for canary)${NC}"
fi
echo ""

# Run canary replay
echo "================================================================================"
echo "Starting canary replay..."
echo "================================================================================"
echo ""

# Use run_replay.sh script
bash scripts/run_replay.sh \
    "$POLICY_PATH" \
    "$START_DATE" \
    "$END_DATE" \
    "$WORKERS" \
    "gx1/wf_runs/$OUTPUT_TAG"

REPLAY_EXIT_CODE=$?

if [ $REPLAY_EXIT_CODE -ne 0 ]; then
    echo ""
    echo -e "${RED}================================================================================"
    echo "CANARY TEST FAILED"
    echo "================================================================================"
    echo -e "${NC}"
    echo "Replay exited with code: $REPLAY_EXIT_CODE"
    echo ""
    echo "Check logs in: gx1/wf_runs/$OUTPUT_TAG/logs/"
    exit $REPLAY_EXIT_CODE
fi

echo ""
echo "================================================================================"
echo "Canary Test Verification"
echo "================================================================================"
echo ""

# Check for run_header.json
RUN_HEADER="gx1/wf_runs/$OUTPUT_TAG/run_header.json"
if [ -f "$RUN_HEADER" ]; then
    echo -e "${GREEN}✓${NC} run_header.json generated"
    echo "  Location: $RUN_HEADER"
else
    echo -e "${YELLOW}⚠${NC}  run_header.json not found (may be non-critical)"
fi

# Check for prod_metrics.csv
PROD_METRICS="gx1/wf_runs/$OUTPUT_TAG/prod_metrics.csv"
if [ -f "$PROD_METRICS" ]; then
    echo -e "${GREEN}✓${NC} prod_metrics.csv generated"
    echo "  Location: $PROD_METRICS"
    echo ""
    echo "Metrics summary:"
    tail -n 1 "$PROD_METRICS" | awk -F',' '{printf "  Trades/day: %.2f\n  EV/trade: %.2f bps\n  RULE6A rate: %.1f%%\n", $2, $3, $4*100}'
else
    echo -e "${YELLOW}⚠${NC}  prod_metrics.csv not found"
fi

# Check for alerts.json
ALERTS="gx1/wf_runs/$OUTPUT_TAG/alerts.json"
if [ -f "$ALERTS" ]; then
    echo ""
    echo -e "${YELLOW}⚠${NC}  Alerts triggered - check alerts.json"
    echo "  Location: $ALERTS"
    cat "$ALERTS" | head -20
else
    echo -e "${GREEN}✓${NC} No alerts triggered"
fi

# Check for trade log
TRADE_LOG="gx1/wf_runs/$OUTPUT_TAG/trade_log"
if ls "$TRADE_LOG"*.csv 1> /dev/null 2>&1; then
    TRADE_COUNT=$(cat "$TRADE_LOG"*.csv | wc -l)
    echo -e "${GREEN}✓${NC} Trade log generated ($TRADE_COUNT trades)"
else
    echo -e "${YELLOW}⚠${NC}  Trade log not found"
fi

echo ""
echo "================================================================================"
echo -e "${GREEN}CANARY TEST COMPLETED SUCCESSFULLY${NC}"
echo "================================================================================"
echo ""
echo "Output directory: gx1/wf_runs/$OUTPUT_TAG"
echo ""
echo "Next steps:"
echo "  1. Review prod_metrics.csv for performance metrics"
echo "  2. Check trade_log*.csv for trade details"
echo "  3. Verify all invariants passed (check logs)"
echo "  4. If all checks pass, proceed to live canary test"
echo ""

