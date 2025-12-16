#!/bin/bash
# Execution Smoke Test Wrapper
#
# Wrapper script for running execution smoke test against OANDA Practice API.
# Tests plumbing (credentials → order → fill → close → journal → reconcile).
#
# Usage:
#   export OANDA_ENV=practice
#   export OANDA_API_TOKEN=...
#   export OANDA_ACCOUNT_ID=...
#   ./scripts/run_oanda_exec_smoke_test.sh [--hold-seconds 180] [--units 1] [--instrument XAU_USD] [--run-tag TAG]
#
# Requirements:
#   - OANDA_ENV=practice (hard check)
#   - OANDA_API_TOKEN must be set
#   - OANDA_ACCOUNT_ID must be set

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
HOLD_SECONDS=180
UNITS=1
INSTRUMENT="XAU_USD"
RUN_TAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --hold-seconds)
            HOLD_SECONDS="$2"
            shift 2
            ;;
        --units)
            UNITS="$2"
            shift 2
            ;;
        --instrument)
            INSTRUMENT="$2"
            shift 2
            ;;
        --run-tag)
            RUN_TAG="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--hold-seconds 180] [--units 1] [--instrument XAU_USD] [--run-tag TAG]"
            exit 1
            ;;
    esac
done

# Hard checks
if [ "${OANDA_ENV:-}" != "practice" ]; then
    echo -e "${RED}ERROR: OANDA_ENV must be 'practice'${NC}"
    echo "Set: export OANDA_ENV=practice"
    exit 1
fi

if [ -z "${OANDA_API_TOKEN:-}" ]; then
    echo -e "${RED}ERROR: OANDA_API_TOKEN must be set${NC}"
    exit 1
fi

if [ -z "${OANDA_ACCOUNT_ID:-}" ]; then
    echo -e "${RED}ERROR: OANDA_ACCOUNT_ID must be set${NC}"
    exit 1
fi

# Generate run tag if not provided
if [ -z "$RUN_TAG" ]; then
    RUN_TAG="EXEC_SMOKE_$(date +%Y%m%d_%H%M%S)"
fi

OUT_DIR="gx1/wf_runs/${RUN_TAG}"

echo -e "${GREEN}=== Execution Smoke Test ===${NC}"
echo "Run Tag: ${RUN_TAG}"
echo "Instrument: ${INSTRUMENT}"
echo "Units: ${UNITS}"
echo "Hold Seconds: ${HOLD_SECONDS}"
echo "Output: ${OUT_DIR}"
echo "OANDA_ENV: ${OANDA_ENV}"
echo ""

# Verify we're in the right directory
if [ ! -f "gx1/execution/exec_smoke_test.py" ]; then
    echo -e "${RED}ERROR: Must run from project root${NC}"
    exit 1
fi

# Create output directory
mkdir -p "${OUT_DIR}"

# Run smoke test
echo -e "${YELLOW}Running execution smoke test...${NC}"
PYTHONPATH=. python3 gx1/execution/exec_smoke_test.py \
    --instrument "${INSTRUMENT}" \
    --units "${UNITS}" \
    --hold-seconds "${HOLD_SECONDS}" \
    --run-tag "${RUN_TAG}" \
    --out-dir "${OUT_DIR}" \
    2>&1 | tee "${OUT_DIR}/smoke_test.log"

SMOKE_EXIT_CODE=${PIPESTATUS[0]}

if [ ${SMOKE_EXIT_CODE} -ne 0 ]; then
    echo -e "${RED}ERROR: Smoke test failed with exit code ${SMOKE_EXIT_CODE}${NC}"
    exit ${SMOKE_EXIT_CODE}
fi

echo ""
echo -e "${GREEN}✓ Smoke test completed${NC}"

# Verify artifacts exist
echo ""
echo -e "${YELLOW}Verifying artifacts...${NC}"

MISSING_ARTIFACTS=0

# Check trade journal JSON file
TRADE_JSON_COUNT=$(find "${OUT_DIR}/trade_journal/trades" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
if [ "${TRADE_JSON_COUNT}" -eq 0 ]; then
    echo -e "${RED}✗ Missing: trade_journal/trades/*.json${NC}"
    MISSING_ARTIFACTS=1
else
    echo -e "${GREEN}✓ Found: ${TRADE_JSON_COUNT} trade journal JSON file(s)${NC}"
fi

# Check trade journal index
if [ ! -f "${OUT_DIR}/trade_journal/trade_journal_index.csv" ]; then
    echo -e "${RED}✗ Missing: trade_journal_index.csv${NC}"
    MISSING_ARTIFACTS=1
else
    echo -e "${GREEN}✓ Found: trade_journal_index.csv${NC}"
fi

# Check run_header.json
if [ ! -f "${OUT_DIR}/run_header.json" ]; then
    echo -e "${RED}✗ Missing: run_header.json${NC}"
    MISSING_ARTIFACTS=1
else
    echo -e "${GREEN}✓ Found: run_header.json${NC}"
fi

# Check exec_smoke_summary.json
if [ ! -f "${OUT_DIR}/exec_smoke_summary.json" ]; then
    echo -e "${RED}✗ Missing: exec_smoke_summary.json${NC}"
    MISSING_ARTIFACTS=1
else
    echo -e "${GREEN}✓ Found: exec_smoke_summary.json${NC}"
    
    # Check status
    STATUS=$(python3 << EOF
import json
try:
    with open("${OUT_DIR}/exec_smoke_summary.json", "r") as f:
        d = json.load(f)
    print(d.get("status", "UNKNOWN"))
except Exception as e:
    print("ERROR: " + str(e))
EOF
)
    
    if [ "$STATUS" = "PASS" ]; then
        echo -e "${GREEN}✓ Test status: PASS${NC}"
    else
        echo -e "${RED}✗ Test status: ${STATUS}${NC}"
        MISSING_ARTIFACTS=1
    fi
fi

if [ ${MISSING_ARTIFACTS} -ne 0 ]; then
    echo -e "${RED}ERROR: Missing required artifacts${NC}"
    exit 1
fi

# Run reconciliation
echo ""
echo -e "${YELLOW}Running reconciliation...${NC}"
RECONCILE_OUT="${OUT_DIR}/reconciliation_report.md"

if [ -f "gx1/monitoring/reconcile_oanda.py" ]; then
    python3 gx1/monitoring/reconcile_oanda.py \
        --run "${OUT_DIR}" \
        --out "${RECONCILE_OUT}" \
        2>&1 | tee "${OUT_DIR}/reconcile.log"
    
    RECONCILE_EXIT_CODE=${PIPESTATUS[0]}
    if [ ${RECONCILE_EXIT_CODE} -eq 0 ]; then
        if [ -f "${RECONCILE_OUT}" ]; then
            echo -e "${GREEN}✓ Reconciliation report generated: ${RECONCILE_OUT}${NC}"
            
            # Hard requirement: If TRADE_CLOSED_OANDA exists, must find transactions
            echo ""
            echo -e "${YELLOW}Verifying reconciliation requirements...${NC}"
            
            # Check if any trade has TRADE_CLOSED_OANDA event
            HAS_CLOSED_TRADE=0
            for trade_json in "${OUT_DIR}/trade_journal/trades"/*.json; do
                if [ -f "$trade_json" ]; then
                    HAS_CLOSED=$(python3 << PYEOF
import json
try:
    with open("${trade_json}", "r") as f:
        d = json.load(f)
    events = d.get("execution_events", [])
    for e in events:
        if e.get("event_type") == "TRADE_CLOSED_OANDA":
            print("1")
            exit(0)
    print("0")
except Exception:
    print("0")
PYEOF
)
                    if [ "$HAS_CLOSED" = "1" ]; then
                        HAS_CLOSED_TRADE=1
                        break
                    fi
                fi
            done
            
            if [ ${HAS_CLOSED_TRADE} -eq 1 ]; then
                # Check if reconciliation found transactions
                TXN_COUNT=$(python3 << PYEOF
import re
try:
    with open("${RECONCILE_OUT}", "r") as f:
        content = f.read()
    # Look for "Transactions Found: X" in queries section
    matches = re.findall(r'Transactions Found.*?(\d+)', content)
    total = sum(int(m) for m in matches)
    print(total)
except Exception:
    print("0")
PYEOF
)
                
                MATCHED_COUNT=$(python3 << PYEOF
import re
try:
    with open("${RECONCILE_OUT}", "r") as f:
        content = f.read()
    # Look for "✅ Matched Trades: X"
    match = re.search(r'✅ Matched Trades.*?(\d+)', content)
    if match:
        print(match.group(1))
    else:
        print("0")
except Exception:
    print("0")
PYEOF
)
                
                PARTIAL_COUNT=$(python3 << PYEOF
import re
try:
    with open("${RECONCILE_OUT}", "r") as f:
        content = f.read()
    # Look for "⚠️ Partial Matches: X"
    match = re.search(r'⚠️ Partial Matches.*?(\d+)', content)
    if match:
        print(match.group(1))
    else:
        print("0")
except Exception:
    print("0")
PYEOF
)
                
                TOTAL_MATCHED=$((MATCHED_COUNT + PARTIAL_COUNT))
                
                if [ "${TXN_COUNT}" -eq 0 ] && [ "${TOTAL_MATCHED}" -eq 0 ]; then
                    echo -e "${RED}✗ RECONCILIATION FAILED: Trade has TRADE_CLOSED_OANDA but 0 transactions found${NC}"
                    echo "  This indicates a problem with transaction fetching logic."
                    echo "  Check reconciliation report: ${RECONCILE_OUT}"
                    exit 1
                elif [ "${TOTAL_MATCHED}" -eq 0 ]; then
                    echo -e "${RED}✗ RECONCILIATION FAILED: Trade has TRADE_CLOSED_OANDA but 0 matched trades${NC}"
                    echo "  Transactions were fetched (${TXN_COUNT}) but none matched."
                    echo "  Check reconciliation report: ${RECONCILE_OUT}"
                    exit 1
                else
                    echo -e "${GREEN}✓ Reconciliation requirement met: ${MATCHED_COUNT} OK + ${PARTIAL_COUNT} PARTIAL = ${TOTAL_MATCHED} matched trade(s)${NC}"
                fi
            else
                echo -e "${YELLOW}⚠ No TRADE_CLOSED_OANDA events found, skipping reconciliation requirement check${NC}"
            fi
        else
            echo -e "${YELLOW}⚠ Reconciliation completed but report not found${NC}"
        fi
    else
        echo -e "${RED}✗ Reconciliation failed with exit code ${RECONCILE_EXIT_CODE}${NC}"
        echo "  Check log: ${OUT_DIR}/reconcile.log"
        exit ${RECONCILE_EXIT_CODE}
    fi
else
    echo -e "${RED}✗ reconcile_oanda.py not found${NC}"
    exit 1
fi

# Final summary
echo ""
echo -e "${GREEN}=== Execution Smoke Test Complete ===${NC}"
echo "Run Tag: ${RUN_TAG}"
echo "Output Directory: ${OUT_DIR}"
echo ""
echo "Artifacts:"
echo "  - trade_journal/trades/*.json (${TRADE_JSON_COUNT} file(s))"
echo "  - trade_journal/trade_journal_index.csv"
echo "  - run_header.json"
echo "  - exec_smoke_summary.json"
if [ -f "${RECONCILE_OUT}" ]; then
    echo "  - reconciliation_report.md"
fi
echo ""
echo "To inspect trade journal:"
echo "  cat ${OUT_DIR}/trade_journal/trades/*.json | python3 -m json.tool"
echo ""
echo "To view reconciliation report:"
if [ -f "${RECONCILE_OUT}" ]; then
    echo "  cat ${RECONCILE_OUT}"
fi
echo ""
echo -e "${GREEN}✓ Self-verification complete${NC}"

