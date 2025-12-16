#!/bin/bash
# Observation-Only Replay for PROD_BASELINE Verification
#
# Runs a replay to verify that PROD_BASELINE system generates trades correctly
# with full trade journal and explainability - WITHOUT execution or parallelism.
#
# Requirements:
#   - Uses PROD_BASELINE policy (same entry model, exit policy, router, guardrail)
#   - REPLAY mode (not live/practice-execution)
#   - n_workers=1 (determinism critical)
#   - No orders sent (observation-only)
#   - Full Trade Journal generated
#
# Usage:
#   ./scripts/run_obs_replay_prod_baseline.sh [start_date] [end_date]
#
# Example:
#   ./scripts/run_obs_replay_prod_baseline.sh 2025-01-01 2025-01-15

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values (last 2 weeks)
END_DATE="${2:-$(date -u +%Y-%m-%d)}"
START_DATE="${1:-$(date -u -v-14d +%Y-%m-%d 2>/dev/null || date -u -d '14 days ago' +%Y-%m-%d)}"

# PROD_BASELINE policy path
PROD_POLICY="gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml"

# Output tag
RUN_TAG="OBS_REPLAY_PROD_BASELINE_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="gx1/wf_runs/${RUN_TAG}"

echo -e "${GREEN}=== Observation-Only Replay (PROD_BASELINE Verification) ===${NC}"
echo "Policy: ${PROD_POLICY}"
echo "Period: ${START_DATE} to ${END_DATE}"
echo "Output: ${OUTPUT_DIR}"
echo "Workers: 1 (deterministic)"
echo "Mode: REPLAY (observation-only, no orders)"
echo ""

# Verify we're in the right directory
if [ ! -f "gx1/execution/oanda_demo_runner.py" ]; then
    echo -e "${RED}ERROR: Must run from project root${NC}"
    exit 1
fi

# Verify policy exists
if [ ! -f "${PROD_POLICY}" ]; then
    echo -e "${RED}ERROR: PROD_BASELINE policy not found: ${PROD_POLICY}${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Policy file found"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run replay using run_replay.sh (which handles REPLAY mode automatically)
echo -e "${YELLOW}Running observation-only replay...${NC}"
echo ""

bash scripts/run_replay.sh \
    "${PROD_POLICY}" \
    "${START_DATE}" \
    "${END_DATE}" \
    1 \
    "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/run.log"

REPLAY_EXIT_CODE=${PIPESTATUS[0]}

if [ ${REPLAY_EXIT_CODE} -ne 0 ]; then
    echo ""
    echo -e "${RED}ERROR: Replay failed with exit code ${REPLAY_EXIT_CODE}${NC}"
    exit ${REPLAY_EXIT_CODE}
fi

echo ""
echo -e "${GREEN}✓ Replay completed${NC}"
echo ""

# Verify artifacts
echo -e "${YELLOW}Verifying Trade Journal artifacts...${NC}"
echo ""

MISSING_ARTIFACTS=0

# Check trade journal index
TRADE_JOURNAL_INDEX="${OUTPUT_DIR}/trade_journal/trade_journal_index.csv"
if [ ! -f "${TRADE_JOURNAL_INDEX}" ]; then
    echo -e "${RED}✗ Missing: trade_journal_index.csv${NC}"
    MISSING_ARTIFACTS=1
else
    TRADE_COUNT=$(tail -n +2 "${TRADE_JOURNAL_INDEX}" 2>/dev/null | wc -l | tr -d ' ')
    echo -e "${GREEN}✓${NC} Found: trade_journal_index.csv (${TRADE_COUNT} trades)"
fi

# Check trade journal JSON files
TRADE_JSON_DIR="${OUTPUT_DIR}/trade_journal/trades"
if [ ! -d "${TRADE_JSON_DIR}" ]; then
    echo -e "${RED}✗ Missing: trade_journal/trades/ directory${NC}"
    MISSING_ARTIFACTS=1
else
    JSON_COUNT=$(find "${TRADE_JSON_DIR}" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
    if [ "${JSON_COUNT}" -eq 0 ]; then
        echo -e "${YELLOW}⚠${NC} No trade journal JSON files found (may be expected if no trades)"
    else
        echo -e "${GREEN}✓${NC} Found: ${JSON_COUNT} trade journal JSON file(s)"
    fi
fi

# Check results.json
RESULTS_JSON="${OUTPUT_DIR}/results.json"
if [ ! -f "${RESULTS_JSON}" ]; then
    echo -e "${YELLOW}⚠${NC} Missing: results.json"
else
    echo -e "${GREEN}✓${NC} Found: results.json"
    # Extract key metrics
    if command -v python3 &> /dev/null; then
        python3 << EOF
import json
import sys
try:
    with open("${RESULTS_JSON}", "r") as f:
        results = json.load(f)
    trades_per_day = results.get("trades_per_day", 0)
    ev_per_trade = results.get("ev_per_trade_bps", 0)
    total_trades = results.get("total_trades", 0)
    print(f"  Trades/day: {trades_per_day:.2f}")
    print(f"  EV/trade: {ev_per_trade:.2f} bps")
    print(f"  Total trades: {total_trades}")
except Exception as e:
    print(f"  Could not parse results.json: {e}")
EOF
    fi
fi

# Check run_header.json
RUN_HEADER="${OUTPUT_DIR}/run_header.json"
if [ ! -f "${RUN_HEADER}" ]; then
    echo -e "${YELLOW}⚠${NC} Missing: run_header.json"
else
    echo -e "${GREEN}✓${NC} Found: run_header.json"
fi

echo ""

# Verify trade journal coverage (if trades exist)
if [ "${JSON_COUNT:-0}" -gt 0 ]; then
    echo -e "${YELLOW}Verifying Trade Journal Coverage...${NC}"
    echo ""
    
    # Sample first trade JSON to verify structure
    FIRST_TRADE_JSON=$(find "${TRADE_JSON_DIR}" -name "*.json" 2>/dev/null | head -1)
    if [ -n "${FIRST_TRADE_JSON}" ]; then
        if command -v python3 &> /dev/null; then
            python3 << EOF
import json
import sys

try:
    with open("${FIRST_TRADE_JSON}", "r") as f:
        trade_data = json.load(f)
    
    trade_id = trade_data.get("trade_id", "UNKNOWN")
    print(f"Sample trade: {trade_id}")
    print("")
    
    # Check required sections
    checks = {
        "entry_snapshot": trade_data.get("entry_snapshot") is not None,
        "feature_context": trade_data.get("feature_context") is not None,
        "router_explainability": trade_data.get("router_explainability") is not None,
        "exit_summary": trade_data.get("exit_summary") is not None,
        "execution_events": len(trade_data.get("execution_events", [])) > 0,
    }
    
    all_present = all(checks.values())
    
    for key, present in checks.items():
        status = "✓" if present else "✗"
        print(f"  {status} {key}")
    
    print("")
    if all_present:
        print("✓ Trade journal coverage: 100%")
    else:
        print("⚠ Trade journal coverage: INCOMPLETE")
        sys.exit(1)
    
    # Check execution events
    execution_events = trade_data.get("execution_events", [])
    event_types = [e.get("event_type") for e in execution_events]
    print(f"  Execution events: {', '.join(event_types) if event_types else 'None'}")
    
    # Check intratrade metrics
    exit_summary = trade_data.get("exit_summary", {})
    has_mfe = exit_summary.get("max_mfe_bps") is not None
    has_mae = exit_summary.get("max_mae_bps") is not None
    has_dd = exit_summary.get("intratrade_drawdown_bps") is not None
    
    if has_mfe and has_mae and has_dd:
        print("  ✓ Intratrade metrics: Complete")
    else:
        print("  ⚠ Intratrade metrics: Missing")
    
except Exception as e:
    print(f"Error verifying trade journal: {e}")
    sys.exit(1)
EOF
            COVERAGE_EXIT=$?
            if [ ${COVERAGE_EXIT} -ne 0 ]; then
                MISSING_ARTIFACTS=1
            fi
        fi
    fi
    echo ""
fi

# Final summary
echo -e "${GREEN}=== Observation Replay Complete ===${NC}"
echo "Run Tag: ${RUN_TAG}"
echo "Output Directory: ${OUTPUT_DIR}"
echo ""

if [ ${MISSING_ARTIFACTS} -eq 0 ]; then
    echo -e "${GREEN}✓ All artifacts verified${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Review trade_journal_index.csv for trade summary"
    echo "  2. Inspect individual trade JSON files in trade_journal/trades/"
    echo "  3. Verify router decisions and guardrail explainability"
    echo "  4. Check intratrade metrics in exit_summary"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some artifacts missing or incomplete${NC}"
    echo ""
    exit 1
fi

