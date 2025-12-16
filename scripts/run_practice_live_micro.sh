#!/bin/bash
# Practice Live Micro Run Script
#
# Dedicated entry-point for real practice orders (not dry-run).
# Runs a short micro-test with automatic verification.
#
# Usage:
#   ./scripts/run_practice_live_micro.sh [--hours 24] [--run-tag TAG] [--policy PATH]
#
# Requirements:
#   - OANDA_ENV=practice (hard check)
#   - dry_run=false
#   - n_workers=1
#   - max_open_trades=1
#   - Short period (default: last 24 hours)
#   - PROD_BASELINE policy snapshot
#
# After run, automatically:
#   - Runs prod_baseline_proof.py
#   - Runs reconcile_oanda.py
#   - Verifies artifacts exist
#   - Verifies execution events in trade journal

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
HOURS=24
RUN_TAG=""
POLICY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --hours)
            HOURS="$2"
            shift 2
            ;;
        --run-tag)
            RUN_TAG="$2"
            shift 2
            ;;
        --policy)
            POLICY="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--hours 24] [--run-tag TAG] [--policy PATH]"
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

# Configuration
if [ -z "$RUN_TAG" ]; then
    RUN_TAG="PRACTICE_LIVE_MICRO_$(date +%Y%m%d_%H%M%S)"
fi

if [ -z "$POLICY" ]; then
    POLICY="gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml"
fi

# Calculate date range (last HOURS hours)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    END_DATE=$(date -u +%Y-%m-%dT%H:%M:%S)
    START_DATE=$(date -u -v-${HOURS}H +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -u -d "${HOURS} hours ago" +%Y-%m-%dT%H:%M:%S)
else
    # Linux
    END_DATE=$(date -u +%Y-%m-%dT%H:%M:%S)
    START_DATE=$(date -u -d "${HOURS} hours ago" +%Y-%m-%dT%H:%M:%S)
fi

OUTPUT_DIR="gx1/wf_runs/${RUN_TAG}"

echo -e "${GREEN}=== Practice Live Micro Run ===${NC}"
echo "Run Tag: ${RUN_TAG}"
echo "Period: ${START_DATE} to ${END_DATE} (${HOURS} hours)"
echo "Policy: ${POLICY}"
echo "Output: ${OUTPUT_DIR}"
echo "OANDA_ENV: ${OANDA_ENV}"
echo ""

# Verify we're in the right directory
if [ ! -f "gx1/execution/oanda_demo_runner.py" ]; then
    echo -e "${RED}ERROR: Must run from project root${NC}"
    exit 1
fi

# Verify policy exists
if [ ! -f "$POLICY" ]; then
    echo -e "${RED}ERROR: Policy file not found: ${POLICY}${NC}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run practice-live with GX1DemoRunner directly
# Note: For practice-live, we use historical data but with real orders (dry_run=false)
echo -e "${YELLOW}Running practice-live execution...${NC}"
echo "  - dry_run=false (REAL ORDERS)"
echo "  - n_workers=1 (deterministic)"
echo "  - max_open_trades=1"
echo "  - mode: REPLAY (historical data, but real orders)"
echo ""

# Create a temporary policy file with live settings
TEMP_POLICY="${OUTPUT_DIR}/policy_live.yaml"
python3 << PYEOF
import yaml
from pathlib import Path

# Load base policy
with open("${POLICY}", "r") as f:
    policy = yaml.safe_load(f)

# Override for practice-live
policy["meta"] = policy.get("meta", {})
policy["meta"]["role"] = "PROD_BASELINE"
policy["mode"] = "REPLAY"  # Use historical data
policy["execution"] = policy.get("execution", {})
policy["execution"]["dry_run"] = False  # REAL ORDERS
policy["execution"]["max_open_trades"] = 1

# Set output directory
policy["output_dir"] = "${OUTPUT_DIR}"

# Save temp policy
with open("${TEMP_POLICY}", "w") as f:
    yaml.dump(policy, f)
PYEOF

# Run with GX1DemoRunner directly (using replay_entry_exit_parallel but modified)
# For practice-live, we need to use replay mode with dry_run=false
# This means we use historical data but send real orders to OANDA Practice API

# Use replay_entry_exit_parallel but ensure it respects dry_run=false
# We'll create a wrapper script that sets dry_run correctly
python3 << PYEOF
import sys
import subprocess
from pathlib import Path

# Use replay_entry_exit_parallel but we need to modify it to support dry_run=false
# For now, we'll use the existing script but ensure dry_run is set correctly in policy
# The policy already has dry_run=false set above

# Actually, replay_entry_exit_parallel always sets dry_run_override=True
# So we need to use GX1DemoRunner directly
sys.path.insert(0, ".")

from gx1.execution.oanda_demo_runner import GX1DemoRunner
import pandas as pd

# Load historical data for the period
data_path = Path("data/raw/xauusd_m5_2025_bid_ask.parquet")
if not data_path.exists():
    print(f"ERROR: Data file not found: {data_path}")
    sys.exit(1)

df = pd.read_parquet(data_path)
df["ts"] = pd.to_datetime(df["ts"])

# Filter to period
start_ts = pd.Timestamp("${START_DATE}")
end_ts = pd.Timestamp("${END_DATE}")
df = df[(df["ts"] >= start_ts) & (df["ts"] <= end_ts)]

if len(df) == 0:
    print(f"ERROR: No data found for period {start_ts} to {end_ts}")
    sys.exit(1)

print(f"Loaded {len(df)} bars from {df['ts'].min()} to {df['ts'].max()}")

# Save filtered data
filtered_path = Path("${OUTPUT_DIR}") / "price_data_filtered.parquet"
df.to_parquet(filtered_path)

# Initialize runner with dry_run=false (REAL ORDERS)
runner = GX1DemoRunner(
    Path("${TEMP_POLICY}"),
    dry_run_override=False,  # REAL ORDERS - this overrides policy dry_run setting
    replay_mode=True,  # Use historical data
    fast_replay=False,
)

# Run replay with real orders
print("Starting practice-live execution with real orders...")
runner.run_replay(filtered_path)
print("Practice-live execution completed.")
PYEOF

RUN_EXIT_CODE=${PIPESTATUS[0]}

if [ ${RUN_EXIT_CODE} -ne 0 ]; then
    echo -e "${RED}ERROR: Replay failed with exit code ${RUN_EXIT_CODE}${NC}"
    exit ${RUN_EXIT_CODE}
fi

echo ""
echo -e "${GREEN}✓ Replay completed${NC}"

# Verify artifacts exist
echo ""
echo -e "${YELLOW}Verifying artifacts...${NC}"

MISSING_ARTIFACTS=0

# Check trade journal index
if [ ! -f "${OUTPUT_DIR}/trade_journal/trade_journal_index.csv" ]; then
    echo -e "${RED}✗ Missing: trade_journal_index.csv${NC}"
    MISSING_ARTIFACTS=1
else
    echo -e "${GREEN}✓ Found: trade_journal_index.csv${NC}"
fi

# Check trade journal JSON files
TRADE_COUNT=$(find "${OUTPUT_DIR}/trade_journal/trades" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
if [ "${TRADE_COUNT}" -eq 0 ]; then
    echo -e "${YELLOW}⚠ No trade journal JSON files found (may be expected if no trades)${NC}"
else
    echo -e "${GREEN}✓ Found: ${TRADE_COUNT} trade journal JSON file(s)${NC}"
fi

# Check run_header.json
if [ ! -f "${OUTPUT_DIR}/run_header.json" ]; then
    echo -e "${RED}✗ Missing: run_header.json${NC}"
    MISSING_ARTIFACTS=1
else
    echo -e "${GREEN}✓ Found: run_header.json${NC}"
fi

# Verify execution events in trade journal (if trades exist)
if [ "${TRADE_COUNT}" -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Verifying execution events in trade journal...${NC}"
    
    # Check first trade for execution events
    FIRST_TRADE=$(find "${OUTPUT_DIR}/trade_journal/trades" -name "*.json" 2>/dev/null | head -1)
    if [ -n "$FIRST_TRADE" ]; then
        EXEC_EVENTS=$(python3 << EOF
import json
try:
    with open("${FIRST_TRADE}", "r") as f:
        d = json.load(f)
    events = d.get("execution_events", [])
    event_types = [e.get("event_type") for e in events if isinstance(e, dict)]
    print(" ".join(event_types))
except Exception as e:
    print("ERROR: " + str(e))
EOF
)
        
        if echo "$EXEC_EVENTS" | grep -q "ORDER_SUBMITTED\|ORDER_FILLED\|TRADE_OPENED_OANDA"; then
            echo -e "${GREEN}✓ Found execution events: ${EXEC_EVENTS}${NC}"
        else
            echo -e "${YELLOW}⚠ No execution events found in sample trade (may be expected if no orders executed)${NC}"
        fi
    fi
fi

if [ ${MISSING_ARTIFACTS} -ne 0 ]; then
    echo -e "${RED}ERROR: Missing required artifacts${NC}"
    exit 1
fi

# Run prod_baseline_proof.py
echo ""
echo -e "${YELLOW}Running prod_baseline_proof.py...${NC}"
PROD_POLICY="${POLICY}"
PROOF_OUT="${OUTPUT_DIR}/prod_baseline_proof.md"

if [ -f "gx1/analysis/prod_baseline_proof.py" ]; then
    python3 gx1/analysis/prod_baseline_proof.py \
        --run "${OUTPUT_DIR}" \
        --prod-policy "${PROD_POLICY}" \
        --out "${PROOF_OUT}" \
        2>&1 | tee "${OUTPUT_DIR}/prod_baseline_proof.log"
    
    PROOF_EXIT_CODE=${PIPESTATUS[0]}
    if [ ${PROOF_EXIT_CODE} -eq 0 ]; then
        if [ -f "${PROOF_OUT}" ]; then
            echo -e "${GREEN}✓ Prod baseline proof generated: ${PROOF_OUT}${NC}"
        else
            echo -e "${YELLOW}⚠ Prod baseline proof completed but report not found${NC}"
        fi
    else
        echo -e "${RED}✗ Prod baseline proof failed with exit code ${PROOF_EXIT_CODE}${NC}"
        echo "  Check log: ${OUTPUT_DIR}/prod_baseline_proof.log"
    fi
else
    echo -e "${RED}✗ prod_baseline_proof.py not found${NC}"
    MISSING_ARTIFACTS=1
fi

# Run reconcile_oanda.py
echo ""
echo -e "${YELLOW}Running reconcile_oanda.py...${NC}"
RECONCILE_OUT="${OUTPUT_DIR}/reconciliation_report.md"

if [ -f "gx1/monitoring/reconcile_oanda.py" ]; then
    python3 gx1/monitoring/reconcile_oanda.py \
        --run "${OUTPUT_DIR}" \
        --out "${RECONCILE_OUT}" \
        2>&1 | tee "${OUTPUT_DIR}/reconcile.log"
    
    RECONCILE_EXIT_CODE=${PIPESTATUS[0]}
    if [ ${RECONCILE_EXIT_CODE} -eq 0 ]; then
        if [ -f "${RECONCILE_OUT}" ]; then
            echo -e "${GREEN}✓ Reconciliation report generated: ${RECONCILE_OUT}${NC}"
        else
            echo -e "${YELLOW}⚠ Reconciliation completed but report not found${NC}"
        fi
    else
        echo -e "${RED}✗ Reconciliation failed with exit code ${RECONCILE_EXIT_CODE}${NC}"
        echo "  Check log: ${OUTPUT_DIR}/reconcile.log"
    fi
else
    echo -e "${RED}✗ reconcile_oanda.py not found${NC}"
    MISSING_ARTIFACTS=1
fi

# Final summary
echo ""
echo -e "${GREEN}=== Practice Live Micro Run Complete ===${NC}"
echo "Run Tag: ${RUN_TAG}"
echo "Output Directory: ${OUTPUT_DIR}"
echo ""

# Check all required artifacts
ALL_ARTIFACTS_OK=1

echo "Artifacts:"
if [ -f "${OUTPUT_DIR}/trade_journal/trade_journal_index.csv" ]; then
    echo -e "  ${GREEN}✓${NC} trade_journal_index.csv"
else
    echo -e "  ${RED}✗${NC} trade_journal_index.csv"
    ALL_ARTIFACTS_OK=0
fi

if [ "${TRADE_COUNT}" -gt 0 ]; then
    echo -e "  ${GREEN}✓${NC} trade_journal/trades/*.json (${TRADE_COUNT} files)"
else
    echo -e "  ${YELLOW}⚠${NC} trade_journal/trades/*.json (0 files - no trades)"
fi

if [ -f "${OUTPUT_DIR}/run_header.json" ]; then
    echo -e "  ${GREEN}✓${NC} run_header.json"
else
    echo -e "  ${RED}✗${NC} run_header.json"
    ALL_ARTIFACTS_OK=0
fi

if [ -f "${PROOF_OUT}" ]; then
    echo -e "  ${GREEN}✓${NC} prod_baseline_proof.md"
else
    echo -e "  ${RED}✗${NC} prod_baseline_proof.md"
    ALL_ARTIFACTS_OK=0
fi

if [ -f "${RECONCILE_OUT}" ]; then
    echo -e "  ${GREEN}✓${NC} reconciliation_report.md"
else
    echo -e "  ${RED}✗${NC} reconciliation_report.md"
    ALL_ARTIFACTS_OK=0
fi

echo ""

if [ ${ALL_ARTIFACTS_OK} -eq 1 ]; then
    echo -e "${GREEN}✓ PASS: All artifacts verified${NC}"
    echo ""
    echo "Reports:"
    echo "  - Prod Baseline Proof: ${PROOF_OUT}"
    echo "  - Reconciliation Report: ${RECONCILE_OUT}"
    echo "  - Trade Journal Index: ${OUTPUT_DIR}/trade_journal/trade_journal_index.csv"
    echo "  - Run Log: ${OUTPUT_DIR}/run.log"
    exit 0
else
    echo -e "${RED}✗ FAIL: Some artifacts are missing${NC}"
    exit 1
fi
