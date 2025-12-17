#!/bin/bash
# GO Practice-Live (PROD_BASELINE) — Controlled Start
#
# Mål: Start GX1 i OANDA practice (ikke dry_run), med PROD_BASELINE fra prod_snapshot,
# SINGLE worker, max_open_trades=1, Asia-window drift, og full verifikasjon + artifacts.
# Dette er "GO"-knappen for practice-live testing.
#
# Usage:
#   ./scripts/go_practice_live_asia.sh [--run-tag TAG] [--debug]
#
# Requirements:
#   - OANDA_ENV=practice (hard check)
#   - OANDA_API_TOKEN and OANDA_ACCOUNT_ID (hard check)
#   - I_UNDERSTAND_LIVE_TRADING must NOT be set (hard check - vi vil IKKE ha live nå)
#   - PROD_BASELINE policy snapshot
#   - dry_run=false (REAL ORDERS)
#   - n_workers=1 (determinisme/parity)
#   - max_open_trades=1
#
# After run:
#   - Runs prod_baseline_proof.py
#   - Runs reconcile_oanda.py
#   - Verifies all artifacts exist
#   - Prints GO SUMMARY with key metrics

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
RUN_TAG=""
DEBUG_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --run-tag)
            RUN_TAG="$2"
            shift 2
            ;;
        --debug)
            DEBUG_FLAG="--debug"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--run-tag TAG] [--debug]"
            exit 1
            ;;
    esac
done

# ============================================================================
# HARD CHECKS - Fail fast if requirements not met
# ============================================================================

echo -e "${BLUE}=== GO Practice-Live (PROD_BASELINE) - Controlled Start ===${NC}"
echo ""

# Check 1: OANDA_ENV must be 'practice'
if [ "${OANDA_ENV:-}" != "practice" ]; then
    echo -e "${RED}ERROR: OANDA_ENV must be 'practice'${NC}"
    echo "Current value: ${OANDA_ENV:-<not set>}"
    echo "Set: export OANDA_ENV=practice"
    exit 1
fi

# Check 2: OANDA_API_TOKEN must be set
if [ -z "${OANDA_API_TOKEN:-}" ]; then
    echo -e "${RED}ERROR: OANDA_API_TOKEN must be set${NC}"
    echo "Set: export OANDA_API_TOKEN=<your_token>"
    exit 1
fi

# Check 3: OANDA_ACCOUNT_ID must be set
if [ -z "${OANDA_ACCOUNT_ID:-}" ]; then
    echo -e "${RED}ERROR: OANDA_ACCOUNT_ID must be set${NC}"
    echo "Set: export OANDA_ACCOUNT_ID=<your_account_id>"
    exit 1
fi

# Check 4: I_UNDERSTAND_LIVE_TRADING must NOT be set (we don't want live trading now)
if [ "${I_UNDERSTAND_LIVE_TRADING:-}" = "YES" ]; then
    echo -e "${RED}ERROR: I_UNDERSTAND_LIVE_TRADING=YES is set${NC}"
    echo "This script is for PRACTICE only. Unset this variable:"
    echo "  unset I_UNDERSTAND_LIVE_TRADING"
    exit 1
fi

echo -e "${GREEN}✓ Environment checks passed${NC}"
echo "  - OANDA_ENV: ${OANDA_ENV}"
echo "  - OANDA_API_TOKEN: <masked>"
echo "  - OANDA_ACCOUNT_ID: $(echo ${OANDA_ACCOUNT_ID} | sed 's/-[^-]*-/-***-/g')"
echo "  - I_UNDERSTAND_LIVE_TRADING: ${I_UNDERSTAND_LIVE_TRADING:-<not set>} (OK)"
echo ""

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set RUN_TAG if not provided
if [ -z "$RUN_TAG" ]; then
    RUN_TAG="GO_PRACTICE_$(date +%Y%m%d_%H%M%S)"
fi

# PROD_BASELINE policy snapshot
PROD_POLICY="gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml"

# Output directory
OUTPUT_DIR="gx1/wf_runs/${RUN_TAG}"

# Calculate date range (last 24 hours from available data)
# For practice-live, we'll use the last available 24 hours from the data file
# The actual live-run will use real-time data from OANDA API
# We'll determine the actual period from the data file itself

echo -e "${BLUE}Configuration:${NC}"
echo "  Run Tag: ${RUN_TAG}"
echo "  Policy: ${PROD_POLICY}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Period: Last 24 hours of available data (will be determined from data file)"
echo "  Mode: REPLAY (historical data, but REAL ORDERS)"
echo "  dry_run: false (REAL ORDERS)"
echo "  n_workers: 1 (deterministic)"
echo "  max_open_trades: 1"
echo ""

# ============================================================================
# VERIFY PREREQUISITES
# ============================================================================

# Verify we're in the right directory
if [ ! -f "gx1/execution/oanda_demo_runner.py" ]; then
    echo -e "${RED}ERROR: Must run from project root${NC}"
    exit 1
fi

# Verify policy exists
if [ ! -f "$PROD_POLICY" ]; then
    echo -e "${RED}ERROR: PROD_BASELINE policy not found: ${PROD_POLICY}${NC}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# ============================================================================
# RUN PRACTICE-LIVE
# ============================================================================

echo -e "${YELLOW}Starting practice-live execution...${NC}"
echo "  - Using PROD_BASELINE policy snapshot"
echo "  - dry_run=false (REAL ORDERS to OANDA Practice API)"
echo "  - n_workers=1 (deterministic)"
echo "  - max_open_trades=1"
echo "  - logging level: ${DEBUG_FLAG:+DEBUG}${DEBUG_FLAG:-INFO}"
echo ""

# Create a temporary policy file with live settings
TEMP_POLICY="${OUTPUT_DIR}/policy_live.yaml"
python3 << PYEOF
import yaml
from pathlib import Path

# Load base policy
with open("${PROD_POLICY}", "r") as f:
    policy = yaml.safe_load(f)

# Override for practice-live
policy["meta"] = policy.get("meta", {})
policy["meta"]["role"] = "PROD_BASELINE"
policy["mode"] = "REPLAY"  # Use historical data
policy["execution"] = policy.get("execution", {})
policy["execution"]["dry_run"] = False  # REAL ORDERS
policy["execution"]["max_open_trades"] = 1

# Set logging level
policy["logging"] = policy.get("logging", {})
if "${DEBUG_FLAG}":
    policy["logging"]["level"] = "DEBUG"
else:
    policy["logging"]["level"] = "INFO"

# Set output directory
policy["output_dir"] = "${OUTPUT_DIR}"

# Save temp policy
with open("${TEMP_POLICY}", "w") as f:
    yaml.dump(policy, f)
PYEOF

# Run with GX1DemoRunner
python3 << PYEOF
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, ".")

from gx1.execution.oanda_demo_runner import GX1DemoRunner
import pandas as pd

# Load historical data for the period
data_path = Path("data/raw/xauusd_m5_2025_bid_ask.parquet")
if not data_path.exists():
    print(f"ERROR: Data file not found: {data_path}")
    sys.exit(1)

df = pd.read_parquet(data_path)

# Handle timestamp column - check if 'time' is index or 'ts' is column
if df.index.name == 'time' or isinstance(df.index, pd.DatetimeIndex):
    # 'time' is the index
    df = df.reset_index()
    if 'time' in df.columns:
        df['ts'] = pd.to_datetime(df['time'])
    elif 'ts' not in df.columns:
        # Use index as ts
        df['ts'] = pd.to_datetime(df.index)
else:
    # 'ts' should be a column
    if 'ts' not in df.columns and 'time' in df.columns:
        df['ts'] = pd.to_datetime(df['time'])
    elif 'ts' in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])

# Ensure ts column is timezone-aware
if df["ts"].dt.tz is None:
    df["ts"] = df["ts"].dt.tz_localize('UTC')

# Use last 24 hours of available data (or all data if less than 24 hours)
max_ts = df["ts"].max()
min_ts = max_ts - pd.Timedelta(hours=24)
if min_ts < df["ts"].min():
    min_ts = df["ts"].min()

df = df[(df["ts"] >= min_ts) & (df["ts"] <= max_ts)]
print(f"Using data period: {df['ts'].min()} to {df['ts'].max()} ({len(df)} bars)")

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
    echo -e "${RED}ERROR: Practice-live execution failed with exit code ${RUN_EXIT_CODE}${NC}"
    exit ${RUN_EXIT_CODE}
fi

echo ""
echo -e "${GREEN}✓ Practice-live execution completed${NC}"

# ============================================================================
# VERIFY ARTIFACTS
# ============================================================================

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

# Check alerts.json (if exists)
if [ -f "${OUTPUT_DIR}/alerts.json" ]; then
    echo -e "${GREEN}✓ Found: alerts.json${NC}"
fi

if [ ${MISSING_ARTIFACTS} -ne 0 ]; then
    echo -e "${RED}ERROR: Missing required artifacts${NC}"
    exit 1
fi

# ============================================================================
# RUN PROD_BASELINE_PROOF
# ============================================================================

echo ""
echo -e "${YELLOW}Running prod_baseline_proof.py...${NC}"
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

# ============================================================================
# RUN RECONCILE_OANDA
# ============================================================================

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

# ============================================================================
# GO SUMMARY
# ============================================================================

echo ""
echo -e "${GREEN}=== GO SUMMARY ===${NC}"
echo "Run Tag: ${RUN_TAG}"
echo "Output Directory: ${OUTPUT_DIR}"
echo ""

# Extract key metrics from trade journal
if [ "${TRADE_COUNT}" -gt 0 ] && [ -f "${OUTPUT_DIR}/trade_journal/trade_journal_index.csv" ]; then
    echo -e "${BLUE}Key Metrics:${NC}"
    
    # Count trades
    TOTAL_TRADES=$(tail -n +2 "${OUTPUT_DIR}/trade_journal/trade_journal_index.csv" 2>/dev/null | wc -l | tr -d ' ')
    echo "  Total Trades: ${TOTAL_TRADES}"
    
    # Calculate RULE6A rate
    RULE6A_COUNT=$(tail -n +2 "${OUTPUT_DIR}/trade_journal/trade_journal_index.csv" 2>/dev/null | grep -c "RULE6A" || echo "0")
    if [ "${TOTAL_TRADES}" -gt 0 ]; then
        RULE6A_RATE=$(python3 -c "print(f'{${RULE6A_COUNT}/${TOTAL_TRADES}*100:.1f}%')" 2>/dev/null || echo "N/A")
        echo "  RULE6A Count: ${RULE6A_COUNT} (${RULE6A_RATE})"
    fi
    
    # Extract fill diffs from reconciliation report (if available)
    if [ -f "${RECONCILE_OUT}" ]; then
        FILL_DIFFS=$(grep -i "fill.*diff\|price.*diff" "${RECONCILE_OUT}" | head -3 || echo "N/A")
        if [ "$FILL_DIFFS" != "N/A" ] && [ -n "$FILL_DIFFS" ]; then
            echo "  Fill Diffs: (see reconciliation_report.md for details)"
        fi
    fi
    echo ""
fi

# Artifacts summary
echo -e "${BLUE}Artifacts:${NC}"
if [ -f "${OUTPUT_DIR}/trade_journal/trade_journal_index.csv" ]; then
    echo -e "  ${GREEN}✓${NC} trade_journal_index.csv"
else
    echo -e "  ${RED}✗${NC} trade_journal_index.csv"
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
fi

if [ -f "${PROOF_OUT}" ]; then
    echo -e "  ${GREEN}✓${NC} prod_baseline_proof.md"
else
    echo -e "  ${RED}✗${NC} prod_baseline_proof.md"
fi

if [ -f "${RECONCILE_OUT}" ]; then
    echo -e "  ${GREEN}✓${NC} reconciliation_report.md"
else
    echo -e "  ${RED}✗${NC} reconciliation_report.md"
fi

if [ -f "${OUTPUT_DIR}/alerts.json" ]; then
    echo -e "  ${GREEN}✓${NC} alerts.json"
fi

echo ""

# Paths summary
echo -e "${BLUE}Reports:${NC}"
echo "  - Prod Baseline Proof: ${PROOF_OUT}"
echo "  - Reconciliation Report: ${RECONCILE_OUT}"
echo "  - Trade Journal Index: ${OUTPUT_DIR}/trade_journal/trade_journal_index.csv"
echo "  - Run Log: ${OUTPUT_DIR}/run.log"
echo ""

# Verification instructions
echo -e "${BLUE}How to verify first trade was sent:${NC}"
echo "  1. Check OANDA Practice Trading Platform (web or desktop) for new trades"
echo "  2. Check trade journal execution events:"
echo "     cat ${OUTPUT_DIR}/trade_journal/trades/*.json | python3 -m json.tool | grep -A 10 execution_events"
echo "  3. Check reconciliation report:"
echo "     cat ${RECONCILE_OUT}"
echo "  4. Look for ORDER_SUBMITTED and ORDER_FILLED events in trade journal"
echo ""

if [ ${MISSING_ARTIFACTS} -eq 0 ]; then
    echo -e "${GREEN}✓ PASS: All artifacts verified${NC}"
    exit 0
else
    echo -e "${RED}✗ FAIL: Some artifacts are missing${NC}"
    exit 1
fi

