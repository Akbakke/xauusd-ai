#!/bin/bash
# LIVE Practice Force-One-Trade Script (CANARY only)
#
# Mål: Få minst én trade i OANDA practice LIVE for å bekrefte plumbing:
#   - live candles kommer inn
#   - entry→router→exit prater sammen
#   - trade journal + execution events + reconciliation produseres
#
# VIKTIG: Dette er KUN for plumbing testing. IKKE for PROD_BASELINE.
# Bruker CANARY policy med debug_force enabled.
#
# Usage:
#   ./scripts/run_live_force_one_trade_asia.sh [--run-tag TAG] [--debug] [--hours HOURS]
#
# Requirements:
#   - OANDA_ENV=practice (hard check)
#   - OANDA_API_TOKEN and OANDA_ACCOUNT_ID (hard check)
#   - I_UNDERSTAND_LIVE_TRADING must NOT be set (hard check - vi vil IKKE ha live nå)

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
HOURS=6  # Default: run for 6 hours or until trade is opened and closed

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
        --hours)
            HOURS="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--run-tag TAG] [--debug] [--hours HOURS]"
            exit 1
            ;;
    esac
done

# ============================================================================
# HARD CHECKS - Fail fast if requirements not met
# ============================================================================

echo -e "${BLUE}=== LIVE Practice Force-One-Trade (CANARY) ===${NC}"
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
    RUN_TAG="LIVE_FORCE_$(date +%Y%m%d_%H%M%S)"
fi

# CANARY policy with force entry
POLICY="gx1/configs/policies/active/GX1_V11_OANDA_PRACTICE_LIVE_FORCE_ONE_TRADE.yaml"

# Output directory
OUTPUT_DIR="gx1/wf_runs/${RUN_TAG}"

echo -e "${BLUE}Configuration:${NC}"
echo "  Run Tag: ${RUN_TAG}"
echo "  Policy: ${POLICY}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Mode: LIVE (real-time candles from OANDA)"
echo "  Duration: ${HOURS} hours (or until trade opened and closed)"
echo "  dry_run: false (REAL ORDERS to OANDA Practice API)"
echo "  max_open_trades: 1"
echo "  debug_force: enabled (force entry after 30 minutes if no trades)"
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
if [ ! -f "$POLICY" ]; then
    echo -e "${RED}ERROR: Policy file not found: ${POLICY}${NC}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# ============================================================================
# RUN LIVE
# ============================================================================

echo -e "${YELLOW}Starting LIVE practice execution...${NC}"
echo "  - Using CANARY policy with debug_force enabled"
echo "  - mode=LIVE (real-time candles from OANDA)"
echo "  - dry_run=false (REAL ORDERS to OANDA Practice API)"
echo "  - max_open_trades=1"
echo "  - Force entry after 30 minutes if no trades"
echo ""

# Run with GX1DemoRunner in LIVE mode
python3 << PYEOF
import sys
import signal
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, ".")

from gx1.execution.oanda_demo_runner import GX1DemoRunner

# Set up signal handler for graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("\n[LIVE] Shutdown requested, stopping gracefully...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Initialize runner with LIVE mode
runner = GX1DemoRunner(
    Path("${POLICY}"),
    dry_run_override=False,  # REAL ORDERS
    replay_mode=False,  # LIVE mode
    fast_replay=False,
)

# Ensure required attributes are initialized (if not set in __init__)
import pandas as pd
from datetime import datetime, timezone

if not hasattr(runner, '_shutdown_requested'):
    runner._shutdown_requested = False
if not hasattr(runner, 'backfill_in_progress'):
    runner.backfill_in_progress = False
if not hasattr(runner, '_last_disk_check'):
    # Initialize to 2 hours ago to force first check
    runner._last_disk_check = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2)
if not hasattr(runner, 'warmup_floor'):
    runner.warmup_floor = None  # Will be set during backfill if needed
if not hasattr(runner, '_last_server_time_check'):
    runner._last_server_time_check = None

# Calculate end time (UTC)
start_time = datetime.now(timezone.utc)
end_time = start_time + timedelta(hours=${HOURS})

print("=" * 80)
print("[LIVE] Starting LIVE practice execution (CANARY mode, force-one-trade)")
print("=" * 80)
print(f"Started at UTC: {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Planned stop at UTC: {end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Duration: ${HOURS} hours (or until trade opened and closed)")
print(f"Mode: LIVE (real-time candles from OANDA Practice API)")
print(f"Policy: ${POLICY}")
print(f"Output: ${OUTPUT_DIR}")
print("=" * 80)
print("")

# Run LIVE (this will run until interrupted or end_time)
# For LIVE mode, we use run_forever() which runs continuously
# We'll monitor for end_time and stop gracefully
import threading

stop_flag = threading.Event()

def monitor_time():
    """Monitor time and set stop flag when end_time is reached."""
    while not stop_flag.is_set():
        if datetime.now(timezone.utc) >= end_time:
            print(f"\n[LIVE] End time reached: {end_time}")
            stop_flag.set()
            # Signal runner to stop (if it supports it)
            if hasattr(runner, 'stop'):
                runner.stop()
            break
        time.sleep(60)  # Check every minute

# Start time monitor in background
time_monitor = threading.Thread(target=monitor_time, daemon=True)
time_monitor.start()

try:
    print("[LIVE] Starting run_forever() - will run until end_time or trade opened and closed")
    runner.run_forever()
except KeyboardInterrupt:
    print("\n[LIVE] Interrupted by user")
    stop_flag.set()
except Exception as e:
    print(f"\n[LIVE] Error: {e}")
    stop_flag.set()
    raise
finally:
    print("\n[LIVE] Execution completed")
PYEOF

RUN_EXIT_CODE=${PIPESTATUS[0]}

if [ ${RUN_EXIT_CODE} -ne 0 ]; then
    echo -e "${RED}ERROR: LIVE execution failed with exit code ${RUN_EXIT_CODE}${NC}"
    exit ${RUN_EXIT_CODE}
fi

echo ""
echo -e "${GREEN}✓ LIVE execution completed${NC}"

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

# ============================================================================
# RUN PROD_BASELINE_PROOF (skal vise CANARY bundle, men fortsatt manifest/router hashes)
# ============================================================================

echo ""
echo -e "${YELLOW}Running prod_baseline_proof.py...${NC}"
PROOF_OUT="${OUTPUT_DIR}/prod_baseline_proof.md"

if [ -f "gx1/analysis/prod_baseline_proof.py" ]; then
    # Use prod_snapshot policy for comparison (for parity check)
    PROD_POLICY="gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml"
    
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
# SUMMARY
# ============================================================================

echo ""
echo -e "${GREEN}=== LIVE Force-One-Trade Summary ===${NC}"
echo "Run Tag: ${RUN_TAG}"
echo "Output Directory: ${OUTPUT_DIR}"
echo ""

# Extract key metrics
if [ "${TRADE_COUNT}" -gt 0 ] && [ -f "${OUTPUT_DIR}/trade_journal/trade_journal_index.csv" ]; then
    echo -e "${BLUE}Key Metrics:${NC}"
    TOTAL_TRADES=$(tail -n +2 "${OUTPUT_DIR}/trade_journal/trade_journal_index.csv" 2>/dev/null | wc -l | tr -d ' ')
    echo "  Total Trades: ${TOTAL_TRADES}"
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
fi

if [ -f "${PROOF_OUT}" ]; then
    echo -e "  ${GREEN}✓${NC} prod_baseline_proof.md"
fi

if [ -f "${RECONCILE_OUT}" ]; then
    echo -e "  ${GREEN}✓${NC} reconciliation_report.md"
fi

echo ""
echo -e "${BLUE}How to verify first trade:${NC}"
echo "  1. Check OANDA Practice Trading Platform (web or desktop) for new trades"
echo "  2. Check trade journal execution events:"
echo "     cat ${OUTPUT_DIR}/trade_journal/trades/*.json | python3 -m json.tool | grep -A 10 execution_events"
echo "  3. Check reconciliation report:"
echo "     cat ${RECONCILE_OUT}"
echo ""

if [ ${MISSING_ARTIFACTS} -eq 0 ]; then
    echo -e "${GREEN}✓ PASS: All artifacts verified${NC}"
    exit 0
else
    echo -e "${RED}✗ FAIL: Some artifacts are missing${NC}"
    exit 1
fi

