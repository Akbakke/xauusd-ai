#!/bin/bash
# LIVE Practice SNIPER Script (EU/London/NY)
#
# Mål: Kjøre SNIPER entry policy i OANDA practice LIVE for EU/London/NY sessions.
# Høyere frekvens enn FARM (flere sessions, HIGH vol tillatt).
#
# Usage:
#   ./scripts/run_practice_live_sniper_london_ny.sh [--run-tag TAG] [--debug] [--hours HOURS]
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
HOURS=6  # Default: run for 6 hours

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

echo -e "${BLUE}=== LIVE Practice SNIPER (EU/London/NY) ===${NC}"
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
    RUN_TAG="SNIPER_LONDON_NY_$(date +%Y%m%d_%H%M%S)"
fi

# SNIPER policy
POLICY="gx1/configs/policies/active/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml"

# Output directory
OUTPUT_DIR="gx1/wf_runs/${RUN_TAG}"

echo -e "${BLUE}Configuration:${NC}"
echo "  Run Tag: ${RUN_TAG}"
echo "  Policy: ${POLICY}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Mode: LIVE (real-time candles from OANDA)"
echo "  Duration: ${HOURS} hours"
echo "  dry_run: false (REAL ORDERS to OANDA Practice API)"
echo "  max_open_trades: 1 (conservative start)"
echo "  Sessions: EU, OVERLAP, US (not ASIA)"
echo "  Vol regimes: LOW, MEDIUM, HIGH (HIGH allowed)"
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

echo -e "${YELLOW}Starting LIVE practice execution (SNIPER)...${NC}"
echo "  - Using SNIPER policy (EU/London/NY sessions)"
echo "  - mode=LIVE (real-time candles from OANDA)"
echo "  - dry_run=false (REAL ORDERS to OANDA Practice API)"
echo "  - max_open_trades=1"
echo ""

# Run with GX1DemoRunner in LIVE mode
python3 << PYEOF
import sys
import signal
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, ".")

from gx1.execution.oanda_demo_runner import GX1DemoRunner

# Set up logging to INFO level for detailed output
logging.basicConfig(level=logging.INFO)

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

if not hasattr(runner, '_shutdown_requested'):
    runner._shutdown_requested = False
if not hasattr(runner, 'backfill_in_progress'):
    runner.backfill_in_progress = False
if not hasattr(runner, '_last_disk_check'):
    runner._last_disk_check = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2)
if not hasattr(runner, 'warmup_floor'):
    runner.warmup_floor = None
if not hasattr(runner, '_last_server_time_check'):
    runner._last_server_time_check = None

# Calculate end time (UTC)
start_time = datetime.now(timezone.utc)
end_time = start_time + timedelta(hours=${HOURS})

print("=" * 80)
print("[LIVE] Starting LIVE practice execution (SNIPER - EU/London/NY)")
print("=" * 80)
print(f"Started at UTC: {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Planned stop at UTC: {end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Duration: ${HOURS} hours")
print(f"Mode: LIVE (real-time candles from OANDA Practice API)")
print(f"Policy: ${POLICY}")
print(f"Output: ${OUTPUT_DIR}")
print("=" * 80)
print("")

# Run LIVE (this will run until interrupted or end_time)
import threading

stop_flag = threading.Event()

def monitor_time():
    """Monitor time and set stop flag when end_time is reached."""
    while not stop_flag.is_set():
        if datetime.now(timezone.utc) >= end_time:
            print(f"\n[LIVE] End time reached: {end_time}")
            stop_flag.set()
            if hasattr(runner, 'stop'):
                runner.stop()
            break
        time.sleep(60)  # Check every minute

# Start time monitor thread
time_thread = threading.Thread(target=monitor_time, daemon=True)
time_thread.start()

# Run forever (will stop when stop_flag is set or interrupted)
try:
    runner.run_forever()
except KeyboardInterrupt:
    print("\n[LIVE] Interrupted by user")
except Exception as e:
    print(f"\n[LIVE] Error during execution: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    stop_flag.set()
    print("\n[LIVE] Execution stopped")

PYEOF

EXIT_CODE=$?

# ============================================================================
# POST-RUN VERIFICATION
# ============================================================================

echo ""
echo -e "${BLUE}=== POST-RUN VERIFICATION ===${NC}"
echo ""

# Check for required artifacts
ARTIFACTS_OK=true

# Check trade journal index
if [ ! -f "${OUTPUT_DIR}/trade_journal/trade_journal_index.csv" ]; then
    echo -e "${RED}✗ Missing: trade_journal_index.csv${NC}"
    ARTIFACTS_OK=false
else
    echo -e "${GREEN}✓ Found: trade_journal_index.csv${NC}"
fi

# Check for trade JSONs
TRADE_COUNT=$(find "${OUTPUT_DIR}/trade_journal/trades" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
if [ "$TRADE_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}⚠ No trades found in trade_journal/trades/${NC}"
else
    echo -e "${GREEN}✓ Found: ${TRADE_COUNT} trade(s) in trade_journal/trades/${NC}"
fi

# Check reconciliation report
if [ ! -f "${OUTPUT_DIR}/reconciliation_report.md" ]; then
    echo -e "${YELLOW}⚠ Missing: reconciliation_report.md (will be generated)${NC}"
else
    echo -e "${GREEN}✓ Found: reconciliation_report.md${NC}"
fi

# Check run_header.json
if [ ! -f "${OUTPUT_DIR}/run_header.json" ]; then
    echo -e "${RED}✗ Missing: run_header.json${NC}"
    ARTIFACTS_OK=false
else
    echo -e "${GREEN}✓ Found: run_header.json${NC}"
    # Verify it contains [BASELINE_FINGERPRINT] info
    if grep -q "BASELINE_FINGERPRINT\|meta.*role\|SNIPER" "${OUTPUT_DIR}/run_header.json" 2>/dev/null; then
        echo -e "${GREEN}✓ run_header.json contains baseline fingerprint info${NC}"
    fi
fi

# Run prod_baseline_proof.py (should show SNIPER_CANARY role)
if [ -f "gx1/analysis/prod_baseline_proof.py" ]; then
    echo ""
    echo -e "${BLUE}Running prod_baseline_proof.py...${NC}"
    python3 gx1/analysis/prod_baseline_proof.py \
        --run "${OUTPUT_DIR}" \
        --prod-policy "gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml" \
        --out "${OUTPUT_DIR}/prod_baseline_proof.md" || echo -e "${YELLOW}⚠ prod_baseline_proof.py failed (non-fatal)${NC}"
fi

# Run reconcile_oanda.py
if [ -f "gx1/monitoring/reconcile_oanda.py" ]; then
    echo ""
    echo -e "${BLUE}Running reconcile_oanda.py...${NC}"
    python3 gx1/monitoring/reconcile_oanda.py \
        --run "${OUTPUT_DIR}" \
        --out "${OUTPUT_DIR}/reconciliation_report.md" || echo -e "${YELLOW}⚠ reconcile_oanda.py failed (non-fatal)${NC}"
fi

# Final summary
echo ""
echo "=================================================================================="
echo -e "${BLUE}=== RUN SUMMARY ===${NC}"
echo "=================================================================================="
echo "Run Tag: ${RUN_TAG}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Trade Count: ${TRADE_COUNT}"
echo "Exit Code: ${EXIT_CODE}"
echo ""

if [ "$ARTIFACTS_OK" = true ] && [ "$TRADE_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ Run completed successfully${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Review trade journal: ${OUTPUT_DIR}/trade_journal/trades/*.json"
    echo "  2. Check reconciliation: ${OUTPUT_DIR}/reconciliation_report.md"
    echo "  3. Verify baseline proof: ${OUTPUT_DIR}/prod_baseline_proof.md"
    echo "  4. Check logs: ${OUTPUT_DIR}/logs/"
    exit 0
else
    echo -e "${YELLOW}⚠ Run completed but some artifacts are missing${NC}"
    echo ""
    echo "Check logs for errors: ${OUTPUT_DIR}/logs/"
    exit 1
fi

