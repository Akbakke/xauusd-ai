#!/bin/bash
# LIVE Practice Force-One-Trade Script (CANARY only, any session)
#
# Mål: Få nøyaktig 1 trade i OANDA practice LIVE for plumbing/strategi-loop verifikasjon.
# Uavhengig av session/regime.
#
# VIKTIG: Dette er KUN for plumbing testing. IKKE for PROD_BASELINE.
# Bruker CANARY policy med debug_force enabled.
#
# Usage:
#   ./scripts/run_live_force_one_trade_any_session.sh [--run-tag TAG] [--debug] [--hours HOURS]
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

echo -e "${BLUE}=== LIVE Practice Force-One-Trade (CANARY, Any Session) ===${NC}"
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
    RUN_TAG="LIVE_FORCE_ANY_$(date +%Y%m%d_%H%M%S)"
fi

# CANARY policy with force entry (any session)
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
echo "  debug_force: enabled (force entry after 30 minutes if no trades, ANY SESSION)"
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
echo "  - Force entry after 30 minutes if no trades (ANY SESSION)"
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
print("[LIVE] Starting LIVE practice execution (CANARY mode, force-one-trade, ANY SESSION)")
print("=" * 80)
print(f"Started at UTC: {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Planned stop at UTC: {end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Duration: ${HOURS} hours (or until trade opened and closed)")
print(f"Mode: LIVE (real-time candles from OANDA Practice API)")
print(f"Policy: ${POLICY}")
print(f"Output: ${OUTPUT_DIR}")
print("=" * 80)
print("")

# Track warmup progress and force entry countdown
warmup_bars_required = 288
warmup_bars_seen = 0
force_timeout_minutes = 30
force_start_time = time.time()

# HARD TIMEOUT: Fail if no trade within X minutes after backfill complete
# This ensures we don't run indefinitely if something is wrong
BACKFILL_TIMEOUT_MINUTES = 60  # Fail if no trade within 60 minutes after backfill
backfill_complete_time = None
backfill_complete = False

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

def monitor_backfill_timeout():
    """Monitor for backfill completion and enforce hard timeout if no trade."""
    global backfill_complete_time, backfill_complete
    import os
    
    # Wait for backfill to complete (check log file)
    log_file = "${OUTPUT_DIR}/logs/gx1_demo_runner.log"
    max_wait_seconds = 300  # Wait up to 5 minutes for backfill to complete
    
    start_wait = time.time()
    while time.time() - start_wait < max_wait_seconds:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read()
                if '[BACKFILL] Backfill complete' in content or '[PHASE] WARMUP_END' in content:
                    backfill_complete = True
                    backfill_complete_time = time.time()
                    print(f"\n[LIVE] Backfill complete detected at {datetime.now(timezone.utc)}")
                    print(f"[LIVE] Hard timeout: Will fail if no trade within {BACKFILL_TIMEOUT_MINUTES} minutes")
                    break
        time.sleep(5)  # Check every 5 seconds
    
    # Now monitor for trades and enforce timeout
    if backfill_complete:
        while not stop_flag.is_set():
            elapsed_minutes = (time.time() - backfill_complete_time) / 60
            
            # Check if trade exists
            trade_dir = "${OUTPUT_DIR}/trade_journal/trades"
            if os.path.exists(trade_dir):
                trade_files = [f for f in os.listdir(trade_dir) if f.endswith('.json')]
                if len(trade_files) > 0:
                    print(f"\n[LIVE] ✓ Trade detected after {elapsed_minutes:.1f} minutes - timeout check passed")
                    break
            
            # Enforce hard timeout
            if elapsed_minutes >= BACKFILL_TIMEOUT_MINUTES:
                print(f"\n[LIVE] ✗ HARD TIMEOUT: No trade within {BACKFILL_TIMEOUT_MINUTES} minutes after backfill complete")
                print(f"[LIVE] This indicates a problem - force entry may not be working or guards are blocking")
                print(f"[LIVE] Check logs for: GUARD BLOCKED, force_entry, FORCED_CANARY_TRADE")
                
                # Dump diagnostic state
                diag_path = "${OUTPUT_DIR}/no_trade_diagnostics.json"
                try:
                    import json
                    from datetime import datetime, timezone
                    
                    # Collect diagnostic state
                    diag = {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_minutes": elapsed_minutes,
                        "backfill_complete_time_utc": datetime.fromtimestamp(backfill_complete_time, tz=timezone.utc).isoformat() if backfill_complete_time else None,
                        "candles_cached": None,
                        "warmup_bars": None,
                        "warmup_complete": None,
                        "session": None,
                        "farm_regime": None,
                        "trend_id": None,
                        "vol_id": None,
                        "spread_pct": None,
                        "atr_bps": None,
                        "atr_pct": None,
                        "force_enabled": None,
                        "force_reason": None,
                        "force_deadline_utc": None,
                        "why_blocked": None,
                    }
                    
                    # Try to get state from runner
                    if hasattr(runner, 'backfill_cache'):
                        diag["candles_cached"] = len(runner.backfill_cache) if runner.backfill_cache is not None else 0
                    
                    if hasattr(runner, 'policy'):
                        diag["warmup_bars"] = runner.policy.get("warmup_bars", None)
                        debug_force = runner.policy.get("debug_force", {})
                        diag["force_enabled"] = debug_force.get("enabled", False)
                        diag["force_deadline_utc"] = None  # Would need to calculate from force_start_time
                    
                    # Try to get last observed state from log file
                    log_file = "${OUTPUT_DIR}/logs/gx1_demo_runner.log"
                    if os.path.exists(log_file):
                        with open(log_file, 'r') as f:
                            log_content = f.read()
                            # Extract last session/regime info
                            import re
                            session_match = re.search(r'session[=:]\s*(\w+)', log_content[-5000:])
                            if session_match:
                                diag["session"] = session_match.group(1)
                            
                            # Extract last GUARD BLOCKED reason
                            blocked_match = re.search(r'\[GUARD\]\s+BLOCKED.*reason=([^\n]+)', log_content[-5000:])
                            if blocked_match:
                                diag["why_blocked"] = blocked_match.group(1).strip()
                    
                    # Write diagnostic file
                    with open(diag_path, 'w') as f:
                        json.dump(diag, f, indent=2)
                    print(f"[LIVE] Diagnostic state dumped to: {diag_path}")
                except Exception as diag_error:
                    print(f"[LIVE] Failed to dump diagnostics: {diag_error}")
                
                stop_flag.set()
                if hasattr(runner, 'stop'):
                    runner.stop()
                raise RuntimeError(f"HARD TIMEOUT: No trade within {BACKFILL_TIMEOUT_MINUTES} minutes after backfill complete. Check {diag_path} for diagnostic state.")
            
            time.sleep(30)  # Check every 30 seconds

# Start time monitor in background
time_monitor = threading.Thread(target=monitor_time, daemon=True)
time_monitor.start()

# Start backfill timeout monitor in background
backfill_timeout_monitor = threading.Thread(target=monitor_backfill_timeout, daemon=True)
backfill_timeout_monitor.start()

try:
    print("[LIVE] Starting run_forever() - will run until end_time or trade opened and closed")
    print("[LIVE] Logging: warmup_progress, session/regime, spread_guard, force_countdown")
    print(f"[LIVE] Hard timeout: Will fail if no trade within {BACKFILL_TIMEOUT_MINUTES} minutes after backfill complete")
    runner.run_forever()
except KeyboardInterrupt:
    print("\n[LIVE] Interrupted by user")
    stop_flag.set()
except RuntimeError as e:
    if "HARD TIMEOUT" in str(e):
        print(f"\n[LIVE] {e}")
        stop_flag.set()
        raise
    else:
        raise
except Exception as e:
    print(f"\n[LIVE] Error: {e}")
    import traceback
    traceback.print_exc()
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
# VERIFY ARTIFACTS (HARD FAIL if missing)
# ============================================================================

echo ""
echo -e "${YELLOW}Verifying artifacts (hard-fail if missing)...${NC}"

MISSING_ARTIFACTS=0

# Check trade journal index (REQUIRED)
if [ ! -f "${OUTPUT_DIR}/trade_journal/trade_journal_index.csv" ]; then
    echo -e "${RED}✗ FAIL: Missing required artifact: trade_journal_index.csv${NC}"
    MISSING_ARTIFACTS=1
else
    echo -e "${GREEN}✓ Found: trade_journal_index.csv${NC}"
fi

# Check trade journal JSON files (REQUIRED - must have at least 1 trade)
TRADE_COUNT=$(find "${OUTPUT_DIR}/trade_journal/trades" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
if [ "${TRADE_COUNT}" -eq 0 ]; then
    echo -e "${RED}✗ FAIL: Missing required artifact: at least one trade JSON file${NC}"
    MISSING_ARTIFACTS=1
else
    echo -e "${GREEN}✓ Found: ${TRADE_COUNT} trade journal JSON file(s)${NC}"
fi

# Check run_header.json (optional but nice to have)
if [ ! -f "${OUTPUT_DIR}/run_header.json" ]; then
    echo -e "${YELLOW}⚠ Missing: run_header.json (optional)${NC}"
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
fi

# ============================================================================
# RUN RECONCILE_OANDA (REQUIRED)
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
            echo -e "${RED}✗ FAIL: Reconciliation completed but report not found${NC}"
            MISSING_ARTIFACTS=1
        fi
    else
        echo -e "${RED}✗ FAIL: Reconciliation failed with exit code ${RECONCILE_EXIT_CODE}${NC}"
        echo "  Check log: ${OUTPUT_DIR}/reconcile.log"
        MISSING_ARTIFACTS=1
    fi
else
    echo -e "${RED}✗ FAIL: reconcile_oanda.py not found${NC}"
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
    
    # Check for forced trades
    if [ "${TRADE_COUNT}" -gt 0 ]; then
        FORCED_TRADES=$(grep -l "FORCED_CANARY_TRADE" "${OUTPUT_DIR}/trade_journal/trades"/*.json 2>/dev/null | wc -l | tr -d ' ')
        if [ "${FORCED_TRADES}" -gt 0 ]; then
            echo "  Forced Trades: ${FORCED_TRADES}"
        fi
    fi
    echo ""
fi

# Artifacts summary
echo -e "${BLUE}Artifacts:${NC}"
if [ -f "${OUTPUT_DIR}/trade_journal/trade_journal_index.csv" ]; then
    echo -e "  ${GREEN}✓${NC} trade_journal_index.csv"
else
    echo -e "  ${RED}✗${NC} trade_journal_index.csv (REQUIRED)"
fi

if [ "${TRADE_COUNT}" -gt 0 ]; then
    echo -e "  ${GREEN}✓${NC} trade_journal/trades/*.json (${TRADE_COUNT} files)"
else
    echo -e "  ${RED}✗${NC} trade_journal/trades/*.json (REQUIRED - at least 1 trade)"
fi

if [ -f "${OUTPUT_DIR}/run_header.json" ]; then
    echo -e "  ${GREEN}✓${NC} run_header.json"
fi

if [ -f "${PROOF_OUT}" ]; then
    echo -e "  ${GREEN}✓${NC} prod_baseline_proof.md"
fi

if [ -f "${RECONCILE_OUT}" ]; then
    echo -e "  ${GREEN}✓${NC} reconciliation_report.md (REQUIRED)"
else
    echo -e "  ${RED}✗${NC} reconciliation_report.md (REQUIRED)"
fi

echo ""
echo -e "${BLUE}How to verify first trade:${NC}"
echo "  1. Check OANDA Practice Trading Platform (web or desktop) for new trades"
echo "  2. Check trade journal execution events:"
echo "     cat ${OUTPUT_DIR}/trade_journal/trades/*.json | python3 -m json.tool | grep -A 10 execution_events"
echo "  3. Check reconciliation report:"
echo "     cat ${RECONCILE_OUT}"
echo "  4. Check for forced trade:"
echo "     grep -l 'FORCED_CANARY_TRADE' ${OUTPUT_DIR}/trade_journal/trades/*.json"
echo ""

if [ ${MISSING_ARTIFACTS} -eq 0 ]; then
    echo -e "${GREEN}✓ PASS: All required artifacts verified${NC}"
    exit 0
else
    echo -e "${RED}✗ FAIL: Some required artifacts are missing${NC}"
    exit 1
fi

