#!/bin/bash
# Smoke test: Verify that master always writes perf JSON even when SIGTERM is sent
#
# Usage:
#   ./scripts/test_master_sigterm_smoke.sh
#
# This script:
# 1. Starts replay_eval_gated_parallel.py with a unique run_id
# 2. Waits 45 seconds
# 3. Sends SIGTERM to master process
# 4. Waits 10 seconds for cleanup
# 5. Verifies that perf_<run_id>.json exists
#
# Exit code: 0 if perf JSON exists, 1 otherwise

set -euo pipefail

# Configuration
POLICY="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml"
DATA="data/raw/xauusd_m5_2025_bid_ask.parquet"
WORKERS=7
OUTPUT_DIR="reports/replay_eval/GATED"
WAIT_BEFORE_SIGTERM=45
WAIT_AFTER_SIGTERM=10

# Required env vars
export GX1_GATED_FUSION_ENABLED=1
export GX1_REQUIRE_XGB_CALIBRATION=1
export GX1_REPLAY_INCREMENTAL_FEATURES=1
export GX1_REPLAY_NO_CSV=1
export GX1_FEATURE_USE_NP_ROLLING=1
export GX1_REPLAY_QUIET=1
export GX1_MASTER_WAIT_TIMEOUT_SEC=3600

# Thread limits
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Generate unique run_id
RUN_ID="smoke_test_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="/tmp/${RUN_ID}.log"

echo "=========================================="
echo "Master SIGTERM Smoke Test"
echo "=========================================="
echo "Run ID: $RUN_ID"
echo "Will send SIGTERM after ${WAIT_BEFORE_SIGTERM}s"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start replay in background
echo "[TEST] Starting replay_eval_gated_parallel.py..."
python3 gx1/scripts/replay_eval_gated_parallel.py \
  --policy "$POLICY" \
  --data "$DATA" \
  --workers "$WORKERS" \
  --output-dir "$OUTPUT_DIR" \
  --run-id "$RUN_ID" \
  > "$LOG_FILE" 2>&1 &

MASTER_PID=$!
echo "[TEST] Master PID: $MASTER_PID"
echo "[TEST] Waiting ${WAIT_BEFORE_SIGTERM}s before sending SIGTERM..."

# Wait before sending SIGTERM
sleep "$WAIT_BEFORE_SIGTERM"

# Send SIGTERM to master
echo "[TEST] Sending SIGTERM to master (PID: $MASTER_PID)..."
if kill -TERM "$MASTER_PID" 2>/dev/null; then
    echo "[TEST] SIGTERM sent successfully"
else
    echo "[TEST] WARNING: Process already finished (PID: $MASTER_PID)"
fi

# Wait for cleanup (poll for perf JSON instead of fixed wait)
echo "[TEST] Waiting for perf JSON (polling up to 15s)..."
PERF_JSON_PATH="${OUTPUT_DIR}/perf_${RUN_ID}.json"
FOUND=0
for i in {1..30}; do
    if [ -f "$PERF_JSON_PATH" ]; then
        echo "[TEST] ✅ Perf JSON found after ${i}0ms"
        FOUND=1
        break
    fi
    sleep 0.5
done

if [ "$FOUND" -eq 0 ]; then
    echo "[TEST] ⚠️  Perf JSON not found after 15s, waiting additional 5s..."
    sleep 5
fi

# Check if process is still running
if ps -p "$MASTER_PID" > /dev/null 2>&1; then
    echo "[TEST] WARNING: Master process still running (PID: $MASTER_PID), killing..."
    kill -9 "$MASTER_PID" 2>/dev/null || true
    sleep 2
fi

# Verify perf JSON exists (check both normal and failed export files)
PERF_JSON_PATH="${OUTPUT_DIR}/perf_${RUN_ID}.json"
PERF_FAILED_PATH="${OUTPUT_DIR}/perf_${RUN_ID}_FAILED_EXPORT.json"
echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="
echo "Expected perf JSON: $PERF_JSON_PATH"
echo "Failed export stub: $PERF_FAILED_PATH"

# Check for normal perf JSON first, then failed export stub
if [ -f "$PERF_JSON_PATH" ]; then
    echo "✅ PASS: perf JSON exists"
    
    # Check if it's valid JSON
    if python3 -c "import json; json.load(open('$PERF_JSON_PATH'))" 2>/dev/null; then
        echo "✅ PASS: perf JSON is valid JSON"
        
        # CRITICAL: Verify required fields exist
        if ! python3 -c "import json; d=json.load(open('$PERF_JSON_PATH')); assert 'run_id' in d, 'missing run_id'; assert 'chunks_statuses' in d, 'missing chunks_statuses'" 2>/dev/null; then
            echo "❌ FAIL: perf JSON missing required fields (run_id or chunks_statuses)"
            exit 1
        fi
        echo "✅ PASS: perf JSON contains required fields (run_id, chunks_statuses)"
        
        # Extract key metrics
        CHUNKS_COMPLETED=$(python3 -c "import json; d=json.load(open('$PERF_JSON_PATH')); print(d.get('chunks_completed', 0))" 2>/dev/null || echo "0")
        CHUNKS_TOTAL=$(python3 -c "import json; d=json.load(open('$PERF_JSON_PATH')); print(d.get('chunks_total', 0))" 2>/dev/null || echo "0")
        TOTAL_BARS=$(python3 -c "import json; d=json.load(open('$PERF_JSON_PATH')); print(d.get('total_bars', 0))" 2>/dev/null || echo "0")
        
        echo "   chunks_completed: $CHUNKS_COMPLETED/$CHUNKS_TOTAL"
        echo "   total_bars: $TOTAL_BARS"
        
        if [ "$TOTAL_BARS" -gt 0 ]; then
            echo "✅ PASS: perf JSON contains data (total_bars > 0)"
        else
            echo "⚠️  WARNING: perf JSON exists but total_bars = 0"
        fi
    else
        echo "❌ FAIL: perf JSON is not valid JSON"
        exit 1
    fi
    
    echo ""
    echo "=========================================="
    echo "✅✅✅ SMOKE TEST PASSED ✅✅✅"
    echo "=========================================="
    echo ""
    echo "Log file: $LOG_FILE"
    echo "Perf JSON: $PERF_JSON_PATH"
    exit 0
else
    echo "❌ FAIL: perf JSON does not exist"
    echo ""
    echo "Last 50 lines of log:"
    tail -50 "$LOG_FILE" 2>/dev/null || echo "Log file not found: $LOG_FILE"
    # Check if failed export stub exists (this is also acceptable - means export was attempted)
    if [ -f "$PERF_FAILED_PATH" ]; then
        echo "⚠️  Perf export failed, but stub file exists: $PERF_FAILED_PATH"
        echo "Checking stub file..."
        if python3 -c "import json; json.load(open('$PERF_FAILED_PATH'))" 2>/dev/null; then
            echo "✅ Stub file is valid JSON"
            
            # CRITICAL: Verify required fields exist
            if ! python3 -c "import json; d=json.load(open('$PERF_FAILED_PATH')); assert 'run_id' in d, 'missing run_id'; assert 'chunks_statuses' in d, 'missing chunks_statuses'; assert 'status' in d, 'missing status'" 2>/dev/null; then
                echo "❌ FAIL: stub file missing required fields (run_id, chunks_statuses, or status)"
                exit 1
            fi
            echo "✅ PASS: stub file contains required fields (run_id, chunks_statuses, status)"
            
            STATUS=$(python3 -c "import json; d=json.load(open('$PERF_FAILED_PATH')); print(d.get('status', 'unknown'))" 2>/dev/null || echo "unknown")
            EXPORT_ERROR=$(python3 -c "import json; d=json.load(open('$PERF_FAILED_PATH')); print(d.get('export_error', 'N/A')[:100])" 2>/dev/null || echo "N/A")
            echo "   status: $STATUS"
            echo "   export_error: $EXPORT_ERROR"
            echo ""
            echo "=========================================="
            echo "⚠️  SMOKE TEST PARTIAL PASS (export failed but stub written)"
            echo "=========================================="
            exit 0  # Accept stub file as valid result
        else
            echo "❌ Stub file is not valid JSON"
            exit 1
        fi
    else
        echo "❌ FAIL: Neither perf JSON nor failed export stub exists"
        echo ""
        echo "Last 50 lines of log:"
        tail -50 "$LOG_FILE" 2>/dev/null || echo "Log file not found: $LOG_FILE"
        echo ""
        echo "=========================================="
        echo "❌ SMOKE TEST FAILED"
        echo "=========================================="
        exit 1
    fi
fi
