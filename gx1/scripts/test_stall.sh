#!/bin/bash
# Intentional stall test script for watchdog verification
# G2: INTENTIONAL STALL TEST (proves watchdog kill)

set -e

# Canonical inputs (same as FULLYEAR)
CANONICAL_PYTHON="/home/andre2/venvs/gx1/bin/python"
CANONICAL_POLICY="/home/andre2/GX1_DATA/configs/policies/canonical/TRUTH_BASELINE_V12AB91.yaml"
CANONICAL_DATA="/home/andre2/GX1_DATA/data/data/raw/xauusd_m5_2025_bid_ask.parquet"
CANONICAL_PREBUILT="/home/andre2/GX1_DATA/data/data/prebuilt/TRIAL160/2025/xauusd_m5_2025_features_v10_ctx.parquet"
CANONICAL_BUNDLE_DIR="/home/andre2/GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2024_2025_V12AB91"

# Test window: 7 days (2025-01-03 to 2025-01-10)
START_TS="2025-01-03T00:00:00Z"
END_TS="2025-01-10T00:00:00Z"

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/home/andre2/GX1_DATA/reports/replay_eval/STALL_TEST_2025_TRUTH_W19_${TIMESTAMP}"

echo "[STALL_TEST] Starting intentional stall test"
echo "  Output: ${OUTPUT_DIR}"
echo "  Window: ${START_TS} to ${END_TS}"
echo "  Workers: 19"
echo "  Strategy: Kill workers after they boot (simulate stall)"

# Export environment variables
export GX1_RUN_MODE=TRUTH
export GX1_CANONICAL_BUNDLE_DIR="${CANONICAL_BUNDLE_DIR}"
export GX1_GATED_FUSION_ENABLED=1
export GX1_XGB_INPUT_FINGERPRINT=1
export GX1_XGB_INPUT_FINGERPRINT_SAMPLE_N=10
export GX1_XGB_INPUT_FINGERPRINT_MAX_PER_SESSION=500
export GX1_WATCHDOG_STALL_TIMEOUT_SEC=60  # Short timeout for test

# Start replay in background
echo "[STALL_TEST] Starting replay in background..."
/home/andre2/src/GX1_ENGINE/gx1/scripts/run_replay_canonical.sh \
    --policy "${CANONICAL_POLICY}" \
    --data "${CANONICAL_DATA}" \
    --prebuilt-parquet "${CANONICAL_PREBUILT}" \
    --workers 19 \
    --chunk-local-padding-days 7 \
    --start-ts "${START_TS}" \
    --end-ts "${END_TS}" \
    --output-dir "${OUTPUT_DIR}" \
    > "${OUTPUT_DIR}/run.log" 2>&1 &
REPLAY_PID=$!

echo "[STALL_TEST] Replay started with PID: ${REPLAY_PID}"

# Wait for workers to boot (check for WORKER_BOOT.json files)
echo "[STALL_TEST] Waiting for workers to boot..."
MAX_WAIT=120
WAITED=0
WORKER_BOOTS=0
while [ ${WAITED} -lt ${MAX_WAIT} ]; do
    WORKER_BOOTS=$(find "${OUTPUT_DIR}" -name "WORKER_BOOT.json" 2>/dev/null | wc -l)
    if [ ${WORKER_BOOTS} -gt 0 ]; then
        echo "[STALL_TEST] Found ${WORKER_BOOTS} worker boot(s), proceeding with stall..."
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
done

if [ ${WORKER_BOOTS} -eq 0 ]; then
    echo "[STALL_TEST] WARNING: No workers booted within ${MAX_WAIT}s, proceeding anyway..."
fi

# Find and kill worker processes (simulate stall)
echo "[STALL_TEST] Killing worker processes to simulate stall..."
WORKER_PIDS=$(pgrep -P ${REPLAY_PID} 2>/dev/null || true)
if [ -n "${WORKER_PIDS}" ]; then
    echo "[STALL_TEST] Found worker PIDs: ${WORKER_PIDS}"
    for PID in ${WORKER_PIDS}; do
        echo "[STALL_TEST] Killing worker PID: ${PID}"
        kill -9 ${PID} 2>/dev/null || true
    done
else
    echo "[STALL_TEST] No worker PIDs found, trying alternative method..."
    # Try to find Python processes running replay_worker.py
    ps aux | grep "[r]eplay_worker.py" | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
fi

# Wait for watchdog to detect stall and kill master
echo "[STALL_TEST] Waiting for watchdog to detect stall (timeout: 120s)..."
STALL_DETECTED=0
for i in {1..24}; do
    if [ -f "${OUTPUT_DIR}/RUN_STALL_FATAL.json" ]; then
        echo "[STALL_TEST] ✅ RUN_STALL_FATAL.json detected!"
        STALL_DETECTED=1
        break
    fi
    sleep 5
done

# Wait for master to exit
echo "[STALL_TEST] Waiting for master to exit..."
wait ${REPLAY_PID} 2>/dev/null || true

# Check results
echo ""
echo "[STALL_TEST] Checking results..."

# Check for stall fatal
if [ -f "${OUTPUT_DIR}/RUN_STALL_FATAL.json" ]; then
    echo "  ✅ RUN_STALL_FATAL.json exists"
    STALL_DETECTED=1
else
    echo "  ❌ RUN_STALL_FATAL.json missing"
fi

# Check for RUN_FAILED
if [ -f "${OUTPUT_DIR}/RUN_FAILED.json" ]; then
    echo "  ✅ RUN_FAILED.json exists"
else
    echo "  ⚠️  RUN_FAILED.json missing (may be normal if watchdog uses os._exit)"
fi

# Check for heartbeat
if [ -f "${OUTPUT_DIR}/HEARTBEAT.json" ]; then
    echo "  ✅ HEARTBEAT.json exists"
    HEARTBEAT_AGE=$(stat -c %Y "${OUTPUT_DIR}/HEARTBEAT.json" 2>/dev/null || echo "0")
    NOW=$(date +%s)
    AGE=$((NOW - HEARTBEAT_AGE))
    echo "  Heartbeat age: ${AGE}s"
else
    echo "  ❌ HEARTBEAT.json missing"
fi

# Run monitor --once
echo ""
echo "[STALL_TEST] Running monitor --once:"
"${CANONICAL_PYTHON}" /home/andre2/src/GX1_ENGINE/gx1/scripts/monitor_run.py \
    --run-dir "${OUTPUT_DIR}" \
    --once

# Generate proof document
PROOF_FILE="${OUTPUT_DIR}/STALL_PROOF.md"
cat > "${PROOF_FILE}" <<EOF
# Intentional Stall Test Proof (G2)

**Test Date:** $(date -u +%Y-%m-%dT%H:%M:%SZ)
**Output Directory:** \`${OUTPUT_DIR}\`
**Test Window:** ${START_TS} to ${END_TS}
**Workers:** 19
**Stall Strategy:** Kill worker processes after boot to simulate stall

## Test Steps

1. Started replay with 19 workers
2. Waited for workers to boot (found ${WORKER_BOOTS} WORKER_BOOT.json files)
3. Killed worker processes to simulate stall
4. Waited for watchdog to detect stall and write RUN_STALL_FATAL.json
5. Verified master process exited

## Test Results

### Stall Detection
- **RUN_STALL_FATAL.json:** $([ -f "${OUTPUT_DIR}/RUN_STALL_FATAL.json" ] && echo "✅ EXISTS" || echo "❌ MISSING")
- **RUN_FAILED.json:** $([ -f "${OUTPUT_DIR}/RUN_FAILED.json" ] && echo "✅ EXISTS" || echo "⚠️  MISSING (may be normal)")
- **HEARTBEAT.json:** $([ -f "${OUTPUT_DIR}/HEARTBEAT.json" ] && echo "✅ EXISTS" || echo "❌ MISSING")

### Stall Fatal Details
\`\`\`
$(if [ -f "${OUTPUT_DIR}/RUN_STALL_FATAL.json" ]; then
    cat "${OUTPUT_DIR}/RUN_STALL_FATAL.json" | "${CANONICAL_PYTHON}" -m json.tool 2>/dev/null || echo "Failed to parse JSON"
else
    echo "RUN_STALL_FATAL.json not found"
fi)
\`\`\`

### Monitor Output
\`\`\`
$("${CANONICAL_PYTHON}" /home/andre2/src/GX1_ENGINE/gx1/scripts/monitor_run.py --run-dir "${OUTPUT_DIR}" --once 2>&1)
\`\`\`

### Artifacts Checklist
- [ ] RUN_STALL_FATAL.json exists
- [ ] Monitor reports FAIL status (exit code 2)
- [ ] Watchdog detected stall within timeout
- [ ] Master process exited (not hanging)

## Conclusion

$(if [ ${STALL_DETECTED} -eq 1 ]; then
    echo "✅ **PASS**: Watchdog successfully detected stall and wrote RUN_STALL_FATAL.json."
else
    echo "❌ **FAIL**: Watchdog did not detect stall. Check logs and watchdog configuration."
fi)
EOF

echo ""
echo "[STALL_TEST] Proof document written: ${PROOF_FILE}"
echo "[STALL_TEST] Test complete!"
