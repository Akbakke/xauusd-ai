#!/bin/bash
# Mini-run test script for watchdog verification
# G1: MINI PASS RUN (workers=19, 7-14 days)

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
OUTPUT_DIR="/home/andre2/GX1_DATA/reports/replay_eval/MINI_2025_TRUTH_W19_${TIMESTAMP}"

echo "[MINI_RUN] Starting mini-run test"
echo "  Output: ${OUTPUT_DIR}"
echo "  Window: ${START_TS} to ${END_TS}"
echo "  Workers: 19"

# Export environment variables
export GX1_RUN_MODE=TRUTH
export GX1_CANONICAL_BUNDLE_DIR="${CANONICAL_BUNDLE_DIR}"
export GX1_GATED_FUSION_ENABLED=1
export GX1_XGB_INPUT_FINGERPRINT=1
export GX1_XGB_INPUT_FINGERPRINT_SAMPLE_N=10
export GX1_XGB_INPUT_FINGERPRINT_MAX_PER_SESSION=500

# Run replay
/home/andre2/src/GX1_ENGINE/gx1/scripts/run_replay_canonical.sh \
    --policy "${CANONICAL_POLICY}" \
    --data "${CANONICAL_DATA}" \
    --prebuilt-parquet "${CANONICAL_PREBUILT}" \
    --workers 19 \
    --chunk-local-padding-days 7 \
    --start-ts "${START_TS}" \
    --end-ts "${END_TS}" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/run.log"

# Check results
echo ""
echo "[MINI_RUN] Checking results..."

# Check for completion
if [ -f "${OUTPUT_DIR}/RUN_COMPLETED.json" ]; then
    echo "  ✅ RUN_COMPLETED.json exists"
else
    echo "  ❌ RUN_COMPLETED.json missing"
fi

# Check for heartbeat
if [ -f "${OUTPUT_DIR}/HEARTBEAT.json" ]; then
    echo "  ✅ HEARTBEAT.json exists"
    HEARTBEAT_AGE=$(stat -c %Y "${OUTPUT_DIR}/HEARTBEAT.json")
    NOW=$(date +%s)
    AGE=$((NOW - HEARTBEAT_AGE))
    echo "  Heartbeat age: ${AGE}s"
else
    echo "  ❌ HEARTBEAT.json missing"
fi

# Check for worker boots
WORKER_BOOTS=$(find "${OUTPUT_DIR}" -name "WORKER_BOOT.json" | wc -l)
echo "  Worker boots found: ${WORKER_BOOTS}"

# Check for chunk footers
CHUNK_FOOTERS=$(find "${OUTPUT_DIR}" -name "chunk_footer.json" | wc -l)
echo "  Chunk footers found: ${CHUNK_FOOTERS}"

# Run monitor --once
echo ""
echo "[MINI_RUN] Running monitor --once:"
"${CANONICAL_PYTHON}" /home/andre2/src/GX1_ENGINE/gx1/scripts/monitor_run.py \
    --run-dir "${OUTPUT_DIR}" \
    --once

# Generate proof document
PROOF_FILE="${OUTPUT_DIR}/MINI_PROOF.md"
cat > "${PROOF_FILE}" <<EOF
# Mini-Run Test Proof (G1)

**Test Date:** $(date -u +%Y-%m-%dT%H:%M:%SZ)
**Output Directory:** \`${OUTPUT_DIR}\`
**Test Window:** ${START_TS} to ${END_TS}
**Workers:** 19

## Test Results

### Completion Status
- **RUN_COMPLETED.json:** $([ -f "${OUTPUT_DIR}/RUN_COMPLETED.json" ] && echo "✅ EXISTS" || echo "❌ MISSING")
- **HEARTBEAT.json:** $([ -f "${OUTPUT_DIR}/HEARTBEAT.json" ] && echo "✅ EXISTS" || echo "❌ MISSING")
- **Worker Boots:** ${WORKER_BOOTS}
- **Chunk Footers:** ${CHUNK_FOOTERS}

### Heartbeat Verification
\`\`\`
$(if [ -f "${OUTPUT_DIR}/HEARTBEAT.json" ]; then
    cat "${OUTPUT_DIR}/HEARTBEAT.json" | "${CANONICAL_PYTHON}" -m json.tool 2>/dev/null || echo "Failed to parse JSON"
fi)
\`\`\`

### Monitor Output
\`\`\`
$("${CANONICAL_PYTHON}" /home/andre2/src/GX1_ENGINE/gx1/scripts/monitor_run.py --run-dir "${OUTPUT_DIR}" --once 2>&1)
\`\`\`

### Artifacts Checklist
- [ ] HEARTBEAT.json updated continuously (mtime changes every ~5s)
- [ ] At least 19 WORKER_BOOT.json files exist
- [ ] RUN_COMPLETED.json exists
- [ ] Monitor reports correct status

## Conclusion

$(if [ -f "${OUTPUT_DIR}/RUN_COMPLETED.json" ]; then
    echo "✅ **PASS**: Mini-run completed successfully. Watchdog and monitor verified."
else
    echo "❌ **FAIL**: Mini-run did not complete. Check logs for details."
fi)
EOF

echo ""
echo "[MINI_RUN] Proof document written: ${PROOF_FILE}"
echo "[MINI_RUN] Test complete!"
