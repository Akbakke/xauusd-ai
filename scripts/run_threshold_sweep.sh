#!/bin/bash
# Entry threshold sweep for FULLYEAR analysis
# Runs FULLYEAR replay with multiple thresholds to analyze trade performance

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
test "$(pwd)" = "$ROOT" || { echo "❌ FAIL: Not in repo-root"; exit 1; }

set -e

# Thresholds to test
THRESHOLDS=(0.20 0.30 0.40 0.50 0.60)

# Output directory
OUTPUT_BASE="reports/replay_eval/THRESHOLD_SWEEP"

echo "=== ENTRY THRESHOLD SWEEP ==="
echo "Thresholds: ${THRESHOLDS[*]}"
echo "Output base: $OUTPUT_BASE"
echo ""

# GUARD CHECK (STOPP-OG-SE-BAKOVER): print what we are about to run
POLICY_YAML="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml"
DATA_PATH="data/raw/xauusd_m5_2025_bid_ask.parquet"
echo "[GUARD_CHECK] policy_yaml=${POLICY_YAML}"
echo "[GUARD_CHECK] policy_sha256=$(shasum -a 256 "${POLICY_YAML}" 2>/dev/null | awk '{print $1}' || echo 'n/a')"
echo "[GUARD_CHECK] dataset=${DATA_PATH}"
echo "[GUARD_CHECK] prebuilt_path=${GX1_REPLAY_PREBUILT_FEATURES_PATH}"
echo "[GUARD_CHECK] mode=PREBUILT"
echo "[GUARD_CHECK] analysis_mode=${GX1_ANALYSIS_MODE:-0}"
echo ""

# Ensure preflight marker exists
PREFLIGHT_MARKER="/tmp/gx1_prebuilt_preflight_passed"
if [ ! -f "${PREFLIGHT_MARKER}" ]; then
    echo "❌ Preflight marker not found. Run go_nogo_prebuilt.sh first."
    exit 1
fi

# Global lock
LOCKFILE="/tmp/gx1_replay_lock"
if [ -f "${LOCKFILE}" ]; then
    LOCK_PID=$(cat "${LOCKFILE}")
    if ps -p "${LOCK_PID}" > /dev/null 2>&1; then
        echo "❌ GLOBAL LOCK FAILED: Another replay is already running (PID: ${LOCK_PID})"
        exit 1
    else
        rm -f "${LOCKFILE}"
    fi
fi

# Set environment variables (same as run_fullyear_prebuilt.sh)
export GX1_REPLAY_USE_PREBUILT_FEATURES=1
export GX1_REPLAY_PREBUILT_FEATURES_PATH="data/features/xauusd_m5_2025_features_v10_ctx.parquet"
export GX1_FEATURE_BUILD_DISABLED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export GX1_XGB_THREADS=1
export GX1_GATED_FUSION_ENABLED=1
export GX1_REQUIRE_XGB_CALIBRATION=1
export GX1_REPLAY_INCREMENTAL_FEATURES=1
export GX1_REPLAY_NO_CSV=1
export GX1_FEATURE_USE_NP_ROLLING=1

# Analysis mode
export GX1_ANALYSIS_MODE=1

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Run each threshold
for threshold in "${THRESHOLDS[@]}"; do
    echo ""
    echo "=== Running threshold=$threshold ==="
    
    # Create unique run directory
    RUN_ID="threshold_${threshold}_$(date +%Y%m%d_%H%M%S)"
    RUN_DIR="$OUTPUT_BASE/$RUN_ID"
    
    # Hard reset - output-dir cannot be reused
    if [ -d "${RUN_DIR}" ] && [ "$(ls -A ${RUN_DIR} 2>/dev/null)" ]; then
        echo "❌ OUTPUT DIR FAILED: Output directory exists: ${RUN_DIR}"
        echo "Removing..."
        rm -rf "${RUN_DIR}"
    fi
    
    mkdir -p "${RUN_DIR}"
    
    echo "Run ID: $RUN_ID"
    echo "Run dir: $RUN_DIR"
    
    # Export threshold override
    export GX1_ENTRY_THRESHOLD_OVERRIDE="$threshold"

    echo "[GUARD_CHECK] run_id=${RUN_ID} output_dir=${RUN_DIR}"
    echo "[GUARD_CHECK] analysis_mode=${GX1_ANALYSIS_MODE} threshold_override=${GX1_ENTRY_THRESHOLD_OVERRIDE}"
    
    # Create lockfile
    echo $$ > "${LOCKFILE}"
    
    # Run FULLYEAR replay with threshold override
    python3 gx1/scripts/replay_eval_gated_parallel.py \
        --policy "${POLICY_YAML}" \
        --data "${DATA_PATH}" \
        --workers 7 \
        --output-dir "${RUN_DIR}" \
        --run-id "${RUN_ID}" \
        2>&1 | tee "/tmp/threshold_sweep_${RUN_ID}.log"
    
    REPLAY_RESULT=$?
    
    # Remove lockfile
    rm -f "${LOCKFILE}"
    
    if [ ${REPLAY_RESULT} -ne 0 ]; then
        echo "❌ Replay failed for threshold=$threshold"
        continue
    fi
    
    # Generate threshold analysis report
    echo "Generating analysis report..."
    python3 scripts/generate_threshold_analysis.py "$RUN_DIR" "$threshold" || true
    
    echo "✅ Threshold $threshold complete"
done

echo ""
echo "=== Generating sweep summary ==="
python3 scripts/generate_threshold_sweep_summary.py "$OUTPUT_BASE" || true

echo ""
echo "=== THRESHOLD SWEEP COMPLETE ==="
echo "Results: $OUTPUT_BASE"
