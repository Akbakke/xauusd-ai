#!/bin/bash
# FULLYEAR PREBUILT REPLAY
# Runs full-year replay with prebuilt features
# REQUIRES: go_nogo_prebuilt.sh must have PASSED (creates /tmp/gx1_prebuilt_preflight_passed)
#
# STANDARD DAILY FLOW:
# 1. ./scripts/go_nogo_prebuilt.sh (preflight, <10 min)
# 2. If GO: ./scripts/run_fullyear_prebuilt.sh (this script, <1 hour)
# 3. Compare results with baseline
# 4. Report
#
# IF FAIL:
# - Check logs for [PREBUILT_FAIL] errors
# - Verify: data/features/xauusd_m5_2025_features_v10_ctx.parquet exists and is valid
# - Re-run go_nogo_prebuilt.sh to regenerate preflight marker

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
test "$(pwd)" = "$ROOT" || { echo "❌ FAIL: Not in repo-root (pwd=$(pwd), root=$ROOT)"; exit 1; }
echo "[RUN_CTX] root=$ROOT"
echo "[RUN_CTX] head=$(git rev-parse --short HEAD)"
echo "[RUN_CTX] whoami=$(whoami) host=$(hostname)"

# GUARD CHECK (STOPP-OG-SE-BAKOVER): print what we are about to run
POLICY_YAML="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml"
DATA_PATH="data/raw/xauusd_m5_2025_bid_ask.parquet"
PREBUILT_PATH="data/features/xauusd_m5_2025_features_v10_ctx.parquet"
echo "[GUARD_CHECK] policy_yaml=${POLICY_YAML}"
echo "[GUARD_CHECK] policy_sha256=$(shasum -a 256 "${POLICY_YAML}" 2>/dev/null | awk '{print $1}' || echo 'n/a')"
echo "[GUARD_CHECK] dataset=${DATA_PATH}"
echo "[GUARD_CHECK] prebuilt_path=${PREBUILT_PATH}"
echo "[GUARD_CHECK] mode=PREBUILT"
echo "[GUARD_CHECK] analysis_mode=${GX1_ANALYSIS_MODE:-0}"
echo "[GUARD_CHECK] threshold_override=${GX1_ENTRY_THRESHOLD_OVERRIDE:-<none>}"

set -e

echo "=== FULLYEAR PREBUILT REPLAY ==="
echo ""

# FASE 3: Sjekk at preflight marker eksisterer
PREFLIGHT_MARKER="/tmp/gx1_prebuilt_preflight_passed"
if [ ! -f "${PREFLIGHT_MARKER}" ]; then
    echo "❌ PREFLIGHT GATE FAILED: Preflight marker not found: ${PREFLIGHT_MARKER}"
    echo ""
    echo "PREBUILT replay requires preflight check to PASS first."
    echo ""
    echo "Run preflight check:"
    echo "  ./scripts/go_nogo_prebuilt.sh"
    echo ""
    echo "This will:"
    echo "  1. Run preflight validation (7 days)"
    echo "  2. Run 2-day sanity check (baseline vs prebuilt)"
    echo "  3. Run 7-day smoke test"
    echo "  4. Run early abort test"
    echo "  5. Create preflight marker if all checks PASS"
    echo ""
    exit 1
fi

PREFLIGHT_TIMESTAMP=$(cat "${PREFLIGHT_MARKER}")
echo "[PREFLIGHT] ✅ Preflight marker found (created: ${PREFLIGHT_TIMESTAMP})"
echo ""

# FASE 0.1: Global lock - sjekk at ingen annen replay kjører
LOCKFILE="/tmp/gx1_replay_lock"
if [ -f "${LOCKFILE}" ]; then
    LOCK_PID=$(cat "${LOCKFILE}")
    if ps -p "${LOCK_PID}" > /dev/null 2>&1; then
        echo "❌ GLOBAL LOCK FAILED: Another replay is already running (PID: ${LOCK_PID})"
        echo ""
        echo "Only one replay can run at a time to prevent resource conflicts."
        echo "Wait for the existing replay to complete, or kill it if stuck:"
        echo "  kill ${LOCK_PID}"
        echo ""
        exit 1
    else
        # Stale lockfile, remove it
        rm -f "${LOCKFILE}"
    fi
fi

# Create lockfile
echo $$ > "${LOCKFILE}"
trap "rm -f ${LOCKFILE}" EXIT
echo "[LOCK] ✅ Global lock acquired (PID: $$)"
echo ""

# FASE 0.2: Hard reset - output-dir kan ikke gjenbrukes
OUTPUT_DIR="reports/replay_eval/PREBUILT_FULLYEAR"
if [ -d "${OUTPUT_DIR}" ] && [ "$(ls -A ${OUTPUT_DIR} 2>/dev/null)" ]; then
    echo "❌ OUTPUT DIR FAILED: Output directory exists and contains files: ${OUTPUT_DIR}"
    echo ""
    echo "Output directory cannot be reused. Remove it first:"
    echo "  rm -rf ${OUTPUT_DIR}"
    echo ""
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
echo "[OUTPUT] ✅ Output directory ready: ${OUTPUT_DIR}"
echo ""

# FASE 0.3: Sett GX1_FEATURE_BUILD_DISABLED når prebuilt er enabled
export GX1_REPLAY_USE_PREBUILT_FEATURES=1
export GX1_REPLAY_PREBUILT_FEATURES_PATH="data/features/xauusd_m5_2025_features_v10_ctx.parquet"
export GX1_FEATURE_BUILD_DISABLED=1
echo "[ENV] ✅ Prebuilt mode enabled"
echo "[ENV] ✅ Feature build disabled: GX1_FEATURE_BUILD_DISABLED=1"
echo ""

# Set thread limits
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export GX1_XGB_THREADS=1

# FASE 5: Quiet mode removed
export GX1_GATED_FUSION_ENABLED=1
export GX1_REQUIRE_XGB_CALIBRATION=1
export GX1_REPLAY_INCREMENTAL_FEATURES=1
export GX1_REPLAY_NO_CSV=1
export GX1_FEATURE_USE_NP_ROLLING=1

# Verify prebuilt features file exists
if [ ! -f "${GX1_REPLAY_PREBUILT_FEATURES_PATH}" ]; then
    echo "❌ PREBUILT PATH FAILED: Prebuilt features file not found: ${GX1_REPLAY_PREBUILT_FEATURES_PATH}"
    echo ""
    echo "Prebuilt features file must exist before running FULLYEAR replay."
    echo "Generate it first, then re-run go_nogo_prebuilt.sh"
    echo ""
    exit 1
fi

echo "[PREBUILT] ✅ Prebuilt features file found: ${GX1_REPLAY_PREBUILT_FEATURES_PATH}"
echo ""

# Run FULLYEAR replay
RUN_ID="prebuilt_fullyear_$(date +%Y%m%d_%H%M%S)"
echo "[REPLAY] Starting FULLYEAR replay..."
echo "[REPLAY] Run ID: ${RUN_ID}"
echo "[REPLAY] Workers: 7"
echo "[REPLAY] Data: data/raw/xauusd_m5_2025_bid_ask.parquet"
echo ""

python3 gx1/scripts/replay_eval_gated_parallel.py \
  --policy "${POLICY_YAML}" \
  --data "${DATA_PATH}" \
  --workers 7 \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}" \
  2>&1 | tee "/tmp/prebuilt_fullyear_${RUN_ID}.log"

REPLAY_RESULT=$?

# Remove lockfile
rm -f "${LOCKFILE}"

if [ ${REPLAY_RESULT} -ne 0 ]; then
    echo ""
    echo "❌ FULLYEAR replay FAILED (exit code: ${REPLAY_RESULT})"
    echo "Check logs: /tmp/prebuilt_fullyear_${RUN_ID}.log"
    exit 1
fi

PERF_JSON="${OUTPUT_DIR}/perf_${RUN_ID}.json"
if [ ! -f "${PERF_JSON}" ]; then
    echo ""
    echo "❌ FULLYEAR replay FAILED: Perf JSON not found: ${PERF_JSON}"
    exit 1
fi

echo ""
echo "=== FULLYEAR PREBUILT REPLAY: ✅ COMPLETE ==="
echo ""
echo "Results:"
echo "  - Perf JSON: ${PERF_JSON}"
echo "  - Log: /tmp/prebuilt_fullyear_${RUN_ID}.log"
echo ""
echo "Expected invariants (verify in perf JSON):"
echo "  - prebuilt_used=true"
echo "  - basic_v1_call_count=0"
echo "  - FEATURE_BUILD_TIMEOUT=0"
echo "  - feature_time_mean_ms <= 5ms"
echo "  - prebuilt_bypass_count == n_model_calls"
echo ""

exit 0
