#!/usr/bin/env bash
# Full-year 2025 A/B: Baseline (ML-exit off / force NORMAL) vs ML-Frozen (default policy).
# Runs both in parallel with 1 worker each. Calls replay_eval_gated_parallel directly
# (main process) so signal handling works; run_fullyear_2025_truth_proof runs replay in a
# thread and would hit "signal only works in main thread".
# Usage: ./scripts/run_fullyear_2025_ab_exit.sh
# Requires: GX1_ENGINE, GX1_DATA (set below if unset).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${GX1_ENGINE:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DATA="${GX1_DATA:-${HOME}/GX1_DATA}"
TRUTH_FILE="${GX1_CANONICAL_TRUTH_FILE:-$ENGINE/gx1/configs/canonical_truth_signal_only.json}"
POLICY_PATH="${GX1_CANONICAL_POLICY_PATH:-$ENGINE/gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml}"
RAW_2025="${GX1_RAW_2025:-$DATA/data/data/raw/xauusd_m5_2025_bid_ask.parquet}"
PYTHON="${GX1_PYTHON:-/home/andre2/venvs/gx1/bin/python}"
START_TS="2025-01-01T00:00:00+00:00"
END_TS="2025-12-31T23:59:59+00:00"

export GX1_ENGINE="$ENGINE"
export GX1_DATA="$DATA"
export GX1_CANONICAL_TRUTH_FILE="$TRUTH_FILE"
export GX1_CANONICAL_POLICY_PATH="$POLICY_PATH"
export GX1_RUN_MODE=TRUTH
export GX1_TRUTH_MODE=1
export GX1_GATED_FUSION_ENABLED=1
export GX1_REPLAY_USE_PREBUILT_FEATURES=1
export GX1_FEATURE_BUILD_DISABLED=1

# Resolve prebuilt and bundle dirs from truth file
PREBUILT=""
BUNDLE_DIR=""
TRANSFORMER_DIR=""
if [[ -f "$TRUTH_FILE" ]]; then
  PREBUILT=$(python3 -c "import json; o=json.load(open('$TRUTH_FILE')); print(o.get('canonical_prebuilt_parquet',''))")
  BUNDLE_DIR=$(python3 -c "import json; o=json.load(open('$TRUTH_FILE')); print(o.get('canonical_xgb_bundle_dir',''))")
  TRANSFORMER_DIR=$(python3 -c "import json; o=json.load(open('$TRUTH_FILE')); print(o.get('canonical_transformer_bundle_dir',''))")
fi
if [[ -z "$PREBUILT" || ! -f "$PREBUILT" ]]; then
  echo "[AB] ERROR: Prebuilt parquet missing. Set GX1_CANONICAL_TRUTH_FILE or fix canonical_prebuilt_parquet in truth file." >&2
  exit 1
fi
export GX1_CANONICAL_BUNDLE_DIR="${BUNDLE_DIR}"
export GX1_CANONICAL_TRANSFORMER_BUNDLE_DIR="${TRANSFORMER_DIR}"

TS=$(date -u +%Y%m%d_%H%M%S)
OUT_ROOT="$DATA/reports/fullyear_truth_proof"
LOG_DIR="$OUT_ROOT/ab_logs"
mkdir -p "$LOG_DIR"
BASELINE_ID="E2E_2025_BASELINE_EXIT_${TS}"
ML_ID="E2E_2025_ML_FROZEN_EXIT_${TS}"
BASELINE_LOG="$LOG_DIR/${BASELINE_ID}.log"
ML_LOG="$LOG_DIR/${ML_ID}.log"

REPLAY_CMD=(
  "$PYTHON" -m gx1.scripts.replay_eval_gated_parallel
  --policy "$POLICY_PATH"
  --data "$RAW_2025"
  --prebuilt-parquet "$PREBUILT"
  --workers 1
  --chunks 1
  --start-ts "$START_TS"
  --end-ts "$END_TS"
)

echo "[AB] TS=$TS"
echo "[AB] Baseline (force NORMAL) run_id=$BASELINE_ID -> $BASELINE_LOG"
echo "[AB] ML-Frozen (default)    run_id=$ML_ID -> $ML_LOG"

GX1_EXIT_FORCE_NORMAL=1 "${REPLAY_CMD[@]}" --output-dir "$OUT_ROOT/$BASELINE_ID" --run-id "$BASELINE_ID" > "$BASELINE_LOG" 2>&1 &
PID_B=$!

"${REPLAY_CMD[@]}" --output-dir "$OUT_ROOT/$ML_ID" --run-id "$ML_ID" > "$ML_LOG" 2>&1 &
PID_M=$!

echo "[AB] Baseline PID=$PID_B  ML-Frozen PID=$PID_M"
wait $PID_B; RC_B=$?
wait $PID_M; RC_M=$?
echo "[AB] Baseline exit=$RC_B  ML-Frozen exit=$RC_M"
exit $(( RC_B | RC_M ))
