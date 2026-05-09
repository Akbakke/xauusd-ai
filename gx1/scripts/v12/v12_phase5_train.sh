#!/usr/bin/env bash
# V12 Phase 5: train Exit-IQL v5 on V12_PER_BAR_V3TRACKED dataset.
#
# Usage:
#   /tmp/v12_phase5_train.sh <V3TRACKED_LOCK>
#   /tmp/v12_phase5_train.sh <V3TRACKED_LOCK> --budget medium --k-primary 480
#
# Argument 1 = V12 final dataset (Phase 4 output). Remaining args passed through.
set -euo pipefail

V3TRACKED_LOCK="${1:?Usage: $0 <V3TRACKED_LOCK> [--budget X] [--k-primary K] [--variants ...]}"
shift
[[ -d "$V3TRACKED_LOCK/per_week" ]] || { echo "ERR: $V3TRACKED_LOCK/per_week missing" >&2; exit 1; }

REPO=/home/andre2/src/GX1_ENGINE
LOG_DIR=/tmp/v12_cascade_logs
mkdir -p "$LOG_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT_LOCK="${V3TRACKED_LOCK}_TRAINED_${TS}_LOCK"

echo "============================================================"
echo "V12 Phase 5: Train Exit-IQL v5"
echo "  input    = $V3TRACKED_LOCK"
echo "  output   = $OUT_LOCK"
echo "  log      = $LOG_DIR/phase5_${TS}.log"
echo "  extra    = $*"
echo "============================================================"

PYTHONPATH=$REPO python3 -u $REPO/gx1/scripts/materialize_build_exit_iql_v5_v12_m1.py \
    --per-bar-dir "$V3TRACKED_LOCK" \
    --out-root "$OUT_LOCK" \
    "$@" \
    > "$LOG_DIR/phase5_${TS}.log" 2>&1

echo "✅ Phase 5 done"
echo "Trained checkpoints: $OUT_LOCK/trained_models_v1/"
echo "Summary: $OUT_LOCK/summary_v1.json"
