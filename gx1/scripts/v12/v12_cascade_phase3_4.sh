#!/usr/bin/env bash
# V12 cascade orchestrator: Phase 3 (V3 v8 augment) → Phase 4 (V3-tracking features)
#
# Usage:
#   /tmp/v12_cascade_phase3_4.sh <V12_PER_BAR_LOCK>
# Example:
#   /tmp/v12_cascade_phase3_4.sh /home/andre2/GX1_DATA/reports/truth_e2e_sanity/BUILD_EXIT_IQL_PER_BAR_DATASET_V12_20260508T093002Z_LOCK
#
# Reads from <V12_PER_BAR_LOCK>/per_week/ and writes:
#   <V12_PER_BAR_LOCK>_V3AUG_<TS>_LOCK/per_week/  (Phase 3 output)
#   <V12_PER_BAR_LOCK>_V3TRACKED_<TS>_LOCK/per_week/  (Phase 4 output, final V12 dataset)
#
# Optional env vars (smoke-mode):
#   WEEK_FILTER="<week_stem>"  # repeat by space-separating; passes --week to Phase 4
#   MAX_WEEKS=N                # passes --max-weeks N to Phase 4
#
# V12.2 multi-TF (V3 v9+) support:
#   V3_BUNDLE        — override default V3 bundle (default = v8 non-multi-TF)
#   M5_PREBUILT_PATH — required when V3 bundle is multi-TF; forwarded to scorer
#   MULTI_TF_SEQ_LEN — optional, default 96 (forwarded only when M5_PREBUILT_PATH set)
set -euo pipefail

V12_LOCK="${1:?Usage: $0 <V12_PER_BAR_LOCK>}"
[[ -d "$V12_LOCK" ]] || { echo "ERR: $V12_LOCK not a directory" >&2; exit 1; }
[[ -d "$V12_LOCK/per_week" ]] || { echo "ERR: $V12_LOCK/per_week missing" >&2; exit 1; }

# FAIL-CLOSED (2026-06-03 audit): resolve the ACTIVE V3 bundle from PROJECT_STATE
# instead of hardcoding the pre-COSTFIX V9 (which would poison a downstream Exit-IQL
# retrain). On any guard error the substitution is empty -> the [[ -d ]] check below exits 1.
V3_BUNDLE="${V3_BUNDLE:-$(PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 -c 'from gx1_guards.artifacts import load_decision_artifact; print(load_decision_artifact("v3_exit"))' 2>/dev/null)}"
V3_DATASET="${V3_DATASET:-/home/andre2/GX1_DATA/data/training/exit_v3_v7_training_2020_2026_canonical_v3}"
[[ -d "$V3_BUNDLE" ]]  || { echo "ERR: V3 bundle missing: $V3_BUNDLE" >&2; exit 1; }
[[ -d "$V3_DATASET" ]] || { echo "ERR: V3 dataset missing: $V3_DATASET" >&2; exit 1; }

# V12.2: detect multi-TF from bundle's transformer_config.json and require M5_PREBUILT_PATH.
V3_CFG="$V3_BUNDLE/transformer_config.json"
V3_IS_MTF=0
if [[ -f "$V3_CFG" ]]; then
    if grep -q '"enabled":[[:space:]]*true' "$V3_CFG"; then
        V3_IS_MTF=1
    fi
fi
if [[ "$V3_IS_MTF" -eq 1 ]]; then
    M5_PREBUILT_PATH="${M5_PREBUILT_PATH:-}"
    if [[ -z "$M5_PREBUILT_PATH" ]]; then
        echo "ERR: V3 bundle is multi-TF (transformer_config.multi_tf.enabled=true)." >&2
        echo "     Set env M5_PREBUILT_PATH=<canonical_v3 M5 parquet> before launch." >&2
        exit 1
    fi
    [[ -f "$M5_PREBUILT_PATH" ]] || { echo "ERR: M5_PREBUILT_PATH not a file: $M5_PREBUILT_PATH" >&2; exit 1; }
fi
MULTI_TF_SEQ_LEN="${MULTI_TF_SEQ_LEN:-96}"

REPO=/home/andre2/src/GX1_ENGINE
LOG_DIR=/tmp/v12_cascade_logs
mkdir -p "$LOG_DIR"

TS=$(date -u +%Y%m%dT%H%M%SZ)
V3AUG_LOCK="${V12_LOCK}_V3AUG_${TS}_LOCK"
V3TRACKED_LOCK="${V12_LOCK}_V3TRACKED_${TS}_LOCK"

echo "============================================================"
echo "V12 cascade  Phase 3 → Phase 4"
echo "  V12_LOCK        = $V12_LOCK"
echo "  V3_BUNDLE       = $V3_BUNDLE"
echo "  V3_DATASET      = $V3_DATASET"
echo "  V3_IS_MTF       = $V3_IS_MTF (multi-TF V3 v9+ if 1)"
if [[ "$V3_IS_MTF" -eq 1 ]]; then
    echo "  M5_PREBUILT     = $M5_PREBUILT_PATH"
    echo "  MTF_SEQ_LEN     = $MULTI_TF_SEQ_LEN"
fi
echo "  V3AUG output    = $V3AUG_LOCK"
echo "  V3TRACKED final = $V3TRACKED_LOCK"
echo "============================================================"

echo
echo "[Phase 3] V3 per-bar inference augment"
echo "  log: $LOG_DIR/phase3_${TS}.log"
P3_EXTRA=()
if [[ -n "${WEEK_FILTER:-}" ]]; then
    for w in $WEEK_FILTER; do P3_EXTRA+=(--week "$w"); done
fi
if [[ "$V3_IS_MTF" -eq 1 ]]; then
    P3_EXTRA+=(--m5-prebuilt-path "$M5_PREBUILT_PATH" --multi-tf-seq-len "$MULTI_TF_SEQ_LEN")
fi
PYTHONPATH=$REPO python3 -u $REPO/gx1/scripts/score_v3_v8_on_per_bar_v1.py \
    --v3-bundle "$V3_BUNDLE" \
    --v3-dataset-dir "$V3_DATASET" \
    --per-bar-dir "$V12_LOCK" \
    --out-root "$V3AUG_LOCK" \
    "${P3_EXTRA[@]}" \
    > "$LOG_DIR/phase3_${TS}.log" 2>&1
echo "  ✅ Phase 3 done"

# Guard: Phase 3 must have produced at least one parquet before Phase 4
P3_OUT_COUNT=$(find "$V3AUG_LOCK/per_week" -maxdepth 1 -name "*.parquet" 2>/dev/null | wc -l)
if [[ "$P3_OUT_COUNT" -eq 0 ]]; then
    echo "  ❌ Phase 3 produced 0 parquets at $V3AUG_LOCK/per_week — aborting before Phase 4" >&2
    exit 2
fi
echo "  Phase 3 output: $P3_OUT_COUNT parquets"

echo
echo "[Phase 4] V3-tracking features"
echo "  log: $LOG_DIR/phase4_${TS}.log"
mkdir -p "$V3TRACKED_LOCK/per_week"
P4_EXTRA=()
if [[ -n "${WEEK_FILTER:-}" ]]; then
    for w in $WEEK_FILTER; do P4_EXTRA+=(--week "$w"); done
fi
if [[ -n "${MAX_WEEKS:-}" ]]; then
    P4_EXTRA+=(--max-weeks "$MAX_WEEKS")
fi
PYTHONPATH=$REPO python3 -u $REPO/gx1/scripts/v12/v12_phase4_v3_tracking_features.py \
    --in-dir  "$V3AUG_LOCK/per_week" \
    --out-dir "$V3TRACKED_LOCK/per_week" \
    "${P4_EXTRA[@]}" \
    > "$LOG_DIR/phase4_${TS}.log" 2>&1
echo "  ✅ Phase 4 done"

echo
echo "============================================================"
echo "V12 cascade complete"
echo "  Final dataset: $V3TRACKED_LOCK/per_week/"
echo "  Phase 3 log:   $LOG_DIR/phase3_${TS}.log"
echo "  Phase 4 log:   $LOG_DIR/phase4_${TS}.log"
echo "============================================================"
