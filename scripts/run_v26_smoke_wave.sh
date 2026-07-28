#!/usr/bin/env bash
# Drive one V26 model-native smoke wave: recipe audit -> smoke train.
#
# The V26 evidence artifacts are immutable and bound by their exact timestamped
# paths below; this launcher never regenerates or substitutes one. Every
# decision-affecting training value is passed explicitly, including all five
# per-timeframe lookback windows (rule 14 - no wrapper default, no ambient
# environment value).
#
# Usage: scripts/run_v26_smoke_wave.sh --run-id <ID> --subsample-rows N --epochs N
set -euo pipefail

REPO=/home/andre2/src/GX1_ENGINE
EVENT=/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v26_6yr_rebuild_20260725_seq513_model_native_v26
DS=$EVENT/dataset
STEM=v10_seq513_dataset__HOLD_03B

RUN_TRAIN=
SUBSAMPLE_ROWS=
EPOCHS=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_TRAIN="$2"; shift 2 ;;
    --subsample-rows) SUBSAMPLE_ROWS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    *) echo "FATAL: unknown argument: $1" >&2; exit 2 ;;
  esac
done
for v in RUN_TRAIN SUBSAMPLE_ROWS EPOCHS; do
  [[ -n "${!v}" ]] || { echo "FATAL: --${v,,} is required" >&2; exit 2; }
done

# Immutable V26 evidence chain.
PR_JSON=$EVENT/post_rebuild_20260725T133748Z/ENTRY_MODEL_NATIVE_SEQ513_POST_REBUILD_READINESS_20260725T134328547802Z.json
FF_JSON=$EVENT/foundation_feature_20260725T134328Z/ENTRY_FEATURE_FOUNDATION_AUDIT_20260725T134653Z.json
FT_JSON=$EVENT/foundation_target_20260725T134653Z/ENTRY_TARGET_FOUNDATION_AUDIT_20260725T134911Z.json
SF_JSON=$EVENT/specialist_feature_20260725T134911Z/ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_20260725T135141Z.json
LIVENESS=$EVENT/audit/ENTRY_FULL_INPUT_LIVENESS_CONTRACT_20260725T122910794355Z.json
PRETRAIN=$EVENT/audit/XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_20260725T131855017916Z.json
SM_JSON=$EVENT/smoke_manifest_20260725T140129Z/ENTRY_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST_20260725T140344858619Z.json
SR_JSON=$EVENT/smoke_readiness_20260725T140344Z/ENTRY_MODEL_NATIVE_SEQ513_SMOKE_READINESS_20260725T140600170904Z.json
TR_JSON=$EVENT/trainability_20260725T140600Z/ENTRY_MODEL_NATIVE_SEQ513_TRAINABILITY_READINESS_20260725T140602243971Z.json

TRAIN_M=$DS/${STEM}_train.manifest.json
TRAIN_P=$DS/${STEM}_train.parquet
VAL_M=$DS/${STEM}_val.manifest.json
VAL_P=$DS/${STEM}_val.parquet
TEST_M=$DS/${STEM}_test.manifest.json
TEST_P=$DS/${STEM}_test.parquet

# Per-timeframe lookback: each band owned by the coarsest timeframe covering it.
# M5 1.3h, M15 16h, H1 4d, H4 16d, D1 one trading year. 252 is the window
# D1_atr_percentile_252 already uses; 64 is the M15 length chosen to drop the M5
# overlap; 96 is the global default.
WINDOWS=(--multi-tf-seq-len 96
         --per-tf-seq-len-m5 16 --per-tf-seq-len-m15 64
         --per-tf-seq-len-h1 96 --per-tf-seq-len-h4 96 --per-tf-seq-len-d1 252)

TRAIN_ARGS=(--profile smoke
  --device cuda --seed 1337 --epochs "$EPOCHS" --batch-size 64
  --learning-rate 0.0003 --early-stop-patience 8 --early-stop-min-delta 0.0
  --grad-clip-norm 1.0 --weight-decay 1e-05 --multi-tf-scale 0.5
  --specialist-fusion-scale 0.25 --subsample-rows "$SUBSAMPLE_ROWS"
  "${WINDOWS[@]}"
  --memory-cap 30G --swap-cap 2G)

EVIDENCE=(--dataset-dir "$DS"
  --train-manifest-json "$TRAIN_M" --train-parquet "$TRAIN_P"
  --val-manifest-json "$VAL_M" --val-parquet "$VAL_P"
  --test-manifest-json "$TEST_M" --test-parquet "$TEST_P"
  --m5-prebuilt-path "$EVENT/FULL_PLUS_CTX_v3src.parquet"
  --multi-tf-cache-manifest-json "$EVENT/MULTI_TF_V2_CACHE/manifest.json"
  --post-rebuild-readiness-json "$PR_JSON"
  --pretrain-audit-json "$PRETRAIN"
  --feature-audit-json "$FF_JSON"
  --target-audit-json "$FT_JSON"
  --specialist-audit-json "$SF_JSON"
  --full-input-liveness-audit-json "$LIVENESS"
  --trainability-readiness-json "$TR_JSON"
  --smoke-manifest-json "$SM_JSON"
  --smoke-readiness-json "$SR_JSON")

require_decision() {
  local path="$1" want="$2"
  local got
  got=$("$REPO/.venv/bin/python" -c "import json,sys;print(json.load(open(sys.argv[1])).get('decision'))" "$path")
  [[ "$got" == "$want" ]] || { echo "FATAL: $path decision=$got expected=$want" >&2; exit 3; }
  echo "[wave] $(basename "$path") decision=$got"
}

cd "$REPO"
RC_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RC_OUT=$EVENT/train_recipe_${RC_STAMP}
BUNDLE_OUT=$EVENT/v12_entry_model_native_seq513_smoke_${RC_STAMP}

echo "[wave] ==== train-recipe-audit @ ${RC_STAMP}"
scripts/entry_next_edge_control.sh model-native-train-recipe-audit \
  --run-id "$RUN_TRAIN" "${EVIDENCE[@]}" \
  --out-bundle-dir "$BUNDLE_OUT" --out-dir "$RC_OUT" \
  --gx1-data-root /home/andre2/GX1_DATA --repo "$REPO" \
  --wrapper-path scripts/run_entry_model_native_seq513_smoke_train.sh \
  "${TRAIN_ARGS[@]}"

RC_JSON=$(ls "$RC_OUT"/ENTRY_MODEL_NATIVE_SEQ513_TRAIN_RECIPE_AUDIT_*.json)
require_decision "$RC_JSON" "PASS"

echo "[wave] ==== smoke-train --execute"
scripts/entry_next_edge_control.sh model-native-smoke-train \
  --run-id "$RUN_TRAIN" "${EVIDENCE[@]}" \
  --recipe-audit-json "$RC_JSON" \
  --out-bundle-dir "$BUNDLE_OUT" \
  --gx1-data-root /home/andre2/GX1_DATA \
  "${TRAIN_ARGS[@]}" --execute

echo "[wave] ==== done; bundle target: $BUNDLE_OUT"
