#!/usr/bin/env bash
set -euo pipefail

MODEL_NATIVE_CONTRACT_MODE=xau_seq513_model_native_direction_v1
MODEL_NATIVE_DIRECTION_LOGIT_MODE=model_native
MODEL_NATIVE_SIGNAL_DIM=513
PROFILE=candidate

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd -P)"
PY=/home/andre2/venvs/gx1/bin/python
CAPPED_RUNNER="$REPO/scripts/gx1_capped_run.sh"

usage() {
  cat <<'EOF'
Usage: run_entry_model_native_seq513_candidate_train.sh [exact arguments] (--dry-run|--execute)

Required identity and immutable evidence:
  --run-id ID
  --dataset-dir PATH
  --train-manifest-json PATH  --val-manifest-json PATH  --test-manifest-json PATH
  --train-parquet PATH        --val-parquet PATH        --test-parquet PATH
  --m5-prebuilt-path PATH
  --full-input-liveness-audit-json PATH
  --feature-audit-json PATH  --target-audit-json PATH  --specialist-audit-json PATH
  --pretrain-audit-json PATH --recipe-audit-json PATH
  --trainability-readiness-json PATH --candidate-readiness-json PATH
  --smoke-bundle-audit-json PATH
  --out-bundle-dir PATH --gx1-data-root PATH

Required audited execution values (there are no wrapper defaults):
  --device cpu|cuda --seed N --epochs N --batch-size N --learning-rate X
  --early-stop-patience N --early-stop-min-delta X --grad-clip-norm X
  --weight-decay X --multi-tf-scale X
  --specialist-fusion-scale X --subsample-rows N
  --memory-cap SIZE --swap-cap SIZE

--dry-run validates and prints the exact capped command without writing files.
--execute additionally requires a clean worktree and runs the capped trainer.
EOF
}

die() {
  printf 'FATAL: %s\n' "$*" >&2
  exit 2
}

declare -A SEEN=()
take_value() {
  local flag="$1" variable="$2" value="$3"
  [[ -z "${SEEN[$flag]+set}" ]] || die "duplicate argument: $flag"
  [[ -n "$value" ]] || die "$flag requires a non-empty value"
  printf -v "$variable" '%s' "$value"
  SEEN[$flag]=1
}

RUN_ID= DATASET_DIR= TRAIN_MANIFEST_JSON= VAL_MANIFEST_JSON= TEST_MANIFEST_JSON=
TRAIN_PARQUET= VAL_PARQUET= TEST_PARQUET= M5_PREBUILT_PATH=
FULL_INPUT_LIVENESS_AUDIT_JSON= FEATURE_AUDIT_JSON= TARGET_AUDIT_JSON=
SPECIALIST_AUDIT_JSON= PRETRAIN_AUDIT_JSON= RECIPE_AUDIT_JSON=
TRAINABILITY_READINESS_JSON= CANDIDATE_READINESS_JSON= SMOKE_BUNDLE_AUDIT_JSON=
OUT_BUNDLE_DIR= GX1_DATA_ROOT= DEVICE= SEED= EPOCHS= BATCH_SIZE= LEARNING_RATE=
EARLY_STOP_PATIENCE= EARLY_STOP_MIN_DELTA= GRAD_CLIP_NORM= WEIGHT_DECAY=
MULTI_TF_SCALE= SPECIALIST_FUSION_SCALE= SUBSAMPLE_ROWS=
MEMORY_CAP= SWAP_CAP= RUN_MODE=

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --dry-run|--execute)
      [[ -z "$RUN_MODE" ]] || die "choose exactly one of --dry-run or --execute"
      RUN_MODE="${1#--}"
      shift
      ;;
    --run-id|--dataset-dir|--train-manifest-json|--val-manifest-json|--test-manifest-json|\
    --train-parquet|--val-parquet|--test-parquet|--m5-prebuilt-path|\
    --full-input-liveness-audit-json|--feature-audit-json|--target-audit-json|\
    --specialist-audit-json|--pretrain-audit-json|--recipe-audit-json|\
    --trainability-readiness-json|--candidate-readiness-json|--smoke-bundle-audit-json|\
    --out-bundle-dir|--gx1-data-root|--device|--seed|--epochs|--batch-size|\
    --learning-rate|--early-stop-patience|--early-stop-min-delta|--grad-clip-norm|\
    --weight-decay|--multi-tf-scale|--specialist-fusion-scale|\
    --subsample-rows|--memory-cap|--swap-cap)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      case "$1" in
        --run-id) variable=RUN_ID ;;
        --dataset-dir) variable=DATASET_DIR ;;
        --train-manifest-json) variable=TRAIN_MANIFEST_JSON ;;
        --val-manifest-json) variable=VAL_MANIFEST_JSON ;;
        --test-manifest-json) variable=TEST_MANIFEST_JSON ;;
        --train-parquet) variable=TRAIN_PARQUET ;;
        --val-parquet) variable=VAL_PARQUET ;;
        --test-parquet) variable=TEST_PARQUET ;;
        --m5-prebuilt-path) variable=M5_PREBUILT_PATH ;;
        --full-input-liveness-audit-json) variable=FULL_INPUT_LIVENESS_AUDIT_JSON ;;
        --feature-audit-json) variable=FEATURE_AUDIT_JSON ;;
        --target-audit-json) variable=TARGET_AUDIT_JSON ;;
        --specialist-audit-json) variable=SPECIALIST_AUDIT_JSON ;;
        --pretrain-audit-json) variable=PRETRAIN_AUDIT_JSON ;;
        --recipe-audit-json) variable=RECIPE_AUDIT_JSON ;;
        --trainability-readiness-json) variable=TRAINABILITY_READINESS_JSON ;;
        --candidate-readiness-json) variable=CANDIDATE_READINESS_JSON ;;
        --smoke-bundle-audit-json) variable=SMOKE_BUNDLE_AUDIT_JSON ;;
        --out-bundle-dir) variable=OUT_BUNDLE_DIR ;;
        --gx1-data-root) variable=GX1_DATA_ROOT ;;
        --device) variable=DEVICE ;;
        --seed) variable=SEED ;;
        --epochs) variable=EPOCHS ;;
        --batch-size) variable=BATCH_SIZE ;;
        --learning-rate) variable=LEARNING_RATE ;;
        --early-stop-patience) variable=EARLY_STOP_PATIENCE ;;
        --early-stop-min-delta) variable=EARLY_STOP_MIN_DELTA ;;
        --grad-clip-norm) variable=GRAD_CLIP_NORM ;;
        --weight-decay) variable=WEIGHT_DECAY ;;
        --multi-tf-scale) variable=MULTI_TF_SCALE ;;
        --specialist-fusion-scale) variable=SPECIALIST_FUSION_SCALE ;;
        --subsample-rows) variable=SUBSAMPLE_ROWS ;;
        --memory-cap) variable=MEMORY_CAP ;;
        --swap-cap) variable=SWAP_CAP ;;
      esac
      take_value "$1" "$variable" "$2"
      shift 2
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -x "$PY" ]] || die "canonical Python is not executable: $PY"
[[ -x "$CAPPED_RUNNER" ]] || die "capped runner is not executable: $CAPPED_RUNNER"
[[ -n "$RUN_MODE" ]] || die "choose exactly one of --dry-run or --execute"
for variable in RUN_ID DATASET_DIR TRAIN_MANIFEST_JSON VAL_MANIFEST_JSON TEST_MANIFEST_JSON \
  TRAIN_PARQUET VAL_PARQUET TEST_PARQUET M5_PREBUILT_PATH FULL_INPUT_LIVENESS_AUDIT_JSON \
  FEATURE_AUDIT_JSON TARGET_AUDIT_JSON SPECIALIST_AUDIT_JSON PRETRAIN_AUDIT_JSON \
  RECIPE_AUDIT_JSON TRAINABILITY_READINESS_JSON CANDIDATE_READINESS_JSON \
  SMOKE_BUNDLE_AUDIT_JSON OUT_BUNDLE_DIR GX1_DATA_ROOT DEVICE SEED EPOCHS BATCH_SIZE \
  LEARNING_RATE EARLY_STOP_PATIENCE EARLY_STOP_MIN_DELTA GRAD_CLIP_NORM WEIGHT_DECAY \
  MULTI_TF_SCALE SPECIALIST_FUSION_SCALE SUBSAMPLE_ROWS MEMORY_CAP SWAP_CAP; do
  [[ -n "${!variable}" ]] || die "missing required argument for $variable"
done
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$ ]] || die "--run-id has invalid format"

VALIDATOR_ARGS=(
  --profile "$PROFILE" --repo "$REPO" --wrapper-path "$SCRIPT_PATH" --run-id "$RUN_ID"
  --dataset-dir "$DATASET_DIR" --out-bundle-dir "$OUT_BUNDLE_DIR"
  --train-manifest-json "$TRAIN_MANIFEST_JSON" --val-manifest-json "$VAL_MANIFEST_JSON"
  --test-manifest-json "$TEST_MANIFEST_JSON" --train-parquet "$TRAIN_PARQUET"
  --val-parquet "$VAL_PARQUET" --test-parquet "$TEST_PARQUET"
  --m5-prebuilt-path "$M5_PREBUILT_PATH"
  --full-input-liveness-audit-json "$FULL_INPUT_LIVENESS_AUDIT_JSON"
  --feature-audit-json "$FEATURE_AUDIT_JSON" --target-audit-json "$TARGET_AUDIT_JSON"
  --specialist-audit-json "$SPECIALIST_AUDIT_JSON" --pretrain-audit-json "$PRETRAIN_AUDIT_JSON"
  --recipe-audit-json "$RECIPE_AUDIT_JSON"
  --trainability-readiness-json "$TRAINABILITY_READINESS_JSON"
  --candidate-readiness-json "$CANDIDATE_READINESS_JSON"
  --smoke-bundle-audit-json "$SMOKE_BUNDLE_AUDIT_JSON"
  --device "$DEVICE" --seed "$SEED" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE" --early-stop-patience "$EARLY_STOP_PATIENCE"
  --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" --grad-clip-norm "$GRAD_CLIP_NORM"
  --weight-decay "$WEIGHT_DECAY" --multi-tf-scale "$MULTI_TF_SCALE"
  --specialist-fusion-scale "$SPECIALIST_FUSION_SCALE" --subsample-rows "$SUBSAMPLE_ROWS"
  --memory-cap "$MEMORY_CAP" --swap-cap "$SWAP_CAP" --gx1-data-root "$GX1_DATA_ROOT"
)
if ! RECIPE_ENV_TEXT=$(cd "$REPO" && "$PY" -m gx1.contracts.entry_model_native_train_launch_v1 "${VALIDATOR_ARGS[@]}"); then
  exit 2
fi
mapfile -t RECIPE_ENV <<<"$RECIPE_ENV_TEXT"
[[ ${#RECIPE_ENV[@]} -gt 0 ]] || die "recipe validator emitted no environment"

ENV_COMMAND=(env)
while IFS= read -r variable; do
  case "$variable" in
    ENTRY_*|GX1_*) ENV_COMMAND+=(-u "$variable") ;;
  esac
done < <(compgen -e)
ENV_COMMAND+=("${RECIPE_ENV[@]}")

TRAIN_CMD=(
  "${ENV_COMMAND[@]}"
  "$PY" -m gx1.models.entry_v10.entry_v10_ctx_train_v3
  --train --profile "$PROFILE" --run-id "$RUN_ID" --seed "$SEED" --device "$DEVICE"
  --train-manifest-json "$TRAIN_MANIFEST_JSON"
  --val-manifest-json "$VAL_MANIFEST_JSON"
  --test-manifest-json "$TEST_MANIFEST_JSON"
  --train-parquet "$TRAIN_PARQUET"
  --val-parquet "$VAL_PARQUET"
  --test-parquet "$TEST_PARQUET"
  --out_bundle_dir "$OUT_BUNDLE_DIR" --gx1-data "$GX1_DATA_ROOT"
  --m5-prebuilt-path "$M5_PREBUILT_PATH"
  --seq_len 96 --epochs "$EPOCHS" --lr "$LEARNING_RATE" --batch_size "$BATCH_SIZE"
  --early-stopping-patience "$EARLY_STOP_PATIENCE"
  --early-stopping-min-delta "$EARLY_STOP_MIN_DELTA"
  --num-workers 0 --grad-accum-steps 1 --subsample-rows "$SUBSAMPLE_ROWS"
  --grad-clip-norm "$GRAD_CLIP_NORM" --weight-decay "$WEIGHT_DECAY"
  --multi-tf-seq-len 96 --per-tf-seq-len-h4 48 --per-tf-seq-len-d1 30
  --multi-tf-scale "$MULTI_TF_SCALE"
  --tf-input-scale-init-m5 1.0 --tf-input-scale-init-m15 1.0
  --tf-input-scale-init-h1 0.7 --tf-input-scale-init-h4 0.5 --tf-input-scale-init-d1 0.3
  --specialist-audit-json "$SPECIALIST_AUDIT_JSON"
  --specialist-contract-mode "$MODEL_NATIVE_CONTRACT_MODE"
  --specialist-num-layers 1 --specialist-fusion-scale "$SPECIALIST_FUSION_SCALE"
)
RUN_CMD=("$CAPPED_RUNNER" --mem "$MEMORY_CAP" --swap "$SWAP_CAP" -- "${TRAIN_CMD[@]}")

if [[ "$RUN_MODE" == dry-run ]]; then
  printf 'Validated model-native seq513 candidate contract: mode=%s direction=%s width=%s\n' \
    "$MODEL_NATIVE_CONTRACT_MODE" "$MODEL_NATIVE_DIRECTION_LOGIT_MODE" "$MODEL_NATIVE_SIGNAL_DIM"
  printf 'Capped candidate train command:'
  printf ' %q' "${RUN_CMD[@]}"
  printf '\n'
  exit 0
fi

[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || \
  die "--execute requires a clean worktree matching the audited source bindings"
exec "${RUN_CMD[@]}"
