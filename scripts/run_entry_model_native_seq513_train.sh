#!/usr/bin/env bash
# Canonical bounded trainability run for the one-bundle Entry/Exit model. The
# explicit profile selects its immutable evidence set; execution remains one
# shared trainer path.
set -euo pipefail

MODEL_NATIVE_DIRECTION_LOGIT_MODE=model_native
PROFILE=

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd -P)"
PY="$REPO/.venv/bin/python"

# Rule 13/14: these two were hand-written literals here and went stale by
# construction -- on 2026-08-19 they read v18/279 while the owner evaluated to
# v20/238, so --specialist-contract-mode failed exact equality and the training
# chain could not start at all.  The width was worse than stale: it was printed
# by --dry-run as "Validated ... width=279", a diagnostic that reads as a
# validation result for a number nothing had checked (rule 2e).  Both are now
# read from the one contract owner at launch, so a contract bump cannot leave
# this wrapper behind again.
read -r MODEL_NATIVE_CONTRACT_MODE MODEL_NATIVE_SIGNAL_DIM < <(
  "$PY" -c 'import gx1.contracts.entry_model_native_signal_v1 as s; print(s.MODEL_NATIVE_CONTRACT_MODE, s.MODEL_NATIVE_SIGNAL_DIM)'
) || { echo "FATAL: cannot read model-native signal contract owner" >&2; exit 1; }
if [[ -z "$MODEL_NATIVE_CONTRACT_MODE" || -z "$MODEL_NATIVE_SIGNAL_DIM" ]]; then
  echo "FATAL: model-native signal contract owner returned no identity" >&2
  exit 1
fi
CAPPED_RUNNER="$REPO/scripts/gx1_capped_run.sh"
ENV_BIN=/usr/bin/env

usage() {
  cat <<'EOF'
Usage: run_entry_model_native_seq513_train.sh --profile smoke|candidate [exact arguments]
       [--attended-smoke|--attended-cpu-smoke] (--dry-run|--execute)

Required identity and immutable evidence:
  --profile smoke|candidate
  --run-id ID
  --dataset-dir PATH
  --train-manifest-json PATH  --val-manifest-json PATH
  --train-parquet PATH        --val-parquet PATH
  --unified-exit-lifecycle-manifest-json PATH
  --m5-prebuilt-path PATH
  --multi-tf-cache-manifest-json PATH
  --post-rebuild-readiness-json PATH
  --prefreeze-test-seal-json PATH --prefreeze-test-seal-sha256 SHA256
  --full-input-liveness-audit-json PATH
  --feature-audit-json PATH  --target-audit-json PATH  --specialist-audit-json PATH
  --pretrain-audit-json PATH --execution-causality-audit-json PATH
  --train-sequence-integrity-audit-json PATH --val-sequence-integrity-audit-json PATH
  --train-sequence-source-audit-json PATH --val-sequence-source-audit-json PATH
  --recipe-audit-json PATH
  --trainability-readiness-json PATH
  --out-bundle-dir PATH --gx1-data-root PATH

Required only for --profile smoke:
  --smoke-manifest-json PATH --smoke-readiness-json PATH

Required only for --profile candidate:
  --candidate-readiness-json PATH --smoke-bundle-audit-json PATH

Required audited execution values (there are no wrapper defaults):
  --device cpu|cuda --seed N --epochs N --batch-size N --learning-rate X
  --early-stop-patience N --early-stop-min-delta X --grad-clip-norm X
  --weight-decay X --dropout X --multi-tf-scale X
  --specialist-fusion-scale X --cross-family-fusion-scale X --subsample-rows N
  --multi-tf-num-layers N --specialist-num-layers N --grad-accum-steps N
  --per-tf-seq-len-m5 N --per-tf-seq-len-m15 N
  --per-tf-seq-len-h1 N --per-tf-seq-len-h4 N --per-tf-seq-len-d1 N
  --memory-cap SIZE --swap-cap SIZE

--dry-run validates and prints the exact capped command without writing files.
--execute additionally requires a clean worktree and runs the capped trainer.
The profile has no default. Evidence from one profile is rejected in the other.
--attended-smoke is accepted only for CUDA smoke runs. --attended-cpu-smoke
is its CPU-only recovery counterpart after a GPU safety stop. Both have a
fixed 10-minute data preflight followed by a separate five-minute model phase,
and are marked in the produced bundle;
that bundle is rejected by smoke-bundle audit and cannot reach candidate, TEST,
promotion, paper or live stages.
Its two source-reconstruction proofs authorize only a memory representation
for the exact TRAIN/VAL inputs; the trainer re-hashes and validates them before
use against the immutable M5 feature surface.
--research-smoke is intentionally disabled after a WSL/GPU reset during the
former long-running route. Historical CUDA training requires a separately
designed low-VRAM, resumable architecture before it may be re-enabled.
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

RUN_ID= DATASET_DIR= TRAIN_MANIFEST_JSON= VAL_MANIFEST_JSON=
TRAIN_PARQUET= VAL_PARQUET= M5_PREBUILT_PATH=
UNIFIED_EXIT_LIFECYCLE_MANIFEST_JSON=
POST_REBUILD_READINESS_JSON=
PREFREEZE_TEST_SEAL_JSON= PREFREEZE_TEST_SEAL_SHA256=
FULL_INPUT_LIVENESS_AUDIT_JSON= FEATURE_AUDIT_JSON= TARGET_AUDIT_JSON=
SPECIALIST_AUDIT_JSON= PRETRAIN_AUDIT_JSON= EXECUTION_CAUSALITY_AUDIT_JSON=
TRAIN_SEQUENCE_INTEGRITY_AUDIT_JSON= VAL_SEQUENCE_INTEGRITY_AUDIT_JSON= RECIPE_AUDIT_JSON=
SMOKE_MANIFEST_JSON= SMOKE_READINESS_JSON= TRAINABILITY_READINESS_JSON=
CANDIDATE_READINESS_JSON= SMOKE_BUNDLE_AUDIT_JSON=
TRAIN_SEQUENCE_SOURCE_AUDIT_JSON= VAL_SEQUENCE_SOURCE_AUDIT_JSON=
OUT_BUNDLE_DIR= GX1_DATA_ROOT= DEVICE= SEED= EPOCHS= BATCH_SIZE= LEARNING_RATE=
EARLY_STOP_PATIENCE= EARLY_STOP_MIN_DELTA= GRAD_CLIP_NORM= WEIGHT_DECAY=
DROPOUT= MULTI_TF_SCALE= SPECIALIST_FUSION_SCALE= CROSS_FAMILY_FUSION_SCALE= SUBSAMPLE_ROWS=
NUM_WORKERS= MULTI_TF_NUM_LAYERS= SPECIALIST_NUM_LAYERS= GRAD_ACCUM_STEPS=
PER_TF_SEQ_LEN_M5= PER_TF_SEQ_LEN_M15=
PER_TF_SEQ_LEN_H1= PER_TF_SEQ_LEN_H4= PER_TF_SEQ_LEN_D1=
MEMORY_CAP= SWAP_CAP= RUN_MODE=
ATTENDED_SMOKE=false
ATTENDED_CPU_SMOKE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --dry-run|--execute)
      [[ -z "$RUN_MODE" ]] || die "choose exactly one of --dry-run or --execute"
      RUN_MODE="${1#--}"
      shift
      ;;
    --attended-smoke)
      [[ "$ATTENDED_SMOKE" == false ]] || die "duplicate argument: --attended-smoke"
      ATTENDED_SMOKE=true
      shift
      ;;
    --attended-cpu-smoke)
      [[ "$ATTENDED_CPU_SMOKE" == false ]] || die "duplicate argument: --attended-cpu-smoke"
      ATTENDED_CPU_SMOKE=true
      shift
      ;;
    --research-smoke)
      die "--research-smoke is disabled after the WSL/GPU reset"
      ;;
    --profile|--run-id|--dataset-dir|--train-manifest-json|--val-manifest-json|\
    --train-parquet|--val-parquet|--unified-exit-lifecycle-manifest-json|\
    --m5-prebuilt-path|--multi-tf-cache-manifest-json|\
    --post-rebuild-readiness-json|--prefreeze-test-seal-json|--prefreeze-test-seal-sha256|\
    --full-input-liveness-audit-json|--feature-audit-json|--target-audit-json|\
    --specialist-audit-json|--pretrain-audit-json|--execution-causality-audit-json|\
    --train-sequence-integrity-audit-json|--val-sequence-integrity-audit-json|--recipe-audit-json|\
    --smoke-manifest-json|--smoke-readiness-json|--trainability-readiness-json|\
    --candidate-readiness-json|--smoke-bundle-audit-json|\
    --train-sequence-source-audit-json|--val-sequence-source-audit-json|\
    --out-bundle-dir|--gx1-data-root|--device|--seed|--epochs|--batch-size|\
    --learning-rate|--early-stop-patience|--early-stop-min-delta|--grad-clip-norm|\
    --weight-decay|--dropout|--multi-tf-scale|--specialist-fusion-scale|--cross-family-fusion-scale|\
    --subsample-rows|--memory-cap|--swap-cap|\
    --num-workers|\
    --multi-tf-num-layers|--specialist-num-layers|--grad-accum-steps|\
    --per-tf-seq-len-m5|--per-tf-seq-len-m15|\
    --per-tf-seq-len-h1|--per-tf-seq-len-h4|--per-tf-seq-len-d1)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      case "$1" in
        --profile) variable=PROFILE ;;
        --run-id) variable=RUN_ID ;;
        --dataset-dir) variable=DATASET_DIR ;;
        --train-manifest-json) variable=TRAIN_MANIFEST_JSON ;;
        --val-manifest-json) variable=VAL_MANIFEST_JSON ;;
        --train-parquet) variable=TRAIN_PARQUET ;;
        --val-parquet) variable=VAL_PARQUET ;;
        --unified-exit-lifecycle-manifest-json) variable=UNIFIED_EXIT_LIFECYCLE_MANIFEST_JSON ;;
        --m5-prebuilt-path) variable=M5_PREBUILT_PATH ;;
        --multi-tf-cache-manifest-json) variable=MULTI_TF_CACHE_MANIFEST_JSON ;;
        --post-rebuild-readiness-json) variable=POST_REBUILD_READINESS_JSON ;;
        --prefreeze-test-seal-json) variable=PREFREEZE_TEST_SEAL_JSON ;;
        --prefreeze-test-seal-sha256) variable=PREFREEZE_TEST_SEAL_SHA256 ;;
        --full-input-liveness-audit-json) variable=FULL_INPUT_LIVENESS_AUDIT_JSON ;;
        --feature-audit-json) variable=FEATURE_AUDIT_JSON ;;
        --target-audit-json) variable=TARGET_AUDIT_JSON ;;
        --specialist-audit-json) variable=SPECIALIST_AUDIT_JSON ;;
        --pretrain-audit-json) variable=PRETRAIN_AUDIT_JSON ;;
        --execution-causality-audit-json) variable=EXECUTION_CAUSALITY_AUDIT_JSON ;;
        --train-sequence-integrity-audit-json) variable=TRAIN_SEQUENCE_INTEGRITY_AUDIT_JSON ;;
        --val-sequence-integrity-audit-json) variable=VAL_SEQUENCE_INTEGRITY_AUDIT_JSON ;;
        --recipe-audit-json) variable=RECIPE_AUDIT_JSON ;;
        --smoke-manifest-json) variable=SMOKE_MANIFEST_JSON ;;
        --smoke-readiness-json) variable=SMOKE_READINESS_JSON ;;
        --trainability-readiness-json) variable=TRAINABILITY_READINESS_JSON ;;
        --candidate-readiness-json) variable=CANDIDATE_READINESS_JSON ;;
        --smoke-bundle-audit-json) variable=SMOKE_BUNDLE_AUDIT_JSON ;;
        --train-sequence-source-audit-json) variable=TRAIN_SEQUENCE_SOURCE_AUDIT_JSON ;;
        --val-sequence-source-audit-json) variable=VAL_SEQUENCE_SOURCE_AUDIT_JSON ;;
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
        --dropout) variable=DROPOUT ;;
        --multi-tf-scale) variable=MULTI_TF_SCALE ;;
        --num-workers) variable=NUM_WORKERS ;;
        --multi-tf-num-layers) variable=MULTI_TF_NUM_LAYERS ;;
        --specialist-num-layers) variable=SPECIALIST_NUM_LAYERS ;;
        --grad-accum-steps) variable=GRAD_ACCUM_STEPS ;;
        --per-tf-seq-len-m5) variable=PER_TF_SEQ_LEN_M5 ;;
        --per-tf-seq-len-m15) variable=PER_TF_SEQ_LEN_M15 ;;
        --per-tf-seq-len-h1) variable=PER_TF_SEQ_LEN_H1 ;;
        --per-tf-seq-len-h4) variable=PER_TF_SEQ_LEN_H4 ;;
        --per-tf-seq-len-d1) variable=PER_TF_SEQ_LEN_D1 ;;
        --specialist-fusion-scale) variable=SPECIALIST_FUSION_SCALE ;;
        --cross-family-fusion-scale) variable=CROSS_FAMILY_FUSION_SCALE ;;
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
[[ -x "$ENV_BIN" ]] || die "environment scrubber is not executable: $ENV_BIN"
[[ -n "$PROFILE" ]] || die "missing required argument: --profile"
case "$PROFILE" in
  smoke|candidate) ;;
  *) die "--profile must be exactly smoke or candidate" ;;
esac
if [[ "$ATTENDED_SMOKE" == true && "$ATTENDED_CPU_SMOKE" == true ]]; then
  die "choose at most one of --attended-smoke or --attended-cpu-smoke"
fi
if [[ "$ATTENDED_SMOKE" == true || "$ATTENDED_CPU_SMOKE" == true ]]; then
  [[ "$PROFILE" == smoke ]] || die "attended smoke is valid only for --profile smoke"
fi
EXECUTION_TIER=canonical
if [[ "$ATTENDED_SMOKE" == true ]]; then
  EXECUTION_TIER=attended_only
elif [[ "$ATTENDED_CPU_SMOKE" == true ]]; then
  EXECUTION_TIER=attended_cpu_only
fi
[[ -n "$RUN_MODE" ]] || die "choose exactly one of --dry-run or --execute"
for variable in RUN_ID DATASET_DIR TRAIN_MANIFEST_JSON VAL_MANIFEST_JSON \
  TRAIN_PARQUET VAL_PARQUET UNIFIED_EXIT_LIFECYCLE_MANIFEST_JSON \
  M5_PREBUILT_PATH MULTI_TF_CACHE_MANIFEST_JSON \
  POST_REBUILD_READINESS_JSON PREFREEZE_TEST_SEAL_JSON PREFREEZE_TEST_SEAL_SHA256 \
  FULL_INPUT_LIVENESS_AUDIT_JSON \
  FEATURE_AUDIT_JSON TARGET_AUDIT_JSON SPECIALIST_AUDIT_JSON PRETRAIN_AUDIT_JSON EXECUTION_CAUSALITY_AUDIT_JSON \
  TRAIN_SEQUENCE_INTEGRITY_AUDIT_JSON VAL_SEQUENCE_INTEGRITY_AUDIT_JSON \
  TRAIN_SEQUENCE_SOURCE_AUDIT_JSON VAL_SEQUENCE_SOURCE_AUDIT_JSON \
  RECIPE_AUDIT_JSON TRAINABILITY_READINESS_JSON \
  OUT_BUNDLE_DIR GX1_DATA_ROOT DEVICE SEED EPOCHS BATCH_SIZE LEARNING_RATE \
  EARLY_STOP_PATIENCE EARLY_STOP_MIN_DELTA GRAD_CLIP_NORM WEIGHT_DECAY MULTI_TF_SCALE \
  DROPOUT \
  SPECIALIST_FUSION_SCALE CROSS_FAMILY_FUSION_SCALE SUBSAMPLE_ROWS MEMORY_CAP SWAP_CAP \
  NUM_WORKERS MULTI_TF_NUM_LAYERS SPECIALIST_NUM_LAYERS GRAD_ACCUM_STEPS \
  PER_TF_SEQ_LEN_M5 PER_TF_SEQ_LEN_M15 PER_TF_SEQ_LEN_H1 \
  PER_TF_SEQ_LEN_H4 PER_TF_SEQ_LEN_D1; do
  [[ -n "${!variable}" ]] || die "missing required argument for $variable"
done
for variable in TRAIN_SEQUENCE_SOURCE_AUDIT_JSON VAL_SEQUENCE_SOURCE_AUDIT_JSON; do
  proof_path="${!variable}"
  [[ "$proof_path" = /* && -f "$proof_path" && ! -L "$proof_path" ]] \
    || die "requires an absolute regular non-symlink ${variable,,}"
done
if [[ "$ATTENDED_SMOKE" == true && "$DEVICE" != cuda ]]; then
  die "--attended-smoke requires --device cuda"
fi
if [[ "$ATTENDED_CPU_SMOKE" == true && "$DEVICE" != cpu ]]; then
  die "--attended-cpu-smoke requires --device cpu"
fi
if [[ "$ATTENDED_SMOKE" == true || "$ATTENDED_CPU_SMOKE" == true ]]; then
  # This lane is a one-off historical CUDA measurement, not a faster smoke
  # profile.  Its exact micro-batch geometry is part of the WSL residency
  # boundary and must agree with the trainer's independent assertion.
  [[ "$BATCH_SIZE" == 8 ]] \
    || die "attended smoke requires the fixed --batch-size 8"
  [[ "$EPOCHS" == 1 && "$GRAD_ACCUM_STEPS" == 1 ]] \
    || die "attended smoke requires exactly --epochs 1 and --grad-accum-steps 1"
fi
PROFILE_VALIDATOR_ARGS=()
case "$PROFILE" in
  smoke)
    for variable in SMOKE_MANIFEST_JSON SMOKE_READINESS_JSON; do
      [[ -n "${!variable}" ]] || die "missing required argument for $variable"
    done
    [[ -z "$CANDIDATE_READINESS_JSON" ]] \
      || die "--candidate-readiness-json is invalid for --profile smoke"
    [[ -z "$SMOKE_BUNDLE_AUDIT_JSON" ]] \
      || die "--smoke-bundle-audit-json is invalid for --profile smoke"
    PROFILE_VALIDATOR_ARGS=(
      --smoke-manifest-json "$SMOKE_MANIFEST_JSON"
      --smoke-readiness-json "$SMOKE_READINESS_JSON"
    )
    ;;
  candidate)
    for variable in CANDIDATE_READINESS_JSON SMOKE_BUNDLE_AUDIT_JSON; do
      [[ -n "${!variable}" ]] || die "missing required argument for $variable"
    done
    [[ -z "$SMOKE_MANIFEST_JSON" ]] \
      || die "--smoke-manifest-json is invalid for --profile candidate"
    [[ -z "$SMOKE_READINESS_JSON" ]] \
      || die "--smoke-readiness-json is invalid for --profile candidate"
    PROFILE_VALIDATOR_ARGS=(
      --candidate-readiness-json "$CANDIDATE_READINESS_JSON"
      --smoke-bundle-audit-json "$SMOKE_BUNDLE_AUDIT_JSON"
    )
    ;;
esac
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$ ]] || die "--run-id has invalid format"

VALIDATOR_ARGS=(
  --profile "$PROFILE" --repo "$REPO" --wrapper-path "$SCRIPT_PATH" --run-id "$RUN_ID"
  --dataset-dir "$DATASET_DIR" --out-bundle-dir "$OUT_BUNDLE_DIR"
  --train-manifest-json "$TRAIN_MANIFEST_JSON" --val-manifest-json "$VAL_MANIFEST_JSON"
  --train-parquet "$TRAIN_PARQUET" --val-parquet "$VAL_PARQUET"
  --unified-exit-lifecycle-manifest-json "$UNIFIED_EXIT_LIFECYCLE_MANIFEST_JSON"
  --m5-prebuilt-path "$M5_PREBUILT_PATH"
  --multi-tf-cache-manifest-json "$MULTI_TF_CACHE_MANIFEST_JSON"
  --post-rebuild-readiness-json "$POST_REBUILD_READINESS_JSON"
  --prefreeze-test-seal-json "$PREFREEZE_TEST_SEAL_JSON"
  --prefreeze-test-seal-sha256 "$PREFREEZE_TEST_SEAL_SHA256"
  --full-input-liveness-audit-json "$FULL_INPUT_LIVENESS_AUDIT_JSON"
  --feature-audit-json "$FEATURE_AUDIT_JSON" --target-audit-json "$TARGET_AUDIT_JSON"
  --specialist-audit-json "$SPECIALIST_AUDIT_JSON" --pretrain-audit-json "$PRETRAIN_AUDIT_JSON"
  --execution-causality-audit-json "$EXECUTION_CAUSALITY_AUDIT_JSON"
  --train-sequence-integrity-audit-json "$TRAIN_SEQUENCE_INTEGRITY_AUDIT_JSON"
  --val-sequence-integrity-audit-json "$VAL_SEQUENCE_INTEGRITY_AUDIT_JSON"
  --train-sequence-source-audit-json "$TRAIN_SEQUENCE_SOURCE_AUDIT_JSON"
  --val-sequence-source-audit-json "$VAL_SEQUENCE_SOURCE_AUDIT_JSON"
  --recipe-audit-json "$RECIPE_AUDIT_JSON"
  --trainability-readiness-json "$TRAINABILITY_READINESS_JSON"
  --device "$DEVICE" --seed "$SEED" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE" --early-stop-patience "$EARLY_STOP_PATIENCE"
  --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" --grad-clip-norm "$GRAD_CLIP_NORM"
  --weight-decay "$WEIGHT_DECAY" --multi-tf-scale "$MULTI_TF_SCALE"
  --dropout "$DROPOUT"
  --specialist-fusion-scale "$SPECIALIST_FUSION_SCALE"
  --cross-family-fusion-scale "$CROSS_FAMILY_FUSION_SCALE" --subsample-rows "$SUBSAMPLE_ROWS"
  --num-workers "$NUM_WORKERS"
  --multi-tf-num-layers "$MULTI_TF_NUM_LAYERS"
  --specialist-num-layers "$SPECIALIST_NUM_LAYERS"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --per-tf-seq-len-m5 "$PER_TF_SEQ_LEN_M5"
  --per-tf-seq-len-m15 "$PER_TF_SEQ_LEN_M15"
  --per-tf-seq-len-h1 "$PER_TF_SEQ_LEN_H1"
  --per-tf-seq-len-h4 "$PER_TF_SEQ_LEN_H4"
  --per-tf-seq-len-d1 "$PER_TF_SEQ_LEN_D1"
  --memory-cap "$MEMORY_CAP" --swap-cap "$SWAP_CAP" --gx1-data-root "$GX1_DATA_ROOT"
)
VALIDATOR_ARGS+=("${PROFILE_VALIDATOR_ARGS[@]}")
if ! RECIPE_ENV_TEXT=$(cd "$REPO" && "$PY" -m gx1.contracts.entry_model_native_train_launch_v1 "${VALIDATOR_ARGS[@]}"); then
  exit 2
fi
mapfile -t RECIPE_ENV <<<"$RECIPE_ENV_TEXT"
[[ ${#RECIPE_ENV[@]} -gt 0 ]] || die "recipe validator emitted no environment"
DATASET_RUN_ID=
declare -A RECIPE_ENV_KEYS=()
for row in "${RECIPE_ENV[@]}"; do
  [[ "$row" =~ ^([A-Z][A-Z0-9_]*)=(.*)$ ]] \
    || die "recipe validator emitted malformed environment assignment"
  recipe_key=${BASH_REMATCH[1]}
  case "$recipe_key" in
    GX1_CAPPED_*|GX1_EXPECTED_*|GX1_CPU_AFFINITY)
      die "recipe validator attempted to override capped-scope control: $recipe_key"
      ;;
    ENTRY_*|GX1_*) ;;
    *) die "recipe validator emitted forbidden environment key: $recipe_key" ;;
  esac
  [[ -z "${RECIPE_ENV_KEYS[$recipe_key]+set}" ]] \
    || die "recipe validator emitted duplicate environment key: $recipe_key"
  RECIPE_ENV_KEYS[$recipe_key]=1
  case "$row" in
    GX1_ENTRY_DATASET_RUN_ID=*)
      [[ -z "$DATASET_RUN_ID" ]] || die "recipe validator emitted duplicate dataset run ID"
      DATASET_RUN_ID=${row#GX1_ENTRY_DATASET_RUN_ID=}
      ;;
  esac
done
[[ "$DATASET_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$ ]] \
  || die "recipe validator emitted invalid dataset run ID"

ENV_COMMAND=("$ENV_BIN")
while IFS= read -r variable; do
  case "$variable" in
    ENTRY_*|GX1_*) ENV_COMMAND+=(-u "$variable") ;;
  esac
done < <(compgen -e)
ENV_COMMAND+=("${RECIPE_ENV[@]}")

TRAIN_CMD=(
  "$PY" -m gx1.models.entry_v10.entry_v10_ctx_train_v3
  --train --profile "$PROFILE" --run-id "$RUN_ID" --dataset-run-id "$DATASET_RUN_ID"
  --seed "$SEED" --device "$DEVICE"
  --execution-tier "$EXECUTION_TIER"
  --train-manifest-json "$TRAIN_MANIFEST_JSON"
  --val-manifest-json "$VAL_MANIFEST_JSON"
  --train-parquet "$TRAIN_PARQUET"
  --val-parquet "$VAL_PARQUET"
  --prefreeze-test-seal-json "$PREFREEZE_TEST_SEAL_JSON"
  --prefreeze-test-seal-sha256 "$PREFREEZE_TEST_SEAL_SHA256"
  --unified-exit-lifecycle-manifest-json "$UNIFIED_EXIT_LIFECYCLE_MANIFEST_JSON"
  --out_bundle_dir "$OUT_BUNDLE_DIR" --gx1-data "$GX1_DATA_ROOT"
  --m5-prebuilt-path "$M5_PREBUILT_PATH"
  --seq_len 96 --epochs "$EPOCHS" --lr "$LEARNING_RATE" --batch_size "$BATCH_SIZE"
  --early-stopping-patience "$EARLY_STOP_PATIENCE"
  --early-stopping-min-delta "$EARLY_STOP_MIN_DELTA"
  --num-workers "$NUM_WORKERS" --grad-accum-steps "$GRAD_ACCUM_STEPS" --subsample-rows "$SUBSAMPLE_ROWS"
  --grad-clip-norm "$GRAD_CLIP_NORM" --weight-decay "$WEIGHT_DECAY"
  --dropout "$DROPOUT"
  --multi-tf-num-layers "$MULTI_TF_NUM_LAYERS"
  --per-tf-seq-len-m5 "$PER_TF_SEQ_LEN_M5"
  --per-tf-seq-len-m15 "$PER_TF_SEQ_LEN_M15"
  --per-tf-seq-len-h1 "$PER_TF_SEQ_LEN_H1"
  --per-tf-seq-len-h4 "$PER_TF_SEQ_LEN_H4"
  --per-tf-seq-len-d1 "$PER_TF_SEQ_LEN_D1"
  --multi-tf-scale "$MULTI_TF_SCALE"
  --specialist-audit-json "$SPECIALIST_AUDIT_JSON"
  --specialist-contract-mode "$MODEL_NATIVE_CONTRACT_MODE"
  --specialist-num-layers "$SPECIALIST_NUM_LAYERS" --specialist-fusion-scale "$SPECIALIST_FUSION_SCALE"
  --cross-family-fusion-scale "$CROSS_FAMILY_FUSION_SCALE"
)
TRAIN_CMD+=(
  --train-sequence-source-audit-json "$TRAIN_SEQUENCE_SOURCE_AUDIT_JSON"
  --val-sequence-source-audit-json "$VAL_SEQUENCE_SOURCE_AUDIT_JSON"
)
CAPPED_RUN_ARGS=(--class trainer --mem "$MEMORY_CAP" --swap "$SWAP_CAP")
if [[ "$ATTENDED_SMOKE" == true || "$ATTENDED_CPU_SMOKE" == true ]]; then
  CAPPED_RUN_ARGS+=(--attended-smoke)
fi
RUN_CMD=(
  "${ENV_COMMAND[@]}"
  "$CAPPED_RUNNER" "${CAPPED_RUN_ARGS[@]}"
  -- "${TRAIN_CMD[@]}"
)

if [[ "$RUN_MODE" == dry-run ]]; then
  printf 'Validated model-native seq513 %s contract: mode=%s direction=%s width=%s execution_tier=%s\n' \
    "$PROFILE" "$MODEL_NATIVE_CONTRACT_MODE" "$MODEL_NATIVE_DIRECTION_LOGIT_MODE" "$MODEL_NATIVE_SIGNAL_DIM" \
    "$EXECUTION_TIER"
  printf 'Capped %s train command:' "$PROFILE"
  printf ' %q' "${RUN_CMD[@]}"
  printf '\n'
  exit 0
fi

[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || \
  die "--execute requires a clean worktree matching the audited source bindings"
exec "${RUN_CMD[@]}"
