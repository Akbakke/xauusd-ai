#!/usr/bin/env bash
# Build one fresh XAU model-native seq513 Entry dataset plus its source-bound
# unified Exit lifecycle surface. This script never trains, promotes, replays,
# or touches live state. Every artifact identity is explicit.
set -euo pipefail

ENG=/home/andre2/src/GX1_ENGINE
PY=$ENG/.venv/bin/python
CAP=("$ENG/scripts/gx1_capped_run.sh" --mem 10G --swap 512M --)

RUN_ID=
SOURCE_PARQUET=
CANONICAL_V2_PARQUET=
SIGNAL_MANIFEST=
FEATURE_RANKING_JSON=
RANK_REFERENCE_NPZ=
MTF_CACHE_DIR=
TAPE_ROOT=
M1_LIFECYCLE_PAIR_MANIFEST_JSON=
M1_LIFECYCLE_PAIR_GENERATION_ROOT=
M1_FEATURE_BASE_PARQUET=
M5_FEATURE_BASE_PARQUET=
EXIT_LIFECYCLE_DIR=
EXIT_TARGET_LOOKAHEAD_M1_STEPS=
EARLY_MOVE_THRESHOLD_BPS=
OUTPUT=
AUDIT_OUT_DIR=
TRAIN_START=
TRAIN_END=
VAL_START=
VAL_END=
TEST_START=
TEST_END=
HISTORY_START=
RESUME_EXACT_CHECKPOINTS=0
EXISTING_RANK_REFERENCE=0

usage() {
  printf '%s\n' \
    "Usage: $0 --run-id ID --source-parquet PATH --canonical-v2-parquet PATH" \
    "  --signal-manifest PATH --feature-ranking-json PATH --rank-reference-npz PATH" \
    "  --mtf-cache-dir PATH --tape-root PATH" \
    "  --m1-lifecycle-pair-manifest-json /immutable/generation/PAIR_MANIFEST.json" \
    "  --m1-lifecycle-pair-generation-root /immutable/generations" \
    "  --m1-feature-base-parquet /immutable/M1_FEATURE_BASE.parquet" \
    "  --m5-feature-base-parquet /immutable/M5_FEATURE_BASE.parquet" \
    "  --exit-lifecycle-dir /new/dir --exit-target-lookahead-m1-steps N" \
    "  --early-move-threshold-bps BPS" \
    "  --output /new/dir/STEM__DIR_H24B.parquet --audit-out-dir /new/report/dir" \
    "  --history-start UTC --train-start UTC --train-end UTC --val-start UTC --val-end UTC" \
    "  --test-start UTC --test-end UTC --existing-rank-reference [--resume-exact-checkpoints]"
}

while (($#)); do
  case "$1" in
    --run-id) RUN_ID=${2:-}; shift 2 ;;
    --source-parquet) SOURCE_PARQUET=${2:-}; shift 2 ;;
    --canonical-v2-parquet) CANONICAL_V2_PARQUET=${2:-}; shift 2 ;;
    --signal-manifest) SIGNAL_MANIFEST=${2:-}; shift 2 ;;
    --feature-ranking-json) FEATURE_RANKING_JSON=${2:-}; shift 2 ;;
    --rank-reference-npz) RANK_REFERENCE_NPZ=${2:-}; shift 2 ;;
    --mtf-cache-dir) MTF_CACHE_DIR=${2:-}; shift 2 ;;
    --tape-root) TAPE_ROOT=${2:-}; shift 2 ;;
    --m1-lifecycle-pair-manifest-json) M1_LIFECYCLE_PAIR_MANIFEST_JSON=${2:-}; shift 2 ;;
    --m1-lifecycle-pair-generation-root) M1_LIFECYCLE_PAIR_GENERATION_ROOT=${2:-}; shift 2 ;;
    --m1-feature-base-parquet) M1_FEATURE_BASE_PARQUET=${2:-}; shift 2 ;;
    --m5-feature-base-parquet) M5_FEATURE_BASE_PARQUET=${2:-}; shift 2 ;;
    --exit-lifecycle-dir) EXIT_LIFECYCLE_DIR=${2:-}; shift 2 ;;
    --exit-target-lookahead-m1-steps) EXIT_TARGET_LOOKAHEAD_M1_STEPS=${2:-}; shift 2 ;;
    --early-move-threshold-bps) EARLY_MOVE_THRESHOLD_BPS=${2:-}; shift 2 ;;
    --output) OUTPUT=${2:-}; shift 2 ;;
    --audit-out-dir) AUDIT_OUT_DIR=${2:-}; shift 2 ;;
    --history-start) HISTORY_START=${2:-}; shift 2 ;;
    --train-start) TRAIN_START=${2:-}; shift 2 ;;
    --train-end) TRAIN_END=${2:-}; shift 2 ;;
    --val-start) VAL_START=${2:-}; shift 2 ;;
    --val-end) VAL_END=${2:-}; shift 2 ;;
    --test-start) TEST_START=${2:-}; shift 2 ;;
    --test-end) TEST_END=${2:-}; shift 2 ;;
    --resume-exact-checkpoints) RESUME_EXACT_CHECKPOINTS=1; shift ;;
    --existing-rank-reference) EXISTING_RANK_REFERENCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf '[ABORT] unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ $EXISTING_RANK_REFERENCE -ne 1 ]]; then
  printf '[ABORT] --existing-rank-reference is required; ranking must bind these upstream bytes\n' >&2
  exit 2
fi

required_values=(
  RUN_ID SOURCE_PARQUET CANONICAL_V2_PARQUET SIGNAL_MANIFEST FEATURE_RANKING_JSON
  RANK_REFERENCE_NPZ MTF_CACHE_DIR TAPE_ROOT M1_LIFECYCLE_PAIR_MANIFEST_JSON
  M1_LIFECYCLE_PAIR_GENERATION_ROOT M1_FEATURE_BASE_PARQUET M5_FEATURE_BASE_PARQUET
  EXIT_LIFECYCLE_DIR EXIT_TARGET_LOOKAHEAD_M1_STEPS EARLY_MOVE_THRESHOLD_BPS
  OUTPUT AUDIT_OUT_DIR
  HISTORY_START TRAIN_START TRAIN_END VAL_START VAL_END TEST_START TEST_END
)
for name in "${required_values[@]}"; do
  if [[ -z ${!name} ]]; then
    printf '[ABORT] required argument missing: %s\n' "$name" >&2
    usage >&2
    exit 2
  fi
done

if [[ $OUTPUT != *.parquet || $OUTPUT != *"__DIR_H24B.parquet" ]]; then
  printf '[ABORT] --output must end in __DIR_H24B.parquet: %s\n' "$OUTPUT" >&2
  exit 2
fi
if [[ ! $RUN_ID =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$ ]]; then
  printf '[ABORT] --run-id has invalid format\n' >&2
  exit 2
fi
if [[ ! $EXIT_TARGET_LOOKAHEAD_M1_STEPS =~ ^[1-9][0-9]*$ ]]; then
  printf '[ABORT] --exit-target-lookahead-m1-steps must be a positive integer\n' >&2
  exit 2
fi
if [[ ! $EARLY_MOVE_THRESHOLD_BPS =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  printf '[ABORT] --early-move-threshold-bps must be a finite non-negative decimal\n' >&2
  exit 2
fi

for name in SOURCE_PARQUET CANONICAL_V2_PARQUET SIGNAL_MANIFEST FEATURE_RANKING_JSON M1_LIFECYCLE_PAIR_MANIFEST_JSON M1_FEATURE_BASE_PARQUET M5_FEATURE_BASE_PARQUET; do
  if [[ ! -f ${!name} ]]; then
    printf '[ABORT] required file missing (%s): %s\n' "$name" "${!name}" >&2
    exit 2
  fi
done
if [[ ! -d $MTF_CACHE_DIR || ! -d $TAPE_ROOT || ! -d $M1_LIFECYCLE_PAIR_GENERATION_ROOT ]]; then
  printf '[ABORT] MTF cache, XAU tape, or pair generation root missing\n' >&2
  exit 2
fi
if [[ -e $EXIT_LIFECYCLE_DIR || -L $EXIT_LIFECYCLE_DIR ]]; then
  printf '[ABORT] Exit lifecycle directory already exists; choose a fresh immutable path: %s\n' "$EXIT_LIFECYCLE_DIR" >&2
  exit 2
fi
if [[ ! -d $(dirname "$EXIT_LIFECYCLE_DIR") ]]; then
  printf '[ABORT] Exit lifecycle parent directory is missing: %s\n' "$(dirname "$EXIT_LIFECYCLE_DIR")" >&2
  exit 2
fi

retired_env=(
  GX1_MTF_CACHE_ALLOW_STALE GX1_ENTRY_ALLOW_TRAIN_ENV_OVERRIDES
  GX1_PERTF_CLOSED_BAR
  GX1_REGIME_V4 GX1_TREND_REGIME_FROM_D1
)
for name in "${retired_env[@]}"; do
  if [[ -v $name ]]; then
    printf '[ABORT] retired environment variable is present: %s\n' "$name" >&2
    exit 2
  fi
done

OUTPUT_DIR=$(dirname "$OUTPUT")
OUTPUT_STEM=$(basename "$OUTPUT" .parquet)
FULL_INPUT_LIVENESS_TIMESTAMP=$("$PY" -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))')
FULL_INPUT_LIVENESS_JSON="$AUDIT_OUT_DIR/ENTRY_FULL_INPUT_LIVENESS_CONTRACT_${FULL_INPUT_LIVENESS_TIMESTAMP}.json"
if [[ -e $AUDIT_OUT_DIR || -L $AUDIT_OUT_DIR ]]; then
  printf '[ABORT] audit output directory already exists; choose a fresh immutable path: %s\n' "$AUDIT_OUT_DIR" >&2
  exit 2
fi
if [[ $RESUME_EXACT_CHECKPOINTS -eq 0 && -e "$OUTPUT_DIR/DATASET_BUILD_PROOF.json" ]]; then
  printf '[ABORT] dataset build proof already exists; choose a fresh immutable output directory: %s\n' "$OUTPUT_DIR/DATASET_BUILD_PROOF.json" >&2
  exit 2
fi
for split in train val test; do
  if [[ -e "$OUTPUT_DIR/${OUTPUT_STEM}_${split}.parquet" || -e "$OUTPUT_DIR/${OUTPUT_STEM}_${split}.manifest.json" ]]; then
    printf '[ABORT] output split already exists; choose a fresh event directory: %s_%s\n' "$OUTPUT" "$split" >&2
    exit 2
  fi
done
cd "$ENG"

"$PY" - "$SIGNAL_MANIFEST" "$FEATURE_RANKING_JSON" "$RUN_ID" "$SOURCE_PARQUET" \
    "$CANONICAL_V2_PARQUET" "$MTF_CACHE_DIR" "$HISTORY_START" "$TEST_END" \
    "$TRAIN_START" "$TRAIN_END" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from gx1.scripts.materialize_entry_model_native_seq513_signal_manifest_v1 import (
    validate_signal_manifest_training_lineage,
)

path = Path(sys.argv[1]).expanduser().resolve()
ranking_path = Path(sys.argv[2]).expanduser().resolve()
source_path = Path(sys.argv[4]).expanduser().resolve()
canonical_v2_path = Path(sys.argv[5]).expanduser().resolve()
mtf_cache_dir = Path(sys.argv[6]).expanduser().resolve()
digest = hashlib.sha256()
with source_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
lineage = validate_signal_manifest_training_lineage(
    manifest_path=path,
    feature_ranking_path=ranking_path,
    expected_run_id=sys.argv[3],
    expected_source_parquet=source_path,
    expected_source_sha256=digest.hexdigest(),
    expected_canonical_v2_parquet=canonical_v2_path,
    expected_mtf_cache_dir=mtf_cache_dir,
    expected_history_start_utc=sys.argv[7],
    expected_time_max_utc=sys.argv[8],
    expected_train_start_utc=sys.argv[9],
    expected_train_end_utc=sys.argv[10],
)
contract = lineage["model_native_signal_contract"]
if len(contract["fields"]) != 513 or contract["bridge_dim"] != 0:
    raise RuntimeError("SEQ513_REBUILD_CONTRACT_INVALID")
print(f"[GATE] exact model-native signal/ranking lineage: {path}")
PY

export GX1_V10_MULTI_TF_V4_CACHE_DIR=$MTF_CACHE_DIR

"$PY" - "$RANK_REFERENCE_NPZ" "$RUN_ID" "$SOURCE_PARQUET" \
    "$HISTORY_START" "$TRAIN_START" "$TRAIN_END" <<'PY'
import hashlib
import sys
from pathlib import Path

from gx1.contracts.entry_model_native_state_v2 import (
    parse_utc,
    validate_train_rank_reference_lineage_v2,
)

rank_path = Path(sys.argv[1]).expanduser().resolve()
run_id = sys.argv[2]
source_path = Path(sys.argv[3]).expanduser().resolve()
history_start = parse_utc(sys.argv[4], field="history_start")
fit_start = parse_utc(sys.argv[5], field="fit_start")
fit_end = parse_utc(sys.argv[6], field="fit_end")
digest = hashlib.sha256()
with source_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
validate_train_rank_reference_lineage_v2(
    rank_path,
    expected_run_id=run_id,
    expected_source_parquet=source_path,
    expected_source_sha256=digest.hexdigest(),
    expected_history_start_utc=history_start,
    expected_fit_start_utc=fit_start,
    expected_fit_end_utc=fit_end,
)
print(f"[GATE] exact existing train-rank identity: {rank_path}")
PY

mkdir -p "$OUTPUT_DIR"
BUILDER_RESUME_ARGS=()
if [[ $RESUME_EXACT_CHECKPOINTS -eq 1 ]]; then
  BUILDER_RESUME_ARGS+=(--resume-exact-checkpoints)
fi
"${CAP[@]}" "$PY" -m gx1.scripts.build_entry_v10_ctx_training_dataset_v3 \
  "${BUILDER_RESUME_ARGS[@]}" \
  --run-id "$RUN_ID" \
  --source-parquet "$SOURCE_PARQUET" \
  --canonical_v2_parquet "$CANONICAL_V2_PARQUET" \
  --seq-structure-manifest "$SIGNAL_MANIFEST" \
  --feature-ranking-json "$FEATURE_RANKING_JSON" \
  --model-native-rank-reference-npz "$RANK_REFERENCE_NPZ" \
  --tape_root "$TAPE_ROOT" \
  --m1-lifecycle-pair-manifest-json "$M1_LIFECYCLE_PAIR_MANIFEST_JSON" \
  --m1-lifecycle-pair-generation-root "$M1_LIFECYCLE_PAIR_GENERATION_ROOT" \
  --m1-feature-base-parquet "$M1_FEATURE_BASE_PARQUET" \
  --m5-feature-base-parquet "$M5_FEATURE_BASE_PARQUET" \
  --exit-lifecycle-dir "$EXIT_LIFECYCLE_DIR" \
  --exit-target-lookahead-m1-steps "$EXIT_TARGET_LOOKAHEAD_M1_STEPS" \
  --output "$OUTPUT" \
  --start "$HISTORY_START" --end "$TEST_END" \
  --seq_len 96 --early_move_threshold_bps "$EARLY_MOVE_THRESHOLD_BPS" --time_split \
  --train_start "$TRAIN_START" --train_end "$TRAIN_END" \
  --val_start "$VAL_START" --val_end "$VAL_END" \
  --test_start "$TEST_START" --test_end "$TEST_END"

"${CAP[@]}" "$PY" -m gx1.scripts.materialize_entry_full_input_liveness_v1 \
  --run-id "$RUN_ID" \
  --dataset-dir "$OUTPUT_DIR" \
  --stem "$OUTPUT_STEM" \
  --mtf-cache-dir "$MTF_CACHE_DIR" \
  --out-json "$FULL_INPUT_LIVENESS_JSON" \
  --quiet

"$PY" - "$FULL_INPUT_LIVENESS_JSON" "$OUTPUT_DIR" <<'PY'
import json
import sys

from gx1.contracts.entry_full_input_liveness_v1 import (
    validate_full_input_liveness_artifact,
)

result = validate_full_input_liveness_artifact(
    sys.argv[1],
    expected_dataset_dir=sys.argv[2],
)
if not result["ok"]:
    raise RuntimeError(
        "FULL_INPUT_LIVENESS_POST_MATERIALIZATION_VALIDATION_FAILED: "
        + json.dumps(result["failures"], sort_keys=True)
    )
print(f"[GATE] exact full-input liveness PASS: {sys.argv[1]}")
PY

"${CAP[@]}" "$PY" -m gx1.scripts.audit_xau_direction_repair_pretrain_v1 \
  --dataset-dir "$OUTPUT_DIR" \
  --stem "$OUTPUT_STEM" \
  --out-dir "$AUDIT_OUT_DIR" \
  --data-splits train,val,test \
  --quiet

printf '[PASS] combined Entry/lifecycle dataset materialized and pretrain-audited; full-input-liveness=%s exit-lifecycle=%s; no training was run. run_id=%s output=%s\n' \
  "$FULL_INPUT_LIVENESS_JSON" "$EXIT_LIFECYCLE_DIR" "$RUN_ID" "$OUTPUT_DIR"
