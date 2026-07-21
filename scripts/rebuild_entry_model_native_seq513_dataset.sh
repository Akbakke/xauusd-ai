#!/usr/bin/env bash
# Build one fresh XAU model-native seq513 Entry dataset. This script never trains,
# promotes, replays, or touches live state. Every artifact identity is explicit.
set -euo pipefail

ENG=/home/andre2/src/GX1_ENGINE
PY=$ENG/.venv/bin/python
CAP=("$ENG/scripts/gx1_capped_run.sh" --mem 30G --swap 2G --)

RUN_ID=
SOURCE_PARQUET=
CANONICAL_V2_PARQUET=
SIGNAL_MANIFEST=
FEATURE_RANKING_JSON=
RANK_REFERENCE_NPZ=
MTF_CACHE_DIR=
TAPE_ROOT=
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

usage() {
  printf '%s\n' \
    "Usage: $0 --run-id ID --source-parquet PATH --canonical-v2-parquet PATH" \
    "  --signal-manifest PATH --feature-ranking-json PATH --rank-reference-npz PATH" \
    "  --mtf-cache-dir PATH --tape-root PATH" \
    "  --output /new/dir/STEM__HOLD_03B.parquet --audit-out-dir /new/report/dir" \
    "  --history-start UTC --train-start UTC --train-end UTC --val-start UTC --val-end UTC" \
    "  --test-start UTC --test-end UTC [--resume-exact-checkpoints]"
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
    -h|--help) usage; exit 0 ;;
    *) printf '[ABORT] unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

required_values=(
  RUN_ID SOURCE_PARQUET CANONICAL_V2_PARQUET SIGNAL_MANIFEST FEATURE_RANKING_JSON
  RANK_REFERENCE_NPZ MTF_CACHE_DIR TAPE_ROOT OUTPUT AUDIT_OUT_DIR
  HISTORY_START TRAIN_START TRAIN_END VAL_START VAL_END TEST_START TEST_END
)
for name in "${required_values[@]}"; do
  if [[ -z ${!name} ]]; then
    printf '[ABORT] required argument missing: %s\n' "$name" >&2
    usage >&2
    exit 2
  fi
done

if [[ $OUTPUT != *.parquet || $OUTPUT != *"__HOLD_03B.parquet" ]]; then
  printf '[ABORT] --output must end in __HOLD_03B.parquet: %s\n' "$OUTPUT" >&2
  exit 2
fi
if [[ ! $RUN_ID =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$ ]]; then
  printf '[ABORT] --run-id has invalid format\n' >&2
  exit 2
fi

for name in SOURCE_PARQUET CANONICAL_V2_PARQUET SIGNAL_MANIFEST FEATURE_RANKING_JSON; do
  if [[ ! -f ${!name} ]]; then
    printf '[ABORT] required file missing (%s): %s\n' "$name" "${!name}" >&2
    exit 2
  fi
done
if [[ ! -d $MTF_CACHE_DIR || ! -d $TAPE_ROOT ]]; then
  printf '[ABORT] MTF cache or XAU tape directory missing\n' >&2
  exit 2
fi

retired_env=(
  GX1_XGB_BUNDLE_DIR
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
if [[ $RESUME_EXACT_CHECKPOINTS -eq 0 && ( -e $RANK_REFERENCE_NPZ || -e ${RANK_REFERENCE_NPZ}.json ) ]]; then
  printf '[ABORT] rank reference already exists; choose a fresh immutable path: %s\n' "$RANK_REFERENCE_NPZ" >&2
  exit 2
fi

cd "$ENG"

"$PY" - "$SIGNAL_MANIFEST" "$FEATURE_RANKING_JSON" "$RUN_ID" "$SOURCE_PARQUET" "$TRAIN_START" "$TRAIN_END" <<'PY'
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
digest = hashlib.sha256()
with source_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
lineage = validate_signal_manifest_training_lineage(
    manifest_path=path,
    feature_ranking_path=ranking_path,
    expected_run_id=sys.argv[3],
    expected_source_sha256=digest.hexdigest(),
    expected_train_start_utc=sys.argv[5],
    expected_train_end_utc=sys.argv[6],
)
contract = lineage["model_native_signal_contract"]
if len(contract["fields"]) != 513 or contract["bridge_dim"] != 0:
    raise RuntimeError("SEQ513_REBUILD_CONTRACT_INVALID")
print(f"[GATE] exact model-native signal/ranking lineage: {path}")
PY

export GX1_V10_MULTI_TF_V2_CACHE_DIR=$MTF_CACHE_DIR

if [[ $RESUME_EXACT_CHECKPOINTS -eq 0 ]]; then
  "${CAP[@]}" "$PY" -m gx1.scripts.materialize_model_native_train_rank_reference_v2 \
    --run-id "$RUN_ID" \
    --source-parquet "$SOURCE_PARQUET" \
    --out "$RANK_REFERENCE_NPZ" \
    --history-start "$HISTORY_START" \
    --fit-start "$TRAIN_START" \
    --fit-end "$TRAIN_END"
else
  "$PY" - "$RANK_REFERENCE_NPZ" "$RUN_ID" "$SOURCE_PARQUET" \
    "$HISTORY_START" "$TRAIN_START" "$TRAIN_END" <<'PY'
import hashlib
import sys
from pathlib import Path

from gx1.contracts.entry_model_native_state_v2 import (
    load_train_rank_reference_v2,
    parse_utc,
)

rank_path = Path(sys.argv[1]).expanduser().resolve()
run_id = sys.argv[2]
source_path = Path(sys.argv[3]).expanduser().resolve()
history_start = parse_utc(sys.argv[4], field="history_start")
fit_start = parse_utc(sys.argv[5], field="fit_start")
fit_end = parse_utc(sys.argv[6], field="fit_end")
reference = load_train_rank_reference_v2(rank_path)
sidecar = reference.sidecar
digest = hashlib.sha256()
with source_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
if str(sidecar.get("entry_run_id") or "") != run_id:
    raise RuntimeError("MODEL_NATIVE_RANK_RESUME_RUN_ID_MISMATCH")
if Path(str(sidecar.get("source_parquet") or "")).expanduser().resolve() != source_path:
    raise RuntimeError("MODEL_NATIVE_RANK_RESUME_SOURCE_PATH_MISMATCH")
if str(sidecar.get("source_parquet_sha256") or "").lower() != digest.hexdigest():
    raise RuntimeError("MODEL_NATIVE_RANK_RESUME_SOURCE_SHA_MISMATCH")
if parse_utc(sidecar.get("history_start_utc"), field="history_start_utc") != history_start:
    raise RuntimeError("MODEL_NATIVE_RANK_RESUME_HISTORY_START_MISMATCH")
if reference.fit_start_utc != fit_start or reference.fit_end_utc != fit_end:
    raise RuntimeError("MODEL_NATIVE_RANK_RESUME_FIT_WINDOW_MISMATCH")
print(f"[GATE] exact train-rank checkpoint resume identity: {rank_path}")
PY
fi

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
  --output "$OUTPUT" \
  --start "$HISTORY_START" --end "$TEST_END" \
  --hold-bars 3 --seq_len 96 --time_split \
  --train_start "$TRAIN_START" --train_end "$TRAIN_END" \
  --val_start "$VAL_START" --val_end "$VAL_END" \
  --test_start "$TEST_START" --test_end "$TEST_END"

"${CAP[@]}" "$PY" -m gx1.scripts.materialize_entry_full_input_liveness_v1 \
  --run-id "$RUN_ID" \
  --dataset-dir "$OUTPUT_DIR" \
  --stem "$OUTPUT_STEM" \
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

"$PY" -m gx1.scripts.audit_xau_direction_repair_pretrain_v1 \
  --dataset-dir "$OUTPUT_DIR" \
  --stem "$OUTPUT_STEM" \
  --out-dir "$AUDIT_OUT_DIR" \
  --data-splits train,val,test \
  --quiet

printf '[PASS] dataset materialized and pretrain-audited; full-input-liveness=%s; no training was run. run_id=%s output=%s\n' "$FULL_INPUT_LIVENESS_JSON" "$RUN_ID" "$OUTPUT_DIR"
