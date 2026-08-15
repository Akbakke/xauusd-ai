#!/usr/bin/env bash
# Build one fresh XAU model-native seq513 Entry dataset plus its source-bound
# unified Exit lifecycle surface. This script never trains, promotes, replays,
# or touches live state. Every artifact identity is explicit.
set -euo pipefail

ENG=/home/andre2/src/GX1_ENGINE
PY=$ENG/.venv/bin/python
CAP=("$ENG/scripts/gx1_capped_run.sh" --class producer --mem 10G --swap 512M --)

RUN_ID=
SOURCE_PARQUET=
CANONICAL_V2_PARQUET=
SIGNAL_MANIFEST=
FEATURE_RANKING_JSON=
MTF_CACHE_DIR=
TAPE_ROOT=
M1_LIFECYCLE_PAIR_MANIFEST_JSON=
M1_LIFECYCLE_PAIR_GENERATION_ROOT=
M1_FEATURE_BASE_PARQUET=
M5_FEATURE_BASE_PARQUET=
EXIT_LIFECYCLE_DIR=
OUTPUT=
AUDIT_OUT_DIR=
REBUILD_TERMINAL_JSON=
PREFREEZE_TEST_SEAL_JSON=
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
    "  --signal-manifest PATH --feature-ranking-json PATH" \
    "  --mtf-cache-dir PATH --tape-root PATH" \
    "  --m1-lifecycle-pair-manifest-json /immutable/generation/PAIR_MANIFEST.json" \
    "  --m1-lifecycle-pair-generation-root /immutable/generations" \
    "  --m1-feature-base-parquet /immutable/M1_FEATURE_BASE.parquet" \
    "  --m5-feature-base-parquet /immutable/M5_FEATURE_BASE.parquet" \
    "  --exit-lifecycle-dir /new/dir (Exit target policy is fit on TRAIN)" \
    "  Entry direction/early-move policy is fit on TRAIN native M5" \
    "  --output /new/dir/STEM__DIR_TRAIN_FIT.parquet --audit-out-dir /new/report/dir" \
    "  --rebuild-terminal-json /event/rebuild_authority/ENTRY_MODEL_NATIVE_SEQ513_DATASET_REBUILD_TERMINAL_<UTC>.json" \
    "  --prefreeze-test-seal-json /event/rebuild_authority/ENTRY_MODEL_NATIVE_SEQ513_UNTOUCHED_TEST_SEAL_<UTC>.json" \
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
    --mtf-cache-dir) MTF_CACHE_DIR=${2:-}; shift 2 ;;
    --tape-root) TAPE_ROOT=${2:-}; shift 2 ;;
    --m1-lifecycle-pair-manifest-json) M1_LIFECYCLE_PAIR_MANIFEST_JSON=${2:-}; shift 2 ;;
    --m1-lifecycle-pair-generation-root) M1_LIFECYCLE_PAIR_GENERATION_ROOT=${2:-}; shift 2 ;;
    --m1-feature-base-parquet) M1_FEATURE_BASE_PARQUET=${2:-}; shift 2 ;;
    --m5-feature-base-parquet) M5_FEATURE_BASE_PARQUET=${2:-}; shift 2 ;;
    --exit-lifecycle-dir) EXIT_LIFECYCLE_DIR=${2:-}; shift 2 ;;
    --output) OUTPUT=${2:-}; shift 2 ;;
    --audit-out-dir) AUDIT_OUT_DIR=${2:-}; shift 2 ;;
    --rebuild-terminal-json) REBUILD_TERMINAL_JSON=${2:-}; shift 2 ;;
    --prefreeze-test-seal-json) PREFREEZE_TEST_SEAL_JSON=${2:-}; shift 2 ;;
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
  MTF_CACHE_DIR TAPE_ROOT M1_LIFECYCLE_PAIR_MANIFEST_JSON
  M1_LIFECYCLE_PAIR_GENERATION_ROOT M1_FEATURE_BASE_PARQUET M5_FEATURE_BASE_PARQUET
  EXIT_LIFECYCLE_DIR
  OUTPUT AUDIT_OUT_DIR REBUILD_TERMINAL_JSON PREFREEZE_TEST_SEAL_JSON
  HISTORY_START TRAIN_START TRAIN_END VAL_START VAL_END TEST_START TEST_END
)
for name in "${required_values[@]}"; do
  if [[ -z ${!name} ]]; then
    printf '[ABORT] required argument missing: %s\n' "$name" >&2
    usage >&2
    exit 2
  fi
done

if [[ $OUTPUT != *.parquet || $OUTPUT != *"__DIR_TRAIN_FIT.parquet" ]]; then
  printf '[ABORT] --output must end in __DIR_TRAIN_FIT.parquet: %s\n' "$OUTPUT" >&2
  exit 2
fi
if [[ ! $RUN_ID =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$ ]]; then
  printf '[ABORT] --run-id has invalid format\n' >&2
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
EXPECTED_AUTHORITY_DIR=$(dirname "$OUTPUT_DIR")/rebuild_authority
if [[ $(dirname "$REBUILD_TERMINAL_JSON") != "$EXPECTED_AUTHORITY_DIR" \
   || $(dirname "$PREFREEZE_TEST_SEAL_JSON") != "$EXPECTED_AUTHORITY_DIR" \
   || $(basename "$REBUILD_TERMINAL_JSON") != ENTRY_MODEL_NATIVE_SEQ513_DATASET_REBUILD_TERMINAL_*.json \
   || $(basename "$PREFREEZE_TEST_SEAL_JSON") != ENTRY_MODEL_NATIVE_SEQ513_UNTOUCHED_TEST_SEAL_*.json ]]; then
  printf '[ABORT] TEST authority paths must be exact event-owned rebuild_authority events\n' >&2
  exit 2
fi
if [[ -e $REBUILD_TERMINAL_JSON || -L $REBUILD_TERMINAL_JSON \
   || -e $PREFREEZE_TEST_SEAL_JSON || -L $PREFREEZE_TEST_SEAL_JSON ]]; then
  printf '[ABORT] TEST authority outputs must be fresh and immutable\n' >&2
  exit 2
fi
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

from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SIGNAL_DIM
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
# The signal width derives from the one contract owner (V29: 592); a
# repeated literal here is exactly the stale-gate class that killed V29F.
if (
    len(contract["fields"]) != MODEL_NATIVE_SIGNAL_DIM
    or contract["bridge_dim"] != 0
):
    raise RuntimeError("SEQ513_REBUILD_CONTRACT_INVALID")
print(f"[GATE] exact model-native signal/ranking lineage: {path}")
PY

export GX1_V10_MULTI_TF_V4_CACHE_DIR=$MTF_CACHE_DIR

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
  --tape_root "$TAPE_ROOT" \
  --m1-lifecycle-pair-manifest-json "$M1_LIFECYCLE_PAIR_MANIFEST_JSON" \
  --m1-lifecycle-pair-generation-root "$M1_LIFECYCLE_PAIR_GENERATION_ROOT" \
  --m1-feature-base-parquet "$M1_FEATURE_BASE_PARQUET" \
  --m5-feature-base-parquet "$M5_FEATURE_BASE_PARQUET" \
  --exit-lifecycle-dir "$EXIT_LIFECYCLE_DIR" \
  --output "$OUTPUT" \
  --rebuild-terminal-json "$REBUILD_TERMINAL_JSON" \
  --prefreeze-test-seal-json "$PREFREEZE_TEST_SEAL_JSON" \
  --start "$HISTORY_START" --end "$TEST_END" \
  --seq_len 96 --time_split \
  --train_start "$TRAIN_START" --train_end "$TRAIN_END" \
  --val_start "$VAL_START" --val_end "$VAL_END" \
  --test_start "$TEST_START" --test_end "$TEST_END"

"$PY" - "$REBUILD_TERMINAL_JSON" "$PREFREEZE_TEST_SEAL_JSON" \
  "$RUN_ID" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


terminal_path = Path(sys.argv[1])
seal_path = Path(sys.argv[2])
run_id = sys.argv[3]
for label, path in (("rebuild terminal", terminal_path), ("TEST seal", seal_path)):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} was not atomically published: {path}")
terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
seal = json.loads(seal_path.read_text(encoding="utf-8"))
if (
    terminal.get("entry_run_id") != run_id
    or terminal.get("terminal_event_path") != str(terminal_path)
    or seal.get("entry_run_id") != run_id
    or seal.get("seal_path") != str(seal_path)
    or seal.get("rebuild_terminal", {}).get("path") != str(terminal_path)
    or seal.get("rebuild_terminal", {}).get("sha256") != sha256(terminal_path)
):
    raise RuntimeError("published pre-freeze TEST authority is inconsistent")
print(
    "[GATE] exact pre-freeze TEST authority: "
    f"terminal={terminal_path} terminal_sha256={sha256(terminal_path)} "
    f"seal={seal_path} seal_sha256={sha256(seal_path)}"
)
PY

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
  --quiet

printf '[PASS] combined Entry/lifecycle dataset materialized and pretrain-audited; full-input-liveness=%s exit-lifecycle=%s; no training was run. run_id=%s output=%s\n' \
  "$FULL_INPUT_LIVENESS_JSON" "$EXIT_LIFECYCLE_DIR" "$RUN_ID" "$OUTPUT_DIR"
