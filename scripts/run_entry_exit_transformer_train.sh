#!/usr/bin/env bash
set -euo pipefail

# Fail-closed launcher contract for the active Entry-bound Exit Transformer.
# This wrapper exists so the control surface can audit the future training
# command before any trainer implementation is allowed to run.

REPO=/home/andre2/src/GX1_ENGINE
DATA=/home/andre2/GX1_DATA
PY=$REPO/.venv/bin/python

TRAINING_PLAN_JSON=$DATA/reports/entry_exit_transformer_training_plan_readiness_20260630_v1/ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READINESS_latest.json
TRAIN_EXECUTION_REVIEW_JSON=$DATA/reports/entry_exit_transformer_train_execution_review_20260630_v1/ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_latest.json
VEDTAK_PREFIX=ENTRY_EXIT_TRANSFORMER_TRAIN_
VEDTAK="${ENTRY_EXIT_TRANSFORMER_TRAIN_VEDTAK:-}"
DEVICE=auto
EPOCHS=1
BATCH_SIZE=32
NUM_WORKERS=0
MEM_CAP="${ENTRY_EXIT_TRANSFORMER_TRAIN_MEM_CAP:-8G}"
SWAP_CAP="${ENTRY_EXIT_TRANSFORMER_TRAIN_SWAP_CAP:-1G}"
DRY_RUN=0
MANIFEST_ONLY=0
TRAINER_IMPLEMENTATION_ENABLED=0

usage() {
  cat <<'EOF'
Usage:
  scripts/run_entry_exit_transformer_train.sh --vedtak <id> [--dry-run]

Options:
  --vedtak <id>      Required; must start with ENTRY_EXIT_TRANSFORMER_TRAIN_.
  --device <auto|cpu|cuda>
  --epochs <n>       Default: 1
  --batch-size <n>   Default: 32
  --dry-run          Print the future capped train command, then stop.
  --manifest-only    Reserved for a later train-execution enablement gate; currently blocked.

Resource caps:
  ENTRY_EXIT_TRANSFORMER_TRAIN_MEM_CAP   Default: 8G
  ENTRY_EXIT_TRANSFORMER_TRAIN_SWAP_CAP  Default: 1G

This wrapper is intentionally fail-closed. It does not train, replay, distill,
promote, shadow or touch live paths until a separate trainer implementation,
pretrain-manifest audit, train-execution review and explicit Exit train vedtak
gate are approved.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vedtak) VEDTAK="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --manifest-only) MANIFEST_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "FATAL: unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${VEDTAK:-}" ]]; then
  echo "FATAL: --vedtak is required for active Exit Transformer train." >&2
  echo "This is a future train command, so it must be an explicit user decision." >&2
  exit 2
fi
if [[ "$VEDTAK" != "$VEDTAK_PREFIX"* ]]; then
  echo "FATAL: --vedtak must start with $VEDTAK_PREFIX" >&2
  exit 2
fi
case "$DEVICE" in
  auto|cpu|cuda) ;;
  *) echo "FATAL: --device must be auto, cpu, or cuda; got $DEVICE" >&2; exit 2 ;;
esac
if [[ "$NUM_WORKERS" != "0" ]]; then
  echo "FATAL: active Exit Transformer training requires --num-workers 0 for RAM guard." >&2
  exit 2
fi

cd "$REPO"

"$PY" - "$TRAINING_PLAN_JSON" "$TRAIN_EXECUTION_REVIEW_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
required = "ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READY_FOR_VEDTAK_REVIEW"
if report.get("decision") != required:
    print("FATAL: active Exit Transformer training plan/readiness is not ready.", file=sys.stderr)
    print(f"Decision: {report.get('decision')}", file=sys.stderr)
    print(f"Required: {required}", file=sys.stderr)
    raise SystemExit(2)
if report.get("exit_training_allowed") is not False:
    print("FATAL: training plan report must keep exit_training_allowed=false.", file=sys.stderr)
    raise SystemExit(2)
review_path = Path(sys.argv[2])
review = json.loads(review_path.read_text(encoding="utf-8"))
review_required = "ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_READY_FOR_EXPLICIT_VEDTAK_PACKAGE"
if review.get("decision") != review_required:
    print("FATAL: active Exit Transformer train-execution review is not ready.", file=sys.stderr)
    print(f"Decision: {review.get('decision')}", file=sys.stderr)
    print(f"Required: {review_required}", file=sys.stderr)
    raise SystemExit(2)
if review.get("exit_training_allowed") is not False or review.get("exit_training_allowed_with_explicit_vedtak") is not False:
    print("FATAL: train-execution review must keep Exit training closed.", file=sys.stderr)
    raise SystemExit(2)
PY

TRAIN_CMD=(
  "$PY" -m gx1.models.exit_sequence_transformer.train_v1
  --training-plan-json "$TRAINING_PLAN_JSON"
  --train-execution-review-json "$TRAIN_EXECUTION_REVIEW_JSON"
  --vedtak "$VEDTAK"
  --device "$DEVICE"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
)

if [[ "$DRY_RUN" = "1" ]]; then
  echo "Active Exit Transformer train wrapper is fail-closed; dry-run only."
  echo "Future resource cap: mem=$MEM_CAP swap=$SWAP_CAP"
  printf 'Future capped train command:'
  printf ' %q' scripts/gx1_capped_run.sh --mem "$MEM_CAP" --swap "$SWAP_CAP" -- "${TRAIN_CMD[@]}"
  printf '\n'
  exit 0
fi

if [[ "$MANIFEST_ONLY" = "1" ]]; then
  echo "FATAL: --manifest-only is blocked until active Exit Transformer train-execution enablement exists." >&2
  exit 2
fi

if [[ "$TRAINER_IMPLEMENTATION_ENABLED" != "1" ]]; then
  echo "FATAL: active Exit Transformer trainer implementation is not enabled." >&2
  echo "Next gate: implement trainer core plus pretrain-manifest audit; no trainer has started." >&2
  exit 2
fi

scripts/gx1_capped_run.sh --mem "$MEM_CAP" --swap "$SWAP_CAP" -- "${TRAIN_CMD[@]}"
