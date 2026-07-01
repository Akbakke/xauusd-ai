#!/usr/bin/env bash
set -euo pipefail

# Fail-closed launcher contract for the active Entry-bound Exit Transformer.
# This wrapper exists so the control surface can audit the future training
# command before any trainer implementation is allowed to run.

REPO=/home/andre2/src/GX1_ENGINE
DATA=/home/andre2/GX1_DATA
PY=$REPO/.venv/bin/python

TRAINING_PLAN_JSON=${ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_JSON:-$DATA/reports/entry_exit_transformer_training_plan_readiness_20260630_v1/ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READINESS_latest.json}
TRAIN_EXECUTION_REVIEW_JSON=${ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_JSON:-$DATA/reports/entry_exit_transformer_train_execution_review_20260630_v1/ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_latest.json}
POST_TRAIN_AUDIT_CONTRACT_JSON=${ENTRY_EXIT_TRANSFORMER_POST_TRAIN_CONTRACT_JSON:-$DATA/reports/entry_exit_transformer_post_train_contract_20260630_v1/ENTRY_EXIT_TRANSFORMER_POST_TRAIN_CONTRACT_latest.json}
FEATURE_ALIGNMENT_JSON=${ENTRY_EXIT_FEATURE_ALIGNMENT_JSON:-$DATA/reports/entry_exit_feature_alignment_20260630_v1/ENTRY_EXIT_FEATURE_ALIGNMENT_latest.json}
VEDTAK_PREFIX=ENTRY_EXIT_TRANSFORMER_TRAIN_
VEDTAK="${ENTRY_EXIT_TRANSFORMER_TRAIN_VEDTAK:-}"
DEVICE=auto
EPOCHS=1
BATCH_SIZE=32
NUM_WORKERS=0
OUT_BUNDLE_DIR=${ENTRY_EXIT_TRANSFORMER_OUT_BUNDLE_DIR:-$DATA/runs/entry_exit_transformer/manual_review_bundle}
MEM_CAP="${ENTRY_EXIT_TRANSFORMER_TRAIN_MEM_CAP:-8G}"
SWAP_CAP="${ENTRY_EXIT_TRANSFORMER_TRAIN_SWAP_CAP:-1G}"
DRY_RUN=0
MANIFEST_ONLY=0
TRAINER_IMPLEMENTATION_ENABLED=0
if [[ "${ENTRY_EXIT_TRANSFORMER_TRAINER_IMPLEMENTATION_ENABLED:-0}" = "1" ]]; then
  TRAINER_IMPLEMENTATION_ENABLED=1
fi

usage() {
  cat <<'EOF'
Usage:
  scripts/run_entry_exit_transformer_train.sh --vedtak <id> [--dry-run]

Options:
  --vedtak <id>      Required; must start with ENTRY_EXIT_TRANSFORMER_TRAIN_.
  --device <auto|cpu|cuda>
  --epochs <n>       Default: 1
  --batch-size <n>   Default: 32
  --out-bundle-dir <path>
  --dry-run          Print the future capped train command, then stop.
  --manifest-only    Reserved for a later train-execution enablement gate; currently blocked.

Resource caps:
  ENTRY_EXIT_TRANSFORMER_TRAIN_MEM_CAP   Default: 8G
  ENTRY_EXIT_TRANSFORMER_TRAIN_SWAP_CAP  Default: 1G

This wrapper is intentionally fail-closed. It does not train, replay, distill,
promote, shadow or touch live paths until a separate trainer implementation,
pretrain-manifest audit, feature-alignment audit, train-execution review,
post-train audit contract and explicit Exit train vedtak gate are approved.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vedtak) VEDTAK="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --out-bundle-dir) OUT_BUNDLE_DIR="$2"; shift 2 ;;
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

"$PY" - "$TRAINING_PLAN_JSON" "$TRAIN_EXECUTION_REVIEW_JSON" "$POST_TRAIN_AUDIT_CONTRACT_JSON" "$FEATURE_ALIGNMENT_JSON" <<'PY'
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
contract_path = Path(sys.argv[3])
if not contract_path.is_file():
    print("FATAL: active Exit Transformer post-train audit contract is missing.", file=sys.stderr)
    print(f"Path: {contract_path}", file=sys.stderr)
    raise SystemExit(2)
contract = json.loads(contract_path.read_text(encoding="utf-8"))
contract_required = "ENTRY_EXIT_TRANSFORMER_POST_TRAIN_AUDIT_CONTRACT_READY"
if contract.get("decision") != contract_required:
    print("FATAL: active Exit Transformer post-train audit contract is not ready.", file=sys.stderr)
    print(f"Decision: {contract.get('decision')}", file=sys.stderr)
    print(f"Required: {contract_required}", file=sys.stderr)
    raise SystemExit(2)
if contract.get("exit_training_allowed") is not False or contract.get("exit_iql_allowed") is not False:
    print("FATAL: post-train audit contract must keep Exit training/IQL closed.", file=sys.stderr)
    raise SystemExit(2)
feature_alignment_path = Path(sys.argv[4])
if not feature_alignment_path.is_file():
    print("FATAL: active Entry-to-Exit feature alignment report is missing.", file=sys.stderr)
    print(f"Path: {feature_alignment_path}", file=sys.stderr)
    raise SystemExit(2)
feature_alignment = json.loads(feature_alignment_path.read_text(encoding="utf-8"))
feature_alignment_required = "ENTRY_EXIT_FEATURE_ALIGNMENT_READY_FOR_EXIT_TRANSFORMER_TRAINING_REVIEW"
if feature_alignment.get("decision") != feature_alignment_required:
    print("FATAL: active Entry-to-Exit feature alignment is not ready.", file=sys.stderr)
    print(f"Decision: {feature_alignment.get('decision')}", file=sys.stderr)
    print(f"Required: {feature_alignment_required}", file=sys.stderr)
    raise SystemExit(2)
if feature_alignment.get("exit_training_allowed") is not False or feature_alignment.get("exit_iql_allowed") is not False:
    print("FATAL: feature-alignment audit must keep Exit training/IQL closed.", file=sys.stderr)
    raise SystemExit(2)
PY

TRAIN_CMD=(
  "$PY" -m gx1.models.exit_sequence_transformer.train_v1
  --training-plan-json "$TRAINING_PLAN_JSON"
  --train-execution-review-json "$TRAIN_EXECUTION_REVIEW_JSON"
  --post-train-contract-json "$POST_TRAIN_AUDIT_CONTRACT_JSON"
  --feature-alignment-json "$FEATURE_ALIGNMENT_JSON"
  --vedtak "$VEDTAK"
  --enable-training
  --out-bundle-dir "$OUT_BUNDLE_DIR"
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
  echo "Next gate: explicit Exit Transformer train-enablement package; no trainer has started." >&2
  exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "FATAL: active Exit Transformer training requires clean git." >&2
  git status --short >&2
  exit 2
fi

scripts/gx1_capped_run.sh --mem "$MEM_CAP" --swap "$SWAP_CAP" -- "${TRAIN_CMD[@]}"
