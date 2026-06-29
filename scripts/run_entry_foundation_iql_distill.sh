#!/usr/bin/env bash
set -euo pipefail

# Vedtak-gated Entry foundation IQL distillation contract launcher.
# This path is fail-closed behind replay-readiness and never promotes, starts
# shadow/live, or writes production adapter artifacts.

REPO=/home/andre2/src/GX1_ENGINE
DATA=/home/andre2/GX1_DATA
PY=$REPO/.venv/bin/python

REPLAY_READINESS_JSON=$DATA/reports/entry_replay_readiness_20260628_v1/ENTRY_REPLAY_READINESS_latest.json
CONTRACT_OUT_DIR=$DATA/reports/entry_iql_distillation_contract_20260628_v1

VEDTAK="${ENTRY_FOUNDATION_IQL_DISTILL_VEDTAK:-}"
DRY_RUN=0
CUSTOM_REPLAY_READINESS_JSON=0
NO_FAIL_ON_NOT_READY=0

usage() {
  cat <<'EOF'
Usage:
  scripts/run_entry_foundation_iql_distill.sh --vedtak <id> [options]

Options:
  --vedtak <id>                 Required explicit user decision id.
  --replay-readiness-json <path>
                                Default: latest Entry replay-readiness report.
  --out-dir <path>              Default: Entry IQL distillation contract report dir.
  --dry-run                     Print the contract command after replay-readiness passes.
  --no-fail-on-not-ready        Write a NOT_READY contract instead of exiting when
                                replay-readiness is not ready.
  --materialize-only            Alias for --no-fail-on-not-ready. This wrapper only
                                materializes the research contract; it never trains.

This is an IQL-distillation research contract path. It will not run until
replay-readiness proves candidate-readiness, selective-edge evidence and 2026
offline replay evidence. It does not promote, shadow, live, or build adapters.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vedtak) VEDTAK="$2"; shift 2 ;;
    --replay-readiness-json) REPLAY_READINESS_JSON="$2"; CUSTOM_REPLAY_READINESS_JSON=1; shift 2 ;;
    --out-dir) CONTRACT_OUT_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-fail-on-not-ready) NO_FAIL_ON_NOT_READY=1; shift ;;
    --materialize-only) NO_FAIL_ON_NOT_READY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "FATAL: unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${VEDTAK:-}" ]]; then
  echo "FATAL: --vedtak is required for foundation IQL distillation." >&2
  echo "This opens a research distillation contract, so it must be an explicit user decision." >&2
  exit 2
fi

cd "$REPO"

if [[ "$CUSTOM_REPLAY_READINESS_JSON" = "0" ]]; then
  if [[ "$NO_FAIL_ON_NOT_READY" = "1" ]]; then
    scripts/entry_next_edge_control.sh replay-readiness --quiet --no-fail-on-not-ready
  elif ! scripts/entry_next_edge_control.sh replay-readiness --quiet; then
    cat >&2 <<'EOF'
FATAL: replay-readiness is NOT_READY; IQL distillation is blocked.

Required next gates:
  scripts/entry_next_edge_control.sh candidate-readiness
  scripts/entry_next_edge_control.sh replay-readiness

No IQL distillation, adapter, promotion, shadow, or live path was started.
EOF
    exit 2
  fi
fi

if ! "$PY" - "$REPLAY_READINESS_JSON" "$NO_FAIL_ON_NOT_READY" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser()
no_fail_on_not_ready = sys.argv[2] == "1"
if not path.exists():
    print(f"missing replay-readiness report: {path}", file=sys.stderr)
    raise SystemExit(2)
report = json.loads(path.read_text(encoding="utf-8"))
ok = (
    report.get("decision") == "READY_FOR_IQL_DISTILLATION_VEDTAK"
    and report.get("iql_distillation_allowed_with_explicit_vedtak") is True
    and report.get("promotion_shadow_live_allowed") is False
)
if not ok:
    print(f"replay-readiness decision is {report.get('decision')}", file=sys.stderr)
    if no_fail_on_not_ready:
        print("writing NOT_READY IQL distillation contract because --no-fail-on-not-ready is set", file=sys.stderr)
        raise SystemExit(0)
    raise SystemExit(2)
PY
then
  cat >&2 <<EOF
FATAL: replay-readiness is NOT_READY; IQL distillation is blocked.

Required next gates:
  scripts/entry_next_edge_control.sh candidate-readiness
  scripts/entry_next_edge_control.sh replay-readiness

No IQL distillation, adapter, promotion, shadow, or live path was started.
EOF
  exit 2
fi

CMD=(
  "$PY" -m gx1.scripts.materialize_entry_iql_distillation_contract_v1
  --vedtak "$VEDTAK"
  --replay-readiness-json "$REPLAY_READINESS_JSON"
  --out-dir "$CONTRACT_OUT_DIR"
)
if [[ "$NO_FAIL_ON_NOT_READY" = "1" ]]; then
  CMD+=(--no-fail-on-not-ready)
fi

if [[ "$DRY_RUN" = "1" ]]; then
  printf 'IQL distillation contract command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  echo "Post-distillation gates still required: offline replay comparison, slice audit, explicit promotion review."
  exit 0
fi

"${CMD[@]}"
echo "IQL distillation contract written under: $CONTRACT_OUT_DIR"
echo "Post-distillation gates still required: offline replay comparison, slice audit, explicit promotion review."
