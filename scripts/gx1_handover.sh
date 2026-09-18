#!/usr/bin/env bash
# Completed scopes are consumed; current_work.next_action is the resume point.
# Read-only current policy/session and exact handover resume point.
# Completed runs remain history; no native launch or model forward occurs.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel)
[[ $# -le 1 ]] || { echo "Expected at most one argument" >&2; exit 2; }
case "${1:-}" in
  ""|--check|--verbose|--source-only) ;;
  -h|--help) echo "Usage: scripts/gx1_handover.sh [--check|--verbose|--source-only]"; exit 0 ;;
  *) echo "Unsupported handover argument: $1" >&2; exit 2 ;;
esac
for required in COMPLETED_RUN.json NEXT_RUN_POLICY.json RUNNING_NATIVE_CALIBRATION.json CURRENT_HANDOVER.md VEIEN_VIDERE.md GX1_ARBEIDSMAAL.md docs/LEARNING_GATE_20260916.md; do
  [[ -f "$REPO/$required" ]] || {
    echo "FATAL: required handover file missing: $required; no legacy fallback" >&2
    exit 78
  }
done
args=(--native-binding "$REPO/COMPLETED_RUN.json")
[[ "${1:-}" != --source-only ]] || args+=(--source-only)
/usr/bin/python3 "$SCRIPT_DIR/collect_gx1_handover_readonly.py" "${args[@]}"
[[ "${1:-}" != --verbose ]] || cat "$REPO/CURRENT_HANDOVER.md"
