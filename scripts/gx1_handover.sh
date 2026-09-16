#!/usr/bin/env bash
# One read-only handover. Missing current evidence never falls back to old runs.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel)
[[ $# -le 1 ]] || { echo "Expected at most one argument" >&2; exit 2; }
case "${1:-}" in
  ""|--check|--verbose|--source-only) ;;
  -h|--help) echo "Usage: scripts/gx1_handover.sh [--check|--verbose|--source-only]"; exit 0 ;;
  *) echo "Unsupported handover argument: $1" >&2; exit 2 ;;
esac
[[ -f "$REPO/COMPLETED_RUN.json" && -f "$REPO/NEXT_RUN_POLICY.json" && -f "$REPO/RUNNING_NATIVE_CALIBRATION.json" && -f "$REPO/CURRENT_HANDOVER.md" ]] || {
  echo "FATAL: completed evidence, current handover/status and next-run policy are required; no legacy fallback" >&2
  exit 78
}
args=(--native-binding "$REPO/COMPLETED_RUN.json")
[[ "${1:-}" != --source-only ]] || args+=(--source-only)
/usr/bin/python3 "$SCRIPT_DIR/collect_gx1_handover_readonly.py" "${args[@]}"
[[ "${1:-}" != --verbose ]] || cat "$REPO/CURRENT_HANDOVER.md"
