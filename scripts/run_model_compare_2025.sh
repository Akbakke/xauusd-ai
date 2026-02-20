#!/usr/bin/env bash
# Model Compare 2025: FULLYEAR + Q1-Q4 for BASE28, PRUNE20, PRUNE14
set -euo pipefail
ENGINE="$(cd "$(dirname "$0")/.." && pwd)"
GX1_DATA="${GX1_DATA:-/home/andre2/GX1_DATA}"
export GX1_DATA
export GX1_CANONICAL_POLICY_PATH="${ENGINE}/gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml"
# Baseline truth (overridden per-model by run_model_compare_2025.py)
export GX1_CANONICAL_TRUTH_FILE="${ENGINE}/gx1/configs/canonical_truth_signal_only.json"

PY="${GX1_PYTHON:-}"
[ -z "$PY" ] && [ -x "${ENGINE}/.venv/bin/python" ] && PY="${ENGINE}/.venv/bin/python"
[ -z "$PY" ] && [ -x "/home/andre2/venvs/gx1/bin/python" ] && PY="/home/andre2/venvs/gx1/bin/python"
[ -z "$PY" ] && PY="python3"

MODE="${1:-all}"
[ $# -gt 0 ] && shift
"$PY" "${ENGINE}/gx1/scripts/run_model_compare_2025.py" --mode "$MODE" "$@"
