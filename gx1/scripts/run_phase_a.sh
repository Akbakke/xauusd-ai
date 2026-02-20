#!/usr/bin/env bash
set -euo pipefail

# Preflight: policy check (makes --no-verify visible)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
make -s gx1-policy || {
  echo "ERROR: Policy check failed. Fix violations or use --no-verify (not recommended)."
  exit 1
}

PY="/home/andre2/venvs/gx1/bin/python"
export GX1_PYTHON="$PY"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Required python interpreter not found or not executable: $PY"
  echo "Hint: source ~/venvs/gx1/bin/activate"
  exit 1
fi

exec "$PY" gx1/scripts/research/run_phase_a_xgb_optuna.py "$@"

