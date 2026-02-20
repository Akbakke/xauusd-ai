#!/usr/bin/env bash
set -euo pipefail

PY="/home/andre2/venvs/gx1/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Required python interpreter not found or not executable: $PY"
  echo "Hint: source ~/venvs/gx1/bin/activate"
  exit 1
fi

# Canonical TRUTH: run_truth_e2e_sanity only; run_replay_eval_gated_parallel.sh removed (ghost purge)
WRAPPERS=(
  "gx1/scripts/run_phase_a.sh"
  "gx1/scripts/run_phase_b.sh"
  "gx1/scripts/run_phase_c.sh"
  "gx1/scripts/run_replay_eval_chain_compute.sh"
  "gx1/scripts/run_build_year_metrics.sh"
  "gx1/scripts/tools/run_env_gate_policy_check.sh"
)

BAD=()
MISSING=()
for w in "${WRAPPERS[@]}"; do
  if [[ ! -e "$w" ]]; then
    MISSING+=("$w")
    continue
  fi
  if [[ ! -x "$w" ]]; then
    BAD+=("$w")
  fi
done

if [[ ${#MISSING[@]} -ne 0 ]]; then
  echo "[GX1_POLICY] FAIL: missing expected wrapper files:"
  for w in "${MISSING[@]}"; do
    echo "  - $w"
  done
  exit 2
fi

if [[ ${#BAD[@]} -ne 0 ]]; then
  echo "[GX1_POLICY] FAIL: wrapper(s) not executable (+x missing):"
  for w in "${BAD[@]}"; do
    echo "  - $w"
  done
  echo ""
  echo "Fix:"
  echo "  - Run: make gx1-fix-perms"
  echo "  - Or:  chmod +x ${BAD[*]}"
  exit 2
fi

set +e
"$PY" gx1/scripts/tools/env_gate_policy_check.py "$@"
rc=$?
set -e

if [[ $rc -eq 0 ]]; then
  echo "[GX1_POLICY] PASS"
else
  echo "[GX1_POLICY] FAIL (rc=$rc)"
fi
exit $rc

