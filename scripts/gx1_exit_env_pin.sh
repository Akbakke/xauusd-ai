#!/usr/bin/env bash
# scripts/gx1_exit_env_pin.sh — ONE-TRUTH exit operating-point env pin
# (user vedtak EXIT_OPERATING_POINT_CONTRACT_PIN_20260707).
#
# Prints one `export VAR=VALUE` line per var in the contract's
# exit_iql.operating_point.live_env (PROJECT_STATE_artifacts.json), so every
# gate/replay launcher measures the EXACT exit policy live runs (hard-stop /
# LWR / Strategy-F / hold-horizon / distilled / AUG64 / regime flags) instead
# of the code defaults (which are OFF for the load-bearing knobs) or a
# hardcoded per-script copy.
#
# Usage (in a launcher, BEFORE any python that reads the exit env):
#   EXIT_ENV_PIN=$(bash /home/andre2/src/GX1_ENGINE/scripts/gx1_exit_env_pin.sh)
#   eval "$EXIT_ENV_PIN"
# (capture-then-eval so a pin failure aborts under `set -e`; a direct
#  eval "$(...)" would swallow the exit status — the emitted FATAL line
#  defends that pattern too.)
#
# Fail-closed: missing/empty live_env or a non-ACTIVE exit_iql contract entry
# exits non-zero — never silently run an unpinned exit policy. Research
# ablations override AFTER eval'ing this, or set GX1_EXIT_ENV_ASSERT=0
# deliberately (the runner startup assert logs a WARNING).
set -euo pipefail

REPO=/home/andre2/src/GX1_ENGINE
PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH} "$REPO/.venv/bin/python" - <<'PYEOF'
import shlex
import sys

from gx1_guards.artifacts import load_decision_entry  # rule 8: contract-only resolution

entry = load_decision_entry("exit_iql")
live_env = (entry.get("operating_point") or {}).get("live_env") or {}
if not isinstance(live_env, dict) or not live_env:
    # Emitted line defends callers that eval "$(...)" directly (status swallowed).
    print("echo 'FATAL: contract exit_iql.operating_point.live_env missing/empty"
          " (vedtak EXIT_OPERATING_POINT_CONTRACT_PIN_20260707) — exit policy is"
          " UNPINNED, refusing.' >&2; exit 1")
    sys.exit(1)
for var in sorted(live_env):
    print(f"export {shlex.quote(str(var))}={shlex.quote(str(live_env[var]))}")
PYEOF
