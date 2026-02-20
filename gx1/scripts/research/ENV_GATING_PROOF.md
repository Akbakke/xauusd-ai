# ENV_IDENTITY_GATING — Proof Pass (Post-rewrite)

**SSoT interpreter**: `/home/andre2/venvs/gx1/bin/python`  
**Contract**: Wrong interpreter must hard-fail early with **`[ENV_IDENTITY_FAIL]`**.

This document is a **static scan / grep proof**. No heavy jobs were run.

---

## A) Shebang & gate coverage

**Scope**: `gx1/scripts/**/*.py`

### Results

- **Python scripts scanned**: 161
- **Missing shebang**: 0
- **Wrong shebang**: 0
- **Missing `[ENV_IDENTITY_FAIL]` gate**: 0
- **Files with coding-cookie on line 2**: 50 (informational only)

### Expected invariants

- Line 1 must be:
  - `#!/home/andre2/venvs/gx1/bin/python`
- Gate presence:
  - `[ENV_IDENTITY_FAIL]` must exist in every script.

**Exceptions**: none.

---

## B) Import-safety (critical)

We must not let import-time gating crash the system for scripts that are used as modules.

### Discovery: scripts imported as modules

Static import scan found **11** modules under `gx1/scripts/` imported by other code:

- `gx1/scripts/backfill_xauusd_m5_from_oanda.py`
- `gx1/scripts/build_truth_report.py`
- `gx1/scripts/preflight_run_identity_provenance_check.py`
- `gx1/scripts/replay_eval_gated.py`
- (replay_eval_gated_parallel removed; canonical TRUTH: run_truth_e2e_sanity only, see GHOST_PURGE_PLAN.md)
- `gx1/scripts/reports_cleanup.py`
- `gx1/scripts/run_depth_ladder_eval_multiyear.py`
- `gx1/scripts/run_us_canonical_drift_benchmark.py`
- `gx1/scripts/selftest_entry_v10_bundle_load.py`
- `gx1/scripts/train_xgb_universal_multihead_v2.py`
- `gx1/scripts/verify_run_index.py`

### Action taken

For all module-used scripts above, ENV gating was moved to:

- `def _env_identity_gate(): ...`
- called only under:
  - `if __name__ == "__main__": _env_identity_gate()`

### Result

- **Module-used scripts with unsafe import-time gate**: 0

---

## C) Wrapper integrity

Wrappers verified:

- `gx1/scripts/run_phase_a.sh`
- `gx1/scripts/run_phase_b.sh`
- `gx1/scripts/run_phase_c.sh`
- `gx1/scripts/run_replay_eval_chain_compute.sh`
- (run_replay_eval_gated_parallel.sh removed; use run_truth_e2e_sanity)
- `gx1/scripts/run_build_year_metrics.sh`

All wrappers satisfy:

- `set -euo pipefail`
- `PY="/home/andre2/venvs/gx1/bin/python"`
- `[[ -x "$PY" ]]` executable check
- `exec "$PY" <target> "$@"` (no subshell, args passthrough)

---

## D) Anti-regression gate (policy check)

Tooling added:

- `gx1/scripts/tools/env_gate_policy_check.py`
- `gx1/scripts/tools/run_env_gate_policy_check.sh`

The policy check scans `gx1/scripts/**` for:

- forbidden shebangs (`#!/usr/bin/env python*`)
- forbidden tokens (py3 / wrong interpreter usage)
- missing/wrong absolute shebang in `*.py`

Run:

```bash
gx1/scripts/tools/run_env_gate_policy_check.sh
```

---

## E) SSoT entrypoint definition

See `GX1_PATHS.md`:

- **Entrypoint**: runnable scripts under `gx1/scripts/**`
- **Library code**: must not enforce interpreter gating at import-time

