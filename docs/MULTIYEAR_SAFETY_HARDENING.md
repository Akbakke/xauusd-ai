# Multiyear Safety Hardening - Temperature Status + Rolling Perf Kill-Rule

## Overview

Enhanced the `run_overlap_sanity_pack_multiyear.py` orchestrator with:
1. **Explicit temperature scaling status logging** (anti-footgun)
2. **Rolling perf-export kill-rule** (spar compute)

This is **OBSERVABILITY + SAFETY**. No changes to trading/model logic.

---

## DEL A: Explicit Temperature Scaling Status

### Purpose
Prevent accidental multiyear runs without temperature scaling by making the effective status **explicitly visible** in logs and artifacts.

### Implementation

**Fields calculated:**
- `temperature_scaling_effective_enabled`: `bool` - What will actually be used
- `temperature_scaling_source`: `"env" | "default" | "flag_override"` - Where the setting came from
- `temperature_scaling_env_value`: `"0" | "1" | None` - Raw env value (if set)

**Logic:**
- If `GX1_TEMPERATURE_SCALING=0` and `--allow-temp-off` → `source="flag_override"`, `enabled=False`
- If `GX1_TEMPERATURE_SCALING=0` and no flag → `source="env"`, `enabled=False` → **FATAL**
- If `GX1_TEMPERATURE_SCALING` not set → `source="default"`, `enabled=True`
- If `GX1_TEMPERATURE_SCALING="1"` → `source="env"`, `enabled=True`

**Artifacts:**
- `master_early.json` (written at orchestrator start)
- `ORCHESTRATOR_SUMMARY.json` (top-level fields)
- Clear INFO log line: `"Temperature scaling ENABLED (source=default, env=None)"`

**Existing FATAL rule preserved:**
- If `effective_enabled == False` and not `--allow-temp-off` → FATAL before starting

---

## DEL B: Rolling Perf-Export Kill-Rule

### Purpose
Abort early if perf-export errors explode, saving compute resources instead of waiting until end-of-run.

### Implementation

**Configuration:**
- `PERF_EXPORT_MAX_RATIO = 0.05` (5% threshold)
- `PERF_EXPORT_MIN_TASKS_FOR_EARLY_EVAL = 20` (minimum tasks before checking)

**Rolling monitoring:**
- Track `tasks_completed` and `perf_export_error_count_rolling` as tasks finish
- After ≥20 tasks completed:
  - Calculate `error_ratio = perf_export_error_count_rolling / tasks_completed`
  - If `error_ratio > 0.05` → **FATAL** with clear message
  - Write snapshot to `ORCHESTRATOR_SUMMARY.json` before abort

**End-of-run evaluation:**
- Still performed (existing behavior)
- Final check after all tasks complete

**Artifacts:**
- `ORCHESTRATOR_SUMMARY.json` contains:
  - `perf_export_monitoring.perf_export_max_ratio`
  - `perf_export_monitoring.perf_export_min_tasks_for_early_eval`
  - `perf_export_monitoring.perf_export_early_abort_triggered` (bool)

---

## Verification

### DEL A Verification
✅ `master_early.json` contains all three fields:
```json
{
  "temperature_scaling_effective_enabled": true,
  "temperature_scaling_source": "default",
  "temperature_scaling_env_value": null
}
```

✅ `ORCHESTRATOR_SUMMARY.json` contains same fields at top-level

✅ Clear INFO log line at start

### DEL B Verification
✅ Early abort logic implemented in task completion loop

✅ Configuration logged in `ORCHESTRATOR_SUMMARY.json`:
```json
{
  "perf_export_monitoring": {
    "perf_export_max_ratio": 0.05,
    "perf_export_min_tasks_for_early_eval": 20,
    "perf_export_early_abort_triggered": false
  }
}
```

---

## Files Modified

- `gx1/scripts/run_overlap_sanity_pack_multiyear.py`
  - DEL A: Temperature scaling status calculation and logging
  - DEL B: Rolling perf-export monitoring in task completion loop
  - Both: Updated `ORCHESTRATOR_SUMMARY.json` structure

---

## No Changes To

- Trading logic
- Model logic
- Features, gates, exits
- Worker code
- `RUN_IDENTITY.json` (per-task, not orchestrator-level)

---

## Usage

The orchestrator now automatically:
1. Logs explicit temperature scaling status at start
2. Monitors perf-export errors rolling and aborts early if >5% after ≥20 tasks
3. Performs end-of-run perf-export evaluation (existing behavior)

No changes to command-line interface or usage patterns.
