# DT_MODULE Fix Verification Guide

## Overview

This document describes how to verify that the `dt_module` bug is fixed and that COST variants are safe to run in multiyear backtests.

## The Problem

COST variants were failing deterministically after ~40 minutes with:
```
NameError: name 'dt_module' is not defined
```

This occurred in `replay_eval_gated_parallel.py` when writing error stubs, where `dt_module` was not in scope.

## The Fix

1. **Created `gx1/utils/dt_module.py`**: Centralized datetime utilities with version stamping
2. **Fixed all datetime usage**: Replaced all `datetime.now()` and `dt_module.now()` with `dt_now_iso()` from `gx1.utils.dt_module`
3. **Added version stamping**: `dt_module_version` is now logged in:
   - `master_early.json`
   - `WORKER_BOOT.json`
   - `chunk_footer.json`
   - `RUN_IDENTITY.json`
4. **Fail-fast validation**: `validate_dt_module_version()` is called in master and workers before replay starts

## Verification Steps

### Step 1: Verify dt_module Version

```bash
python3 -c "from gx1.utils.dt_module import get_dt_module_version, validate_dt_module_version; validate_dt_module_version(); print(f'Version: {get_dt_module_version()}')"
```

Expected output:
```
Version: 2026-01-18_fix2
```

### Step 2: Run COST Smoke Test

```bash
python3 gx1/scripts/run_cost_smoke_test.py --year 2020 --variant THR_BASE__COST_S12_A80__WIN_W0 --days 7
```

This should:
- Complete in a few minutes
- Not fail with `dt_module` errors
- Produce valid `RUN_IDENTITY.json`, `WORKER_BOOT.json`, and `chunk_footer.json` with `dt_module_version`

### Step 3: Verify Version Stamps

After running the smoke test, verify that all artifacts contain the correct `dt_module_version`:

```bash
# Check RUN_IDENTITY.json
python3 -c "import json; d=json.load(open('reports/replay_eval/COST_SMOKE_TEST/THR_BASE__COST_S12_A80__WIN_W0/YEAR_2020/RUN_IDENTITY.json')); print(f\"dt_module_version: {d.get('dt_module_version')}\")"

# Check WORKER_BOOT.json
python3 -c "import json; d=json.load(open('reports/replay_eval/COST_SMOKE_TEST/THR_BASE__COST_S12_A80__WIN_W0/YEAR_2020/chunk_0/WORKER_BOOT.json')); print(f\"dt_module_version: {d.get('dt_module_version')}\")"

# Check chunk_footer.json
python3 -c "import json; d=json.load(open('reports/replay_eval/COST_SMOKE_TEST/THR_BASE__COST_S12_A80__WIN_W0/YEAR_2020/chunk_0/chunk_footer.json')); print(f\"dt_module_version: {d.get('dt_module_version')}\")"
```

All should show: `dt_module_version: 2026-01-18_fix2`

### Step 4: Verify No dt_module Errors

Check that no `FATAL_ERROR.txt` contains `dt_module` errors:

```bash
grep -r "dt_module" reports/replay_eval/COST_SMOKE_TEST/*/FATAL_ERROR.txt 2>/dev/null || echo "No dt_module errors found"
```

## Before Multiyear Restart

**CRITICAL**: Do NOT restart the full multiyear backtest until:

1. ✅ COST smoke test passes
2. ✅ All version stamps are correct
3. ✅ No `dt_module` errors in logs
4. ✅ `chunk_footer.json` status is "ok"

## Expected Behavior

- **Before fix**: COST variants fail after ~40 minutes with `NameError: name 'dt_module' is not defined`
- **After fix**: COST variants complete successfully, all artifacts contain `dt_module_version: 2026-01-18_fix2`

## Troubleshooting

If the smoke test fails:

1. Check that `gx1/utils/dt_module.py` exists and has `DT_MODULE_VERSION = "2026-01-18_fix2"`
2. Check that all imports in `replay_eval_gated_parallel.py` use `from gx1.utils.dt_module import ...`
3. Check that `validate_dt_module_version()` is called in master and workers
4. Check logs for any import errors
