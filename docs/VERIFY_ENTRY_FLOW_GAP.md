# Entry Flow Gap Verification Guide

## Purpose

This verification tool exists as a **plumbing completion gate** for the entry evaluation flow. It deterministically answers what happened in the critical gap between:

- `AFTER_SOFT_ELIGIBILITY_PASSED` (soft eligibility check passed)
- `BEFORE_STAGE0_CHECK` (Stage-0 filter check)

This gap is instrumented with try/except to catch any hidden exceptions that prevent Stage-0 from being reached. The verification tool reads telemetry artifacts and outputs one of four terminal states:

1. **EXCEPTION_CAUGHT** - Exception was caught and logged
2. **STAGE0_REACHED** - Stage-0 was reached (entry flow verified)
3. **UNKNOWN_EXIT** - Soft eligibility passed but Stage-0 not reached (structural issue)
4. **INSUFFICIENT_EVIDENCE** - No telemetry or soft eligibility never passed

## Usage

### Basic Usage

```bash
python3 gx1/scripts/verify_entry_flow_gap.py <path>
```

### With JSON Report

```bash
python3 gx1/scripts/verify_entry_flow_gap.py <path> --write-json <output_path>
```

### Verbose Output

```bash
python3 gx1/scripts/verify_entry_flow_gap.py <path> --verbose
```

## Path Resolution

The tool accepts three types of paths:

### 1. Run Root Directory

```bash
python3 gx1/scripts/verify_entry_flow_gap.py reports/replay_eval/run_20250108
```

Searches for:
- `ENTRY_FEATURES_USED_MASTER.json` (highest priority)
- Falls back to `chunk_*/ENTRY_FEATURES_USED.json` if master not found

### 2. Chunk Directory

```bash
python3 gx1/scripts/verify_entry_flow_gap.py reports/replay_eval/run_20250108/chunk_0
```

Searches for:
- `ENTRY_FEATURES_USED.json`
- `chunk_footer.json` (fallback)
- `CHUNK_FAIL_CAPSULE.json` (exception details)

### 3. Direct JSON File

```bash
python3 gx1/scripts/verify_entry_flow_gap.py reports/replay_eval/run_20250108/ENTRY_FEATURES_USED_MASTER.json
```

Reads the specified file directly.

## Terminal States

### EXCEPTION_CAUGHT (Exit Code: 10)

**When:** Exception occurred in the gap and was caught by try/except.

**Output:**
```
EXCEPTION CAUGHT:
  Type: <exc_type>
  Message: <exc_msg>
  Line: <line>
  Timestamp: <ts>
```

**Next Action:**
1. Review exception details in `CHUNK_FAIL_CAPSULE.json`
2. Classify:
   - **Bug:** Fix exception and re-run test
   - **Expected:** Document and accept (e.g., missing context feature)
3. Re-run test after fix/acceptance

**Example Legitimate Exceptions:**
- `KeyError` in feature lookup
- `AttributeError` in policy_state access
- `ValueError` in feature alignment

**Example Bugs:**
- `NoneType` access without None-check
- Wrong code order
- Stale state access

### STAGE0_REACHED (Exit Code: 0)

**When:** Stage-0 was reached and all verification criteria are met.

**Output:**
```
✓ STAGE-0 REACHED
✓ SOFT ELIGIBILITY PASSED
✓ ROUTING OK (v10_hybrid)
✓ TRANSFORMER CALLED (N calls)
```

**Verification Criteria:**
- `BEFORE_STAGE0_CHECK > 0`
- `transformer_forward_calls > 0`
- `entry_routing.selected_model == "v10_hybrid"` (if routing data available)

**Next Action:**
→ **Entry-flow is verified**
→ **Ready for A/B tests (XGB ↔ Transformer)**

### STAGE0_REACHED_BUT_NOT_FULLY_VERIFIED (Exit Code: 20)

**When:** Stage-0 was reached but some verification criteria are missing.

**Output:**
```
⚠ PARTIAL VERIFICATION
Stage-0 reached but some criteria missing:
  Missing: transformer_forward_calls == 0
```

**Next Action:**
1. Investigate missing criteria
2. Check Stage-0 filter (may block transformer call)
3. Check routing logic (may select legacy path)
4. Check model entry gates

### UNKNOWN_EXIT (Exit Code: 30)

**When:** Soft eligibility passed but Stage-0 not reached, and no exception or early return recorded.

**Output:**
```
✗ UNKNOWN EXIT (SOFT->STAGE0)
AFTER_SOFT_ELIGIBILITY_PASSED > 0 but BEFORE_STAGE0_CHECK == 0
and no EXCEPTION_IN_SOFT_TO_STAGE0_GAP or EARLY_RETURN_IN_GAP recorded
```

**Next Action:**
Investigate structural control-flow issue:
- `sys.exit()` / `os._exit()` calls
- Multiprocessing abort
- C-extension side-effects
- Signal handlers

Check `CHUNK_FAIL_CAPSULE.json` for full traceback.

### INSUFFICIENT_EVIDENCE (Exit Code: 40)

**When:** No telemetry found or soft eligibility never passed.

**Output:**
```
✗ INSUFFICIENT EVIDENCE
  Soft eligibility never passed - check eligibility gates
```

**Next Action:**
1. Check telemetry collection
2. Check file paths
3. Verify test run completed successfully

## Output Files

### JSON Report

The tool writes a JSON report (default: `verify_entry_flow_gap_report.json` in script directory):

```json
{
  "status": "STAGE0_REACHED",
  "exit_code": 0,
  "paths_used": ["/path/to/ENTRY_FEATURES_USED_MASTER.json"],
  "soft_passed_count": 100,
  "stage0_count": 100,
  "gap_exception_count": 0,
  "early_return_count": 0,
  "exception_present": false,
  "exception_gap": null,
  "transformer_forward_calls": 100,
  "routing_v10_count": 100,
  "callsite_entered": 100,
  "callsite_returned": 100,
  "callsite_exception": 0,
  "model_attempts": 100,
  "model_forwards": 100,
  "hints": ["All verification criteria met"]
}
```

## Source File Priority

The tool reads telemetry in this priority order:

1. **ENTRY_FEATURES_USED_MASTER.json** (run root)
   - Aggregated telemetry from all chunks
   - Highest priority

2. **ENTRY_FEATURES_USED.json** (chunk)
   - Per-chunk telemetry
   - Used if master not available

3. **chunk_footer.json** (chunk)
   - Fallback for control_flow summary
   - Used if entry features file incomplete

4. **CHUNK_FAIL_CAPSULE.json** (chunk)
   - SSoT for exception_gap details
   - Used when exception occurred

## Definition of "Plumbing Done"

Entry-flow plumbing is considered **complete** when:

1. ✓ `BEFORE_STAGE0_CHECK > 0` (Stage-0 reached)
2. ✓ `transformer_forward_calls > 0` (Transformer called)
3. ✓ `entry_routing.selected_model == "v10_hybrid"` (if routing data available)
4. ✓ `v10_callsite.entered > 0` (V10 callsite entered)
5. ✓ `model_entry.attempts > 0` (Model entry attempted)

When all criteria are met:
- Entry evaluation flow is verified
- Ready for A/B tests (XGB ↔ Transformer)
- Can proceed to feature/model tuning

## Integration with Test Pipeline

### Example: After Replay Run

```bash
# Run replay
export GX1_REQUIRE_ENTRY_TELEMETRY=1
export GX1_REPLAY_USE_PREBUILT_FEATURES=1

python3 gx1/scripts/replay_eval_gated_parallel.py \
  --data <data_path> \
  --prebuilt-parquet <prebuilt_path> \
  --policy <policy_path> \
  --bundle-dir <bundle_dir> \
  --output-dir reports/replay_eval/run_20250108 \
  --workers 1

# Verify entry flow
python3 gx1/scripts/verify_entry_flow_gap.py reports/replay_eval/run_20250108

# Check exit code
if [ $? -eq 0 ]; then
    echo "Entry flow verified - ready for A/B tests"
elif [ $? -eq 10 ]; then
    echo "Exception caught - review and fix"
elif [ $? -eq 20 ]; then
    echo "Stage-0 reached but not fully verified"
elif [ $? -eq 30 ]; then
    echo "Unknown exit - investigate structural issue"
else
    echo "Insufficient evidence"
fi
```

## Troubleshooting

### Exception Not Caught

If `AFTER_SOFT_ELIGIBILITY_PASSED > 0` but `EXCEPTION_IN_SOFT_TO_STAGE0_GAP == 0`:
1. Check that try/except block is correctly placed
2. Check that exception doesn't occur in a subprocess
3. Check for `sys.exit()` calls

### Transformer Not Called

If `BEFORE_STAGE0_CHECK > 0` but `transformer_forward_calls == 0`:
1. Check Stage-0 filter (may block)
2. Check routing logic (may select legacy path)
3. Check model entry gates

### Routing Wrong

If `entry_routing.selected_model != "v10_hybrid"`:
1. Check V10 enable state
2. Check bundle loading
3. Check routing conditions

## Exit Codes Summary

| Exit Code | Status | Meaning |
|-----------|--------|---------|
| 0 | STAGE0_REACHED | All criteria met - ready for A/B tests |
| 10 | EXCEPTION_CAUGHT | Exception occurred - needs fix |
| 20 | STAGE0_REACHED_BUT_NOT_FULLY_VERIFIED | Stage-0 reached but criteria missing |
| 30 | UNKNOWN_EXIT | Structural control-flow issue |
| 40 | INSUFFICIENT_EVIDENCE | No telemetry or soft never passed |
