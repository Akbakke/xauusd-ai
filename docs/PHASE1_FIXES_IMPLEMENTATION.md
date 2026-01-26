# Phase 1 Fixes - Implementation Report
**Date:** 2026-01-18  
**Purpose:** Critical fixes before multiyear backtest (2020-2025)  
**Status:** ✅ **ALL FIXES IMPLEMENTED**

---

## SUMMARY

All 4 critical fixes from the sanity scan have been implemented with fail-fast semantics and proper telemetry integration.

---

## DEL 1: Wall-Clock Time Removal (DETERMINISM BLOCKER) ✅

### Files Changed
- `gx1/features/basic_v1.py`

### Implementation
1. **Added helper function** `_assert_valid_datetime_index()`:
   - Validates `df.index` is a `DatetimeIndex`
   - Validates `len(df.index) > 0`
   - Hard-fails in replay mode with clear error message
   - Hard-fails in live mode (treating as fatal, not silent fallback)

2. **Replaced all 4 instances** of `pd.Timestamp.now()` fallback:
   - Line 1114: H1 RSI z-score (stateful aligner)
   - Line 1165: H1 RSI z-score (legacy aligner)
   - Line 1260: H4 RSI z-score (stateful aligner)
   - Line 1294: H4 RSI z-score (legacy aligner)

### Code Changes
```python
# BEFORE:
current_m5_ts = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0 else pd.Timestamp.now(tz='UTC')

# AFTER:
current_m5_ts = _assert_valid_datetime_index(df, ctx="H1 RSI z-score (stateful aligner)")
```

### Verification
- ✅ No linter errors
- ✅ All 4 instances replaced
- ✅ Helper function properly validates and fails fast

---

## DEL 2: Exit Invariants (CORRECTNESS) ✅

### Files Changed
- `gx1/execution/oanda_demo_runner.py`
- `gx1/runtime/run_identity.py`

### Implementation
1. **Added tracking for exited trades**:
   - `_exited_trade_ids: set[str]` - tracks all exited trade IDs
   - `_exit_monotonicity_violations: int` - counter for exit_time < entry_time
   - `_duplicate_exit_attempts: int` - counter for duplicate exit attempts

2. **Added validation in `request_close()`**:
   - **Single-exit invariant**: Check if trade already in `_exited_trade_ids` before closing
   - **Monotonicity invariant**: Validate `exit_time >= entry_time` before closing
   - Hard-fail in replay mode on violations
   - Warn and ignore in live mode (but don't double-register PnL)

3. **Updated RUN_IDENTITY.json**:
   - Added `exit_monotonicity_violations` field
   - Added `duplicate_exit_attempts` field
   - Counters updated after replay completes

### Code Changes
```python
# In request_close():
# 1. Check single-exit invariant
if trade_id in self._exited_trade_ids:
    self._duplicate_exit_attempts += 1
    if is_replay:
        raise RuntimeError(f"[EXIT_INVARIANT_VIOLATION] Duplicate exit attempt...")

# 2. Check monotonicity invariant
if now_ts < entry_time:
    self._exit_monotonicity_violations += 1
    if is_replay:
        raise RuntimeError(f"[EXIT_INVARIANT_VIOLATION] Exit time < entry time...")

# 3. Mark as exited after successful close
self._exited_trade_ids.add(trade_id)
```

### Verification
- ✅ No linter errors
- ✅ Invariants enforced in canonical exit path (`request_close()`)
- ✅ Counters tracked and written to RUN_IDENTITY.json

---

## DEL 3: NaN/Inf Hard-Fail in Replay (DETERMINISM) ✅

### Files Changed
- `gx1/execution/entry_manager.py`
- `gx1/execution/oanda_demo_runner.py`
- `gx1/runtime/run_identity.py`

### Implementation
1. **Predictions (entry_manager.py)**:
   - Hard-fail in replay mode on first NaN/Inf prediction
   - In live mode: log warning, skip entry (treat as no-trade), increment counter
   - Added telemetry: `nan_inf_pred_count`, `nan_inf_pred_first_ts`

2. **XGB Features (oanda_demo_runner.py:6797)**:
   - Hard-fail in replay mode if XGB features contain NaN/Inf
   - In live mode: log warning (first 3 occurrences), convert with `nan_to_num`, increment counter

3. **Sequence Features (oanda_demo_runner.py:7671, 7676)**:
   - Hard-fail in replay mode if sequence features contain NaN/Inf (before and after scaling)
   - In live mode: log warning (first 3 occurrences), convert with `nan_to_num`, increment counter

4. **Updated RUN_IDENTITY.json**:
   - Added `nan_inf_pred_count` field
   - Added `nan_inf_pred_first_ts` field
   - Added `nan_inf_pred_mode_handling` field (always "FATAL_IN_REPLAY")

### Code Changes
```python
# In entry_manager.py:
elif not np.isfinite(v9_pred.prob_long) or not np.isfinite(v9_pred.prob_short):
    error_msg = f"[NAN_INF_PRED_FATAL] Prediction has NaN/Inf..."
    if is_replay:
        raise RuntimeError(error_msg)  # Hard-fail in replay
    else:
        log.warning(f"{error_msg} (live mode: skipping entry)")
        return None  # Skip entry in live mode

# In oanda_demo_runner.py (XGB features):
if is_replay:
    if not np.all(np.isfinite(xgb_features)):
        raise RuntimeError(f"[NAN_INF_INPUT_FATAL] XGB features contain NaN/Inf...")
```

### Verification
- ✅ No linter errors
- ✅ Hard-fail in replay mode for predictions and features
- ✅ Counters tracked and written to RUN_IDENTITY.json

---

## DEL 4: Temperature Scaling Re-Enabled (CALIBRATION) ✅

### Files Changed
- `gx1/execution/oanda_demo_runner.py`
- `gx1/runtime/run_identity.py`

### Implementation
1. **Re-enabled temperature scaling**:
   - Removed hardcoded `T = 1.0`
   - Restored: `T = float(temp_map.get(session_key, 1.0))`
   - Removed TODO and "TEMPORARY TEST" comments

2. **Made configurable via env var**:
   - `GX1_TEMPERATURE_SCALING=1` (default ON)
   - `GX1_TEMPERATURE_SCALING=0` (explicitly disable)
   - Logs clearly when disabled

3. **Added telemetry**:
   - `temperature_scaling_enabled` - whether scaling is enabled
   - `temperature_map_loaded` - whether temperature map exists
   - `temperature_defaults_used_count` - count of times default T=1.0 was used

4. **Updated RUN_IDENTITY.json**:
   - All temperature fields written after replay completes

### Code Changes
```python
# BEFORE:
# TEMPORARY TEST: Force T=1.0 for all sessions (disable temperature scaling)
# TODO: Remove this after diagnosis - restore: T = float(temp_map.get(session_key, 1.0))
T = 1.0

# AFTER:
temp_scaling_disabled = os.getenv("GX1_TEMPERATURE_SCALING", "1") == "0"
if temp_scaling_disabled:
    T = 1.0
    log.info(f"[TEMPERATURE] Scaling disabled via GX1_TEMPERATURE_SCALING=0")
else:
    T = float(temp_map.get(session_key, 1.0))
    if session_key not in temp_map:
        # Log warning once per session
        self._temp_defaults_used_count += 1
```

### Verification
- ✅ No linter errors
- ✅ Temperature scaling re-enabled (default ON)
- ✅ Configurable via env var
- ✅ Status tracked and written to RUN_IDENTITY.json

---

## FILES MODIFIED SUMMARY

1. **gx1/features/basic_v1.py**
   - Added `_assert_valid_datetime_index()` helper
   - Replaced 4 instances of `pd.Timestamp.now()` fallback

2. **gx1/execution/oanda_demo_runner.py**
   - Added exit invariant tracking and validation in `request_close()`
   - Added NaN/Inf validation for XGB and sequence features
   - Re-enabled temperature scaling with env var control
   - Updated RUN_IDENTITY.json after replay completes

3. **gx1/execution/entry_manager.py**
   - Added NaN/Inf hard-fail for predictions in replay mode
   - Added telemetry counters for NaN/Inf predictions

4. **gx1/runtime/run_identity.py**
   - Added `exit_monotonicity_violations` field
   - Added `duplicate_exit_attempts` field
   - Added `nan_inf_pred_count`, `nan_inf_pred_first_ts`, `nan_inf_pred_mode_handling` fields
   - Added `temperature_scaling_enabled`, `temperature_map_loaded`, `temperature_defaults_used_count` fields
   - Updated `create_run_identity()` to accept new parameters
   - Updated `load_run_identity()` to load new fields

5. **docs/GX1_SANITY_SCAN_REPORT.md**
   - Updated Section 1 to mark all 4 critical issues as FIXED
   - Updated GO/NO-GO checklist
   - Updated conclusion

---

## VERIFICATION CHECKLIST

### Pre-Smoke Test
- [x] ✅ All files modified without syntax errors
- [x] ✅ No linter errors
- [x] ✅ All 4 fixes implemented
- [x] ✅ RUN_IDENTITY.json fields added
- [x] ✅ Documentation updated

### Smoke Test (To Be Run)
- [ ] Run baseline replay smoke (2024, 1 worker, 2-7 days)
- [ ] Run COST smoke test (2020, COST_S12_A80, 1 worker, 2-7 days)
- [ ] Verify RUN_IDENTITY.json contains all new fields
- [ ] Verify no exit invariant violations
- [ ] Verify no NaN/Inf predictions or features
- [ ] Verify temperature scaling is active (unless disabled)

---

## NEXT STEPS

1. **Run smoke tests** to verify fixes work correctly
2. **If smoke tests pass**, proceed with full multiyear backtest (2020-2025)
3. **Monitor RUN_IDENTITY.json** for any violations or warnings

---

## NOTES

- All fixes follow fail-fast semantics: hard-fail in replay, warn/ignore in live
- All counters are tracked in RUN_IDENTITY.json for auditability
- No trading logic changed (only safety/validation improvements)
- No refactoring beyond what's necessary for fixes

---

**Implementation Complete:** 2026-01-18  
**Ready for Smoke Tests:** ✅ YES
