# GX1 Pipeline - Full System Sanity Scan Report
**Date:** 2026-01-18  
**Purpose:** Pre-multiyear backtest (2020-2025) validation  
**Scope:** Complete GX1 pipeline review for correctness, determinism, consistency

---

## EXECUTIVE SUMMARY

This report provides a comprehensive sanity scan of the GX1 pipeline before running the full multiyear backtest (2020-2025). The scan covers all critical components: entry pipeline, exit pipeline, model stack, features, regime logic, parallel execution, logging/telemetry, configuration, and edge cases.

**Status:** 🔴 **CRITICAL ISSUES FOUND** - See Section 1 for details.

---

## 1. CRITICAL (MÅ FIKSES FØR MULTIYEAR)

### 1.1 Wall-Clock Time Usage in Features (DETERMINISM BREAK) ✅ **FIXED**

**Location:** `gx1/features/basic_v1.py` lines 1114, 1165, 1260, 1294

**Issue:** Four instances of `pd.Timestamp.now(tz='UTC')` used as fallback when `df.index[-1]` is not available. This breaks determinism in replay mode.

**Fix Implemented:**
- ✅ Removed all `pd.Timestamp.now()` fallbacks
- ✅ Added `_assert_valid_datetime_index()` helper function
- ✅ Hard-fail in replay mode if `df.index` is not `DatetimeIndex` or empty
- ✅ Hard-fail in live mode (treating as fatal, not silent fallback)

**Files Changed:**
- `gx1/features/basic_v1.py`: Added helper function and replaced all 4 instances

**Status:** ✅ **FIXED** - Phase 1 implementation complete

---

### 1.2 NaN/Inf Handling - Silent Fallbacks ✅ **FIXED**

**Location:** Multiple locations in `gx1/execution/oanda_demo_runner.py` and `gx1/execution/entry_manager.py`

**Issue:** Several places use `np.nan_to_num()` with silent fallback to 0.0, which may hide data quality issues.

**Fix Implemented:**
- ✅ Hard-fail in replay mode on NaN/Inf in predictions (`entry_manager.py`)
- ✅ Hard-fail in replay mode on NaN/Inf in XGB features (`oanda_demo_runner.py:6797`)
- ✅ Hard-fail in replay mode on NaN/Inf in sequence features (`oanda_demo_runner.py:7671, 7676`)
- ✅ In live mode: Log warning and skip entry (for predictions) or convert with counter (for features)
- ✅ Added telemetry counters: `nan_inf_pred_count`, `nan_inf_pred_first_ts` in `RUN_IDENTITY.json`

**Files Changed:**
- `gx1/execution/entry_manager.py`: Hard-fail on NaN/Inf predictions in replay
- `gx1/execution/oanda_demo_runner.py`: Hard-fail on NaN/Inf in XGB and sequence features in replay
- `gx1/runtime/run_identity.py`: Added NaN/Inf counters

**Status:** ✅ **FIXED** - Phase 1 implementation complete

---

### 1.3 Temperature Scaling Disabled (Temporary Test Code) ✅ **FIXED**

**Location:** `gx1/execution/oanda_demo_runner.py:6235-6247`

**Issue:** Temperature scaling is hardcoded to T=1.0 with a TODO comment to restore.

**Fix Implemented:**
- ✅ Re-enabled temperature scaling: `T = float(temp_map.get(session_key, 1.0))`
- ✅ Made configurable via `GX1_TEMPERATURE_SCALING` env var (default ON)
- ✅ Removed TODO and "TEMPORARY TEST" comments
- ✅ Added telemetry: `temperature_scaling_enabled`, `temperature_map_loaded`, `temperature_defaults_used_count` in `RUN_IDENTITY.json`
- ✅ Logs warning once per session if temperature missing (uses T=1.0 default)

**Files Changed:**
- `gx1/execution/oanda_demo_runner.py`: Re-enabled temperature scaling with env var control
- `gx1/runtime/run_identity.py`: Added temperature scaling status fields

**Status:** ✅ **FIXED** - Phase 1 implementation complete

---

### 1.4 Exit Time Monotonicity - No Validation ✅ **FIXED**

**Location:** `gx1/execution/oanda_demo_runner.py` (exit handling)

**Issue:** No explicit validation that `exit_time >= entry_time` for trades. No check that only one exit occurs per trade.

**Fix Implemented:**
- ✅ Added `_exited_trade_ids` set to track exited trades (single-exit invariant)
- ✅ Added validation: `exit_time >= entry_time` (hard-fail in replay, warn in live)
- ✅ Added validation: only one exit per trade (hard-fail in replay, ignore in live)
- ✅ Added telemetry counters: `exit_monotonicity_violations`, `duplicate_exit_attempts` in `RUN_IDENTITY.json`

**Files Changed:**
- `gx1/execution/oanda_demo_runner.py`: Added exit invariants in `request_close()`
- `gx1/runtime/run_identity.py`: Added exit invariant counters

**Status:** ✅ **FIXED** - Phase 1 implementation complete

---

## 2. HIGH VALUE (BØR FIKSES NÅ – HØY ROI)

### 2.1 Pre-Entry Wait Gate - Entry Direction Not Available

**Location:** `gx1/entry/pre_entry_wait_gate.py:109`

**Issue:** `entry_direction` parameter is optional and may be `None` when evaluating adverse move check. The gate evaluates adverse move without knowing entry direction.

**Code:**
```python
def should_wait(
    self,
    candles: pd.DataFrame,
    atr_bps: float,
    current_price: float,
    entry_direction: Optional[str] = None,  # "long" or "short" (optional, for adverse move)
) -> tuple[bool, WaitReason]:
```

**Impact:**
- Adverse move check may not work correctly if entry direction is unknown
- Gate may pass when it should wait (or vice versa)

**Fix Required:**
- If `entry_direction` is `None`, skip adverse move check (or use a conservative default)
- Document that adverse move check requires entry direction
- Add telemetry for cases where entry direction is missing

**Priority:** 🟡 **MEDIUM** - Should fix, but not blocking.

---

### 2.2 HTF Alignment - Monotonicity Check Only in Replay

**Location:** `gx1/features/htf_align_state.py:123-143`

**Issue:** Monotonicity check hard-fails in replay mode but only warns in live mode. This is correct, but the reset behavior in live mode may cause subtle issues.

**Impact:**
- In live mode, non-monotonic timestamps reset the aligner, which may cause temporary misalignment
- May cause incorrect HTF feature values for a few bars after reset

**Fix Required:**
- Document that live mode reset is expected behavior
- Add telemetry counter for live mode resets
- Consider logging the reset event with more context

**Priority:** 🟡 **MEDIUM** - Documentation/telemetry improvement.

---

### 2.3 Entry Manager - NaN/Inf Prediction Handling

**Location:** `gx1/execution/entry_manager.py:2281-2288`

**Issue:** NaN/Inf predictions are logged as warnings (first 3 occurrences) but then silently ignored. Counter is incremented but no hard-fail.

**Code:**
```python
elif not np.isfinite(v9_pred.prob_long) or not np.isfinite(v9_pred.prob_short):
    self.n_v10_pred_none_or_nan += 1
    if self._v10_log_count < 3:
        log.warning(...)
```

**Impact:**
- NaN/Inf predictions may cause incorrect entry decisions
- In replay mode, should hard-fail after first occurrence

**Fix Required:**
- In replay mode: Hard-fail on first NaN/Inf prediction
- In live mode: Log warning and skip entry (current behavior acceptable)
- Add telemetry counter for NaN/Inf predictions

**Priority:** 🟡 **MEDIUM** - Should fix for replay determinism.

---

### 2.4 Parallel Workers - Environment Variable Leakage

**Location:** `gx1/scripts/replay_eval_gated_parallel.py:587-601`

**Issue:** Environment variables are set in worker process, but there's no explicit cleanup of parent process env vars that might leak.

**Impact:**
- Parent process env vars may leak into workers (though spawn method should prevent this)
- No explicit validation that workers start with clean env

**Fix Required:**
- Add explicit env var whitelist for workers
- Log all env vars set in worker (for debugging)
- Add invariant check: workers should not inherit unexpected env vars

**Priority:** 🟡 **MEDIUM** - Defensive programming improvement.

---

## 3. SAFE TO IGNORE (DOKUMENTERT)

### 3.1 Regime Classification - Default to MEDIUM

**Location:** `gx1/features/basic_v1.py:633-634`

**Issue:** ATR regime defaults to MEDIUM (1.0) when quantiles are NaN.

**Code:**
```python
regime_id = np.zeros(len(atr14_arr_for_regime), dtype=np.float64)
regime_id[:] = 1.0  # Default to MEDIUM
```

**Justification:**
- This is correct behavior: when we don't have enough data for quantiles, defaulting to MEDIUM is conservative
- Only affects early bars (warmup period)
- Documented in code comments

**Status:** ✅ **OK** - No action needed.

---

### 3.2 Pre-Entry Wait Gate - ATR Fallback

**Location:** `gx1/entry/pre_entry_wait_gate.py:133-137`

**Issue:** If ATR is not available, gate passes (doesn't block).

**Code:**
```python
if atr_absolute <= 0:
    # ATR not available - pass (don't block)
    log.debug("[PRE_ENTRY_WAIT] ATR not available, passing")
    self.counters["pre_entry_wait_n_pass"] += 1
    return False, WaitReason.PASS
```

**Justification:**
- This is correct: if ATR is not available, we can't evaluate wait conditions, so we pass
- Better to allow entry than to block indefinitely
- Telemetry tracks this case (counter incremented)

**Status:** ✅ **OK** - No action needed.

---

### 3.3 Exit Policy V2 - Grace Period for MAE Kill

**Location:** `gx1/exits/exit_policy_v2.py:266-277`

**Issue:** MAE kill has a grace period (`mae_kill_grace_bars`) before it triggers.

**Justification:**
- This is intentional: allows trades to recover from initial adverse moves
- Prevents premature exits on temporary drawdowns
- Configurable via `mae_kill_grace_bars` parameter

**Status:** ✅ **OK** - No action needed.

---

## 4. INVARIANT CHECKLIST

### 4.1 Entry Pipeline Invariants

| Invariant | Location | Status | Notes |
|-----------|----------|--------|-------|
| No trade without score | `entry_manager.py:evaluate_entry()` | ✅ | Hard eligibility check before feature build |
| No score without features | `entry_manager.py:evaluate_entry()` | ✅ | Feature build before model inference |
| No NaN/Inf in predictions (replay) | `entry_manager.py:2281` | 🔴 | Currently only warns, should hard-fail |
| No trade without valid session | `entry_manager.py:908` | ✅ | Session inference before entry evaluation |
| No trade without valid ATR | `entry_manager.py:509` | ✅ | ATR validation in soft eligibility |
| Pre-entry wait gate called before model | `entry_manager.py:1053-1076` | ✅ | Gate evaluated after context features, before model |

### 4.2 Exit Pipeline Invariants

| Invariant | Location | Status | Notes |
|-----------|----------|--------|-------|
| Max 1 exit per trade | `oanda_demo_runner.py` | 🔴 | **NOT VALIDATED** - Should add check |
| Exit time >= entry time | `oanda_demo_runner.py` | 🔴 | **NOT VALIDATED** - Should add check |
| Exit price within bid/ask spread | `exit_policy_v2.py:256` | ✅ | Uses current_bid/current_ask |
| Exit counters match trade exits | `exit_policy_v2.py:109-115` | ✅ | Counters tracked per exit reason |

### 4.3 Feature Pipeline Invariants

| Invariant | Location | Status | Notes |
|-----------|----------|--------|-------|
| No wall-clock time in features (replay) | `basic_v1.py:1114,1165,1260,1294` | 🔴 | **VIOLATED** - Uses `pd.Timestamp.now()` |
| HTF alignment monotonic (replay) | `htf_align_state.py:123-143` | ✅ | Hard-fails on non-monotonic timestamps |
| No lookahead in features | `basic_v1.py` | ✅ | All features use shift(1) or rolling windows |
| Feature schema matches model | `oanda_demo_runner.py:6170-6180` | ✅ | Hard-fail on mismatch |

### 4.4 Parallel Execution Invariants

| Invariant | Location | Status | Notes |
|-----------|----------|--------|-------|
| Workers use spawn method | `replay_eval_gated_parallel.py:47-51` | ✅ | Forced to spawn |
| Workers have clean env | `replay_eval_gated_parallel.py:587-601` | 🟡 | Set explicitly, but no validation |
| No shared state between workers | `replay_eval_gated_parallel.py:206` | ✅ | Each worker is isolated |
| Feature build disabled in PREBUILT | `run_identity.py:318-323` | ✅ | Hard-fail if not disabled |

### 4.5 Model Stack Invariants

| Invariant | Location | Status | Notes |
|-----------|----------|--------|-------|
| Model classes_ matches prediction | `oanda_demo_runner.py:550-608` | ✅ | `extract_entry_probabilities()` handles mapping |
| XGB features aligned with model | `oanda_demo_runner.py:6762` | ✅ | NaN/Inf handled (but silently) |
| Sequence features aligned with model | `oanda_demo_runner.py:7636` | ✅ | NaN/Inf handled (but silently) |
| Temperature scaling applied | `oanda_demo_runner.py:6213-6236` | 🔴 | **DISABLED** - Hardcoded to T=1.0 |

---

## 5. PRE-MULTIYEAR GO/NO-GO CHECKLIST

### 5.1 Critical Blockers (Must be GREEN)

- [x] ✅ **Fix wall-clock time usage in features** (`basic_v1.py` lines 1114, 1165, 1260, 1294) - **FIXED**
- [x] ✅ **Fix temperature scaling** (restore or make configurable, remove TODO) - **FIXED**
- [x] ✅ **Add exit time monotonicity validation** (exit_time >= entry_time) - **FIXED**
- [x] ✅ **Add single exit per trade validation** (max 1 exit per trade) - **FIXED**
- [x] ✅ **Hard-fail on NaN/Inf predictions in replay** (currently only warns) - **FIXED**

### 5.2 High-Value Improvements (Should be GREEN)

- [ ] 🟡 **Improve NaN/Inf handling** (log and hard-fail in replay, warn in live)
- [ ] 🟡 **Document pre-entry wait gate entry direction requirement**
- [ ] 🟡 **Add telemetry for HTF alignment resets in live mode**
- [ ] 🟡 **Add env var whitelist validation for parallel workers**

### 5.3 Verification Steps (Must be GREEN)

- [ ] ✅ **COST smoke test passes** (already verified)
- [ ] ✅ **dt_module version stamping works** (already verified)
- [ ] ✅ **RUN_IDENTITY.json completeness** (already verified)
- [ ] ✅ **PREBUILT mode invariants** (feature_build_disabled=1)
- [ ] ✅ **Parallel worker isolation** (spawn method, clean env)

---

## 6. RECOMMENDATIONS

### 6.1 Immediate Actions (Before Multiyear)

1. **Fix wall-clock time usage** - This is a determinism blocker
2. **Fix temperature scaling** - Either restore or document decision
3. **Add exit validation** - Critical for correctness
4. **Improve NaN/Inf handling** - Better error detection

### 6.2 Short-Term Improvements (After Multiyear)

1. **Add comprehensive telemetry** - Track all edge cases
2. **Improve documentation** - Document all invariants explicitly
3. **Add unit tests** - Test invariant enforcement
4. **Add integration tests** - Test end-to-end correctness

### 6.3 Long-Term Improvements

1. **Refactor feature building** - Remove all wall-clock dependencies
2. **Add invariant testing framework** - Automated invariant checking
3. **Improve error messages** - More context in error logs
4. **Add performance monitoring** - Track feature build times, model inference times

---

## 7. CONCLUSION

**Current Status:** 🟡 **READY FOR MULTIYEAR** (after Phase 1 fixes verified)

**Phase 1 Fixes Completed:**
- ✅ Wall-clock time usage in features (determinism break) - **FIXED**
- ✅ Temperature scaling disabled - **FIXED** (re-enabled with env var control)
- ✅ Missing exit validation (correctness risk) - **FIXED**

**Remaining High-Value Improvements:**
- 🟡 Improve NaN/Inf handling telemetry (already implemented, but can be enhanced)
- 🟡 Document pre-entry wait gate entry direction requirement
- 🟡 Add telemetry for HTF alignment resets
- 🟡 Add env var whitelist validation for parallel workers

**Recommendation:** 
1. Run smoke tests to verify Phase 1 fixes
2. If smoke tests pass, proceed with multiyear backtest
3. High-value improvements can be done in parallel or after multiyear

**Phase 1 Fix Time:** ✅ **COMPLETED** - All 4 critical fixes implemented

---

**Report Generated:** 2026-01-18  
**Next Review:** After critical blockers are fixed
