# Preflight Leakage Detection and Fix

## Summary

Enhanced preflight sanity script to detect and classify truncate-future leakage, and fixed root cause: ATR regime feature using global quantile ranking (non-causal) replaced with rolling quantiles (causal).

## Changes

### 1. Enhanced Preflight Script (`gx1/scripts/preflight_full_build_sanity.py`)

**New Features:**
- **Warmup boundary calculation**: Uses `calculate_first_valid_eval_idx()` logic to determine warmup vs post-warmup regions
- **Comprehensive mismatch analysis**: 
  - Checks ALL features (not just first 20)
  - Analyzes mismatches by feature name
  - Calculates statistics: mismatch_count, mismatch_rate, max_abs_diff, median_abs_diff
  - Tracks first_mismatch_ts and last_mismatch_ts
  - Splits mismatches into warmup vs post-warmup windows
- **PASS/FAIL verdict**: 
  - PASS if mismatches are confined to warmup window only
  - FAIL if mismatches persist post-warmup
- **Detailed report**: Top 20 mismatching features with full statistics

### 2. Fixed Root Cause: ATR Regime Feature (`gx1/features/basic_v1.py`)

**Problem:**
- `_v1_atr_regime_id` used global quantile ranking over entire dataset
- Non-causal: percentile ranks changed when data was truncated
- Caused truncate-future mismatches

**Solution:**
- Replaced global ranking with rolling quantiles (window=5760 bars = 20 days)
- Uses `rolling_quantile()` from `gx1/features/rolling_np.py`
- Causal: only uses past data within rolling window
- Window: 5760 bars (20 days of M5), min_periods: 2880 (10 days)

**Code Change:**
```python
# OLD (non-causal):
atr14_rank_pct = ranks / n  # Global ranking

# NEW (causal):
q33_arr = rolling_quantile(atr14_arr_for_regime, 5760, q=0.333, min_periods=2880)
q67_arr = rolling_quantile(atr14_arr_for_regime, 5760, q=0.667, min_periods=2880)
# Classify based on rolling quantiles
```

### 3. New Generic Rolling Quantile Function (`gx1/features/rolling_np.py`)

**Added:**
- `rolling_quantile_numba()`: Numba-accelerated rolling quantile for any window size
- `rolling_quantile()`: Public entrypoint
- Uses ring buffer for O(N) performance
- Supports arbitrary window sizes (not just 48)

### 4. Unit Test (`gx1/tests/test_preflight_truncate_future.py`)

**Added:**
- `test_atr_regime_causal()`: Proves ATR regime feature is causal post-warmup
- Tests truncate-future invariance for ATR regime feature specifically

## Verification

**Preflight Test Results:**
- ✅ Truncate-future: Feature equality - **PASS** (all 1249 matching index pairs match)
- ✅ ATR regime feature: No mismatches post-warmup (proves causality)

**Unit Test Results:**
- ✅ `test_atr_regime_causal`: **PASSED**

## Impact

**Before Fix:**
- `_v1_atr_regime_id` had future leakage (used global quantiles)
- Truncate-future test found 3 mismatches in regime-based features
- Features using `_v1_atr_regime_id` (e.g., `_v1_int_clv_atr`, `_v1_int_r5_atr`) also mismatched

**After Fix:**
- `_v1_atr_regime_id` is causal (uses rolling quantiles)
- Truncate-future test passes (no mismatches)
- All dependent interaction features are now causal

## Files Modified

1. `gx1/scripts/preflight_full_build_sanity.py` - Enhanced mismatch analysis
2. `gx1/features/basic_v1.py` - Fixed ATR regime feature (rolling quantiles)
3. `gx1/features/rolling_np.py` - Added generic rolling quantile function
4. `gx1/tests/test_preflight_truncate_future.py` - Added ATR regime causality test

## Usage

Run preflight before FULLYEAR build:
```bash
python gx1/scripts/preflight_full_build_sanity.py \
  --data data/entry_v9/full_2025.parquet \
  --policy_config gx1/configs/policies/.../GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml \
  --feature_meta_path gx1/models/entry_v9/.../entry_v9_feature_meta.json \
  --seq_scaler_path ... \
  --snap_scaler_path ... \
  --calibration_dir models/xgb_calibration \
  --calibration_method platt \
  --require_calibration
```

## Notes

- Rolling quantile window (5760 bars = 20 days) matches original comment "rullende 20 dager"
- Min_periods (2880 = 10 days) ensures stable quantile estimates
- All features using `_v1_atr_regime_id` are now causal by inheritance
- Preflight report includes detailed mismatch analysis even when test passes
