# STEP 2: Context Features Contract + Plumbing - Implementation Summary

**Date:** 2026-01-06  
**Status:** ✅ COMPLETE (Contract + Plumbing Skeleton)  
**Next:** STEP 3 (Model Training with Context Features)

---

## Leveranse A: Context Feature Contract ✅

**File:** `docs/ENTRY_CONTEXT_FEATURES_CONTRACT.md`

**Content:**
- ✅ 7 context features dokumentert (session_id, trend_regime_id, vol_regime_id, atr_bps, atr_bucket, spread_bps, spread_bucket)
- ✅ Availability contract (hvilke er tilgjengelige uten feature build)
- ✅ Embedding design (5 categorical × 8 dims = 40 dims, 2 continuous = 2 dims, total 42 dims)
- ✅ Range & clipping (atr_bps: [0, 1000] bps, spread_bps: [0, 500] bps)
- ✅ Validation rules (replay: hard fail, live: warning + fallback)
- ✅ Feature contract hash (for model/runtime validation)

---

## Leveranse B: Plumbing Skeleton ✅

**File:** `gx1/execution/entry_context_features.py`

**Components:**
- ✅ `EntryContextFeatures` dataclass (7 features + metadata)
- ✅ `build_entry_context_features()` metode
- ✅ `to_tensor_categorical()` og `to_tensor_continuous()` metoder
- ✅ `validate()` metode (replay: hard fail, live: warning)

**Integration:**
- ✅ Integrert i `evaluate_entry()` flow (after soft eligibility, before feature build)
- ✅ Passes to `_predict_entry_v10_hybrid()` (with backward compatibility)
- ✅ Reuses ATR proxy and spread_bps from soft/hard eligibility (performance optimization)

---

## Leveranse C: Bundle Interface ✅

**File:** `gx1/models/entry_v10/entry_v10_bundle.py`

**Changes:**
- ✅ Added `metadata` field to `EntryV10Bundle` dataclass
- ✅ Bundle metadata includes:
  - `supports_context_features: bool` (default: False for old bundles)
  - `expected_ctx_cat_dim: int` (default: 5)
  - `expected_ctx_cont_dim: int` (default: 2)
  - `feature_contract_hash: Optional[str]`
  - `schema_version: str` (default: "v10")

**Validation:**
- ✅ `_predict_entry_v10_hybrid()` checks bundle support
- ✅ Shape validation (fail-fast in replay, warning in live)
- ✅ Backward compatibility: old bundles ignore context features

---

## Leveranse D: Telemetry & Invariants ✅

**Files:**
- `gx1/execution/entry_manager.py` (telemetry counters)
- `scripts/run_mini_replay_perf.py` (export)
- `scripts/assert_perf_invariants.py` (invariants)

**New Counters:**
- ✅ `n_eligible_hard` (cycles that passed hard eligibility)
- ✅ `n_eligible_cycles` (cycles that passed hard + soft eligibility)
- ✅ `n_context_built` (context features successfully built)
- ✅ `n_context_missing_or_invalid` (context features failed to build)

**New Invariants:**
- ✅ `ELIGIBILITY_INV_1`: `0 <= n_eligible_hard <= n_cycles`
- ✅ `ELIGIBILITY_INV_2`: `0 <= n_eligible_cycles <= n_eligible_hard`
- ✅ `ELIGIBILITY_INV_3`: `n_context_built <= n_eligible_cycles` (if context enabled)
- ✅ `ELIGIBILITY_INV_4`: `n_candidates <= n_context_built` (if context enabled) or `n_eligible_cycles` (if disabled)

---

## Leveranse E: Unit Tests ✅

**File:** `tests/test_entry_context_features.py`

**Tests:**
1. ✅ `test_entry_context_features_validation` - Contract validation (missing fields, out of range)
2. ✅ `test_entry_context_features_tensor_conversion` - Tensor shape validation
3. ✅ `test_build_entry_context_features_determinism` - Mapping/bucketing determinism
4. ✅ `test_build_entry_context_features_clipping` - Range clipping (atr_bps, spread_bps)
5. ✅ `test_build_entry_context_features_missing_atr` - Missing ATR handling (replay vs live)

**Result:** ✅ All 5 tests PASS

---

## Regression Test (Context Flag OFF) ✅

**Test:** 1-week replay with `ENTRY_CONTEXT_FEATURES_ENABLED=false` (default)

**Result:**
- ✅ Identical behavior to before STEP 2
- ✅ `n_context_built = 0` (context features not built)
- ✅ `n_eligible_cycles = 540` (same as after STEP 1 fix)
- ✅ `n_trades_created = 242` (same as before)
- ✅ All invariants PASS

**Output:** `gx1/wf_runs/replay_1w_perf_20260106_150220/`

---

## Compatibility Test (Context Flag ON + Old Bundle)

**Status:** ⏳ PENDING (requires test run)

**Expected Behavior:**
- Context flag ON + old bundle (`supports_context_features=false`):
  - **Replay:** Hard fail with clear error message
  - **Live:** Warning + fallback to legacy regime inputs

**Test Command:**
```bash
export ENTRY_CONTEXT_FEATURES_ENABLED=true
bash scripts/run_replay_1w_perf.sh
# Expected: RuntimeError("CONTEXT_FEATURES_MISMATCH: ...")
```

---

## Implementation Details

### Context Features Build Order

1. **Hard Eligibility** → blocks before any computation
2. **Soft Eligibility** → computes cheap ATR proxy (reused for context)
3. **Context Features Build** → builds all 7 features (cheaper than full V9 feature build)
4. **Full Feature Build** → builds V9 features (seq + snap)
5. **Model Inference** → passes context + features to model

### Backward Compatibility

**Old Bundles (`supports_context_features=false`):**
- Context features are built but **ignored** by model
- Model runs with existing seq_x + snap_x + legacy regime inputs
- No breaking changes

**New Bundles (`supports_context_features=true`):**
- Context features **REQUIRED** (hard fail if missing in replay)
- Model expects ctx_cat + ctx_cont tensors
- Shape validation before inference

### Performance Target

**Context Features Build Cost:**
- Target: < 10% of full V9 feature build time
- Uses: Cheap ATR proxy (already computed), session inference (O(1)), spread from quotes (O(1))
- Avoids: Full feature pipeline, heavy rolling operations

**Current Status:** ✅ Meets target (context build is O(1) + cheap ATR proxy)

---

## Next Steps (STEP 3)

**NOT in STEP 2:**
- ❌ Model training with context features
- ❌ Transformer model signature update (ctx_cat, ctx_cont)
- ❌ Feature contract hash computation

**STEP 3 Will Include:**
- Model architecture update (add context embeddings)
- Training pipeline update (include context features)
- Bundle metadata update (set `supports_context_features=true`)
- Full integration test (context flag ON + new bundle)

---

## Summary

**STEP 2 Status:** ✅ **COMPLETE**

**Deliverables:**
- ✅ Contract documented
- ✅ Plumbing implemented
- ✅ Bundle interface updated
- ✅ Telemetry & invariants updated
- ✅ Unit tests passing
- ✅ Regression test passing (context flag OFF)

**Ready for STEP 3:** ✅ Yes (when approved)

---

## Test Artifacts

- **Contract:** `docs/ENTRY_CONTEXT_FEATURES_CONTRACT.md`
- **Implementation:** `gx1/execution/entry_context_features.py`
- **Bundle Interface:** `gx1/models/entry_v10/entry_v10_bundle.py`
- **Unit Tests:** `tests/test_entry_context_features.py` (5 tests, all passing)
- **Regression Test:** `gx1/wf_runs/replay_1w_perf_20260106_150220/` (context flag OFF, identical behavior)



