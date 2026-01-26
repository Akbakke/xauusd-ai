# ROOT CAUSE ANALYSIS: FULLYEAR 2025 - 0 TRADES & CHUNK FAILURES

**Date:** 2026-01-04  
**Run:** FULLYEAR_2025_20260104_172047  
**Analysis:** Based on chunk summaries, logs, and config inspection

---

## PROBLEM A: 0 TRADES

### Observation from Chunk 2 Log:
```
[ERROR] [ENTRY] require_v9_for_entry=False but neither V9 nor V10 enabled. No entry possible.
[SNIPER_CYCLE] ... reason=no_entry_model decision=NO-TRADE
```

### Root Cause:
**CONFIG-LOGIC MISMATCH (Bug Type: b - Logical Bug)**

Config file `ENTRY_V10_1_SNIPER_FLAT_THRESHOLD_0_18.yaml` specifies:
- `entry_models.v10.enabled: true`
- `require_v9_for_entry: false`

But `GX1DemoRunner.__init__` in `oanda_demo_runner.py` (lines 2044-2119) **only loads ENTRY_V9**, not ENTRY_V10.

**Evidence:**
- Runner checks `entry_models_cfg.get("v9", {})` only
- No code path that reads `entry_models_cfg.get("v10", {})`
- `self.entry_v10_enabled` is never set to `True`
- EntryManager checks `if not self.entry_v9_enabled and not (hasattr(self._runner, "entry_v10_enabled") and self._runner.entry_v10_enabled)` → always fails when `require_v9_for_entry=False`

**Entry Counters:** All 0 because `evaluate_entry` returns `None` immediately (line 1252 in entry_manager.py) before any cycles/candidates are counted.

### Minimal Fix:
**Option 1 (Fastest test):** Temporarily set `require_v9_for_entry: true` in config to use V9 instead of V10.

**Option 2 (Proper fix):** Implement V10 loading in `GX1DemoRunner.__init__` similar to V9 loading (check `entry_models.v10.enabled`, load transformer model from `entry_models.v10.model_path`, set `self.entry_v10_enabled=True`).

---

## PROBLEM B: CHUNK FAILURES

### Chunks 0/3/5: Hard Crash (No Summary Files)

**Evidence:**
- No `REPLAY_PERF_SUMMARY.json` in chunks 0/3/5
- No `fault.log` files (faulthandler never activated = crash before Python exception handling)
- Only `WINDOW.json` exists (written before replay starts)

**Classification:** **Segfault or OOM before Python exception handling**

**Root Cause:** Likely NumPy threading deadlock (same as chunk 2, but crashes harder).

### Chunk 2: Incomplete (7,834/9,931 bars, feat_time=0.00s)

**Evidence from log:**
```
Timeout (0:00:30)!
Thread 0x0000000209bfa0c0 (most recent call first):
  File "gx1/features/rolling_np.py", line 133 in rolling_mean_w48
  File "gx1/features/basic_v1.py", line 273 in build_basic_v1
```

**Classification:** **Deadlock in NumPy `rolling_mean_w48`**

**Root Cause:**
- NumPy operation hangs in `rolling_mean_w48` (Numba-accelerated rolling mean)
- Timeout after 30 seconds (faulthandler dump_traceback_later)
- Exception: `'NoneType' object cannot be interpreted as an integer` (likely from exception handling code trying to process failed NumPy operation)

**Why feat_time_sec=0.00s:** Exception occurs during feature building (before perf instrumentation completes its first cycle), so `perf_feat_time` never accumulates.

### Minimal Fix:
**Option 1:** Set `OMP_NUM_THREADS=1` in FULLYEAR script (already done, but verify it's actually exported to child processes).

**Option 2:** If still hangs, wrap `rolling_mean_w48` call with timeout or disable Numba for that specific function temporarily.

**Option 3:** Reduce number of workers from 7 → 4 to reduce contention (empirical fix).

---

## PROBLEM C: VALIDITY CHECK

### Config Comparison:

**Current Config (SNIPER_V10_1_THRESHOLD_0_18):**
- `entry_models.v10.enabled: true` ← **NOT LOADED BY RUNNER**
- `require_v9_for_entry: false` ← **CAUSES FAILURE**
- `entry_gating.p_side_min.long: 0.18` ← Threshold would be fine if models loaded
- `entry_models.v9.enabled: false` ← Intentionally disabled

**Expected Config for ~2300 trades (from historical analysis):**
- `entry_models.v9.enabled: true` (V9 model loaded)
- OR V10 loading implemented
- Thresholds typically 0.55-0.72 for p_long

**Critical Deviation:** Config requires V10 but runner cannot load V10 → **no entry model available → 0 trades**.

---

## CONSOLIDATED ROOT CAUSE SUMMARY

### 0 Trades:
**"Trades=0 fordi ENTRY_V10 model ikke lastes av runner selv om config spesifiserer `entry_models.v10.enabled: true`. EntryManager returnerer `None` ved linje 1252 i `entry_manager.py` fordi `self.entry_v10_enabled` aldri settes til `True`."**

**Fix:** Implementer V10 loading i `GX1DemoRunner.__init__` ELLER endre config til `require_v9_for_entry: true`.

### Chunk Failures:
**"Chunks 0/3/5 krasjer hardt (segfault/OOM) før faulthandler aktiveres. Chunk 2 henger i NumPy `rolling_mean_w48` deadlock pga threading."**

**Fix:** Verifiser at `OMP_NUM_THREADS=1` faktisk eksporteres til worker-prosesser, reduser workers fra 7→4, eller wrap NumPy calls med timeout.

---

## EXACT COMMAND FOR CONFIRMATION TEST

```bash
# Test 1: Fix V10 loading issue (use V9 instead as quick test)
# Edit: gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/ENTRY_V10_1_SNIPER_FLAT_THRESHOLD_0_18.yaml
# Change: require_v9_for_entry: true
# Change: entry_models.v9.enabled: true
# Change: entry_models.v10.enabled: false

# Then run 1-week replay to verify trades > 0:
bash scripts/run_replay_1w_perf.sh

# Expected change:
# - n_entry_candidates > 0
# - n_entry_accepted > 0  
# - trades_total > 0
```

---

## MINIMAL FIX PRIORITY

1. **IMMEDIATE (Test):** Set `require_v9_for_entry: true` + `entry_models.v9.enabled: true` in config → verify trades > 0.
2. **PROPER:** Implement V10 loading in `GX1DemoRunner.__init__` (read `entry_models.v10.*`, load transformer, set `entry_v10_enabled=True`).
3. **CHUNK FAILURES:** Verify thread limits, reduce workers to 4, add timeout wrapper for NumPy calls.


