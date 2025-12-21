# REPLAY UNKNOWN Analysis

**Date:** 2025-12-17  
**Problem:** SNIPER replay gives 0 trades because `trend=UNKNOWN` and `vol=UNKNOWN`  
**Root Cause:** SNIPER path doesn't compute trend/vol tags in replay mode

---

## Where UNKNOWN is Set

### 1. `gx1/execution/entry_manager.py` - Main Entry Manager

**Location:** Lines 280-484

**Flow:**
1. **Big Brain V1 path (lines 280-319):**
   - If Big Brain V1 is enabled and has enough bars → computes trend/vol
   - If error or not enough bars → sets `UNKNOWN`
   - **Condition:** `hasattr(self, "big_brain_v1") and self.big_brain_v1 is not None`

2. **FARM_V2B path (lines 320-450):**
   - If `is_farm_v2b == True` → computes ATR regime from features or candles fallback
   - Sets `brain_trend_regime` and `brain_vol_regime` based on FARM regime
   - **REPLAY FIX (lines 359-414):** If ATR regime is UNKNOWN in replay, computes from candles
   - **Condition:** `is_farm_v2b == True` (determined by config check)

3. **Fallback path (lines 482-484):**
   - If NOT FARM_V2B → sets `UNKNOWN`
   - **Problem:** SNIPER falls through here because `is_farm_v2b == False`

**Key Functions:**
- `infer_farm_regime(current_session, atr_regime_id)` - computes FARM regime
- `infer_session_tag(current_ts)` - computes session tag
- ATR regime fallback (lines 361-414) - computes ATR from candles in replay

### 2. `gx1/execution/entry_manager.py` - Feature Building

**Location:** Lines 900-955

**Flow:**
- Adds `session` column if missing (lines 906-922)
- Adds `atr_regime`/`vol_regime` column if missing (lines 924-934)
- Adds `trend_regime` column if missing (lines 936-955)
- **Problem:** These are added to `current_row` DataFrame, but `policy_state` dict (used by guards) may not have them

**Key Functions:**
- `infer_session_tag(ts)` - computes session from timestamp
- Maps `_v1_atr_regime_id` → `atr_regime` (LOW/MEDIUM/HIGH/EXTREME)
- Maps `trend_regime_tf24h` → `trend_regime` (TREND_UP/TREND_DOWN/TREND_NEUTRAL)

### 3. `gx1/policy/farm_guards.py` - Guard Functions

**Location:** `sniper_guard_v1()` and `farm_brutal_guard_v2()`

**Flow:**
- Guards check `row["session"]` and `row["vol_regime"]` or `row["atr_regime"]`
- **Problem:** If these are UNKNOWN, guard fails with AssertionError
- Guards are called BEFORE policy evaluation (lines 795-802 for SNIPER)

---

## Conditions for UNKNOWN

### Trend/Vol UNKNOWN when:
1. **Not FARM_V2B mode** → falls through to line 482-484
2. **Big Brain V1 disabled or error** → sets UNKNOWN (lines 306-318)
3. **Not enough bars for Big Brain** → sets UNKNOWN (line 316-318)
4. **ATR regime missing from features** → sets UNKNOWN (lines 333, 344, 347, 354, 357)
5. **Replay mode without candle fallback** → stays UNKNOWN (if fallback fails)

### Session UNKNOWN when:
1. **`session_id` column missing** → fills with UNKNOWN (line 911)
2. **`_v1_session_tag` missing** → no fallback
3. **Timestamp inference fails** → no fallback

---

## Current Fix (FARM_V2B only)

**Location:** Lines 359-414

**What it does:**
- If `atr_regime_id == "UNKNOWN"` and `replay_mode == True`
- Computes ATR14 from candles using rolling window
- Maps to regime: LOW/MEDIUM/HIGH based on percentile

**Limitation:**
- Only works for FARM_V2B mode (`is_farm_v2b == True`)
- SNIPER doesn't use FARM_V2B mode, so this fix doesn't apply

---

## Required Fix

**Goal:** Ensure trend/vol/session tags are set BEFORE guards run, for BOTH FARM and SNIPER

**Approach:**
1. Create `ensure_replay_tags()` function that computes tags from candles/history
2. Call this BEFORE guards (before line 795 for SNIPER, before line 966 for FARM)
3. Ensure tags are set in both `policy_state` dict AND `current_row` DataFrame
4. Make it backward-compatible: only set if missing/UNKNOWN

---

*Last Updated: 2025-12-17*

