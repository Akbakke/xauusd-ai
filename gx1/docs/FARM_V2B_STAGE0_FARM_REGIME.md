# FARM_V2B Stage-0 FARM-Only Regime

## Overview

FARM_V2B now operates in "FARM-only" mode without requiring Big Brain V1 runtime, while maintaining strict regime control through a simple session + ATR-regime mapping.

## Problem Statement

Previously, when Big Brain V1 was unavailable:
- Regime detection defaulted to `UNKNOWN`
- Stage-0 filter blocked all entries with `UNKNOWN` regimes
- Result: **0 trades** for FARM_V2B, even though it has its own brutal guard

## Solution

### 1. FARM-Only Regime Module

**File:** `gx1/regime/farm_regime.py`

Simple mapping function that determines FARM scope based on:
- **Session ID**: ASIA, EU, US, OVERLAP
- **ATR Regime ID**: LOW, MEDIUM, HIGH, EXTREME (or numeric: 0, 1, 2, 3)

**Mapping:**
- `ASIA + LOW` → `FARM_ASIA_LOW` ✅
- `ASIA + MEDIUM` → `FARM_ASIA_MEDIUM` ✅
- All other combinations → `FARM_OUT_OF_SCOPE` ❌

### 2. Integration in Entry Manager

**File:** `gx1/execution/entry_manager.py`

When Big Brain V1 is disabled and FARM_V2B mode is detected:

1. **Detect FARM_V2B mode:**
   - Check `self.farm_v2b_mode` (set in `GX1DemoRunner.__init__`)
   - Check `entry_v9_policy_farm_v2b.enabled` in policy
   - Check if entry config path contains "FARM_V2B"

2. **Extract session and ATR regime:**
   - Session: From `infer_session_tag(current_ts)`
   - ATR regime: From `entry_bundle.features["_v1_atr_regime_id"]` or `atr_regime_id`
   - Map numeric IDs: 0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME

3. **Infer FARM regime:**
   - Call `infer_farm_regime(session, atr_regime)`
   - Store result in `policy_state["farm_regime"]`

4. **Set policy_state for Stage-0:**
   - If `FARM_ASIA_LOW` or `FARM_ASIA_MEDIUM`:
     - Set `brain_trend_regime = "TREND_UP"` (FARM assumes uptrend bias)
     - Set `brain_vol_regime = atr_regime` (LOW or MEDIUM)
     - Set `brain_risk_score = 0.5` (moderate risk)
   - If `FARM_OUT_OF_SCOPE`:
     - Set `brain_trend_regime = "UNKNOWN"`
     - Set `brain_vol_regime = "UNKNOWN"`
     - Let Stage-0 or brutal guard handle filtering

### 3. Stage-0 Behavior

**Modified behavior for FARM_V2B:**

- **Before:** Stage-0 blocked all entries when `trend=UNKNOWN` or `vol=UNKNOWN`
- **After:** For FARM_V2B with valid FARM regime (`FARM_ASIA_LOW` or `FARM_ASIA_MEDIUM`):
  - Stage-0 **does NOT block** based on UNKNOWN
  - Logs at DEBUG level: `"[STAGE_0] FARM_V2B mode with valid regime: skipping Stage-0 filter"`
  - Further filtering handled by:
    - `farm_brutal_guard_v2` (session + vol regime check)
    - FARM_V2B policy thresholds (`p_long`, `p_profitable`)

**For other policies:**
- Behavior unchanged
- If they expect Big Brain and it's unavailable, UNKNOWN still blocks

## Boot Logging

On first bar with FARM_V2B mode:
```
[BOOT] FARM_V2B mode detected: using FARM-only regime (session+ATR) instead of Big Brain. 
First bar: session=ASIA atr_regime=LOW -> farm_regime=FARM_ASIA_LOW
```

## Limitations

1. **Only ASIA + LOW/MEDIUM:**
   - FARM scope is intentionally narrow
   - Other sessions/regimes are `FARM_OUT_OF_SCOPE`
   - Brutal guard will still filter these

2. **No trend detection:**
   - FARM regime assumes uptrend bias (`TREND_UP`)
   - No actual trend analysis (Big Brain would provide this)
   - Acceptable for FARM's conservative scope

3. **ATR regime only:**
   - Uses ATR-based volatility classification
   - No complex regime modeling
   - Simple and deterministic

## Testing

After implementation, run:
```bash
bash scripts/active/run_replay.sh \
  gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_FULL.yaml \
  2025-08-04 \
  2025-09-30 \
  2
```

**Expected results:**
- ✅ > 0 trades (should match previous hygiene test volume)
- ✅ Exit reasons dominated by `FARM_EXIT_V2_RULES_A_*`
- ✅ No Stage-0 logs saying "UNKNOWN regime blocked" for FARM_V2B
- ✅ No new exceptions
- ✅ Boot log shows FARM regime detection

## Files Changed

1. **Created:**
   - `gx1/regime/__init__.py`
   - `gx1/regime/farm_regime.py`

2. **Modified:**
   - `gx1/execution/entry_manager.py`:
     - Added FARM_V2B detection logic
     - Added FARM regime inference when Big Brain is unavailable
     - Modified Stage-0 to skip blocking for FARM_V2B with valid regime

## Future Improvements

1. **Expand FARM scope** (if needed):
   - Add other session/regime combinations
   - Update `infer_farm_regime` mapping

2. **Add trend detection** (optional):
   - Simple trend filter based on price action
   - Not critical for FARM's conservative scope

3. **Regime persistence:**
   - Track regime changes over time
   - Add regime transition logging

