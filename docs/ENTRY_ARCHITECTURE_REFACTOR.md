# Entry Architecture Refactor Plan

**Date:** 2026-01-06  
**Goal:** Færre, bedre entry-kandidater. Markedsregimer skal være INPUT til modellen, ikke etter-gating.

---

## Current State Analysis

### Current Gating Flow (entry_manager.py)

**Stage-0 (Precheck - before feature build):**
- Warmup check
- Session check (ASIA/EU/US/OVERLAP allowed)
- Vol regime check (EXTREME blocked)
- Spread check (hard cap)
- Kill-switch check
- Model missing check
- NaN features check

**Stage-1 (After prediction - before trade):**
- Threshold check (p_long >= threshold)
- Risk guard check
- Max concurrent trades
- Big Brain veto

### Current Context Features (used as gates):
- `session` → gated (ASIA/EU/US/OVERLAP)
- `vol_regime` → gated (EXTREME blocked)
- `trend_regime` → not gated (but used in policy)
- `atr_bps` / `atr_regime` → gated (EXTREME blocked)
- `spread_bps` → gated (hard cap)

---

## Target Architecture

### A) HARD ELIGIBILITY (before feature build, before inference)
**Purpose:** Stop immediately if conditions are unsafe/illegal

**Gates:**
- `session not allowed` → STOP (hard eligibility)
- `vol_regime == EXTREME` → STOP (hard eligibility)
- `spread > hard cap` → STOP (hard eligibility)
- `kill-switch` → STOP (hard eligibility)
- `warmup not ready` → STOP (hard eligibility)

**Implementation:**
- Move to `_check_hard_eligibility()` method
- Called BEFORE `build_live_entry_features()`
- Returns `(eligible: bool, reason: str)`
- Increments `veto_pre_*` counters (not `veto_cand_*`)

### B) CONTEXT FEATURES (input to model)
**Purpose:** Let model learn from market context

**Features to add:**
- `trend_regime_id` (categorical: 0=DOWN, 1=NEUTRAL, 2=UP)
- `vol_regime_id` (categorical: 0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME)
- `atr_bucket` / `atr_bps` (continuous)
- `session_id` (categorical: 0=ASIA, 1=EU, 2=US, 3=OVERLAP)
- `spread_bucket` (categorical or continuous)

**Implementation:**
- Add to `feature_meta` in `build_live_entry_features()`
- Include in sequence features (for transformer)
- Use as embeddings (categorical features)
- Model sees these as INPUT, not gates

### C) POST-MODEL SANITY (minimal)
**Purpose:** Final safety checks after model prediction

**Gates:**
- `max_concurrent_positions` → VETO (post-model)
- `catastrophic account risk` → VETO (post-model)

**Implementation:**
- Keep in `evaluate_entry()` after prediction
- Increments `veto_cand_*` counters
- Threshold check moved here (weakest filter)

---

## Migration Plan

### Step 1: Fast Path Invariant ✅
- [x] Implement `verify_fast_path_enabled()`
- [x] Hard fail in replay if not enabled
- [x] Export `fast_path_enabled` in perf summary

### Step 2: Hard Eligibility Gate
- [ ] Create `_check_hard_eligibility()` method
- [ ] Move session/vol_regime/spread/kill-switch checks here
- [ ] Call BEFORE feature build
- [ ] Update `veto_pre_*` counters

### Step 3: Context Features
- [ ] Add context features to `build_live_entry_features()`
- [ ] Add to `feature_meta.json`
- [ ] Include in sequence features
- [ ] Update model input contract

### Step 4: Post-Model Sanity
- [ ] Keep only max_concurrent_positions and catastrophic risk
- [ ] Move threshold check here (weakest filter)
- [ ] Update `veto_cand_*` counters

### Step 5: Documentation
- [ ] Update `docs/SNIPER_NY_AS_BUILT_OVERVIEW.md`
- [ ] Update `feature_meta.json`
- [ ] Document feature contract

---

## Expected Impact

### Entry Count Reduction
- **Current:** `n_candidates ≈ n_cycles` (most bars produce candidates)
- **Target:** `n_candidates << n_cycles` (only high-quality bars)
- **Expected reduction:** 50-70% fewer candidates

### Threshold Sensitivity
- **Current:** High sensitivity (small threshold change → large candidate change)
- **Target:** Low sensitivity (plateau around optimal value)
- **Reason:** Model learns context, threshold is weak filter

### Model Learning
- **Current:** Model doesn't see session/regime/spread
- **Target:** Model learns optimal entries per context
- **Benefit:** Better generalization, fewer false positives

---

## Risks & Open Questions

### Risks
1. **Model retraining required:** New context features need model update
2. **Feature compatibility:** Existing models may not have context features
3. **Performance impact:** Additional features may slow inference slightly

### Open Questions
1. **How to handle existing models?** (backward compatibility)
2. **What embedding dimensions for categorical features?**
3. **Should spread be categorical or continuous?**

---

## Implementation Status

- [x] OPPGAVE 1: Fast path invariant
- [ ] OPPGAVE 2: Gate refaktor
- [ ] OPPGAVE 3: Feature contract
- [ ] OPPGAVE 4: Entry count target verification



