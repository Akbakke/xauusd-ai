# Entry Architecture Refactor - Implementation Plan

**Date:** 2026-01-06  
**Status:** Plan (ikke implementert enda)  
**Goal:** Færre, bedre entry-kandidater. Markedsregimer som INPUT, ikke gates.

---

## 1. PRINSIPPAVKLARING

### A) HARD ELIGIBILITY (før modell, fail-fast)
**Prinsipp:** Dette er ikke trading, dette er markedet er uegnet nå.

**Flyttes hit:**
- `session ikke tillatt` (SNIPER: ASIA blocked)
- `vol_regime == EXTREME`
- `spread > hard cap`
- `kill-switch / system safety`
- `warmup not ready`

**Implementering:**
- `_check_hard_eligibility()` i `entry_manager.py`
- Kalles FØR `build_live_entry_features()`
- Hvis `False`: ingen feature build, ingen modellkall, ingen kandidat
- Returnerer `(eligible: bool, reason: str)`

### B) CONTEXT FEATURES (input til modellen)
**Prinsipp:** Dette er ikke gates lenger. Dette er modellens verdensbilde.

**Flyttes fra gating → modell-input:**
- `trend_regime` (categorical / embedding)
- `vol_regime` (categorical / embedding, ekskl. EXTREME)
- `session` (categorical / embedding)
- `atr_bps` / `atr_bucket`
- `spread_bps` / `spread_bucket`

**Implementering:**
- Legges til i `build_live_entry_features()`
- Defineres eksplisitt i `feature_meta.json`
- Mates inn i både XGB og Transformer (hybrid)
- Mål: Modellen lærer "Dette signalet er dårlig i dette regimet"

### C) POST-MODEL SANITY (minimalt, beholdes)
**Prinsipp:** Dette er kontobeskyttelse, ikke signalfiltrering.

**Beholdes:**
- `max_concurrent_positions`
- `catastrophic risk / drawdown guards`
- `threshold` (flyttes hit, svakeste filter)

**Implementering:**
- Flyttes etter modell og threshold
- Skal være sjeldne (kontobeskyttelse, ikke signalfiltrering)

---

## 2. KONKRET IMPLEMENTASJONSPLAN

### STEP 1: Implementer `_check_hard_eligibility()`

**Fil:** `gx1/execution/entry_manager.py`

**Lokasjon:** Før `build_live_entry_features()` i `evaluate_entry()`

**Kode-struktur:**
```python
def _check_hard_eligibility(
    self,
    candles: pd.DataFrame,
    policy_state: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """
    Check hard eligibility BEFORE feature build and model inference.
    
    Returns:
        (eligible: bool, reason: Optional[str])
        If eligible=False, reason explains why (for logging)
    """
    current_ts = candles.index[-1] if len(candles) > 0 else None
    
    # 1. Warmup check
    warmup_bars = getattr(self, "warmup_bars", 288)
    if len(candles) < warmup_bars:
        return False, "warmup_not_ready"
    
    # 2. Session check (SNIPER: ASIA blocked)
    current_session = policy_state.get("session")
    if not current_session:
        from gx1.execution.live_features import infer_session_tag
        current_session = infer_session_tag(current_ts).upper()
        policy_state["session"] = current_session
    
    policy_sniper_cfg = self.policy.get("entry_v9_policy_sniper", {})
    use_sniper = policy_sniper_cfg.get("enabled", False)
    if use_sniper:
        allowed_sessions = policy_sniper_cfg.get("allowed_sessions", ["EU", "OVERLAP", "US"])
        if current_session not in allowed_sessions:
            return False, f"session_not_allowed_{current_session}"
    
    # 3. Vol regime check (EXTREME blocked)
    vol_regime = policy_state.get("brain_vol_regime", "UNKNOWN")
    if vol_regime == "EXTREME":
        return False, "vol_regime_extreme"
    
    # 4. Spread check (hard cap)
    # Get spread from candles or features (before feature build)
    spread_bps = self._get_spread_bps_before_features(candles)
    spread_hard_cap = policy_sniper_cfg.get("spread_hard_cap_bps", 100.0)  # Default 100 bps
    if spread_bps is not None and spread_bps > spread_hard_cap:
        return False, f"spread_too_high_{spread_bps:.1f}bps"
    
    # 5. Kill-switch check
    if self._is_kill_switch_active():
        return False, "kill_switch_active"
    
    return True, None
```

**Endringer i `evaluate_entry()`:**
```python
def evaluate_entry(self, candles: pd.DataFrame) -> Optional[LiveTrade]:
    # ... existing code ...
    
    # OPPGAVE 2: Hard eligibility check BEFORE feature build
    eligible, eligibility_reason = self._check_hard_eligibility(candles, policy_state)
    if not eligible:
        self.entry_telemetry["n_cycles"] += 1
        self.stage0_reasons[eligibility_reason] += 1
        self.veto_pre[f"veto_pre_{eligibility_reason}"] += 1
        return None  # STOP - no feature build, no model call
    
    # Only proceed if eligible
    self.entry_telemetry["n_eligible_cycles"] += 1  # NEW counter
    
    # NOW build features
    feat_start = time.perf_counter()
    entry_bundle = build_live_entry_features(candles)
    # ... rest of code ...
```

### STEP 2: Definer Context Feature Contract

**Fil:** `docs/ENTRY_ARCHITECTURE_REFACTOR.md` (oppdater)

**Context Features:**
```python
CONTEXT_FEATURES = {
    "trend_regime_id": {
        "type": "categorical",
        "values": [0, 1, 2],  # 0=DOWN, 1=NEUTRAL, 2=UP
        "embedding_dim": 8,  # For transformer
        "normalization": None,  # Categorical
    },
    "vol_regime_id": {
        "type": "categorical",
        "values": [0, 1, 2],  # 0=LOW, 1=MEDIUM, 2=HIGH (EXTREME excluded by hard eligibility)
        "embedding_dim": 8,
        "normalization": None,
    },
    "session_id": {
        "type": "categorical",
        "values": [0, 1, 2, 3],  # 0=ASIA, 1=EU, 2=US, 3=OVERLAP
        "embedding_dim": 8,
        "normalization": None,
    },
    "atr_bps": {
        "type": "continuous",
        "range": [0, 200],  # Typical range in bps
        "normalization": "standard",  # Z-score
    },
    "atr_bucket": {
        "type": "categorical",
        "values": [0, 1, 2, 3],  # 0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME
        "embedding_dim": 8,
        "normalization": None,
    },
    "spread_bps": {
        "type": "continuous",
        "range": [0, 100],  # Typical range in bps
        "normalization": "standard",
    },
    "spread_bucket": {
        "type": "categorical",
        "values": [0, 1, 2],  # 0=LOW, 1=MEDIUM, 2=HIGH
        "embedding_dim": 8,
        "normalization": None,
    },
}
```

### STEP 3: Plumb Context Features til Modellen

**Fil:** `gx1/execution/live_features.py`

**Endringer i `build_live_entry_features()`:**
```python
def build_live_entry_features(
    candles: pd.DataFrame,
    policy_state: Optional[Dict[str, Any]] = None,
    enable_context_features: bool = False,  # NEW flag
) -> EntryFeatureBundle:
    # ... existing feature building ...
    
    # OPPGAVE 2: Add context features if enabled
    if enable_context_features and policy_state:
        # Get context from policy_state (computed before feature build)
        trend_regime = policy_state.get("brain_trend_regime", "UNKNOWN")
        vol_regime = policy_state.get("brain_vol_regime", "UNKNOWN")
        session = policy_state.get("session", "UNKNOWN")
        
        # Map to IDs
        trend_regime_id = {"TREND_DOWN": 0, "TREND_NEUTRAL": 1, "TREND_UP": 2}.get(trend_regime, 1)
        vol_regime_id = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(vol_regime, 1)
        session_id = {"ASIA": 0, "EU": 1, "US": 2, "OVERLAP": 3}.get(session, 0)
        
        # Add to features DataFrame
        aligned_last["trend_regime_id"] = trend_regime_id
        aligned_last["vol_regime_id"] = vol_regime_id
        aligned_last["session_id"] = session_id
        
        # Add ATR/spread buckets if available
        if "atr_bps" in aligned_last.columns:
            atr_bps = aligned_last["atr_bps"].iloc[-1]
            atr_bucket = self._bucket_atr(atr_bps)
            aligned_last["atr_bucket"] = atr_bucket
        
        if "spread_bps" in aligned_last.columns:
            spread_bps = aligned_last["spread_bps"].iloc[-1]
            spread_bucket = self._bucket_spread(spread_bps)
            aligned_last["spread_bucket"] = spread_bucket
```

**Fil:** `gx1/execution/entry_manager.py`

**Endringer i `evaluate_entry()`:**
```python
# After hard eligibility check, before model inference
enable_context_features = os.getenv("ENTRY_CONTEXT_FEATURES_ENABLED", "false").lower() == "true"

entry_bundle = build_live_entry_features(
    candles,
    policy_state=policy_state,
    enable_context_features=enable_context_features,
)
```

### STEP 4: Telemetry Justering

**Fil:** `gx1/execution/entry_manager.py`

**Nye counters:**
```python
self.entry_telemetry = {
    "n_cycles": 0,  # Total bar cycles
    "n_eligible_cycles": 0,  # NEW: Cycles that passed hard eligibility
    "n_precheck_pass": 0,  # Deprecated (replaced by n_eligible_cycles)
    "n_predictions": 0,  # Predictions produced
    "n_candidates": 0,  # Valid predictions
    "n_candidate_pass": 0,  # Candidates that passed post-model sanity
    "n_trades_created": 0,  # Trades actually created
    # ... existing counters ...
}
```

**Nye metrics:**
- `eligibility_rate = n_eligible_cycles / n_cycles`
- `candidate_rate = n_candidates / n_eligible_cycles` (should be << 1.0)
- `trade_rate = n_trades_created / n_candidates` (should be ≈ 1.0 after refactor)

### STEP 5: Bakoverkompatibilitet

**Environment Variable:**
```bash
ENTRY_CONTEXT_FEATURES_ENABLED=true  # Enable context features
```

**Fallback for gamle modeller:**
```python
# In model inference path
if enable_context_features:
    # Check if model expects context features
    if hasattr(model, "expects_context_features") and model.expects_context_features:
        # Use context features
        pass
    else:
        # Old model - drop context features before inference
        features_without_context = features.drop(columns=CONTEXT_FEATURE_COLS)
        prediction = model.predict(features_without_context)
else:
    # Context features disabled - old behavior
    pass
```

---

## 3. RISIKOANALYSE

### Risiko 1: Modell-kompatibilitet
**Sannsynlighet:** Høy  
**Impact:** Høy  
**Mitigering:**
- `ENTRY_CONTEXT_FEATURES_ENABLED` flag (default: false)
- Fallback: drop context features for gamle modeller
- Test med eksisterende modeller før aktivering

### Risiko 2: Feature mismatch i training
**Sannsynlighet:** Medium  
**Impact:** Høy  
**Mitigering:**
- Eksplisitt feature contract dokumentasjon
- Feature validation i `build_live_entry_features()`
- Unit tests for feature alignment

### Risiko 3: Performance impact
**Sannsynlighet:** Lav  
**Impact:** Medium  
**Mitigering:**
- Context features er små (7 features)
- Embeddings er effektive (categorical)
- Benchmark før/etter

### Risiko 4: Hard eligibility for streng
**Sannsynlighet:** Medium  
**Impact:** Medium  
**Mitigering:**
- Hard eligibility er konfigurerbar
- Logging av alle veto_reasons
- Kan justeres basert på telemetry

### Risiko 5: Regime inference feil
**Sannsynlighet:** Lav  
**Impact:** Høy  
**Mitigering:**
- Regime inference skjer før hard eligibility
- Fallback til UNKNOWN hvis inference feiler
- UNKNOWN behandles som hard eligibility fail (sikker side)

---

## 4. TESTPLAN

### Unit Tests
1. `_check_hard_eligibility()` - alle edge cases
2. Context feature encoding - alle mappings
3. Feature alignment - gamle vs nye modeller

### Integration Tests
1. Full entry flow med context features enabled
2. Full entry flow med context features disabled (backward compat)
3. Hard eligibility blocking feature build

### Replay Tests
1. Mini replay (1 uke) med context features enabled
2. Verifiser: `n_eligible_cycles << n_cycles`
3. Verifiser: `n_candidates << n_eligible_cycles`
4. Verifiser: alle guards passerer

---

## 5. MIGRASJONSVEI

### Fase 1: Hard Eligibility (Lav risiko)
- Implementer `_check_hard_eligibility()`
- Flytt eksisterende gates dit
- Test med eksisterende modeller (ingen breaking changes)

### Fase 2: Context Feature Contract (Dokumentasjon)
- Dokumenter feature contract
- Oppdater `feature_meta.json`
- Ingen kode-endringer enda

### Fase 3: Context Features Plumbing (Medium risiko)
- Legg til context features i `build_live_entry_features()`
- `ENTRY_CONTEXT_FEATURES_ENABLED=false` (default)
- Test med dummy-modeller

### Fase 4: Modell Training (Høy risiko)
- Tren nye modeller med context features
- Valider feature alignment
- Test med nye modeller

### Fase 5: Aktivering (Høy risiko)
- Sett `ENTRY_CONTEXT_FEATURES_ENABLED=true` for nye modeller
- Monitor telemetry
- Rollback plan klar

---

## 6. FORVENTET IMPACT

### Entry Count Reduction
- **Current:** `n_candidates ≈ n_cycles` (mange bars produserer kandidater)
- **Target:** `n_candidates << n_cycles` (færre, bedre kandidater)
- **Expected:** 50-70% reduksjon i kandidater

### Threshold Sensitivity
- **Current:** Høy sensitivitet (liten threshold-endring → stor kandidat-endring)
- **Target:** Lav sensitivitet (plateau rundt optimal verdi)
- **Reason:** Modellen lærer context, threshold er svak filter

### Model Learning
- **Current:** Modellen ser ikke session/regime/spread
- **Target:** Modellen lærer optimale entries per context
- **Benefit:** Bedre generalisering, færre false positives

---

## 7. OPEN QUESTIONS

1. **Embedding dimensions:** Hvilke dimensjoner for categorical features? (forslag: 8)
2. **Spread bucketing:** Kategorisk eller kontinuerlig? (forslag: begge - bucket for embedding, bps for continuous)
3. **ATR bucketing:** Hvilke buckets? (forslag: LOW/MEDIUM/HIGH/EXTREME, men EXTREME excluded by hard eligibility)
4. **Model retraining:** Når skal nye modeller trenes? (forslag: etter Fase 3)
5. **Rollback plan:** Hvordan ruller vi tilbake hvis noe går galt? (forslag: `ENTRY_CONTEXT_FEATURES_ENABLED=false`)

---

## 8. NESTE STEG

1. ✅ OPPGAVE 1: Fast path invariant (ferdig)
2. ⏳ OPPGAVE 2A: Implementer `_check_hard_eligibility()` (neste)
3. ⏳ OPPGAVE 2B: Definer context feature contract (dokumentasjon)
4. ⏳ OPPGAVE 2C: Plumb context features (med flag)
5. ⏳ OPPGAVE 3: Oppdater dokumentasjon
6. ⏳ OPPGAVE 4: Verifiser entry count reduction

**Start med STEP 1 (hard eligibility) - lav risiko, høy verdi.**



