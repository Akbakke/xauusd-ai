# PREFLIGHT SANITY RESULT - Gate-SSoT Overrides

**Dato:** 2025-01-14  
**Status:** ✅ Overrides fungerer, entry-stage nås, prebuilt bypass fungerer

## RESULTAT FRA LOGG

### Warmup Override (✅ FUNGERER)
```
INFO:gx1.execution.oanda_demo_runner:[PREFLIGHT] Using warmup override: GX1_PREFLIGHT_WARMUP_BARS=100 (preflight-only, does not affect FULLYEAR/prod)
INFO:gx1.execution.oanda_demo_runner:[REPLAY] first_valid_eval_idx=100 (out of 576 total bars). Skipping first 100 bars due to HTF warmup requirements.
```
- ✅ Warmup override anvendt: `warmup_required_effective=100` (ikke 288)
- ✅ `first_valid_eval_idx=100` (ikke 288)
- ✅ `bars_processed=476` (576 - 100 = 476)

### Pregate Override (✅ FUNGERER)
```
INFO:gx1.execution.oanda_demo_runner:[REPLAY_PREGATE] Env override: GX1_REPLAY_PREGATE_ENABLED=0 -> enabled=False (preflight=True)
```
- ✅ Pregate override anvendt: `pregate_enabled_effective=False`
- ✅ `pregate_skips=0` (pregate disabled)

### Entry-Stage (✅ NÅS)
```
INFO:gx1.execution.oanda_demo_runner:[REPLAY_PERF_SUMMARY] ... n_model_calls=476 ... prebuilt_bypass_count=180
```
- ✅ `bars_reaching_entry_stage=476` (alle bars etter warmup)
- ✅ `eval_calls_total=476` (alle bars nådde entry-stage)
- ✅ `n_model_calls=476` (bekrefter at entry-stage nås)

### Prebuilt Bypass (✅ FUNGERER DELVIS)
```
INFO:gx1.execution.oanda_demo_runner:[REPLAY_PERF_SUMMARY] ... prebuilt_bypass_count=180
WARNING:gx1.execution.oanda_demo_runner:[PREBUILT] prebuilt_bypass_count=180 != n_model_calls=476 (warmup_bars=100). This may indicate some bars did not use prebuilt features.
```
- ✅ `prebuilt_bypass_count=180 > 0` (prebuilt bypass fungerer)
- ⚠️ `prebuilt_bypass_count=180 < n_model_calls=476` (ikke alle bars brukte prebuilt)
- ⚠️ `lookup_attempts` og `lookup_hits` ikke logget (må sjekkes)

## PROBLEM IDENTIFISERT

### FASE_2_TRIPWIRE Feiler
```
ERROR: [PREBUILT_FAIL] FASE_2_TRIPWIRE: prebuilt_bypass_count=180 < expected_min=376 (bars_processed=476, warmup=100). 
This indicates prebuilt features were not used for all processed bars.
```

**Årsak:**
- `prebuilt_bypass_count=180` (faktisk antall bars som brukte prebuilt)
- `expected_min=376` (bars_processed=476 - warmup=100 = 376)
- Forskjell: 196 bars nådde entry-stage men brukte ikke prebuilt features

**Mulige årsaker:**
1. Noen bars feilet i prebuilt lookup
2. Noen bars brukte fallback til feature-building (skal ikke skje i PREBUILT mode)
3. `prebuilt_bypass_count` telles ikke riktig

## BEVIS FOR MÅL

### ✅ Warmup Override
- `warmup_required_effective=100` (ikke 288)
- `warmup_override_applied=true` (bekreftet i logg)

### ✅ Pregate Override  
- `pregate_enabled_effective=false` (ikke true)
- `pregate_override_applied=true` (bekreftet i logg)

### ✅ Entry-Stage Nås
- `bars_reaching_entry_stage=476 > 0` ✅
- `eval_calls_total=476 > 0` ✅

### ✅ Prebuilt Bypass Fungerer
- `prebuilt_bypass_count=180 > 0` ✅
- `lookup_attempts` må sjekkes (ikke logget)

## GJENSTÅENDE ARBEID

1. **Fikse prebuilt_bypass_count mismatch:**
   - Undersøk hvorfor 196 bars ikke brukte prebuilt features
   - Sjekk lookup telemetry (`lookup_attempts`, `lookup_hits`)
   - Verifiser at alle bars som når entry-stage faktisk bruker prebuilt

2. **Fikse JSON-serialiseringsfeil i stub footer:**
   - Bruk `convert_to_json_serializable()` for alle numeriske verdier i stub footer

3. **Verifisere lookup telemetry:**
   - Sjekk at `lookup_attempts` og `lookup_hits` eksporteres i footer
   - Verifiser at `lookup_attempts == bars_reaching_entry_stage`

## KONKLUSJON

**Gate-SSoT overrides fungerer:**
- ✅ Warmup override anvendt (100 bars i stedet for 288)
- ✅ Pregate override anvendt (disabled)
- ✅ Entry-stage nås (476 bars)
- ✅ Prebuilt bypass fungerer (180 bars)

**Gjenstående:**
- ⚠️ Prebuilt bypass fungerer ikke for alle bars (180/476)
- ⚠️ FASE_2_TRIPWIRE feiler (forventet, siden ikke alle bars bruker prebuilt)

**Neste steg:**
- Undersøk hvorfor 196 bars ikke brukte prebuilt features
- Fikse prebuilt lookup/logging for å sikre at alle bars bruker prebuilt
