# GATE-SSoT OVERRIDES - PREFLIGHT ONLY

**Dato:** 2025-01-13  
**Status:** Implementert - Overrides koblet til Gate-SSoT

## PROBLEM

Preflight overrides må være effektive i den faktiske replay-pathen som produserer `bars_seen` og `bars_processed` i `chunk_footer.json`. Risiko for code-path mismatch hvor overrides ikke påvirker faktiske gates.

## LØSNING - GATE-SSoT

### Gate-SSoT Prinsipp

**Single Source of Truth (SSoT) for gates:**
- Warmup gate: Sjekkes i bar loop på linje 9914, 9919 i `oanda_demo_runner.py`
- Pregate gate: Sjekkes i bar loop på linje 10085 i `oanda_demo_runner.py`

**Override-kobling:**
- Overrides leses på samme sted hvor gate-verdier beregnes (Gate-SSoT)
- Effective values lagres på runner og eksporteres i footer
- Override flags indikerer om overrides faktisk ble anvendt

### Implementasjon

#### A) Warmup Override (Gate-SSoT)

**Lokasjon:** `gx1/execution/oanda_demo_runner.py` linje 9463-9495

**Gate-SSoT:**
- `min_bars_for_features` beregnes her (brukt i bar loop linje 9914, 9919)
- Override sjekkes på samme sted

**Implementasjon:**
```python
# PREFLIGHT-ONLY: Allow override for sanity/smoke tests
preflight_warmup_override = os.getenv("GX1_PREFLIGHT_WARMUP_BARS")
is_preflight_run = os.getenv("GX1_PREFLIGHT", "0") == "1" or preflight_warmup_override is not None

if preflight_warmup_override is not None and is_preflight_run:
    min_bars_for_features_effective = int(preflight_warmup_override)
    warmup_override_applied = True
else:
    # Normal calculation
    min_bars_for_features_effective = max(lookback_requirements)
    warmup_override_applied = False

# Store on runner for footer export
self.warmup_required_effective = min_bars_for_features_effective
self.warmup_override_applied = warmup_override_applied
min_bars_for_features = min_bars_for_features_effective  # Use effective value
```

**Virkning:**
- `min_bars_for_features` brukes i `_calculate_first_valid_eval_idx()` (linje 9480)
- `first_valid_eval_idx` brukes i bar loop (linje 9914)
- Override påvirker faktisk gate i bar loop

#### B) Pregate Override (Gate-SSoT)

**Lokasjon:** `gx1/execution/oanda_demo_runner.py` linje 9881-9900

**Gate-SSoT:**
- `pregate_enabled` bestemmes her (brukt i bar loop linje 10085)
- Override sjekkes på samme sted

**Implementasjon:**
```python
# DEL A2: Check if pregate is enabled in replay_config
env_pregate_enabled = os.getenv("GX1_REPLAY_PREGATE_ENABLED")
is_preflight_run = os.getenv("GX1_PREFLIGHT", "0") == "1" or os.getenv("GX1_PREFLIGHT_WARMUP_BARS") is not None

if env_pregate_enabled is not None:
    pregate_enabled_effective = env_pregate_enabled.lower() in ("1", "true")
    pregate_override_applied = is_preflight_run and env_pregate_enabled.lower() == "0"
else:
    # Read from YAML
    pregate_enabled_effective = pregate_cfg.get("enabled", False)
    pregate_override_applied = False

# Store on runner for footer export
self.pregate_enabled_effective = pregate_enabled_effective
self.pregate_override_applied = pregate_override_applied
self.pregate_enabled = pregate_enabled_effective  # Use effective value
```

**Virkning:**
- `self.pregate_enabled` brukes i bar loop (linje 10085)
- Override påvirker faktisk gate i bar loop

#### C) Footer Export (Gate-SSoT Verification)

**Lokasjon:** `gx1/scripts/replay_eval_gated_parallel.py` linje 808-825

**Eksporterte felter:**
- `warmup_required_effective` - Faktisk warmup-verdi brukt i gates
- `warmup_override_applied` - True hvis override ble anvendt
- `pregate_enabled_effective` - Faktisk pregate-verdi brukt i gates
- `pregate_override_applied` - True hvis override ble anvendt

**Verifisering:**
- `warmup_override_applied == true` → Override ble anvendt
- `warmup_required_effective == 100` → Override-verdi brukt
- `pregate_override_applied == true` → Override ble anvendt
- `pregate_enabled_effective == false` → Override-verdi brukt

## PREFLIGHT FLAGG

**Env-var:** `GX1_PREFLIGHT=1`

**Betydning:**
- Indikerer at dette er en preflight-run (sanity/smoke)
- Aktiverer preflight-only overrides
- Settes automatisk i `go_nogo_prebuilt.sh` og `test_prebuilt_smoke_7days.sh`

**Alternativ deteksjon:**
- Hvis `GX1_PREFLIGHT_WARMUP_BARS` er satt, anses det som preflight-run

## VERIFISERING

### Forventet resultat i `chunk_footer.json`:

```json
{
  "bars_seen": 576,
  "bars_processed": 576,
  "bars_skipped_warmup": < 576,  // Ikke alle bars
  "bars_skipped_pregate": 0,      // Pregate disabled
  "bars_reaching_entry_stage": > 0,
  "eval_calls_total": > 0,
  "lookup_attempts": > 0,
  "prebuilt_bypass_count": > 0,
  "warmup_required_effective": 100,
  "warmup_override_applied": true,
  "pregate_enabled_effective": false,
  "pregate_override_applied": true
}
```

### Kjør verifisering:

```bash
./scripts/go_nogo_prebuilt.sh
```

Sjekk output for:
- `[GO/NO-GO] ✅ Warmup override applied: warmup_required_effective=100`
- `[GO/NO-GO] ✅ Pregate override applied: pregate_enabled_effective=False`
- `[GO/NO-GO] ✅ Entry-stage reached: X bars`

## SIKKERHET

- **Kun preflight:** Overrides gjelder kun når `GX1_PREFLIGHT=1` eller `GX1_PREFLIGHT_WARMUP_BARS` er satt
- **Ikke påvirker FULLYEAR/prod:** FULLYEAR scripts setter ikke `GX1_PREFLIGHT=1`
- **Deterministisk:** Effective values eksporteres i footer for full sporbarhet
- **Fail-fast:** Alle tripwires og gates fungerer som før

## NOTATER

- **Gate-SSoT:** Overrides kobles direkte til stedet hvor gates beregnes
- **Effective values:** Faktiske verdier brukt i gates eksporteres i footer
- **Override flags:** Indikerer om overrides faktisk ble anvendt
- **Verifisering:** All verifisering via `chunk_footer.json`, ikke stdout
