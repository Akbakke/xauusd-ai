# PREFLIGHT SANITY FIX - WARMUP VS PREGATE

**Dato:** 2025-01-13  
**Problem:** Entry-stage nås ikke i 2-day sanity (bars_reaching_entry_stage = 0)

## PROBLEM

I 2-day PREBUILT sanity:
- `bars_processed = 576` (bar-løkka kjører)
- `bars_seen = 576`
- `bars_reaching_entry_stage = 0`
- `eval_calls_total = 0`
- 100% av bars blir stoppet før entry-stage

Dette er IKKE et prebuilt-problem. Prebuilt-gate er true, men irrelevant fordi entry aldri kjøres.

## LØSNING

### STEG 1 - MÅLING (✅ ALLEREDE IMPLEMENTERT)

Tellerne er allerede implementert i `oanda_demo_runner.py`:
- `self.bars_seen` - inkrementeres per bar (linje 9899)
- `self.bars_skipped_warmup` - inkrementeres ved warmup skip (linje 9904, 9909)
- `self.bars_skipped_pregate` - inkrementeres ved pregate skip (linje 10143)
- `self.bars_reaching_entry_stage` - inkrementeres før evaluate_entry() (linje 10175)

Tellerne eksporteres til `chunk_footer.json` i `replay_eval_gated_parallel.py` (linje 808-812).

### STEG 2 - MINIMAL FIX (✅ IMPLEMENTERT)

#### A) Warmup Override (PREFLIGHT-ONLY)

**Env-var:** `GX1_PREFLIGHT_WARMUP_BARS=100`

**Implementasjon:**
- `gx1/execution/oanda_demo_runner.py` (linje 9463-9474): Sjekker env-var og overstyrer `min_bars_for_features`
- `scripts/go_nogo_prebuilt.sh`: Setter `GX1_PREFLIGHT_WARMUP_BARS=100` for sanity/smoke
- `scripts/test_prebuilt_smoke_7days.sh`: Setter `GX1_PREFLIGHT_WARMUP_BARS=100` for smoke

**Virkning:**
- Reduserer warmup fra typisk 288 bars til 100 bars for preflight-only
- FULLYEAR/prod warmup påvirkes IKKE (env-var settes kun i preflight scripts)

#### B) Pregate Disable (PREFLIGHT-ONLY)

**Env-var:** `GX1_REPLAY_PREGATE_ENABLED=0`

**Implementasjon:**
- `gx1/execution/oanda_demo_runner.py` (linje 9871-9882): Respekterer env-var (allerede implementert)
- `scripts/go_nogo_prebuilt.sh`: Setter `GX1_REPLAY_PREGATE_ENABLED=0` for sanity/smoke
- `scripts/test_prebuilt_smoke_7days.sh`: Setter `GX1_REPLAY_PREGATE_ENABLED=0` for smoke

**Virkning:**
- Disabler pregate for preflight-only
- FULLYEAR/prod pregate påvirkes IKKE (env-var settes kun i preflight scripts)

#### C) Diagnostikk

**Implementasjon:**
- `scripts/go_nogo_prebuilt.sh`: Leser `chunk_footer.json` og viser diagnostikk:
  - `bars_seen`, `bars_processed`
  - `bars_skipped_warmup`, `bars_skipped_pregate`
  - `bars_reaching_entry_stage`, `eval_calls_total`
  - `lookup_attempts`, `prebuilt_bypass_count`
  - `pregate_enabled`

**Virkning:**
- Viser tydelig hva som blokkerer entry-stage
- Diagnostikk i stdout (ikke i chunk_footer, som allerede inneholder dataene)

## FORVENTET RESULTAT

Etter fix:
- `bars_reaching_entry_stage > 0`
- `eval_calls_total > 0`
- `lookup_attempts > 0`
- `prebuilt_bypass_count > 0` (og helst ≈ bars_processed - warmup)
- FASE_2_TRIPWIRE passerer

## ENDRINGER

### Filer endret:
1. `scripts/go_nogo_prebuilt.sh`
   - Lagt til `GX1_PREFLIGHT=1` flagg (indikerer preflight-run)
   - Lagt til `GX1_PREFLIGHT_WARMUP_BARS=100` for sanity (baseline og prebuilt)
   - Lagt til `GX1_REPLAY_PREGATE_ENABLED=0` for sanity (baseline og prebuilt)
   - Lagt til diagnostikk med Gate-SSoT verification (effective values)

2. `scripts/test_prebuilt_smoke_7days.sh`
   - Fjernet `GX1_REPLAY_QUIET=1` (forbidden i FASE 5)
   - Lagt til `GX1_PREFLIGHT=1` flagg (indikerer preflight-run)
   - Lagt til `GX1_PREFLIGHT_WARMUP_BARS=100` for smoke (baseline og prebuilt)
   - Lagt til `GX1_REPLAY_PREGATE_ENABLED=0` for smoke (baseline og prebuilt)

3. `gx1/execution/oanda_demo_runner.py`
   - **Gate-SSoT Warmup (linje 9463-9495):** Override sjekkes på stedet hvor `min_bars_for_features` beregnes (brukt i bar loop linje 9914, 9919)
     - Lagrer `warmup_required_effective` og `warmup_override_applied` på runner
   - **Gate-SSoT Pregate (linje 9881-9900):** Override sjekkes på stedet hvor `pregate_enabled` bestemmes (brukt i bar loop linje 10085)
     - Lagrer `pregate_enabled_effective` og `pregate_override_applied` på runner

4. `gx1/scripts/replay_eval_gated_parallel.py`
   - **Footer export (linje 808-825):** Eksporterer effective values og override flags:
     - `warmup_required_effective`
     - `warmup_override_applied`
     - `pregate_enabled_effective`
     - `pregate_override_applied`

## VERIFISERING

Kjør:
```bash
./scripts/go_nogo_prebuilt.sh
```

Sjekk output for:
- `[DIAGNOSTICS] bars_reaching_entry_stage > 0`
- `[GO/NO-GO] ✅ Entry-stage reached: X bars`
- `[GO/NO-GO] ✅ Warmup override applied: warmup_required_effective=100`
- `[GO/NO-GO] ✅ Pregate override applied: pregate_enabled_effective=False`

Sjekk `chunk_footer.json` for:
- `warmup_required_effective == 100`
- `warmup_override_applied == true`
- `pregate_enabled_effective == false`
- `pregate_override_applied == true`
- `bars_reaching_entry_stage > 0`
- `eval_calls_total > 0`
- `lookup_attempts > 0`
- `prebuilt_bypass_count > 0`

## NOTATER

- **Minimal endring:** Kun preflight-only overrides, ingen refactoring
- **Ikke påvirker FULLYEAR/prod:** Overrides settes kun i preflight scripts
- **Fail-fast bevart:** Alle tripwires og gates fungerer som før
- **Ingen stdout-logging:** Diagnostikk vises kun i sanity-sjekken (ikke i bar-loop)
