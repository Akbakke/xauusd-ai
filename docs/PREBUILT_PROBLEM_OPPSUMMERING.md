# PREBUILT FEATURES - PROBLEMOPPSUMMERING

**Dato:** 2025-01-13  
**Status:** Kritiske sikkerhetssjekker implementert, system er fail-fast

## 🔴 PROBLEMER SOM VAR IDENTIFISERT

### 1. Risiko for feil replay-kjøring
- **Problem:** Replay kunne kjøre feil eller parallelt uten å stoppe tidlig
- **Konsekvens:** TIMER kunne brenne 100% CPU i timevis
- **Risiko:** Ikke-deterministisk oppførsel, umulig å verifisere at prebuilt features faktisk ble brukt

### 2. Manglende separasjon mellom PREBUILT og BASELINE
- **Problem:** Feature-building kode kunne importeres selv i PREBUILT-modus
- **Konsekvens:** Risiko for at `build_basic_v1()` ble kalt selv når prebuilt features skulle brukes
- **Risiko:** Silent fallback til feature-building uten at det ble oppdaget

### 3. Ingen systematiske gates
- **Problem:** Ingen obligatorisk preflight-sjekk før FULLYEAR replay
- **Konsekvens:** Kunne kjøre full replay uten å verifisere at prebuilt features var gyldige
- **Risiko:** Bruk av feil eller utdaterte prebuilt features

### 4. Output-dir gjenbruk
- **Problem:** Output-dir kunne gjenbrukes, noe som kunne føre til forvirring
- **Konsekvens:** Vanskelig å skille mellom nye og gamle resultater
- **Risiko:** Feilaktig sammenligning med baseline

### 5. Parallell replay
- **Problem:** Ingen mekanisme for å forhindre at flere replays kjørte samtidig
- **Konsekvens:** Ressurskonflikter, uforutsigbar oppførsel
- **Risiko:** Data-race conditions, korrupte resultater

## ✅ LØSNINGER SOM ER IMPLEMENTERT

### FASE 0 - TOTAL RENS

#### 0.1: Global Lock (✅ IMPLEMENTERT)
- **Fil:** `gx1/scripts/replay_eval_gated_parallel.py` (linje 1320-1360)
- **Implementasjon:** PID-basert lockfile (`/tmp/gx1_replay_lock`)
- **Funksjonalitet:** 
  - Sjekker før replay starter at ingen annen replay kjører
  - Bruker både psutil-basert prosess-sjekk og bombproof pidfile
- **Resultat:** Hard-fail hvis eksisterende replay kjører

#### 0.2: Hard Reset (✅ IMPLEMENTERT)
- **Fil:** `gx1/scripts/replay_eval_gated_parallel.py` (linje 1385-1399)
- **Implementasjon:** Sjekker at output-dir ikke eksisterer eller er tom
- **Funksjonalitet:**
  - Hard-fail hvis output-dir eksisterer og inneholder filer
  - Ingen overwrite, ingen merge
- **Resultat:** Hver replay starter med ren output-dir

#### 0.3: Global Kill-Switch (✅ IMPLEMENTERT)
- **Fil:** 
  - `gx1/scripts/replay_eval_gated_parallel.py` (linje 1612-1616) - setter env var
  - `scripts/run_fullyear_prebuilt.sh` (linje 95) - setter env var
  - `gx1/features/basic_v1.py` (linje 465-471) - sjekker i build_basic_v1()
  - `gx1/features/runtime_v10_ctx.py` (linje 58-64) - sjekker i build_v10_ctx_runtime_features()
  - `gx1/features/runtime_sniper_core.py` (linje 238-243) - sjekker i build_sniper_core_base_features()
  - `gx1/execution/live_features.py` (linje 135-141) - sjekker i build_live_entry_features()
- **Funksjonalitet:**
  - `GX1_FEATURE_BUILD_DISABLED=1` settes automatisk når `GX1_REPLAY_USE_PREBUILT_FEATURES=1`
  - Alle feature-building funksjoner sjekker denne og hard-failer umiddelbart
- **Resultat:** Umulig å kalle feature-building i PREBUILT-modus

### FASE 1 - PREBUILT = EGEN KODEVEI

#### sys.modules-sjekk (✅ IMPLEMENTERT)
- **Fil:** `gx1/scripts/replay_eval_gated_parallel.py` (linje 1944-1961)
- **Funksjonalitet:**
  - Sjekker `sys.modules` før workers spawn (i master process)
  - Verifiserer at følgende moduler IKKE er importert:
    - `gx1.features.basic_v1`
    - `gx1.execution.live_features`
    - `gx1.features.runtime_v10_ctx`
    - `gx1.features.runtime_sniper_core`
  - Hard-fail hvis noen av disse er funnet
- **Resultat:** Garanti om at feature-building kode ikke er importert i PREBUILT-modus

### FASE 2 - TRIPWIRES (HARD-FAIL)

#### basic_v1_call_count == 0 (✅ IMPLEMENTERT)
- **Fil:** `gx1/scripts/replay_eval_gated_parallel.py` (linje 716-733)
- **Funksjonalitet:** Sjekker at `basic_v1_call_count == 0` i chunk footer
- **Resultat:** Hard-fail hvis > 0 i PREBUILT-modus

#### FEATURE_BUILD_TIMEOUT == 0 (✅ IMPLEMENTERT)
- **Fil:** `gx1/scripts/replay_eval_gated_parallel.py` (linje 735-741)
- **Funksjonalitet:** Sjekker at `feature_timeout_count == 0` i chunk footer
- **Resultat:** Hard-fail hvis > 0 i PREBUILT-modus

#### feature_time_mean_ms <= 5 (✅ IMPLEMENTERT)
- **Fil:** `gx1/scripts/replay_eval_gated_parallel.py` (linje 539-543, 743-749)
- **Funksjonalitet:** Sjekker at `feature_time_mean_ms <= 5` i chunk footer
- **Resultat:** Hard-fail hvis > 5ms i PREBUILT-modus (indikerer at feature-building skjer)

#### prebuilt_bypass_count >= total_bars - warmup (✅ IMPLEMENTERT)
- **Fil:** `gx1/scripts/replay_eval_gated_parallel.py` (linje 751-760)
- **Funksjonalitet:** Sjekker at prebuilt features faktisk ble brukt
- **Resultat:** Hard-fail hvis ikke oppfylt

#### prebuilt_enabled=1 && prebuilt_used=0 (✅ IMPLEMENTERT)
- **Fil:** `gx1/scripts/replay_eval_gated_parallel.py` (linje 1013-1020, 1675-1693)
- **Funksjonalitet:** Validerer at prebuilt path eksisterer og at prebuilt faktisk ble brukt
- **Resultat:** Hard-fail hvis prebuilt enabled men ikke brukt

### FASE 3 - OBLIGATORISK PREFLIGHT

#### go_nogo_prebuilt.sh (✅ IMPLEMENTERT)
- **Fil:** `scripts/go_nogo_prebuilt.sh`
- **Funksjonalitet:**
  - Kjører preflight check (7 dager)
  - Fast sanity sample (2 dager, baseline vs prebuilt)
  - Smoke test (7 dager)
  - Early abort test (første chunk, full dataset)
  - Oppretter `/tmp/gx1_prebuilt_preflight_passed` marker ved PASS
- **Resultat:** Obligatorisk gate før FULLYEAR replay kan kjøres

#### run_fullyear_prebuilt.sh (✅ IMPLEMENTERT)
- **Fil:** `scripts/run_fullyear_prebuilt.sh`
- **Funksjonalitet:**
  - Sjekker at preflight marker eksisterer (hard-fail hvis ikke)
  - Global lock (PID-sjekk)
  - Output-dir hard-fail
  - Setter `GX1_FEATURE_BUILD_DISABLED=1`
  - Validerer prebuilt path
- **Resultat:** Umulig å kjøre FULLYEAR replay uten å ha passert preflight

### FASE 4 - HARD LOGGING (✅ IMPLEMENTERT)
- **Fil:** `gx1/scripts/replay_eval_gated_parallel.py` (linje 1414-1417)
- **Funksjonalitet:** Logger kritiske invariants ved start:
  - run_id
  - workers
  - prebuilt_enabled
  - prebuilt_path
- **Resultat:** Full sporbarhet av replay-konfigurasjon

### FASE 5 - OPPRYDDING (✅ IMPLEMENTERT)
- **Fil:** `gx1/scripts/replay_eval_gated_parallel.py` (linje 1378-1380)
- **Funksjonalitet:** Hard-fail hvis `GX1_REPLAY_QUIET=1`
- **Resultat:** Quiet mode er forbudt - alle feil må logges

## 📋 STATUS SAMMENDRAG

### ✅ FULLSTENDIG IMPLEMENTERT:
- ✅ FASE 0: Global lock, hard reset, kill-switch
- ✅ FASE 1: sys.modules-sjekk før workers spawn
- ✅ FASE 2: Alle tripwire-sjekker (5 invariants)
- ✅ FASE 3: Preflight gate (go_nogo + run_fullyear)
- ✅ FASE 4: Hard logging
- ✅ FASE 5: Quiet mode fjernet
- ✅ Feature-building hard-fail i alle funksjoner:
  - ✅ `build_basic_v1()`
  - ✅ `build_v10_ctx_runtime_features()`
  - ✅ `build_sniper_core_base_features()`
  - ✅ `build_live_entry_features()`

## ✅ RESULTAT

Systemet er nå **fail-fast og deterministisk**:
- Enhver feil i prebuilt-bruk stopper innen 30-90 sekunder
- Ingen quiet mode, ingen silent fallback, ingen "continue on error"
- Umulig å kjøre en "feil" replay i PREBUILT-modus
- Alle invariants håndheves med hard-fail
- Full sporbarhet gjennom logging

## 🎯 GJENSTÅENDE ARBEID

**INGEN KRITISKE GJENSTÅENDE OPPGAVER**

Alle planlagte sikkerhetssjekker er implementert og verifisert. Systemet er klar for produksjon.

### Eventuelle forbedringer (ikke kritiske):
- Ytterligere dokumentasjon av tripwire-logikken
- Ytterligere test-dekning for edge cases
- Performance-optimalisering av preflight-sjekker

## 📝 BRUKSANVISNING

### Standard daglig flyt:
1. **GO/NO-GO Check** (<10 min):
   ```bash
   ./scripts/go_nogo_prebuilt.sh
   ```

2. **Hvis GO: FULLYEAR Prebuilt Replay** (<1 time):
   ```bash
   ./scripts/run_fullyear_prebuilt.sh
   ```

3. **Sammenlign resultater** med baseline

4. **Rapporter** basert på perf JSON

### Hvis FAIL:
- Sjekk logs for `[PREBUILT_FAIL]` errors
- Kjør: `python3 gx1/scripts/prebuilt_preflight.py --days 7`
- Sjekk: `grep "FEATURE_BUILD_TIMEOUT\|build_basic_v1" /tmp/gx1_replay_*.log`
- Verifiser: `data/features/xauusd_m5_2025_features_v10_ctx.parquet` eksisterer og er gyldig
