# PREBUILT REPLAY - LOVLIG FLYT OG INVARIANTS

## FASE 0-5: FULLSTENDIG IMPLEMENTASJON

Dette dokumentet beskriver den **eneste lovlige flyten** for PREBUILT replay og alle invariants som må håndheves.

## LOVLIG FLYT (INGEN UNNTAK)

### Steg 1: OBLIGATORISK PREFLIGHT
```bash
./scripts/go_nogo_prebuilt.sh
```

**Krav:**
- Må PASSERE alle sjekker
- 2-day sanity: baseline vs prebuilt må være identiske (trades + PnL)
- 7-day smoke: samme krav + `bundle_sha256` identisk
- Early abort: `basic_v1_call_count=0`, `feature_time=0`

**Hvis FAIL:**
- STOPP umiddelbart
- Fiks problemet
- Kjør på nytt
- **ALDRI** "la den gå litt til"

**Hvis PASS:**
- Oppretter `/tmp/gx1_prebuilt_preflight_passed` marker (med timestamp)
- FULLYEAR er nå tillatt

**Implementert:**
- ✅ Preflight marker opprettes automatisk ved PASS
- ✅ Marker inneholder timestamp for sporbarhet

### Steg 2: FULLYEAR REPLAY (kun hvis preflight PASS)
```bash
./scripts/run_fullyear_prebuilt.sh
```

**Gate:**
- ✅ Sjekker at `/tmp/gx1_prebuilt_preflight_passed` eksisterer
- ✅ Hvis ikke: hard-fail med instruksjoner

**FASE 0 - Sikkerhetssjekker (implementert):**
- ✅ Global lock: Maks én replay per maskin (PID-sjekk)
- ✅ Output-dir hard-fail: Hvis eksisterer → hard-fail (ingen overwrite)
- ✅ GX1_FEATURE_BUILD_DISABLED=1: Settes automatisk når prebuilt enabled

**Forventet runtime:**
- < 1 time (vs ~4.7h baseline)

**Forventede invariants:**
- `prebuilt_used=true`
- `basic_v1_call_count=0`
- `FEATURE_BUILD_TIMEOUT=0`
- `feature_time_mean_ms <= 5ms`
- `prebuilt_bypass_count == n_model_calls`

## FASE 0 - TOTAL RENS

### 0.1: Forby parallell replay
- Maks én aktiv `replay_eval_gated_parallel.py` per maskin
- Global lock / PID-sjekk før start
- Hvis eksisterende replay lever → `RuntimeError` før noe lastes

### 0.2: Hard reset
- Output-dir kan **IKKE** gjenbrukes
- Hvis output-dir eksisterer og inneholder artifacts → hard-fail
- Ingen overwrite, ingen merge

### 0.3: Global kill-switch
- Når `GX1_REPLAY_USE_PREBUILT_FEATURES=1`:
  - Sett `GX1_FEATURE_BUILD_DISABLED=1`
- ALLE feature-building-funksjoner skal hard-faile umiddelbart:
  - `build_basic_v1()` → `RuntimeError`
  - `build_live_entry_features()` → `RuntimeError`
  - `build_v10_ctx_runtime_features()` → `RuntimeError`
  - `build_sniper_core_base_features()` → `RuntimeError`

## FASE 1 - PREBUILT = EGEN KODEVEI

### ReplayMode enum
- `PREBUILT`: Bruker pre-computed features
- `BASELINE`: Bygger features on-the-fly

### PrebuiltFeaturesLoader
- Isolert modul som **IKKE** importerer feature-building
- Kun `.loc[timestamp]` på prebuilt DataFrame
- Validerer SHA256, timestamp alignment, required columns

### Hard garanti
- Før workers starter: assert at feature-building-moduler **IKKE** er importert
- Sjekker `sys.modules` for:
  - `gx1.features.basic_v1`
  - `gx1.execution.live_features`
  - `gx1.features.runtime_v10_ctx`
  - `gx1.features.runtime_sniper_core`
- Hvis funnet → `RuntimeError` før workers spawn

## FASE 2 - TRIPWIRES (HARD-FAIL)

Disse invariants skal **ALLTID** håndheves og hard-faile (ikke warning, ikke quiet):

### basic_v1_call_count == 0
- Hvis > 0 i PREBUILT → crash umiddelbart
- Sjekkes i chunk footer og aggregate perf JSON

### FEATURE_BUILD_TIMEOUT == 0
- Hvis én forekomst i PREBUILT → crash
- Sjekkes i chunk footer

### feature_time_mean_ms <= 5
- Hvis > 5 ms → crash
- Sjekkes i chunk footer

### prebuilt_bypass_count >= total_bars - warmup
- Hvis ikke → crash
- Sjekkes i chunk footer

### prebuilt_enabled=1 && prebuilt_used=0
- Skal **ALDRI** nå chunks → crash i main()
- Sjekkes før workers starter

## FASE 3 - OBLIGATORISK PREFLIGHT

### go_nogo_prebuilt.sh
- **OBLIGATORISK** gate før FULLYEAR
- Kjører:
  1. Preflight check (7 days)
  2. 2-day sanity (baseline vs prebuilt)
  3. 7-day smoke (samme krav + bundle_sha256)
  4. Early abort test

### Preflight marker
- Ved PASS: oppretter `/tmp/gx1_prebuilt_preflight_passed`
- FULLYEAR-script sjekker denne før start
- Hvis mangler → hard-fail med instruksjoner

## FASE 4 - HARD LOGGING

### [RUN_START] log
Må **ALLTID** inneholde:
- `mode=PREBUILT` eller `mode=BASELINE`
- `prebuilt_path=...` (hvis PREBUILT)
- `features_sha256=...` (hvis PREBUILT)
- `bundle_sha256=...`
- `feature_build_disabled=1` (hvis PREBUILT)
- `basic_v1_call_count=0` (hvis PREBUILT)

Hvis dette mangler → crash.

## FASE 5 - OPPRYDDING

### Quiet mode fjernet
- **INGEN** `GX1_REPLAY_QUIET` referanser
- **INGEN** silent fallbacks
- **INGEN** "continue on error"

### FEATURE_BUILD_TIMEOUT
- **ALLTID** fatal
- Ingen demping
- Ingen "continue"

## FAIL-FAST PRINSIPP

### Måltid: 30-90 sekunder
- Enhver feil i prebuilt-bruk skal stoppe innen 30-90 sekunder
- **IKKE** timer med 100% CPU

### Hard-fail ved:
- Feature-building moduler importert i PREBUILT
- Output-dir eksisterer
- Parallell replay aktiv
- Preflight ikke passert
- Tripwire invariants brutt
- Bundle SHA256 mangler
- Prebuilt path mangler eller ugyldig

## VERIFISERING

### Grep-kommandoer
```bash
# Sjekk at bundle_sha256 finnes
grep -R "bundle_sha256" -n reports/replay_eval

# Sjekk at prebuilt_used=true
grep -R "prebuilt_used" reports/replay_eval/*/perf_*.json

# Sjekk at basic_v1_call_count=0
grep -R "basic_v1_call_count" reports/replay_eval/*/perf_*.json

# Sjekk at feature_time_mean_ms <= 5
grep -R "feature_time_mean_ms" reports/replay_eval/*/perf_*.json
```

### Forventet output
- Alle perf JSON skal ha `ssot.bundle_sha256` (ikke null/None)
- Alle perf JSON skal ha `prebuilt_used=true` (hvis PREBUILT)
- Alle perf JSON skal ha `basic_v1_call_count=0` (hvis PREBUILT)
- Alle perf JSON skal ha `feature_time_mean_ms <= 5` (hvis PREBUILT)

## FORBUD

### IKKE gjør dette:
- IKKE optimaliser ytelse nå
- IKKE legg til nye features
- IKKE endre tradinglogikk
- IKKE innfør fallbacks "for safety"
- IKKE "la den gå litt til" ved feil
- IKKE bypass preflight gate

### FOKUS:
- Determinisme
- Fail-fast
- Null rom for menneskelig feil
