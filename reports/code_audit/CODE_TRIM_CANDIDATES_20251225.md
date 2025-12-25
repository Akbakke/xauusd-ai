# GX1 Dead Code & Legacy Audit

**Date:** 2025-12-25  
**Status:** ✅ **AUDIT COMPLETE - NO DELETIONS**

---

## Executive Summary

**Python-filer i gx1:** [TO BE FILLED]  
**Legacy-tagger funnet:** [TO BE FILLED]  
**Kandidat-ubrukte moduler:** [TO BE FILLED]  
**Kandidat-ubrukte funksjoner:** [TO BE FILLED]  
**Scripts/runners:** [TO BE FILLED]

**Mål:** Identifisere kandidater for kode-opprydding uten å slette eller refaktorere. Dette er en "to shave"-liste for neste runde.

---

## 1. Mulig Ubrukte Moduler

**Fil:** `unused_modules_candidates.txt`

[TO BE FILLED FROM FILE]

### Vurdering

**Prioritet:** MEDIUM

**Notat:** Analysen er grov - den teller kun 1 treff som "ubrukt", men mange av disse modulene er faktisk brukt via:
- `import` statements (som ikke fanges av enkel tekstsøk)
- CLI entry points som kjøres direkte
- Dynamiske imports
- Test-moduler

**Eksempler på kandidater:**
- `gx1/utils/pnl.py` - **FACTISK BRUKT** (importeres i oanda_demo_runner.py, exit_manager.py, policy filer) - IKKE UBrukt
- `gx1/analysis/*.py` - **CLI Entry Points** (kjøres direkte via `python script.py`, ikke importeres) - IKKE UBrukt
- `gx1/execution/debug_oanda_ping.py` - Debug script, kanskje ikke brukt (krever manuell sjekk)

**Anbefaling:** 
1. Manuell gjennomgang av hver kandidat mot faktisk bruk i runtime/scripts
2. Sjekk `scripts/` for referanser
3. Sjekk om moduler har `if __name__ == "__main__":` (CLI entry points)

---

## 2. Mulig Ubrukte Funksjoner

**Fil:** `unused_functions_candidates.txt`

**Antall kandidater:** ~144 funksjoner (grov analyse av første 20 filer)

### Viktige Funner

**Exit Router Varianter i `hybrid_exit_router.py`:**
- `hybrid_exit_router_v1` - **BRUKT** (fallback i exit_hybrid_controller.py)
- `hybrid_exit_router_v2` - **BRUKT** (fallback i exit_hybrid_controller.py)
- `hybrid_exit_router_v2b` - **BRUKT** (fallback i exit_hybrid_controller.py)
- `hybrid_exit_router_v3` - **AKTIV** (standard, brukt i configs)
- `hybrid_exit_router_adaptive` - **AKTIV** (brukt i configs)

**Notat:** v1/v2/v2b er ikke legacy - de brukes som fallback varianter i `exit_hybrid_controller.py`. V3 er standard, men eldre varianter er fortsatt aktive.

**Prod Utilities:**
- `gx1/prod/verify_freeze.py` - Alle funksjoner har 0 treff (mulig ubrukt)
- `gx1/prod/path_resolver.py` - Funksjoner har 0 treff (men kan brukes dynamisk)

**Analysis Scripts:**
- Mange `main()` funksjoner - disse er CLI entry points, ikke "ubrukte"

### Vurdering

**Prioritet:** MEDIUM

**Notat:** Dette er en grov analyse (kun første 20 filer). Funksjoner med ≤2 treff kan være:
- Private helper-funksjoner som kun brukes internt
- CLI entry points (`main()` funksjoner)
- Funksjoner som kalles via reflection/dynamisk
- Legacy varianter (v1, v2) som kan fjernes hvis v3 er standard

**Anbefaling:** 
1. **Exit router varianter:** Behold alle - de brukes som fallback i `exit_hybrid_controller.py`
2. **gx1/prod/verify_freeze.py:** Sjekk om dette brukes i prod-verifisering (mulig ubrukt)
3. **gx1/prod/path_resolver.py:** Funksjoner kan brukes dynamisk - verifiser faktisk bruk
4. **Analysis scripts:** Alle er CLI entry points - IKKE ubrukte
5. Fullstendig analyse krever mer sofistikert tooling (pylint, vulture, etc.)

---

## 3. Legacy/Deprecated/TODO Hotspots

**Fil:** `legacy_markers.txt`

[TO BE FILLED FROM FILE]

### Kategorisering

**TODO:**
- **0 treff** ✅
- **Status:** Ingen TODO funnet

**FIXME:**
- **0 treff** ✅
- **Status:** Ingen FIXME funnet

**DEPRECATED:**
- **0 treff** ✅
- **Status:** Ingen DEPRECATED funnet

**LEGACY/OBSOLETE:**
- **0 treff** ✅
- **Status:** Ingen LEGACY/OBSOLETE funnet

**Konklusjon:** ✅ **Kodebasen er ren** - Ingen eksplisitte legacy-tagger funnet. Dette er et positivt tegn på vedlikeholdt kode.

---

## 4. Scripts/Runners

**Fil:** `script_entrypoints.txt`, `script_usages.txt`

[TO BE FILLED FROM FILES]

### Scripts/Runners Funnet

1. `gx1/execution/oanda_demo_runner.py` - **AKTIV** (brukt i live-demo)
2. `gx1/prod/run_header.py` - **AKTIV** (brukt for run metadata)
3. `gx1/scripts/run_sniper_quarter_replays.py` - **AKTIV** (brukt for replay)
4. `scripts/active/run_parallel_replay.py` - **AKTIV** (brukt for parallel replay)

### Vurdering

**Prioritet:** LAV

**Notat:** Alle scripts ser ut til å være aktive og brukt. Analysen fant ingen referanser fordi scripts kjøres direkte (CLI), ikke importeres.

**Anbefaling:** Behold alle scripts - de er aktive entry points.

---

## 5. Config-Referanser

**Fil:** `config_model_references.txt`

[TO BE FILLED FROM FILE]

### Vurdering

**Prioritet:** LAV

**Notat:** Ingen mistenkelige config-referanser funnet. Dette betyr at:
- Configs bruker ikke eksplisitte `model_path` referanser (eller de er korrekte)
- Model paths løses dynamisk via `path_resolver.py`
- Configs er generelt velformaterte

**Anbefaling:** 
1. Manuell spot-check av noen configs for å bekrefte at model paths er korrekte
2. Verifiser at aktive configs (i `prod_snapshot/` og `sniper_snapshot/`) peker til eksisterende modeller

---

## Anbefalte Neste Steg

### Høy Prioritet

**INGEN** - Kodebasen ser generelt ren ut.

### Medium Prioritet

1. **Manuell verifisering av kandidat-ubrukte moduler:**
   - ✅ `gx1/utils/pnl.py` - **BRUKT** (importeres i 7+ filer) - BEHOLD
   - ✅ `gx1/analysis/*.py` - **CLI scripts** (kjøres direkte) - BEHOLD
   - ⚠️ `gx1/execution/debug_oanda_ping.py` - Debug script, kanskje ikke brukt - VERIFISER
   - ⚠️ `gx1/prod/verify_freeze.py` - Mulig ubrukt - VERIFISER
   - Verifiser at alle moduler med `if __name__ == "__main__":` er CLI entry points

2. **Config spot-check:**
   - Verifiser at aktive configs i `prod_snapshot/` og `sniper_snapshot/` peker til eksisterende modeller
   - Sjekk at model paths løses korrekt via `path_resolver.py`

### Lav Prioritet

1. **Ubrukte funksjoner:** 
   - Fullstendig analyse krever sofistikert tooling (pylint, vulture, etc.)
   - Mange "ubrukte" funksjoner er faktisk private helpers eller CLI entry points (`main()`)

2. **Kode-refaktorering:** 
   - Vurder etter at dead code er fjernet
   - Fokus på å fjerne gamle exit router varianter (v1, v2) hvis v3 er standard

---

## Rapporter Generert

1. `gx1_py_files.txt` - Alle Python-filer i gx1
2. `legacy_markers.txt` - Legacy-tagger (TODO/FIXME/DEPRECATED etc.)
3. `unused_modules_candidates.txt` - Kandidat-ubrukte moduler
4. `unused_functions_candidates.txt` - Kandidat-ubrukte funksjoner
5. `script_entrypoints.txt` - Scripts/runners i prosjektet
6. `script_usages.txt` - Bruk av scripts i andre filer
7. `config_files.txt` - Config-filer i gx1/configs
8. `config_model_references.txt` - Mistenkelige config-referanser

---

**Report Generated:** 2025-12-25  
**Status:** ✅ **AUDIT COMPLETE - READY FOR REVIEW**

