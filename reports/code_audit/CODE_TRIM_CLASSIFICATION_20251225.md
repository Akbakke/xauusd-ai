# GX1 Code Trim Classification

**Date:** 2025-12-25  
**Status:** ✅ **CLASSIFICATION COMPLETE**

---

## Executive Summary

**Filer analysert:** 7 filer  
**Kategorisering:**
- **[CORE]:** 2 filer (kritisk runtime)
- **[TOOL]:** 5 filer (nyttige dev-tools)
- **[LEGACY_CANDIDATE]:** 0 filer

**Konklusjon:** Ingen filer er kandidater for sletting. 2 filer er kritiske for runtime, 5 filer er nyttige dev-tools som bør beholdes eller flyttes til `gx1/tools/`.

---

## Klassifiseringstabell

| Fil | Kategori | Begrunnelse | Foreslått handling |
|-----|--------|-------------|-------------------|
| `gx1/prod/path_resolver.py` | **[CORE]** | Brukes i `oanda_demo_runner.py` for PROD_BASELINE path resolution. Kritisk for runtime når `meta.role == PROD_BASELINE`. | **Behold som er** |
| `gx1/prod/run_header.py` | **[CORE]** | Brukes i `oanda_demo_runner.py` for å generere `run_header.json` ved startup. Også brukt i mange analysis scripts (`parity_audit.py`, `prod_baseline_proof.py`, `reconcile_oanda.py`). Kritisk for runtime observability. | **Behold som er** |
| `gx1/prod/verify_freeze.py` | **[TOOL]** | Verifiserer PROD freeze struktur (sjekker at alle artifacts er på plass). Har CLI interface (`if __name__ == "__main__"`). Nevnt i `gx1/prod/current/README.md` som manuell verifiseringsverktøy. Ikke brukt i runtime. | **Flytt til `gx1/tools/`** (eller behold i `gx1/prod/` hvis det er logisk) |
| `gx1/execution/debug_oanda_ping.py` | **[TOOL]** | Enkel connectivity check mot OANDA REST API. Tester credentials, account summary, candles, og open trades. Har CLI interface. Nyttig for debugging OANDA connectivity issues. Ikke brukt i runtime. | **Flytt til `gx1/tools/`** |
| `gx1/execution/exec_smoke_test.py` | **[TOOL]** | Execution smoke test som sender en test trade til OANDA Practice API. Tester full execution pipeline (order → fill → close → journal). Har CLI interface med argparse. Nyttig for å verifisere execution plumbing. Ikke brukt i runtime. | **Flytt til `gx1/tools/`** eller behold i `gx1/execution/` hvis det er logisk (det er execution-relatert) |
| `gx1/tests/build_smoke_dataset.py` | **[TOOL]** | Utility script for å generere en liten bid/ask M5 dataset for CI smoke tests. Har CLI interface. Ikke brukt i runtime. | **Behold som er** (allerede i `gx1/tests/` som er riktig plassering) |
| `gx1/sniper/policy/test_q4_atrend_disable_policy.py` | **[TOOL]** | Pytest test for Q4 A_TREND disable policy. Ikke brukt i runtime. | **Behold som er** (allerede en test-fil, riktig plassering) |

---

## Detaljert Analyse

### [CORE] Filer

#### `gx1/prod/path_resolver.py`

**Bruk i runtime:**
- Brukes i `gx1/execution/oanda_demo_runner.py:4068` for PROD_BASELINE path resolution
- Resolver paths relative til `gx1/prod/current/` når `meta.role == PROD_BASELINE`
- Kritisk for å sikre at PROD_BASELINE runs bruker frozen artifacts

**Funksjoner:**
- `resolve_prod_path()` - Resolver paths for PROD_BASELINE mode
- `resolve_model_path()` - Resolver model paths for router

**Status:** ✅ **KRITISK** - Må bevares

---

#### `gx1/prod/run_header.py`

**Bruk i runtime:**
- Brukes i `gx1/execution/oanda_demo_runner.py:4005` for å generere `run_header.json` ved startup
- Brukes i `gx1/execution/exec_smoke_test.py:33` for smoke test run headers
- Brukes i mange analysis scripts:
  - `gx1/analysis/parity_audit.py` (load_run_header)
  - `gx1/analysis/prod_baseline_proof.py` (load_run_header)
  - `gx1/monitoring/reconcile_oanda.py` (load_run_header)
  - `gx1/analysis/diff_runs.py` (load_run_header)

**Funksjoner:**
- `generate_run_header()` - Genererer run_header.json med SHA256 hashes for artifacts
- `load_run_header()` - Laster run_header.json fra run directory
- `get_git_commit_hash()` - Henter git commit hash
- `compute_file_hash()` - Beregner SHA256 hash av fil

**Status:** ✅ **KRITISK** - Må bevares

---

### [TOOL] Filer

#### `gx1/prod/verify_freeze.py`

**Bruk:**
- Ikke brukt i runtime
- Har CLI interface (`if __name__ == "__main__"`)
- Nevnt i `gx1/prod/current/README.md` som manuelt verifiseringsverktøy

**Funksjoner:**
- `verify_prod_freeze()` - Verifiserer at alle required artifacts er på plass i PROD freeze directory
- `compute_file_hash()` - Beregner SHA256 hash av fil

**Relevans:**
- Nyttig for å verifisere PROD freeze struktur før deployment
- Kan brukes manuelt eller i CI/CD pipelines

**Status:** ✅ **NYTTIG TOOL** - Bør beholdes eller flyttes til `gx1/tools/`

---

#### `gx1/execution/debug_oanda_ping.py`

**Bruk:**
- Ikke brukt i runtime
- Har CLI interface (`if __name__ == "__main__"`)

**Funksjoner:**
- `main()` - Tester OANDA connectivity (account summary, candles, open trades)

**Relevans:**
- Nyttig for debugging OANDA connectivity issues
- Rask sanity check av credentials og API tilgang

**Status:** ✅ **NYTTIG TOOL** - Bør flyttes til `gx1/tools/`

---

#### `gx1/execution/exec_smoke_test.py`

**Bruk:**
- Ikke brukt i runtime
- Har CLI interface med argparse

**Funksjoner:**
- `run_smoke_test()` - Sender en test trade til OANDA Practice API og tester full execution pipeline
- `main()` - CLI entry point

**Relevans:**
- Nyttig for å verifisere execution plumbing (credentials → order → fill → close → journal)
- Tester at hele execution pipeline fungerer korrekt

**Status:** ✅ **NYTTIG TOOL** - Bør flyttes til `gx1/tools/` eller beholdes i `gx1/execution/` (execution-relatert)

---

#### `gx1/tests/build_smoke_dataset.py`

**Bruk:**
- Ikke brukt i runtime
- Har CLI interface (`if __name__ == "__main__"`)

**Funksjoner:**
- `build_smoke_dataset()` - Genererer en liten bid/ask M5 dataset for CI smoke tests

**Relevans:**
- Nyttig for CI/CD pipelines
- Allerede i `gx1/tests/` som er riktig plassering

**Status:** ✅ **NYTTIG TOOL** - Behold som er (riktig plassering)

---

#### `gx1/sniper/policy/test_q4_atrend_disable_policy.py`

**Bruk:**
- Ikke brukt i runtime
- Pytest test

**Funksjoner:**
- `test_q4_atrend_disable_default()` - Tester at Q4 × A_TREND defaults til disable
- `test_q4_atrend_disable_explicit()` - Tester eksplisitt disable config
- `test_q4_atrend_non_q4_not_blocked()` - Tester at non-Q4 ikke blokkeres
- `test_q4_atrend_non_atrend_not_blocked()` - Tester at non-A_TREND ikke blokkeres

**Relevans:**
- Unit test for Q4 A_TREND overlay
- Allerede en test-fil, riktig plassering

**Status:** ✅ **NYTTIG TOOL** - Behold som er (riktig plassering)

---

## Anbefalte Neste Steg

### Umiddelbar Handling

**Ingen kritiske endringer nødvendig.** Alle filer er enten kritiske for runtime eller nyttige dev-tools.

### Valgfri Reorganisering

1. **Flytt debug/dev tools til `gx1/tools/`:**
   - `gx1/execution/debug_oanda_ping.py` → `gx1/tools/debug_oanda_ping.py`
   - `gx1/execution/exec_smoke_test.py` → `gx1/tools/exec_smoke_test.py` (eller behold i `gx1/execution/` hvis det er logisk)

2. **Vurder flytting av `gx1/prod/verify_freeze.py`:**
   - Kan flyttes til `gx1/tools/verify_freeze.py` hvis det ikke er logisk å ha det i `gx1/prod/`
   - Eller behold i `gx1/prod/` hvis det er tett knyttet til PROD freeze struktur

### Opprett `gx1/tools/` Directory (hvis ikke eksisterer)

Hvis vi flytter filer, bør vi opprette `gx1/tools/` directory og kanskje legge til en `__init__.py` og `README.md` som dokumenterer at dette er dev-tools, ikke runtime-kode.

---

## Konklusjon

**Ingen filer er kandidater for sletting.** Alle analyserte filer er enten:
- Kritiske for runtime (2 filer)
- Nyttige dev-tools (5 filer)

**Anbefaling:** Behold alle filer. Valgfri reorganisering kan gjøres for å samle dev-tools i `gx1/tools/`, men dette er ikke kritisk.

---

**Report Generated:** 2025-12-25  
**Status:** ✅ **CLASSIFICATION COMPLETE - NO DELETIONS RECOMMENDED**

