# PREBUILT KARTLEGGING - RESULTAT

**Dato:** 2025-01-13  
**Status:** Systematisk gjennomgang utført

## A) KONKRET SJEKKLISTE (UTFØRT)

### ✅ 1. Finn alle referanser til "replay_eval_gated_parallel"
**Resultat:** 26 treff funnet
- `scripts/go_nogo_prebuilt.sh`: linje 69, 88, 184
- `scripts/run_fullyear_prebuilt.sh`: linje 129
- `docs/PREBUILT_REPLAY_FLOW.md`: linje 61
- `docs/PREBUILT_IMPLEMENTATION_STATUS.md`: flere referanser
- `reports/replay_eval/*/chunk_*/chunk_footer.json`: tracebacks viser linje 424, 1418, 1288, 1622, 1107, 1002
- `reports/replay_eval/GATED/perf_*.json`: tracebacks viser linje 842, 1622

### ✅ 2. Finn alle replay-entrypoints som kjøres fra scripts/
**Resultat:** 
- `scripts/go_nogo_prebuilt.sh`: kaller `python3 gx1/scripts/replay_eval_gated_parallel.py` med args:
  - `--policy`, `--data`, `--workers`, `--output-dir`, `--run-id`, `--days`, `--abort-after-first-chunk`
- `scripts/run_fullyear_prebuilt.sh`: kaller samme script med:
  - `--policy`, `--data`, `--workers`, `--output-dir`, `--run-id`
- Andre scripts bruker `scripts/run_mini_replay_perf.py` (forskjellig interface)

### ✅ 3. Finn alle kandidater til gated replay runner
**Resultat:**
- `gx1/scripts/replay_with_shadow_parallel.py`: parallel replay, men ikke gated
- `scripts/active/replay_entry_exit_parallel.py`: parallel replay, men ikke gated
- `scripts/run_mini_replay_perf.py`: replay med perf tracking, men ikke gated parallel
- **INGEN fil funnet som matcher argumentene i go_nogo_prebuilt.sh**

### ✅ 4. Finn hvor worker-spawn skjer
**Resultat:**
- `gx1/scripts/replay_with_shadow_parallel.py`: bruker subprocess.Popen
- `scripts/active/replay_entry_exit_parallel.py`: bruker joblib.Parallel
- `gx1/inference/model_loader_worker.py`: bruker multiprocessing.Process
- **INGEN fil funnet som matcher gated parallel worker spawn**

### ✅ 5. Finn hvor chunk footer / perf JSON / invariants evalueres
**Resultat:**
- `reports/replay_eval/GATED/chunk_0/chunk_footer.json`: EKSISTERER og viser:
  - `prebuilt_used: true`
  - `basic_v1_call_count: 0`
  - `feature_time_mean_ms: 0.0`
  - `prebuilt_bypass_count: 6299`
- `reports/replay_eval/GATED/perf_*.json`: EKSISTERER og viser chunk-struktur
- **Men ingen Python-fil funnet som genererer disse**

### ✅ 6. Sjekk om filen eksisterer under annet navn
**Resultat:**
- `find gx1 -maxdepth 4 -type f -name "*gated*replay*"`: INGEN
- `find gx1 -maxdepth 4 -type f -name "*replay*parallel*"`: INGEN
- `find gx1 -maxdepth 4 -type f -name "*replay_eval*"`: INGEN

### ✅ 7. Git-historikk
**Resultat:**
- `git log --all --name-only --follow -- "**/replay_eval_gated_parallel.py"`: INGEN
- `git log --all --diff-filter=A --name-only -- "**/replay_eval_gated_parallel.py"`: INGEN
- `git log --all --diff-filter=D --name-only -- "**/replay_eval_gated_parallel.py"`: INGEN
- `git log --all --diff-filter=R --name-only -- "**/replay_eval_gated_parallel.py"`: INGEN
- **Filen har ALDRI vært i git (ikke tracked)**

## B) FUNN

### Bevis for at filen HAR EKSISTERT og BLITT KJØRT:
1. **Tracebacks i chunk_footer.json:**
   - `File "/Users/andrekildalbakke/Desktop/GX1 XAUUSD/gx1/scripts/replay_eval_gated_parallel.py", line 424, in process_chunk`
   - `File "/Users/andrekildalbakke/Desktop/GX1 XAUUSD/gx1/scripts/replay_eval_gated_parallel.py", line 1418, in watchdog_thread`
   - `File "/Users/andrekildalbakke/Desktop/GX1 XAUUSD/gx1/scripts/replay_eval_gated_parallel.py", line 842, in export_perf_json_from_footers`

2. **Perf JSON-filer i reports/replay_eval/GATED/:**
   - Struktur viser at systemet faktisk har kjørt
   - Chunk footers viser prebuilt_used=true, basic_v1_call_count=0

3. **Absolutt path i tracebacks:**
   - `/Users/andrekildalbakke/Desktop/GX1 XAUUSD/gx1/scripts/replay_eval_gated_parallel.py`
   - Dette er UTENFOR worktree (`/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/cia`)

### Filen finnes IKKE i workspace:
- `ls -la "/Users/andrekildalbakke/Desktop/GX1 XAUUSD/gx1/scripts/replay_eval_gated_parallel.py"`: INGEN
- `find . -name "replay_eval_gated_parallel.py"`: INGEN
- `git ls-files | grep replay_eval_gated_parallel`: INGEN

### Alternativer funnet:
- `scripts/run_mini_replay_perf.py`: Replay med perf tracking, men ikke gated parallel
- `gx1/scripts/replay_with_shadow_parallel.py`: Parallel replay, men ikke gated
- `scripts/active/replay_entry_exit_parallel.py`: Parallel replay, men ikke gated

## C) KONKLUSJON

### Hovedfunn:
**`gx1/scripts/replay_eval_gated_parallel.py` finnes IKKE i workspace, men HAR eksistert og blitt kjørt tidligere.**

### Bevis:
1. Tracebacks viser at filen faktisk har blitt kjørt (linje 424, 842, 1418, 1622, etc.)
2. Perf JSON-filer og chunk footers viser at gated replay faktisk har kjørt
3. Filen har ALDRI vært i git (ikke tracked)
4. Filen ligger utenfor worktree (absolutt path i tracebacks)

### Sannsynlig forklaring:
1. Filen ligger i hovedrepo (ikke i worktree)
2. Filen er git-ignored
3. Filen er flyttet/omdøpt, men scripts ikke oppdatert
4. Filen er i en submodule eller ekstern path

### Hvilken fil ER gated runner i dag?
**UKLART** - Ingen fil i workspace matcher argumentene eller funksjonaliteten som beskrevet i tracebacks.

## D) TILTAK (MINIMALE ENDRINGER)

### Umiddelbare tiltak:
1. **Verifiser om filen ligger i hovedrepo:**
   - Sjekk `/Users/andrekildalbakke/Desktop/GX1 XAUUSD/gx1/scripts/replay_eval_gated_parallel.py`
   - Hvis den finnes der: kopier til worktree eller oppdater scripts til å bruke absolutt path

2. **Hvis filen faktisk mangler:**
   - Opprett `gx1/scripts/replay_eval_gated_parallel.py` basert på:
     - Argumentene i go_nogo_prebuilt.sh
     - Strukturen i chunk_footer.json og perf JSON
     - Funksjonaliteten i lignende scripts (replay_with_shadow_parallel.py)

3. **Oppdater dokumentasjon:**
   - Dokumenter hvor filen faktisk ligger
   - Oppdater referanser hvis filen er flyttet

### Spøkelsesreferanser som må oppdateres:
- `scripts/go_nogo_prebuilt.sh`: linje 69, 88, 184
- `scripts/run_fullyear_prebuilt.sh`: linje 129
- `docs/PREBUILT_REPLAY_FLOW.md`: linje 61

## E) AKTIV FILINVENTAR

| Component | Active file path | Called by | Notes |
|-----------|------------------|----------|-------|
| **Scripts** | `scripts/go_nogo_prebuilt.sh` | Manuell | Preflight check |
| **Scripts** | `scripts/run_fullyear_prebuilt.sh` | Manuell | FULLYEAR replay |
| **Runner** | `gx1/scripts/replay_eval_gated_parallel.py` | **MANGLER** | Referert men ikke funnet |
| **Runner (alt)** | `scripts/run_mini_replay_perf.py` | Mange scripts | Replay med perf tracking |
| **Runner (alt)** | `gx1/scripts/replay_with_shadow_parallel.py` | Manuell | Shadow replay parallel |
| **Worker** | **UKLART** | **UKLART** | Må finnes i replay_eval_gated_parallel.py |
| **Aggregator** | **UKLART** | **UKLART** | Må finnes i replay_eval_gated_parallel.py |
| **Loader** | **UKLART** | **UKLART** | Prebuilt loader må finnes |
| **Docs** | `docs/PREBUILT_REPLAY_FLOW.md` | Referanse | Kanonisk dokumentasjon |

## F) NESTE STEG

1. **Verifiser hovedrepo:**
   ```bash
   ls -la "/Users/andrekildalbakke/Desktop/GX1 XAUUSD/gx1/scripts/replay_eval_gated_parallel.py"
   ```

2. **Hvis filen finnes i hovedrepo:**
   - Kopier til worktree, eller
   - Oppdater scripts til å bruke absolutt path

3. **Hvis filen mangler:**
   - Opprett basert på tracebacks og argumentene
   - Implementer funksjonalitet som matcher chunk_footer.json struktur

4. **Verifiser at filen faktisk er aktiv:**
   - Kjør go_nogo_prebuilt.sh og se om den finner filen
   - Hvis ikke: opprett eller oppdater path
