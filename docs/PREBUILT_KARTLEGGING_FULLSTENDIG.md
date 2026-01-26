# PREBUILT KARTLEGGING - FULLSTENDIG RESULTAT

**Dato:** 2025-01-13  
**Status:** Systematisk gjennomgang fullført

## A) KONKRET SJEKKLISTE (UTFØRT)

### ✅ 1. Finn alle referanser til "replay_eval_gated_parallel"
**Kommando:** `grep -r "replay_eval_gated_parallel" .`
**Resultat:** 26 treff
- `scripts/go_nogo_prebuilt.sh`: linje 69, 88, 184
- `scripts/run_fullyear_prebuilt.sh`: linje 129
- `docs/PREBUILT_REPLAY_FLOW.md`: linje 61
- `reports/replay_eval/*/chunk_*/chunk_footer.json`: tracebacks
- `reports/replay_eval/GATED/perf_*.json`: tracebacks

**Traceback-linjer funnet:**
- Linje 424: `process_chunk`
- Linje 842: `export_perf_json_from_footers`
- Linje 1002, 1107, 1288, 1418, 1622: `watchdog_thread`, `main`

### ✅ 2. Finn alle replay-entrypoints som kjøres fra scripts/
**Kommando:** `grep -n "python.*replay" scripts gx1`
**Resultat:**
- `scripts/go_nogo_prebuilt.sh` linje 69, 88, 184:
  ```bash
  python3 gx1/scripts/replay_eval_gated_parallel.py \
    --policy ... \
    --data ... \
    --workers ... \
    --output-dir ... \
    --run-id ... \
    --days ... \
    --abort-after-first-chunk ...
  ```
- `scripts/run_fullyear_prebuilt.sh` linje 129: samme interface

### ✅ 3. Finn alle kandidater til gated replay runner
**Kommando:** `grep -n "(gated|GATED|gated_fusion)" gx1`
**Resultat:**
- `gx1/models/entry_v10/gated_fusion.py`: GatedFusion klasse
- `gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py`: bruker gated fusion
- `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml`: GATED policy
- **INGEN replay runner funnet**

**Kommando:** `grep -n "ReplayMode|PREBUILT|BASELINE" gx1`
**Resultat:**
- `gx1/core/hybrid_exit_router.py`: PROD_BASELINE referanser
- `gx1/configs/policies/`: PROD_BASELINE i policy-filer
- **INGEN ReplayMode enum funnet**

**Kommando:** `grep -n "(worker|spawn|multiprocessing|chunk|footer|aggregat|perf_json)" gx1/scripts`
**Resultat:**
- `gx1/scripts/replay_with_shadow_parallel.py`: worker, chunk, spawn
- `gx1/scripts/run_sniper_quarter_replays.py`: multiprocessing
- **INGEN gated parallel runner funnet**

### ✅ 4. Finn hvor worker-spawn skjer
**Kommando:** `grep -n "(apply_async|Pool\\(|Process\\(|multiprocessing)" gx1`
**Resultat:**
- `gx1/inference/model_loader_worker.py`: multiprocessing.Process
- `gx1/rl/entry_v10/train_entry_transformer_v10.py`: multiprocessing
- `gx1/scripts/run_sniper_quarter_replays.py`: multiprocessing
- **INGEN gated parallel worker spawn funnet**

### ✅ 5. Finn hvor chunk footer / perf JSON / invariants evalueres
**Kommando:** `grep -n "(chunk.*footer|footer|perf_json|FEATURE_BUILD_TIMEOUT|basic_v1_call_count|feature_time_mean_ms|prebuilt_bypass_count)" gx1`
**Resultat:**
- `gx1/scripts/preflight_full_build_sanity.py`: FEATURE_BUILD_TIMEOUT_MS
- **INGEN chunk footer generator funnet**

**Eksisterende chunk_footer.json:**
- `reports/replay_eval/GATED/chunk_0/chunk_footer.json`: EKSISTERER
- Viser: `prebuilt_used: true`, `basic_v1_call_count: 0`, `feature_time_mean_ms: 0.0`

### ✅ 6. Sjekk om filen eksisterer under annet navn
**Kommando:** `find gx1 -maxdepth 4 -type f -name "*gated*replay*" -o -name "*replay*parallel*" -o -name "*replay_eval*"`
**Resultat:** INGEN

### ✅ 7. Git-historikk
**Kommando:** `git log --all --name-only --follow -- "**/replay_eval_gated_parallel.py"`
**Resultat:** INGEN (filen har ALDRI vært i git)

**Kommando:** `git log --all --diff-filter=A/D/R --name-only -- "**/replay_eval_gated_parallel.py"`
**Resultat:** INGEN (filen har ALDRI vært tracked)

## B) FUNN

### Bevis for at filen HAR EKSISTERT:
1. **Tracebacks i chunk_footer.json:**
   ```
   File "/Users/andrekildalbakke/Desktop/GX1 XAUUSD/gx1/scripts/replay_eval_gated_parallel.py", line 424, in process_chunk
   File "/Users/andrekildalbakke/Desktop/GX1 XAUUSD/gx1/scripts/replay_eval_gated_parallel.py", line 842, in export_perf_json_from_footers
   File "/Users/andrekildalbakke/Desktop/GX1 XAUUSD/gx1/scripts/replay_eval_gated_parallel.py", line 1622, in main
   ```

2. **Perf JSON-filer:**
   - `reports/replay_eval/GATED/perf_prebuilt_fullyear_*.json`: EKSISTERER
   - Struktur viser: `chunks`, `chunks_statuses`, `chunks_completed`, `total_model_calls`

3. **Chunk footers:**
   - `reports/replay_eval/GATED/chunk_*/chunk_footer.json`: EKSISTERER
   - Viser prebuilt invariants: `prebuilt_used`, `basic_v1_call_count`, `feature_time_mean_ms`

### Filen finnes IKKE i workspace:
- `find . -name "replay_eval_gated_parallel.py"`: INGEN
- `git ls-files | grep replay_eval_gated_parallel`: INGEN
- `ls -la "/Users/andrekildalbakke/Desktop/GX1 XAUUSD/gx1/scripts/replay_eval_gated_parallel.py"`: MÅ VERIFISERES

### Absolutt path i tracebacks:
- `/Users/andrekildalbakke/Desktop/GX1 XAUUSD/gx1/scripts/replay_eval_gated_parallel.py`
- Dette er UTENFOR worktree (`/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/cia`)

## C) KONKLUSJON

### Hovedfunn:
**`gx1/scripts/replay_eval_gated_parallel.py` finnes IKKE i workspace, men HAR eksistert og blitt kjørt tidligere fra absolutt path.**

### Sannsynlig forklaring:
1. **Filen ligger i hovedrepo (utenfor worktree):**
   - Absolutt path: `/Users/andrekildalbakke/Desktop/GX1 XAUUSD/gx1/scripts/replay_eval_gated_parallel.py`
   - Worktree: `/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/cia`
   - Filen er IKKE i worktree, men scripts refererer til den

2. **Filen er git-ignored eller untracked:**
   - Git-historikk viser at filen ALDRI har vært tracked
   - Kan være i .gitignore eller untracked

3. **Filen er flyttet/omdøpt:**
   - Men scripts ikke oppdatert (spøkelsesreferanse)

### Hvilken fil ER gated runner i dag?
**UKLART** - Ingen fil i workspace matcher. Filen må:
- Ligger i hovedrepo (utenfor worktree), ELLER
- Må opprettes basert på tracebacks og argumentene

## D) TILTAK (MINIMALE ENDRINGER)

### Umiddelbare tiltak:
1. **Verifiser hovedrepo:**
   ```bash
   test -f "/Users/andrekildalbakke/Desktop/GX1 XAUUSD/gx1/scripts/replay_eval_gated_parallel.py"
   ```
   - Hvis EXISTS: kopier til worktree eller oppdater scripts
   - Hvis NOT_FOUND: filen må opprettes

2. **Hvis filen finnes i hovedrepo:**
   - Kopier til: `gx1/scripts/replay_eval_gated_parallel.py` i worktree
   - ELLER oppdater scripts til å bruke absolutt path

3. **Hvis filen mangler:**
   - Opprett `gx1/scripts/replay_eval_gated_parallel.py` basert på:
     - Argumentene: `--policy`, `--data`, `--workers`, `--output-dir`, `--run-id`, `--days`, `--abort-after-first-chunk`
     - Funksjoner: `process_chunk` (linje 424), `export_perf_json_from_footers` (linje 842), `watchdog_thread` (linje 1418), `main` (linje 1622)
     - Struktur: chunk footers, perf JSON, worker spawn

### Spøkelsesreferanser som må oppdateres:
- `scripts/go_nogo_prebuilt.sh`: linje 69, 88, 184
- `scripts/run_fullyear_prebuilt.sh`: linje 129
- `docs/PREBUILT_REPLAY_FLOW.md`: linje 61

## E) AKTIV FILINVENTAR

| Component | Active file path | Called by | Notes |
|-----------|------------------|-----------|-------|
| **Scripts** | `scripts/go_nogo_prebuilt.sh` | Manuell | Preflight check, kaller replay_eval_gated_parallel.py |
| **Scripts** | `scripts/run_fullyear_prebuilt.sh` | Manuell | FULLYEAR replay, kaller replay_eval_gated_parallel.py |
| **Runner** | `gx1/scripts/replay_eval_gated_parallel.py` | **MANGLER I WORKTREE** | Referert, tracebacks viser eksistens |
| **Runner (alt)** | `scripts/run_mini_replay_perf.py` | Mange scripts | Replay med perf tracking, forskjellig interface |
| **Runner (alt)** | `gx1/scripts/replay_with_shadow_parallel.py` | Manuell | Shadow replay parallel, ikke gated |
| **Runner (alt)** | `scripts/active/replay_entry_exit_parallel.py` | scripts/run_replay.sh | Parallel replay, ikke gated |
| **Worker** | **I replay_eval_gated_parallel.py** | **UKLART** | Må finnes i filen (linje 424: process_chunk) |
| **Aggregator** | **I replay_eval_gated_parallel.py** | **UKLART** | Må finnes i filen (linje 842: export_perf_json_from_footers) |
| **Loader** | **UKLART** | **UKLART** | Prebuilt loader må finnes |
| **Docs** | `docs/PREBUILT_REPLAY_FLOW.md` | Referanse | Kanonisk dokumentasjon |

## F) NESTE STEG

1. **Verifiser hovedrepo:**
   ```bash
   test -f "/Users/andrekildalbakke/Desktop/GX1 XAUUSD/gx1/scripts/replay_eval_gated_parallel.py"
   ```

2. **Hvis filen finnes:**
   - Kopier til worktree: `cp "/Users/andrekildalbakke/Desktop/GX1 XAUUSD/gx1/scripts/replay_eval_gated_parallel.py" gx1/scripts/`
   - ELLER oppdater scripts til absolutt path

3. **Hvis filen mangler:**
   - Opprett basert på tracebacks og argumentene
   - Implementer funksjonalitet som matcher chunk_footer.json struktur

4. **Verifiser at filen faktisk er aktiv:**
   - Kjør go_nogo_prebuilt.sh og se om den finner filen
   - Hvis ikke: opprett eller oppdater path
