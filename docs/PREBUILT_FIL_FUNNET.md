# PREBUILT FIL FUNNET - RESULTAT

**Dato:** 2025-01-13  
**Status:** Filen funnet og kopiert til cia worktree

## FUNN

### Filen funnet i:
- ✅ `aaq/gx1/scripts/replay_eval_gated_parallel.py` - EKSISTERER
- ✅ `muo/gx1/scripts/replay_eval_gated_parallel.py` - EKSISTERER
- ❌ `cia/gx1/scripts/replay_eval_gated_parallel.py` - MANGLER (nå kopiert)

### Filen kopiert til cia:
- ✅ Kopiert fra `aaq` til `cia`
- ✅ Verifisert at filen eksisterer i cia worktree

## FILSTRUKTUR

### Funksjoner identifisert:
- `process_chunk` (linje 195): Prosesserer en chunk i worker process
- `export_perf_json_from_footers` (linje 845): Eksporterer perf JSON fra chunk footers
- `watchdog_thread` (linje 1449): Watchdog thread for graceful shutdown
- `main` (linje 1305): Main entry point

### Argumenter støttet:
- `--policy`: Policy YAML path
- `--data`: Input data (parquet)
- `--workers`: Number of parallel workers
- `--output-dir`: Output directory
- `--run-id`: Run ID
- `--days`: Use only first N days
- `--abort-after-first-chunk`: Stop after first chunk
- `--abort-after-n-bars-per-chunk`: Stop each chunk after N bars
- `--dry-run-prebuilt-check`: Only load prebuilt + run checks

### FASE 0.1 implementert:
- Global lock via pidfile (linje 1343-1349)
- psutil-basert parallel replay detection (linje 1320-1341)

## NESTE STEG

Nå som filen er kopiert til cia worktree, må vi:
1. Verifisere at alle sikkerhetssjekker er på plass
2. Legge inn PREBUILT-spesifikke sjekker (sys.modules, GX1_FEATURE_BUILD_DISABLED, etc.)
3. Verifisere tripwire-sjekker i chunk footer/aggregator
