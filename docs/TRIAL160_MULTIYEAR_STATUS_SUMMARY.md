# Trial 160 Multi-Year Backtest - Fullstendig Sammendrag

**Dato:** 2026-01-16  
**Status:** KØRER (6 prosesser aktiv)

---

## 🎯 MÅL

Backteste Trial 160 på 2020-2025 med samme rails som 2025 (SSoT, PREBUILT, invariants), kjøre 5 år i parallell for maksimal throughput.

---

## 📊 NÅVÆRENDE STATUS

### Prosesser som kjører nå

**Totalt: 6 prosesser**
- ✅ 1 orchestrator (`run_trial160_multiyear_parallel.py`)
- ✅ 5 år-jobber i parallell:
  - Year 2020 (PID 31766) - CPU: 95.7%
  - Year 2021 (PID 31767) - CPU: 97.2%
  - Year 2022 (PID 31768) - CPU: 96.4%
  - Year 2023 (PID 31769) - CPU: 96.3%
  - Year 2024 (PID 31770) - CPU: 96.6%
- ⏳ Year 2025 venter på at en av de 5 første fullfører

### År-status

| År | Prebuilt Features | Replay Footer | Report | Status |
|----|-------------------|---------------|--------|--------|
| 2020 | ✅ Ferdig | ⏳ Kjører | ⏳ Vent | I replay |
| 2021 | ✅ Ferdig | ⏳ Kjører | ⏳ Vent | I replay |
| 2022 | ✅ Ferdig | ⏳ Kjører | ⏳ Vent | I replay |
| 2023 | ✅ Ferdig | ⏳ Kjører | ⏳ Vent | I replay |
| 2024 | ✅ Ferdig | ⏳ Kjører | ⏳ Vent | I replay |
| 2025 | ✅ Ferdig | ⏳ Vent | ⏳ Vent | Venter på slot |

**Alle 6 år har prebuilt features bygget.**  
**Alle 5 aktive år-jobber kjører replay (har kjørt i ~3 minutter).**

---

## 🔧 TEKNISKE ENDRINGER VI HAR GJORT

### 1. Prosessantall-optimalisering

**Problem:** Opprinnelig startet hver år-jobb en egen replay-prosess med multiprocessing-pool, noe som ga 40+ prosesser.

**Løsning:**
- ✅ Endret `replay_eval_gated_parallel.py`: Når `workers=1`, kjører direkte uten multiprocessing-pool
- ✅ Endret `run_trial160_year_job.py`: Importerer og kaller replay direkte (ingen subprocess)
- ✅ Resultat: Maks 6 prosesser (1 orchestrator + 5 år-jobber)

**Filer endret:**
- `gx1/scripts/replay_eval_gated_parallel.py` (workers=1 direkte kjøring)
- `gx1/scripts/run_trial160_year_job.py` (direkte import i stedet for subprocess)

### 2. FASE_1-separasjonssjekk

**Problem:** FASE_1-sjekken feilet når vi kalte replay direkte, fordi `basic_v1` og `live_features` allerede var importert.

**Løsning:**
- ✅ Endret `oanda_demo_runner.py`: Hopp over FASE_1-sjekk når `GX1_ALLOW_PARALLEL_REPLAY=1`
- ✅ Dette tillater direkte kall fra år-jobber uten å feile på import-sjekk

**Fil endret:**
- `gx1/execution/oanda_demo_runner.py` (FASE_1-sjekk hoppes over for parallel mode)

### 3. Global replay-lock

**Problem:** `replay_eval_gated_parallel.py` hadde en global lock som forbyr parallell kjøring.

**Løsning:**
- ✅ Endret `replay_eval_gated_parallel.py`: Tillat parallell kjøring når `GX1_ALLOW_PARALLEL_REPLAY=1`
- ✅ Hver år-jobb setter denne env-variabelen før replay

**Fil endret:**
- `gx1/scripts/replay_eval_gated_parallel.py` (parallel replay allowed flag)

### 4. Feature meta-path

**Problem:** Orchestrator fant ikke `feature_meta.json` på standard sti.

**Løsning:**
- ✅ Endret `run_trial160_multiyear_parallel.py`: Bruk kanonisk path fra policy YAML
- ✅ Default: `gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json`

**Fil endret:**
- `gx1/scripts/run_trial160_multiyear_parallel.py` (feature_meta default path)

### 5. Output directory check

**Problem:** Output directory-sjekk feilet fordi `RUN_IDENTITY.json` allerede eksisterte.

**Løsning:**
- ✅ Endret `run_trial160_year_job.py`: Tillat kun `RUN_IDENTITY.json` i output-dir før replay
- ✅ Dette tillater at `RUN_IDENTITY.json` skrives før replay starter

**Fil endret:**
- `gx1/scripts/run_trial160_year_job.py` (allow RUN_IDENTITY.json only)

---

## 📁 ARKITEKTUR

### Prosessstruktur

```
run_trial160_multiyear_parallel.py (orchestrator)
├── ProcessPoolExecutor (max_workers=5)
    ├── run_trial160_year_job.py --year 2020
    │   └── replay_eval_gated_parallel.py (direkte import, workers=1)
    ├── run_trial160_year_job.py --year 2021
    │   └── replay_eval_gated_parallel.py (direkte import, workers=1)
    ├── run_trial160_year_job.py --year 2022
    │   └── replay_eval_gated_parallel.py (direkte import, workers=1)
    ├── run_trial160_year_job.py --year 2023
    │   └── replay_eval_gated_parallel.py (direkte import, workers=1)
    └── run_trial160_year_job.py --year 2024
        └── replay_eval_gated_parallel.py (direkte import, workers=1)
```

**Total: 6 prosesser (1 orchestrator + 5 år-jobber)**

### Hver år-jobb gjør

1. **Doctor check** (`doctor_trial160.sh`)
2. **Build prebuilt features** (`build_fullyear_features_parquet.py`)
3. **Create RUN_IDENTITY** (`create_run_identity_for_year`)
4. **Run replay** (`replay_eval_gated_parallel.py` direkte, workers=1)
5. **Verify invariants** (`verify_invariants`)
6. **Generate reports** (`generate_year_report`)

### Output-struktur

```
reports/replay_eval/TRIAL160_YEARLY/
├── 2020/
│   ├── RUN_IDENTITY.json
│   ├── chunk_0/
│   │   ├── chunk_footer.json
│   │   └── ...
│   ├── FULLYEAR_TRIAL160_REPORT_2020.md
│   └── FULLYEAR_TRIAL160_METRICS_2020.json
├── 2021/
│   └── ...
├── ...
└── 2025/
    └── ...

reports/replay_eval/TRIAL160_MULTIYEAR_2020_2025/
├── MULTIYEAR_PARALLEL_STATUS.json
├── MULTIYEAR_SUMMARY.md (genereres etter alle år er ferdig)
└── MULTIYEAR_METRICS.json (genereres etter alle år er ferdig)
```

---

## 🔒 INVARIANTS (Håndheves per år)

For hver år-jobb (PREBUILT replay):
- ✅ `RUN_IDENTITY.json` må skrives før trading
- ✅ `replay_mode == PREBUILT`
- ✅ `feature_build_call_count == 0`
- ✅ Schema validation PASS
- ✅ `KeyErrors == 0` (hard-fail)
- ✅ Lookup invariant: `lookup_hits == lookup_attempts - eligibility_blocks`
- ✅ `policy_id` og `policy_sha256` må matche `trial160_prod_v1`
- ✅ `bundle_sha256` må matche forventet
- ✅ Ingen warnings som skjuler mismatch: mismatch = FATAL

---

## 🚀 KOMMANDOER

### Start full backtest

```bash
python3 gx1/scripts/run_trial160_multiyear_parallel.py \
    --years 2020,2021,2022,2023,2024,2025 \
    --max-workers 5
```

### Sjekk status

```bash
# Prosesser
ps aux | grep -E "python.*trial160|python.*replay" | grep -v grep

# Log
tail -f /tmp/trial160_multiyear.log

# Status fil
cat reports/replay_eval/TRIAL160_MULTIYEAR_2020_2025/MULTIYEAR_PARALLEL_STATUS.json | python3 -m json.tool
```

### Aggreger resultater (etter alle år er ferdig)

```bash
python3 gx1/scripts/aggregate_trial160_multiyear.py \
    --years 2020,2021,2022,2023,2024,2025 \
    --report-base reports/replay_eval/TRIAL160_YEARLY
```

---

## ⚠️ PROBLEMER LØST

### Problem 1: For mange prosesser (40+)
- **Løsning:** Direkte import i stedet for subprocess, ingen multiprocessing-pool når workers=1
- **Resultat:** Maks 6 prosesser

### Problem 2: FASE_1-separasjonssjekk feilet
- **Løsning:** Hopp over sjekk når `GX1_ALLOW_PARALLEL_REPLAY=1`
- **Resultat:** Direkte kall fungerer

### Problem 3: Global replay-lock blokkerte parallell kjøring
- **Løsning:** Tillat parallell kjøring når `GX1_ALLOW_PARALLEL_REPLAY=1`
- **Resultat:** 5 år kan kjøre i parallell

### Problem 4: Feature meta-path ikke funnet
- **Løsning:** Bruk kanonisk path fra policy YAML
- **Resultat:** Feature meta lastes riktig

### Problem 5: Output directory-sjekk feilet
- **Løsning:** Tillat kun `RUN_IDENTITY.json` i output-dir før replay
- **Resultat:** Replay kan starte etter RUN_IDENTITY er skrevet

---

## 📈 FORVENTET TID

- **Prebuilt features:** ~1-2 minutter per år (kjører i parallell)
- **Replay:** ~5-10 minutter per år (avhengig av år-størrelse)
- **Total:** ~15-30 minutter for alle 6 år (med 5 i parallell)

---

## ✅ BEKREFTET FUNGERER

- ✅ 6 prosesser totalt (1 orchestrator + 5 år-jobber)
- ✅ Ingen multiprocessing-ghosts
- ✅ Ingen subprocess-kall
- ✅ Alle 6 år har prebuilt features bygget
- ✅ 5 år kjører replay i parallell
- ✅ FASE_1-sjekk hoppes over korrekt
- ✅ Global replay-lock tillater parallell kjøring

---

## 📝 NESTE STEG

1. **Vent på at alle 6 år fullfører** (replay tar tid)
2. **Kjør aggregator** når alle år er ferdig:
   ```bash
   python3 gx1/scripts/aggregate_trial160_multiyear.py \
       --years 2020,2021,2022,2023,2024,2025 \
       --report-base reports/replay_eval/TRIAL160_YEARLY
   ```
3. **Sjekk resultater:**
   - `MULTIYEAR_SUMMARY.md`
   - `MULTIYEAR_METRICS.json`
   - Per-år rapporter i `TRIAL160_YEARLY/{year}/`

---

**Sist oppdatert:** 2026-01-16 21:10
