# TRIAL 160 Storage Policy

**Dato:** 2026-01-17  
**Status:** PERMANENT POLICY  
**Formål:** Etablere varige regler for hva som er kanonisk vs. forbruk

---

## GRUNNREGEL

**"Rå replay-artefakter er forbruksvare. Kun aggregerte rapporter er SSoT."**

---

## KLASSIFISERING

### SSoT (Single Source of Truth) — ALDRI SLETT

**Definisjon:** Kanoniske artefakter som er endelig output og ikke kan regenereres uten å re-kjøre hele backtesten.

**Eksempler:**
- `RUN_IDENTITY.json` (audit trail med git sha, policy sha, bundle sha, fingerprint)
- `FULLYEAR_TRIAL160_REPORT_*.md` (aggregert rapport per år)
- `FULLYEAR_TRIAL160_METRICS_*.json` (aggregert metrics per år)
- `MULTIYEAR_SUMMARY.md` (aggregert multiyear rapport)
- `MULTIYEAR_METRICS.json` (aggregert multiyear metrics)
- `MULTIYEAR_PARALLEL_STATUS.json` (execution status)
- `policies/sniper_trial160_prod.json` (kanonisk policy)
- `gx1/configs/**/*.yaml` (kanonisk konfigurasjon)
- `docs/**/*.md` (dokumentasjon)
- `models/**/*.pt`, `models/**/*.json` (trenede modeller)
- `data/prebuilt/**/*.parquet`, `data/prebuilt/**/*.manifest.json` (prebuilt features)

**Handling:** ✅ PRESERVE — Aldri slett eller arkiver

---

### DERIVAT — ARKIVER (Kan Regenereres)

**Definisjon:** Rå replay-artefakter som kan regenereres ved å re-kjøre replay.

**Eksempler:**
- `reports/replay_eval/**/chunk_*/` (chunk directories)
- `reports/replay_eval/**/raw_signals_*.parquet`
- `reports/replay_eval/**/policy_decisions_*.parquet`
- `reports/replay_eval/**/trade_outcomes_*.parquet`
- `reports/replay_eval/**/attribution_*.json`
- `reports/replay_eval/**/perf_*.json` (redundant med FULLYEAR_METRICS)
- `reports/replay_eval/**/chunk_footer.json`
- `reports/replay_eval/**/run_header.json`
- `reports/replay_eval/**/trade_journal/trades/*.json` (individuelle trades)

**Handling:** 📦 ARCHIVE → DELETE (flytt til `archive/REPLAY_RAW_YYYY_MM/`)

**Regenerering:**
- Alle DERIVAT-artefakter kan regenereres ved å re-kjøre replay
- Replay er deterministisk (PREBUILT + samme policy → samme output)

---

### RUNTIME STØY — SLETT DIREKTE

**Definisjon:** Midlertidige filer som ikke er nødvendige etter kjøring.

**Eksempler:**
- `runs/`
- `outputs/`
- `logs/*.log` (unntatt `logs/*.md` som er dokumentasjon)
- `.cache/`
- `.tmp/`
- `__pycache__/`
- `*.tmp`, `*.pyc`, `*.pyo`
- `.DS_Store`, `Thumbs.db`

**Handling:** 🗑️ DELETE — Slett direkte uten arkivering

---

## AUTOMATISK OPPRYDDING

### Default Policy: Auto-Archive After Replay

**Etter fullført multiyear/backtest kjøres auto-archive automatisk, med mindre `GX1_KEEP_RAW_ARTIFACTS=1` er eksplisitt satt.**

**Implementasjon:**
- `scripts/post_replay_auto_archive.sh` kjøres automatisk etter replay completion
- Scriptet er idempotent og trygt å kjøre flere ganger
- Scriptet hard-failer hvis SSoT-artefakter mangler eller er korrupte

**Workflow:**
1. Replay completion → trigger `post_replay_auto_archive.sh`
2. Script sjekker `GX1_KEEP_RAW_ARTIFACTS`
3. Hvis satt → skip archive, logg og exit
4. Hvis ikke satt → kjør inventory → archive → verify
5. Hard-fail hvis verify feiler

### Environment Variable Guard

**`GX1_KEEP_RAW_ARTIFACTS=1`**

Hvis satt:
- Rå replay-artefakter (DERIVAT) bevares etter replay
- Ingen automatisk arkivering
- Nyttig for debugging eller når raw data trengs lokalt

Hvis ikke satt (default):
- Rå replay-artefakter (DERIVAT) arkiveres automatisk etter ferdig replay
- Kun SSoT-artefakter bevares i `reports/replay_eval/`
- Normal drift = rent repo

**Manuell kjøring:**
```bash
# Auto-archive (default behavior)
./scripts/post_replay_auto_archive.sh

# Keep raw artifacts (debug mode)
GX1_KEEP_RAW_ARTIFACTS=1 ./scripts/post_replay_auto_archive.sh
```

**Arkiveringslogg:**
- Skrives til `archive/REPLAY_RAW_YYYY_MM/ARCHIVE_LOG.md`
- Inkluderer antall filer flyttet, diskstørrelse frigjort, source → destination paths

---

## HARD VERIFIKASJON (Fail-Fast)

Før flytting av DERIVAT-artefakter, må følgende SSoT-paths verifiseres:

1. ✅ `RUN_IDENTITY.json` må eksistere for hvert år
2. ✅ `FULLYEAR_TRIAL160_REPORT_*.md` må eksistere for hvert år
3. ✅ `FULLYEAR_TRIAL160_METRICS_*.json` må eksistere for hvert år
4. ✅ `MULTIYEAR_SUMMARY.md` må eksistere
5. ✅ `MULTIYEAR_METRICS.json` må eksistere
6. ✅ `MULTIYEAR_PARALLEL_STATUS.json` må eksistere

**Hvis noen av disse mangler → HARD-FAIL, ikke flytt noe.**

---

## REVERSIBILITET

Alle DERIVAT-artefakter flyttes til:
```
archive/REPLAY_RAW_YYYY_MM/
├── TRIAL160_YEARLY/
│   └── {year}/
│       └── chunk_*/                        # Flyttet hit
└── ARCHIVE_LOG.md                          # Logg over hva som ble flyttet
```

**Gjenoppretting:**
```bash
# Flytt tilbake fra archive
mv archive/REPLAY_RAW_YYYY_MM/TRIAL160_YEARLY/{year}/chunk_* \
   reports/replay_eval/TRIAL160_YEARLY/{year}/
```

**Alternativ: Re-generer**
```bash
# Re-kjør replay for å regenerere DERIVAT-artefakter
python3 gx1/scripts/run_trial160_year_job.py --year {year} ...
```

---

## GITIGNORE REGLER

`.gitignore` er konfigurert til å:
- ✅ Ignorere alle DERIVAT-paths (`chunk_*`, `trade_journal/`, `perf_*.json`)
- ✅ Ignorere alle parquet-filer (unntatt prebuilt manifest JSON)
- ✅ Unnta eksplisitt SSoT-rapporter (RUN_IDENTITY.json, FULLYEAR_TRIAL160_REPORT_*.md, etc.)

**Se `.gitignore` for detaljerte patterns.**

---

## ESTIMAT REDUKSJON

**Basert på faktiske data (2026-01-17):**
- **Filer:** 131,623 → 364 (99.7% reduksjon)
- **Størrelse:** 740.48 MB → 1.25 MB (99.8% reduksjon)

**Etter opprydding:**
- Kun SSoT-artefakter bevares i `reports/replay_eval/`
- Alle DERIVAT-artefakter flyttes til `archive/REPLAY_RAW_YYYY_MM/`
- RUNTIME STØY slettes direkte

---

## PERMANENTE REGLER

1. **Rå replay-artefakter er forbruksvare**
   - Kun aggregerte rapporter er SSoT
   - Chunk directories kan alltid regenereres

2. **RUN_IDENTITY.json er alltid SSoT**
   - Dette er audit trail, ikke regenererbar
   - Må alltid bevares

3. **Policy/config er alltid SSoT**
   - Dette er kanonisk konfigurasjon
   - Må alltid bevares

4. **Dokumentasjon er alltid SSoT**
   - `docs/` directory er alltid SSoT
   - Må alltid bevares

5. **Prebuilt features er SSoT**
   - Deterministisk input for replay
   - Må bevares (men kan regenereres hvis nødvendig)

---

## SCRIPTS

### Inventory (Dry-Run)
```bash
python3 scripts/inspect_storage_trial160.py
```
- Teller filer per kategori
- Estimerer % reduksjon
- Skriver `reports/storage/TRIAL160_STORAGE_INVENTORY.md`

### Archive (Reversibel)
```bash
./scripts/archive_trial160_raw_artifacts.sh
```
- Hard-verifiserer SSoT før flytting
- Flytter DERIVAT-artefakter til `archive/REPLAY_RAW_YYYY_MM/`
- Skriver `archive/REPLAY_RAW_YYYY_MM/ARCHIVE_LOG.md`

### Verification (Post-Cleanup)
```bash
./scripts/verify_trial160_post_cleanup.sh
```
- Verifiserer at alle SSoT-artefakter er intakte
- Validerer JSON og policy_id/policy_sha256
- Hard-fail hvis noe mangler
- Skriver `reports/storage/TRIAL160_POST_CLEANUP_CHECK.md`

---

## STOPP KRITERIER

Oppryddingen er ferdig når:
- ✅ ≥90% av filer er flyttet/ryddet (mål: 99.7%)
- ✅ Alle SSoT-rapporter åpner uten feil
- ✅ Repo er merkbart raskere i Cursor/Finder
- ✅ Alle scripts er dokumentert og kjørbare
- ✅ `.gitignore` er oppdatert
- ✅ Storage policy er dokumentert

**IKKE:**
- ❌ Ikke slett SSoT
- ❌ Ikke endre replay-kode
- ❌ Ikke re-kjør backtests
- ❌ Ikke introduser nye output-formater

**Dette er ren hygiene + kontroll.**
