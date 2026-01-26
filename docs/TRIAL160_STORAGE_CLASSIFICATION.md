# TRIAL 160 Storage Classification

**Dato:** 2026-01-17  
**Status:** KANONISK KLASSIFISERING  
**Formål:** Definere hva som er SSoT (bevares) vs DERIVAT (arkiveres) vs RUNTIME STØY (slettes)

---

## PRINSIPPER

1. **SSoT (Single Source of Truth)**: Kanoniske artefakter som aldri slettes
   - Aggregerte rapporter (markdown, JSON)
   - RUN_IDENTITY.json (audit trail)
   - Policy/config filer
   - Dokumentasjon

2. **DERIVAT**: Rå replay-artefakter som kan regenereres
   - chunk_* directories
   - Raw signals, policy decisions, trade outcomes (parquet)
   - Intratrade data
   - Intermediate artifacts

3. **RUNTIME STØY**: Midlertidige filer som kan slettes direkte
   - runs/
   - outputs/
   - logs/
   - .cache/
   - .tmp/

---

## SSoT (KEEP) — Kanoniske Artefakter

### Trial 160 Multi-Year Reports
```
reports/replay_eval/TRIAL160_MULTIYEAR_2020_2025/
├── MULTIYEAR_SUMMARY.md                    # ✅ SSoT
├── MULTIYEAR_METRICS.json                  # ✅ SSoT
├── MULTIYEAR_PARALLEL_STATUS.json          # ✅ SSoT
└── FULLSTENDIG_SAMMENDRAG.md              # ✅ SSoT
```

### Trial 160 Per-Year Reports
```
reports/replay_eval/TRIAL160_YEARLY/{2020..2025}/
├── RUN_IDENTITY.json                       # ✅ SSoT (audit trail)
├── FULLYEAR_TRIAL160_REPORT_{year}.md     # ✅ SSoT
└── FULLYEAR_TRIAL160_METRICS_{year}.json  # ✅ SSoT
```

### Trial 160 Canary/Test Reports
```
reports/replay_eval/TRIAL160_YEARLY/2023_CANARY_7D/
├── RUN_IDENTITY.json                       # ✅ SSoT
├── FULLYEAR_TRIAL160_REPORT_2023.md       # ✅ SSoT
└── FULLYEAR_TRIAL160_METRICS_2023.json   # ✅ SSoT
```

### Policy & Config (SSoT)
```
policies/
└── sniper_trial160_prod.json              # ✅ SSoT

gx1/configs/
├── policies/**/*.yaml                     # ✅ SSoT
├── exits/**/*.yaml                        # ✅ SSoT
└── entry_configs/**/*.yaml                # ✅ SSoT

gx1/scripts/**/*.py                        # ✅ SSoT (kode)
```

### Documentation (SSoT)
```
docs/
├── TRIAL160_*.md                          # ✅ SSoT
├── SSOT_*.md                              # ✅ SSoT
├── TRIAL160_AUDIT_CHECKLIST.md            # ✅ SSoT
└── **/*.md                                # ✅ SSoT (all dokumentasjon)
```

### Models & Bundles (SSoT)
```
models/
├── entry_v10_ctx/**/*.pt                  # ✅ SSoT
├── entry_v10_ctx/**/*.json                 # ✅ SSoT
└── **/*.json                               # ✅ SSoT

gx1/models/**/*.json                       # ✅ SSoT (feature meta, etc.)
```

### Prebuilt Features (SSoT)
```
data/prebuilt/TRIAL160/{2020..2025}/
├── *.parquet                               # ✅ SSoT
└── *.manifest.json                         # ✅ SSoT
```

---

## DERIVAT (ARCHIVE → DELETE) — Rå Replay-Artefakter

### Chunk Directories (Kan Regenereres)
```
reports/replay_eval/TRIAL160_YEARLY/{2020..2025}/chunk_*/
├── raw_signals_*.parquet                  # ❌ DERIVAT
├── policy_decisions_*.parquet              # ❌ DERIVAT
├── trade_outcomes_*.parquet               # ❌ DERIVAT
├── attribution_*.json                     # ❌ DERIVAT
├── metrics_*.json                         # ❌ DERIVAT (aggregert i FULLYEAR_METRICS)
├── summary_*.md                            # ❌ DERIVAT (aggregert i FULLYEAR_REPORT)
├── chunk_footer.json                      # ❌ DERIVAT
├── run_header.json                         # ❌ DERIVAT
└── trade_journal/                         # ❌ DERIVAT
    └── trades/*.json                      # ❌ DERIVAT
```

### Legacy Replay Artifacts
```
reports/replay_eval/TRIAL160_FULLYEAR/chunk_*/          # ❌ DERIVAT
reports/replay_eval/TRIAL160_SMOKE_*/chunk_*/            # ❌ DERIVAT
reports/replay_eval/GATED/chunk_*/                       # ❌ DERIVAT
reports/replay_eval/PREBUILT_FULLYEAR/chunk_*/          # ❌ DERIVAT
reports/replay_eval/PREBUILT_SANITY/chunk_*/             # ❌ DERIVAT
```

### Intermediate Analysis Artifacts
```
reports/replay_eval/**/intratrade/                       # ❌ DERIVAT
reports/replay_eval/**/raw/                              # ❌ DERIVAT
reports/replay_eval/**/chunks/                           # ❌ DERIVAT
```

### Performance JSON (Redundant med Metrics)
```
reports/replay_eval/TRIAL160_YEARLY/{year}/perf_*.json  # ❌ DERIVAT (data i FULLYEAR_METRICS)
```

---

## RUNTIME STØY (DELETE) — Midlertidige Filer

### Runtime Directories
```
runs/                                       # ❌ RUNTIME STØY
outputs/                                    # ❌ RUNTIME STØY
logs/                                       # ❌ RUNTIME STØY (unntatt logs/*.md som er dokumentasjon)
.cache/                                     # ❌ RUNTIME STØY
.tmp/                                       # ❌ RUNTIME STØY
__pycache__/                                # ❌ RUNTIME STØY
*.pyc                                       # ❌ RUNTIME STØY
*.pyo                                       # ❌ RUNTIME STØY
```

### Temporary Files
```
*.tmp                                       # ❌ RUNTIME STØY
*.log                                       # ❌ RUNTIME STØY (unntatt logs/*.md)
.DS_Store                                   # ❌ RUNTIME STØY
Thumbs.db                                   # ❌ RUNTIME STØY
```

---

## REGLER FOR KLASSIFISERING

### SSoT Kriterier:
1. **Aggregerte rapporter** (markdown, JSON) som er endelig output
2. **RUN_IDENTITY.json** (audit trail, ikke regenererbar)
3. **Policy/config filer** (kanonisk konfigurasjon)
4. **Dokumentasjon** (docs/)
5. **Models/bundles** (trenede modeller)
6. **Prebuilt features** (deterministisk input)

### DERIVAT Kriterier:
1. **Chunk directories** (chunk_0, chunk_1, etc.)
2. **Raw replay artifacts** (raw_signals, policy_decisions, trade_outcomes parquet)
3. **Trade journal JSON** (individuelle trade-filer)
4. **Intermediate metrics** (per-chunk metrics, redundant med aggregerte)
5. **Performance JSON** (redundant med FULLYEAR_METRICS)

### RUNTIME STØY Kriterier:
1. **Temporary directories** (runs/, outputs/, logs/)
2. **Cache files** (.cache/, __pycache__/)
3. **System files** (.DS_Store, Thumbs.db)
4. **Temporary files** (*.tmp, *.log)

---

## ESTIMAT REDUKSJON

**Basert på faktiske data:**
- Chunk directories: ~880MB (6 år × ~140MB per chunk_0)
- SSoT reports: ~50KB (markdown + JSON per år)
- **Estimert reduksjon: >95%** (880MB → ~50KB)

**Filer:**
- Chunk directories: 6774+ directories
- SSoT files: ~20 files (6 år × 3 files + multiyear summary)
- **Estimert reduksjon: >99%** (6774+ → ~20)

---

## HARD VERIFIKASJON (Fail-Fast)

Før flytting av DERIVAT-artefakter, må følgende SSoT-paths verifiseres:

1. **RUN_IDENTITY.json** må eksistere for hvert år
2. **FULLYEAR_TRIAL160_REPORT_*.md** må eksistere for hvert år
3. **FULLYEAR_TRIAL160_METRICS_*.json** må eksistere for hvert år
4. **MULTIYEAR_SUMMARY.md** må eksistere
5. **MULTIYEAR_METRICS.json** må eksistere
6. **MULTIYEAR_PARALLEL_STATUS.json** må eksistere

Hvis noen av disse mangler → **HARD-FAIL**, ikke flytt noe.

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
- Alle filer kan flyttes tilbake fra archive/
- Replay kan re-kjøres for å regenerere DERIVAT-artefakter
- SSoT-artefakter forblir uendret

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
   - docs/ directory er alltid SSoT
   - Må alltid bevares
