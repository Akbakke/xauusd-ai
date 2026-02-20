# XGB → Transformer Truth Verification Ritual

## Oversikt

End-to-end runner som kjører hele "Truth Verification Ritual" i én kommando:
1. 1-day PREBUILT replay (microscope run)
2. GO/NO-GO check på 1-day
3. Hvis GO/CONDITIONAL GO: 7-day PREBUILT replay + GO/NO-GO check
4. Final status report

## Bruk

### Full ritual (1-day + 7-day)

```bash
/home/andre2/venvs/gx1/bin/python gx1/scripts/run_xgb_transformer_truth_ritual.py \
    --date-anchor 2025-01-01 \
    --replay-output-dir-1day /path/to/1day/replay/output \
    --replay-output-dir-7day /path/to/7day/replay/output
```

### Quick check (kun 1-day)

```bash
/home/andre2/venvs/gx1/bin/python gx1/scripts/run_xgb_transformer_truth_ritual.py \
    --date-anchor 2025-01-01 \
    --skip-7day \
    --replay-output-dir-1day /path/to/1day/replay/output
```

### Dry run

```bash
/home/andre2/venvs/gx1/bin/python gx1/scripts/run_xgb_transformer_truth_ritual.py \
    --date-anchor 2025-01-01 \
    --dry-run \
    --replay-output-dir-1day /path/to/1day/replay/output
```

## Parametre

- `--date-anchor`: Anchor date (YYYY-MM-DD)
  - 1-day = anchor
  - 7-day = anchor±3 days (anchor-3 to anchor+3)
- `--skip-7day`: Skip 7-day run (kun 1-day)
- `--dry-run`: Print hva som ville blitt kjørt (ikke kjør)
- `--workers`: Number of workers (for replay, default: 1)
- `--reports-output-dir`: Output directory for reports (default: $GX1_DATA/reports/truth)
- `--replay-output-dir-1day`: Existing 1-day replay output directory
- `--replay-output-dir-7day`: Existing 7-day replay output directory

## Prosess

### Step 1: 1-day PREBUILT replay

- Validerer at replay output eksisterer
- Sjekker for RUN_COMPLETED.json eller ENTRY_FEATURES_USED.json
- Hvis mangler: feiler med tydelig melding

### Step 2: GO/NO-GO check på 1-day

- Kjører preflight checker
- Kjører truth report generator
- Kjører GO/NO-GO evaluator
- Hvis NO-GO: stopper og rapporterer
- Hvis GO/CONDITIONAL GO: fortsetter til 7-day

### Step 3: 7-day PREBUILT replay (hvis ikke --skip-7day)

- Samme validering som 1-day
- Hvis feiler: stopper og rapporterer NO-GO

### Step 4: GO/NO-GO check på 7-day

- Samme prosess som 1-day
- Final status basert på 7-day resultat

## Kontrollflyt

### Hvis 1-day == NO-GO:
- ❌ Print: "NO-GO (1-day)"
- Print sti til GO/NO-GO MD
- Exit 2
- **Stopper** - ikke kjører 7-day

### Hvis 1-day ∈ {GO, CONDITIONAL GO}:
- ✅ Fortsetter til 7-day

### Hvis 7-day == NO-GO:
- ❌ Print: "NO-GO (7-day)"
- Exit 2

### Hvis 7-day == CONDITIONAL GO:
- ⚠️ Print: "CONDITIONAL GO (7-day)"
- Exit 1

### Hvis 7-day == GO:
- ✅ Print: "GO (7-day)"
- Exit 0

## Output

### Final Status Banner

```
================================================================================
✅ GO
(7-day)
================================================================================

GO/NO-GO Report (MD): /path/to/XGB_TRANSFORMER_GO_NOGO_<run_id>.md
Truth Report (MD): /path/to/XGB_TRANSFORMER_TRUTH_<run_id>.md
Lock File: /path/to/LOCK_XGB_TRANSFORMER_TRUTH_<run_id>.txt

Replay Output: /path/to/replay/output
Reports Output: /path/to/reports/truth
```

## Exit Codes

- `0` = GO (7-day)
- `1` = CONDITIONAL GO (7-day)
- `2` = NO-GO (1-day eller 7-day)

## Notater

### Replay må kjøres separat

Dette scriptet fokuserer på orkestrering av GO/NO-GO checks. Replay må kjøres separat:

```bash
# Eksempel: Kjør replay (canonical TRUTH: run_truth_e2e_sanity only, see docs/GHOST_PURGE_PLAN.md)
# python -m gx1.scripts.run_truth_e2e_sanity --start-ts 2025-01-01 --end-ts 2025-01-02

# Deretter: Kjør ritual
/home/andre2/venvs/gx1/bin/python gx1/scripts/run_xgb_transformer_truth_ritual.py \
    --date-anchor 2025-01-01 \
    --replay-output-dir-1day /path/to/output
```

### CI/CD Integrasjon

Exit codes kan brukes som hard gate:

```bash
/home/andre2/venvs/gx1/bin/python gx1/scripts/run_xgb_transformer_truth_ritual.py ... || exit 1
```

Hvis exit code != 0, stopper pipeline.

## Ferdigkriterier

- ✅ Én kommando kan kjøres på ny maskin uten manuell stitching
- ✅ Ingen menneskelig tolkning nødvendig for status
- ✅ Scriptet bruker kun eksisterende byggesteiner
- ✅ Exit codes kan brukes i CI eller som hard gate

**Når ritualet er grønt, er XGB → Transformer-kjeden verifisert og låst.**
