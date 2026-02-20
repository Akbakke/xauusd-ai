# XGB → Transformer GO/NO-GO Report

## Oversikt

Kanonisk beslutningsartefakt som konsoliderer preflight + truth report til én entydig GO/NO-GO beslutning.

## Bruk

```bash
/home/andre2/venvs/gx1/bin/python gx1/scripts/build_xgb_transformer_go_nogo.py \
    --date-from 2025-01-01 \
    --date-to 2025-01-08 \
    --mode PREBUILT \
    --output-dir $GX1_DATA/reports/truth \
    --existing-output-dir /path/to/replay/output
```

## Prosess

Scriptet kjører i rekkefølge:

1. **Preflight Check** (`preflight_truth_report_requirements.py`)
   - Validerer at alle krav for TRUSTED correlations er oppfylt
   - Sjekker TS invariants, required inputs, hard invariants

2. **Truth Report** (`build_xgb_transformer_truth_report.py`)
   - Genererer fullstendig truth report
   - Beregner korrelasjoner, validerer kontrakt, sjekker invariants

3. **GO/NO-GO Evaluation**
   - Konsoliderer resultater fra begge
   - Anvender eksplisitte regler for beslutning

## GO/NO-GO Regler

### ✅ GO

Alle følgende må være sanne:

- ✅ `preflight_status == PASS`
- ✅ `ts_validation_trusted == true` (FULL TS scan)
- ✅ `correlation_trusted == true` (ikke "conditional")
- ✅ Alle REQUIRED inputs tilgjengelig
- ✅ Ingen hard invariants brutt
- ✅ Ingen NaN/Inf i outputs

**Exit code:** 0

### ⚠️ CONDITIONAL GO

Alle REQUIRED krav oppfylt, men:

- ⚠️ `preflight_status == WARN` (sample check, ikke full)
- ⚠️ `correlation_trusted == "conditional"` (optional inputs mangler)
- ⚠️ Kun OPTIONAL inputs mangler (confidence/entropy)

**Exit code:** 1

### ❌ NO-GO

Hvis noen av følgende:

- ❌ `preflight_status == FAIL`
- ❌ `ts_validation_trusted == false`
- ❌ `correlation_trusted == false`
- ❌ REQUIRED inputs mangler
- ❌ Hard invariants brutt
- ❌ NaN/Inf i outputs

**Exit code:** 2

## Output

### JSON Rapport

`XGB_TRANSFORMER_GO_NOGO_<run_id>.json`

Struktur:
```json
{
  "run_id": "...",
  "timestamp": "...",
  "decision": "GO" | "CONDITIONAL_GO" | "NO_GO",
  "details": {
    "checks": {...},
    "blockers": [...],
    "warnings": [...],
    "passed": [...]
  },
  "preflight_summary": {...},
  "truth_report_summary": {...}
}
```

### Markdown Rapport

`XGB_TRANSFORMER_GO_NOGO_<run_id>.md`

Inneholder:
- Tydelig banner (✅ GO / ⚠️ CONDITIONAL GO / ❌ NO-GO)
- Kort begrunnelse (auto-generert)
- Checks summary
- Passed checks / Warnings / Blockers
- Next actions (hvis NO-GO)
- Pekere til truth report og preflight report

### Lock File

Hvis GO eller CONDITIONAL GO:

`LOCK_XGB_TRANSFORMER_TRUTH_<run_id>.txt`

Inneholder:
- Timestamp
- Git SHA
- Session tokens enabled status
- Statement: "XGB → Transformer information chain verified and locked."

**Betydning:** Lock file indikerer at informasjonskjeden er verifisert og låst. Videre endringer krever ny GO/NO-GO run.

## Eksempler

### GO

```bash
$ /home/andre2/venvs/gx1/bin/python gx1/scripts/build_xgb_transformer_go_nogo.py ...
GO/NO-GO Decision: GO
✅ All checks passed. XGB → Transformer information chain verified.
```

**Resultat:**
- Exit code: 0
- Lock file skrevet
- Kan brukes som arkitektonisk referansepunkt

### CONDITIONAL GO

```bash
$ /home/andre2/venvs/gx1/bin/python gx1/scripts/build_xgb_transformer_go_nogo.py ...
GO/NO-GO Decision: CONDITIONAL_GO
⚠️  Checks passed with warnings. Use with caution.
```

**Resultat:**
- Exit code: 1
- Lock file skrevet
- Warnings dokumentert i rapport

### NO-GO

```bash
$ /home/andre2/venvs/gx1/bin/python gx1/scripts/build_xgb_transformer_go_nogo.py ...
GO/NO-GO Decision: NO_GO
❌ Critical checks failed. Review blockers and fix before proceeding.
```

**Resultat:**
- Exit code: 2
- Ingen lock file
- Blockers dokumentert i rapport
- Next actions gitt

## Arkivering og Låsing

Ved GO eller CONDITIONAL GO:

1. Lock file skrives automatisk
2. Lock file dokumenterer:
   - Run ID
   - Timestamp
   - Git SHA
   - Session tokens status
   - Verifiseringsstatement

3. Lock file betyr:
   - Informasjonskjeden er verifisert
   - Videre endringer krever ny verifisering
   - Kan brukes som arkitektonisk referansepunkt

## Integrasjon

GO/NO-GO rapporten er den kanoniske verifiseringsartefakten:

1. **Pre-run:** Kjør preflight checker først
2. **Post-run:** Kjør GO/NO-GO rapport
3. **Ved GO:** Systemet er verifisert og låst
4. **Ved NO-GO:** Fiks blockers og re-run

## Ferdigkriterier

- ✅ Én kommando gir entydig beslutning
- ✅ Ingen menneskelig tolkning nødvendig for status
- ✅ Resultatet kan brukes som arkitektonisk referansepunkt
- ✅ Dette kapitlet kan lukkes uten tvil

**Systemet vet hva det ser, og vi vet at det vet det.**
