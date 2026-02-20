# Truth Report - Kjøringsinstruksjoner

## Status

Truth report-generatoren er klar, men krever **eksisterende replay outputs** med full telemetri.

## Forutsetninger

1. **Replay outputs** må være kjørt med `GX1_REPLAY=1` og `GX1_REPLAY_USE_PREBUILT_FEATURES=1`
2. Output directories må inneholde `ENTRY_FEATURES_USED.json` i roten
3. Telemetri må inneholde:
   - `run_identity` (med `run_date`, `run_id`)
   - `transformer_outputs` (med `timestamp` og `session`)
   - `xgb_flows` (med `timestamp` og `xgb_session`)

## Kjøring: 7-dagers PREBUILT

```bash
cd /home/andre2/src/GX1_ENGINE

# Sett GX1_DATA hvis nødvendig
export GX1_DATA=/path/to/gx1_data  # eller bruk default /tmp/gx1_data

# Kjør truth report
/home/andre2/venvs/gx1/bin/python gx1/scripts/build_xgb_transformer_truth_report.py \
    --date-from 2025-01-01 \
    --date-to 2025-01-08 \
    --mode PREBUILT \
    --output-dir $GX1_DATA/reports/truth
```

## Kjøring: 1-dag mikroskop-run

```bash
/home/andre2/venvs/gx1/bin/python gx1/scripts/build_xgb_transformer_truth_report.py \
    --date-from 2025-01-01 \
    --date-to 2025-01-01 \
    --mode PREBUILT \
    --output-dir $GX1_DATA/reports/truth
```

## TS Validation Scope

Truth report utfører alltid **FULL_TS_CHECK** (hele datasettet) siden vi analyserer eksisterende data.

**Trust status:**
- `trusted: true` = Full TS check passert, alle required inputs tilgjengelig
- `trusted: "conditional"` = Full TS check passert, men optional inputs (confidence) mangler
- `trusted: false` = TS check feilet eller kritiske krav mangler

**Correlation inputs:**
- **Required:** `prob_long` (må være tilgjengelig)
- **Optional:** `confidence`, `entropy` (hvis mangler, markeres korrelasjoner som SKIPPED)

## Verifisering etter kjøring

### 1. Sjekk JSON rapport

```bash
cat $GX1_DATA/reports/truth/XGB_TRANSFORMER_TRUTH_*.json | jq '.transformer_behavior_truth.xgb_correlations'
```

**Forventet:**
- `trusted: true`
- `join_type: "ts+session"`
- `join_coverage: > 0.95` (for mikroskop-run skal dette være ~1.0)
- `by_session.EU.transformer_confidence_vs_xgb_uncertainty.pearson_r: < 0` (negativ)

### 2. Sjekk Markdown rapport

```bash
cat $GX1_DATA/reports/truth/XGB_TRANSFORMER_TRUTH_*.md
```

**Se etter:**
- "## 3. Correlation Validity" seksjon
- Status: "✅ TRUSTED"
- Join coverage høy (>95% for 7-dagers, ~100% for 1-dag)

### 3. Verifiser korrelasjoner

**7-dagers run skal vise:**
- ✅ TRUSTED korrelasjoner for alle sessions (EU, OVERLAP, US)
- ✅ Negativ korrelasjon: `transformer_confidence_vs_xgb_uncertainty.pearson_r < 0`
- ✅ Ingen session-lekkasje (distribusjoner skal være forskjellige)

**1-dag mikroskop-run skal vise:**
- ✅ Join coverage ~100% (`n_joined_rows ≈ min(n_xgb_rows, n_transformer_rows)`)
- ✅ Alle korrelasjoner TRUSTED
- ✅ Tidsriktig matching bekreftet

## Troubleshooting

### "No matching replay outputs found"
- Sjekk at `$GX1_DATA/outputs` eksisterer
- Sjekk at output directories inneholder `ENTRY_FEATURES_USED.json`
- Sjekk at `run_identity.run_date` matcher dato-området

### "Correlation skipped: missing ts"
- Telemetri må ha `timestamp` i både `xgb_flows` og `transformer_outputs`
- Sjekk at replay ble kjørt med full telemetri

### "Join mismatch rate too high"
- Dette indikerer at XGB og Transformer outputs ikke matcher tidsriktig
- For mikroskop-run bør dette være ~100%
- For 7-dagers run kan det være lavere pga. gates/filtering

## Neste steg

Når riktige replay outputs er tilgjengelige, kjør kommandoene over og verifiser resultatene.
