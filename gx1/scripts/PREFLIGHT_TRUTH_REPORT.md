# Preflight Truth Report Requirements Checker

## Oversikt

Preflight checker som validerer at alle krav for TRUSTED correlations er oppfylt før man kjører dyre replays.

## Bruk

### Sjekk eksisterende output (post-run)

```bash
/home/andre2/venvs/gx1/bin/python gx1/scripts/preflight_truth_report_requirements.py \
    --existing-output-dir /path/to/output
```

### Sjekk planlagt replay-konfig (pre-run)

```bash
/home/andre2/venvs/gx1/bin/python gx1/scripts/preflight_truth_report_requirements.py \
    --pre-run-check
```

## TS Check Scope: SAMPLE vs FULL

### SAMPLE_CHECK
- Brukes når telemetry-fil > 50MB
- Sjekker første 500 rader
- **Trust implication:** PARTIAL
- **Exit code:** WARN (1)

### FULL_TS_CHECK
- Brukes når telemetry-fil ≤ 50MB
- Sjekker hele datasettet
- **Trust implication:** FULL
- **Exit code:** PASS (0) hvis success, FAIL (2) hvis failure

### Trust Levels

- **TRUSTED:** Full TS check passert, alle krav oppfylt
- **PARTIAL:** Sample check passert, eller optional inputs mangler
- **UNTRUSTED:** TS check feilet, eller kritiske krav mangler

**Note:** UNTRUSTED er ikke en feil, men en bevisst tilstand som indikerer at korrelasjoner ikke kan beregnes pålitelig.

## Validerte krav

### 1. RUN_IDENTITY.json
- ✅ `replay_mode`
- ✅ `output_mode`
- ✅ `git_head_sha`
- ✅ `xgb_session_policy_hash`
- ⚠️ `xgb_us_disabled` (warning hvis mangler)

### 2. ENTRY_FEATURES_USED.json
- ✅ Fil må eksistere
- ✅ Gyldig JSON struktur

### 3. XGB Output Telemetry
- ✅ `xgb_flows[]` må eksistere
- ✅ `timestamp` per event (sample check på første 500)
- ✅ `xgb_session` per event
- ✅ `xgb_p_long_cal` (p_long output)
- ✅ `xgb_uncertainty_score`
- ⚠️ `xgb_p_hat` (optional, warning hvis mangler)

### 4. Transformer Output Telemetry
- ✅ `transformer_outputs[]` må eksistere
- ✅ `timestamp` per event (sample check)
- ✅ `session` per event
- ✅ `prob_long` (transformer output)
- ⚠️ `confidence` (optional, warning hvis mangler - PARTIAL correlation)
- ⚠️ `entropy` (optional)

### 5. Session Tokens (hvis enabled)
- ✅ Sjekker om `GX1_TRANSFORMER_SESSION_TOKEN=1`
- ✅ Validerer `session_token_values` i telemetry
- ✅ One-hot invariant check (sample)

### 6. Timestamp Invariants
- ✅ Monotonic per session (XGB og Transformer)
- ✅ Ingen duplikate timestamps innen session
- ✅ Timestamp parsing (pd.Timestamp/ISO)

## Output

### JSON Rapport
Skrives til: `$GX1_DATA/reports/truth_preflight/TRUTH_PREFLIGHT_<timestamp>.json`

Struktur:
```json
{
  "timestamp": "2025-01-30T12:00:00Z",
  "check_type": "truth_report_preflight",
  "status": "PASS|FAIL|WARN",
  "trust_level_estimate": "TRUSTED|UNTRUSTED|PARTIAL",
  "missing_artifacts": [...],
  "missing_fields": [...],
  "recommended_actions": [...],
  "notes": [...],
  "warnings": [...]
}
```

### Human Summary
Skrives til stdout med:
- Status og trust level
- Missing artifacts/fields
- Warnings
- Recommended actions
- Notes

## Exit Codes

- `0` = PASS (forvent TRUSTED correlations mulig)
- `1` = WARN (correlations mulig men PARTIAL)
- `2` = FAIL (vil bli UNTRUSTED; stopp før dyr replay)

## Eksempler

### PASS (TRUSTED)
```bash
$ /home/andre2/venvs/gx1/bin/python gx1/scripts/preflight_truth_report_requirements.py --existing-output-dir /path/to/output
Status: PASS
Trust Level Estimate: TRUSTED
```

### FAIL (UNTRUSTED)
```bash
$ /home/andre2/venvs/gx1/bin/python gx1/scripts/preflight_truth_report_requirements.py --existing-output-dir /path/to/output
Status: FAIL
Trust Level Estimate: UNTRUSTED
Missing Artifacts:
  ❌ ENTRY_FEATURES_USED.json
Missing Fields:
  ❌ xgb_flows[].timestamp
```

### WARN (PARTIAL)
```bash
$ /home/andre2/venvs/gx1/bin/python gx1/scripts/preflight_truth_report_requirements.py --existing-output-dir /path/to/output
Status: WARN
Trust Level Estimate: PARTIAL
Warnings:
  ⚠️  Transformer outputs missing confidence (optional but recommended for correlation)
```

## Integrasjon med Truth Report

Preflight checker garanterer at truth report kan produsere TRUSTED correlations:

1. **Pre-run**: Kjør `--pre-run-check` før replay
2. **Post-run**: Kjør `--existing-output-dir` etter replay
3. **Hvis PASS**: Truth report vil kunne produsere TRUSTED correlations
4. **Hvis FAIL**: Stopp før dyr replay, fiks manglende krav
5. **Hvis WARN**: Truth report vil produsere PARTIAL correlations

## Troubleshooting

### "Missing timestamp"
- Sjekk at telemetry collection er enabled
- Sjekk at replay ble kjørt med full telemetry

### "Timestamp not monotonic"
- Dette indikerer at timestamps ikke er korrekt sortert per session
- Kan skyldes async logging eller manglende sortering

### "Duplicate timestamps"
- Dette indikerer at samme timestamp forekommer flere ganger
- Kan skyldes logging-bug eller manglende deduplication
