# Truth Verification - First E2E Run

## Status: Klar for kjøring

Alle scripts er implementert og klar. For å kjøre første E2E 1-day Truth Verification trengs:

### Forutsetninger

1. **Policy file** (absolutt path):
   - Eksempel: `/home/andre2/src/GX1_ENGINE/gx1/configs/entry_configs/ENTRY_V10_CTX_SNIPER_REPLAY.yaml`

2. **Data file** (absolutt path til .parquet med PREBUILT features):
   - Må inneholde PREBUILT features for datoen du kjører
   - Eksempel: `/path/to/data.parquet`

3. **GX1_DATA** (absolutt path):
   - Satt til: `/tmp/gx1_data` (eller permanent location hvis du vil bevare runs)

## Kjøringsinstruksjoner

### Step 0: Sett environment variables

```bash
export GX1_REPLAY=1
export GX1_REPLAY_USE_PREBUILT_FEATURES=1
export GX1_TRANSFORMER_SESSION_TOKEN=1
export GX1_TRUTH_TELEMETRY=1
export GX1_DATA=/tmp/gx1_data  # eller permanent location
```

### Step 1: Kjør 1-day PREBUILT replay

```bash
cd /home/andre2/src/GX1_ENGINE

./gx1/scripts/run_one_day_truth_replay.sh \
    --date-anchor 2025-01-01 \
    --policy <ABS_PATH_TIL_POLICY.yaml> \
    --data <ABS_PATH_TIL_DATA.parquet> \
    --workers 1
```

**Forventet output:**
- Scriptet printer final `OUTPUT_DIR`
- Output inneholder:
  - `RUN_IDENTITY.json`
  - `ENTRY_FEATURES_USED.json` (med `xgb_flows[]` + `transformer_outputs[]` + `ts`+`session`)

**Hvis wrapper feiler:**
- Ikke gå videre
- Fiks akkurat feilmeldingen (policy-path, data-path, prebuilt DF, imports, telemetri-mangler)

### Step 2: Post-run preflight

```bash
/home/andre2/venvs/gx1/bin/python gx1/scripts/preflight_truth_report_requirements.py \
    --existing-output-dir <OUTPUT_DIR>
```

**Regel:**
- `exit 0` = PASS → fortsett
- `exit 1` = WARN → fortsett (CONDITIONAL GO mulig)
- `exit 2` = FAIL → stopp, fiks diagnose, rerun 1-day

### Step 3: Kjør GO/NO-GO på 1-day

```bash
/home/andre2/venvs/gx1/bin/python gx1/scripts/run_xgb_transformer_truth_ritual.py \
    --skip-7day \
    --date-anchor 2025-01-01 \
    --replay-output-dir-1day <OUTPUT_DIR>
```

**Forventet:**
- ✅ GO eller ⚠️ CONDITIONAL GO
- Rapport-paths printes:
  - GO/NO-GO MD
  - Truth report MD/JSON
  - Lock-fil (hvis skrevet)

### Step 4: Hvis resultat == NO-GO

- Åpne GO/NO-GO MD
- Utfør kun de blokkende punktene
- Rerun fra Step 1 til status ≠ NO-GO

### Step 5: Når 1-day er GO/COND-GO: klargjør 7-day

```bash
# Kjør 7-day replay separat (samme policy/data, date_from=anchor, date_to=anchor+6)
./gx1/scripts/run_one_day_truth_replay.sh \
    --date-anchor 2025-01-01 \
    --policy <ABS_PATH_TIL_POLICY.yaml> \
    --data <ABS_PATH_TIL_DATA.parquet> \
    --workers 1
# (Note: må justere date range manuelt eller lage 7-day wrapper)

# Deretter full ritual:
/home/andre2/venvs/gx1/bin/python gx1/scripts/run_xgb_transformer_truth_ritual.py \
    --date-anchor 2025-01-01 \
    --replay-output-dir-1day <ONE_DAY_OUTPUT_DIR> \
    --replay-output-dir-7day <SEVEN_DAY_OUTPUT_DIR>
```

## Ferdigkriterier

- ✅ 1-day: GO eller CONDITIONAL GO med ts+session TRUSTED/conditional korrekt begrunnet
- ✅ Alle hard invariants grønne:
  - PREBUILT=on
  - feature_build_call_count==0
  - US XGB disabled
  - session tokens one-hot
- ✅ Correlation Validity viser ts+session join coverage og trust-status

## Notat

Dette er første "real-world proof run". Når den er grønn, går vi videre til 7-day uten å endre noe annet.
