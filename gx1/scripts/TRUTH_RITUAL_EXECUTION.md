# Truth Verification Ritual - Execution Guide

## Status: Klar for kjøring

Alle scripts er implementert og klar. For å kjøre ritualet trengs **faktiske PREBUILT replay outputs** med full telemetri.

## Kanonisk Run-Output Struktur (SSoT)

Alle PREBUILT replay outputs må skrive følgende filer i `<output_dir>/`:

| Path | Innhold | Required/Optional |
|------|---------|-------------------|
| `RUN_IDENTITY.json` | Run metadata (run_id, run_date, git_head_sha, xgb_session_policy_hash, xgb_us_disabled) | **REQUIRED** |
| `ENTRY_FEATURES_USED.json` | Run-level telemetri (aggregert fra chunks):<br>- `xgb_flows[]` (med `timestamp`, `xgb_session`, `xgb_p_long_cal`, `xgb_uncertainty_score`)<br>- `transformer_outputs[]` (med `timestamp`, `session`, `prob_long`, `confidence`)<br>- `run_identity` (eller separat RUN_IDENTITY.json) | **REQUIRED** |
| `ENTRY_FEATURES_USED_MASTER.json` | Samme som ENTRY_FEATURES_USED.json (for bakoverkompatibilitet) | Optional |
| `ENTRY_FEATURES_TELEMETRY_MANIFEST.json` | Manifest over chunk telemetri | Optional |
| `chunk_*/ENTRY_FEATURES_USED.json` | Chunk-level telemetri (per chunk) | Optional (brukes for aggregering) |
| `chunk_*/ENTRY_FEATURES_TELEMETRY.json` | Chunk-level full telemetri (inkluderer xgb_flows og transformer_outputs) | Optional (brukes for aggregering) |

**Notat:** Master aggregerer `xgb_flows` og `transformer_outputs` fra `chunk_*/ENTRY_FEATURES_TELEMETRY.json` filer og skriver dem til run-level `ENTRY_FEATURES_USED.json`.

## Policy Requirements

### Bundle Artifacts

| Artifact | Required/Optional | Notes |
|----------|------------------|-------|
| `bundle_dir` | **REQUIRED** | Path to bundle directory (e.g., FULLYEAR_2025_GATED_FUSION) |
| `feature_meta_path` | **REQUIRED** | Path to feature metadata JSON (e.g., entry_v10_ctx_feature_meta.json) |
| `seq_scaler_path` | **NOT USED** | Legacy training artifact - NOT used in PREBUILT/TRUTH mode |
| `snap_scaler_path` | **NOT USED** | Legacy training artifact - NOT used in PREBUILT/TRUTH mode |

**Code Proof:**
- `load_entry_v10_ctx_bundle` accepts scalers as `Optional[Path] = None`
- Scalers are NOT passed to `EntryV10CtxBundle` constructor
- `EntryV10CtxBundle` does NOT have seq_scaler or snap_scaler fields
- PREBUILT mode loads features directly from parquet (no transformation)

**Policy:** Set `seq_scaler_path: null` and `snap_scaler_path: null` in policy.

## Forutsetninger (må være sanne)

### 1. Environment Variables

```bash
export GX1_REPLAY=1
export GX1_REPLAY_USE_PREBUILT_FEATURES=1
export GX1_TRANSFORMER_SESSION_TOKEN=1
export GX1_TRUTH_TELEMETRY=1  # Enable truth telemetry (fail-safe for PREBUILT)
export GX1_REQUIRE_ENTRY_TELEMETRY=1  # Require telemetry (fail-fast if missing)
export GX1_DATA=/path/to/gx1_data  # eller /tmp/gx1_data
```

**Notat:** `GX1_TRUTH_TELEMETRY=1` aktiverer automatisk `GX1_REQUIRE_ENTRY_TELEMETRY=1` hvis ikke allerede satt (fail-safe).

### 2. PREBUILT Replay Outputs

Replay må skrive følgende i output directory:

- ✅ `RUN_IDENTITY.json` med:
  - `run_id`
  - `run_date`
  - `git_head_sha`
  - `xgb_session_policy_hash`
  - `xgb_us_disabled`

- ✅ `ENTRY_FEATURES_USED.json` med:
  - `run_identity` (eller separat RUN_IDENTITY.json)
  - `transformer_outputs[]` (med `timestamp`, `session`, `prob_long`, `confidence`)
  - `xgb_flows[]` (med `timestamp`, `xgb_session`, `xgb_p_long_cal`, `xgb_uncertainty_score`)

## Kjøringsinstruksjoner

### Step 1: Kjør 1-day PREBUILT replay

**Mest enkelt:** Bruk wrapper scriptet:

```bash
./gx1/scripts/run_one_day_truth_replay.sh \
    --date-anchor 2025-01-01 \
    --policy /path/to/policy.yaml \
    --data /path/to/data.parquet \
    --workers 1
```

**Eller manuelt:**

```bash
# Sett environment variables
export GX1_REPLAY=1
export GX1_REPLAY_USE_PREBUILT_FEATURES=1
export GX1_TRANSFORMER_SESSION_TOKEN=1
export GX1_TRUTH_TELEMETRY=1
export GX1_REQUIRE_ENTRY_TELEMETRY=1
export GX1_DATA=/tmp/gx1_data

# Kjør replay (canonical: run_truth_e2e_sanity only)
# python -m gx1.scripts.run_truth_e2e_sanity --start-ts 2025-01-01 --end-ts 2025-01-02
```

**Verifiser outputs:**
```bash
OUTPUT_DIR=/tmp/gx1_data/reports/truth_runs/ONE_DAY_2025-01-01_*

# Sjekk at filer eksisterer
ls $OUTPUT_DIR/RUN_IDENTITY.json
ls $OUTPUT_DIR/ENTRY_FEATURES_USED.json

# Sjekk telemetri struktur
/home/andre2/venvs/gx1/bin/python -c "
import json
with open('$OUTPUT_DIR/ENTRY_FEATURES_USED.json') as f:
    d = json.load(f)
print('transformer_outputs:', len(d.get('transformer_outputs', [])))
print('xgb_flows:', len(d.get('xgb_flows', [])))
if d.get('transformer_outputs'):
    print('First output has timestamp:', 'timestamp' in d['transformer_outputs'][0])
"
```

### Step 2: Kjør ritual på 1-day

```bash
/home/andre2/venvs/gx1/bin/python gx1/scripts/run_xgb_transformer_truth_ritual.py \
    --date-anchor 2025-01-01 \
    --skip-7day \
    --replay-output-dir-1day $OUTPUT_DIR
```

**Forventet resultat:**
- ✅ GO eller ⚠️ CONDITIONAL GO → Fortsett til Step 3
- ❌ NO-GO → Følg "NEXT_ACTIONS" i GO/NO-GO MD rapporten

### Step 3: Hvis NO-GO - Fiks blockers

1. Åpne GO/NO-GO MD rapporten (sti printes i output)
2. Se "Blockers" seksjon
3. Fiks nøyaktig det som blokkerer (ingen sidequests)
4. Re-run Step 2 til GO/COND-GO

**Tillatte fiks:**
- Manglende telemetry fields (ts/session, transformer_outputs)
- Feil path-resolve / output policy
- Broken invariants (one-hot, US disabled)
- TS duplikater/ikke-monotonic (root-cause i pipeline)

**Ikke tillatt:**
- Ny modell/tuning
- Nye features
- Endre threshold/logikk "for å få grønt"
- Disabling av checks

### Step 4: Kjør 7-day PREBUILT replay

Når 1-day er GO/COND-GO:

```bash
# 7-day range: use run_truth_e2e_sanity (canonical TRUTH)
# python -m gx1.scripts.run_truth_e2e_sanity --start-ts 2024-12-29 --end-ts 2025-01-05
```

### Step 5: Kjør ritual på 1-day + 7-day

```bash
/home/andre2/venvs/gx1/bin/python gx1/scripts/run_xgb_transformer_truth_ritual.py \
    --date-anchor 2025-01-01 \
    --replay-output-dir-1day $OUTPUT_DIR_1DAY \
    --replay-output-dir-7day $OUTPUT_DIR_7DAY
```

## Beslutningsregel

### ✅ GO (7-day)
- **LÅS DETTE KAPITTELT**
- Behold lock-fil som referansepunkt
- Ingen flere endringer i truth-pipeline

### ⚠️ CONDITIONAL GO (7-day)
- Akseptabelt hvis årsak kun er OPTIONAL telemetri (confidence/entropy)
- Logg dette og fortsett (men noter hva som mangler)

### ❌ NO-GO
- Fiks kun blockers fra NEXT_ACTIONS
- Re-run 7-day (ikke endre andre ting)

## Nåværende Status

**Eksisterende outputs mangler nødvendig struktur.**

Eksisterende telemetri-filer i `/tmp/test_worker/` og `/tmp/gx1_single_chunk_4_debug/` 
mangler:
- `run_identity`
- `transformer_outputs[]`
- `xgb_flows[]`

**Neste steg:** Kjør faktisk PREBUILT replay med full telemetri, deretter kjør ritual.

## Verifisering

Etter kjøring, sjekk:

1. **GO/NO-GO MD rapport** (sti printes i output)
2. **Lock file** (hvis GO/COND-GO): `LOCK_XGB_TRANSFORMER_TRUTH_<run_id>.txt`
3. **Exit code:**
   - 0 = GO
   - 1 = CONDITIONAL GO
   - 2 = NO-GO

**Når ritualet er grønt, er XGB → Transformer-kjeden verifisert og låst.**
