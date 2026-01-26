# FULLYEAR_2025 Robust Pipeline

**Date:** 2026-01-07  
**Status:** ✅ Implemented

---

## Overview

Ekstremt robust og lett å verifisere FULLYEAR-pipeline med hygiene, observability og CLI-utvidelser. Ingen arkitekturendringer.

---

## 1. Build-Script: Mini-Build og Tidsavgrensing

**File:** `gx1/scripts/build_entry_v10_ctx_training_dataset.py`

**Nye CLI Args:**
- `--start` (ISO date/time): Filtrer input data før feature-bygging
- `--end` (ISO date/time): Filtrer input data før feature-bygging
- `--max_rows` (int): Ta de første N radene etter filtrering (deterministisk)
- `--dry_run`: Bare parse config og sjekk calibrators, ikke bygg

**Krav:**
- Hvis `--start/--end` er satt: Filtrer input data **før** HTF/XGB feature-bygging
- Hvis `--max_rows` er satt: Ta de første N radene etter filtrering (deterministisk)
- Skriv output til samme `--output` path, men tillat "mini" navngiving hvis brukeren angir det

**Example:**
```bash
# Mini-build for testing
python gx1/scripts/build_entry_v10_ctx_training_dataset.py \
    --data data/entry_v9/full_2025.parquet \
    --output data/entry_v10_ctx/FULLYEAR_2025_mini.parquet \
    --start 2025-01-01T00:00:00Z \
    --end 2025-01-07T23:59:59Z \
    --max_rows 1000 \
    --policy_config gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml \
    --feature_meta_path gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json \
    --seq_scaler_path gx1/models/entry_v9/nextgen_2020_2025_clean/seq_scaler.joblib \
    --snap_scaler_path gx1/models/entry_v9/nextgen_2020_2025_clean/snap_scaler.joblib \
    --calibration_dir models/xgb_calibration \
    --calibration_method platt

# Dry run (verify only)
python gx1/scripts/build_entry_v10_ctx_training_dataset.py \
    --data data/entry_v9/full_2025.parquet \
    --output data/entry_v10_ctx/FULLYEAR_2025.parquet \
    --dry_run \
    --policy_config ... \
    --feature_meta_path ...
```

---

## 2. Dataset Sanity-Check (Hard Fail)

**File:** `gx1/scripts/build_entry_v10_ctx_training_dataset.py`

**Function:** `validate_built_dataset()`

**Hard Fails:**
- `calibration_applied != true`
- NaN/inf i kritiske kolonner: `seq_features`, `snap_features`, `p_cal`, `margin`, `p_hat`, `uncertainty_score`
- Range sanity:
  - `p_cal` ∉ [0, 1]
  - `p_hat` ∉ [0, 1]
  - `uncertainty_score` ∉ [0, 1]

**Soft Warnings:**
- `p_cal` vs `p_raw`: Hvis `corr > 0.999` og `mean_abs_diff < 1e-6` → WARN (calibration may not have been applied)

**Statistics Block:**
```
[DATASET_VALIDATE STATISTICS]
  p_cal: mean=0.5234, p5=0.4123, p95=0.6234
  uncertainty_score: mean=0.2345, p5=0.1234, p95=0.3456
  ⚠️  WARNING: p_cal and p_raw are nearly identical (if applicable)
```

**Note:** Shape mismatch checks (seq_x.shape[-1] == 16, etc.) er ikke mulig fra DataFrame alene. Disse sjekkes i training script når dataset lastes.

---

## 3. Calibrator Path SSoT + Usage Stats

**File:** `gx1/models/entry_v10/calibration_paths.py` (NEW)

**SSoT Function:**
```python
get_calibrator_path(
    calibration_dir: Path,
    policy_id: str,
    session: str,
    bucket_key: Optional[str],
    method: str = "platt",
) -> Path
```

**Usage:**
- Brukt i build script, runtime, og eval
- Logger eksakt paths som søkes
- Hierarki: `session+bucket` → `session-only`

**Usage Stats Tracking:**
- `track_calibrator_usage()`: Teller hvilken tier som brukes
- `get_calibrator_usage_stats()`: Returnerer stats per session/bucket
- Skrives til manifest

**Hard Fail:**
- Hvis `GX1_REQUIRE_XGB_CALIBRATION=1` og vi ender opp i en fallback-tier som ikke er tillatt → hard fail
- Logger eksakt paths som ble søkt

---

## 4. Manifest for Prebuilt Dataset (SSoT)

**File:** `gx1/scripts/build_entry_v10_ctx_training_dataset.py`

**Function:** `write_manifest()`

**Output:** `<output_basename>.manifest.json` (ved siden av parquet)

**Felter:**
- `created_at`: ISO timestamp
- `git_commit`: Git commit hash
- `input_data_path`: Input data path
- `output_data_path`: Output data path
- `build_command`: Full argv
- `policy_config_path`: Policy config path
- `calibration_dir`, `calibration_method`
- `feature_meta_path`, `seq_scaler_path`, `snap_scaler_path`
- `feature_contract`:
  - `seq_len`, `seq_dim=16`, `snap_dim=88`, `ctx_cat_dim=5`, `ctx_cont_dim=2`
  - `feature_contract_hash` (hvis finnes)
- `splits`: `train/val/test` start/end + counts (hvis `--time_split`)
- `calibrator_usage_stats`: Stats fra punkt 3
- `xgb_model_paths`: EU/US/OVERLAP model paths
- `validation_stats`: Stats fra `validate_built_dataset()`
- `notes`: Fri tekst

**Logging:**
```
MANIFEST WRITTEN: data/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION_train.manifest.json
```

---

## 5. Training: Worker Benchmark + Throughput Logging

**File:** `gx1/models/entry_v10/entry_v10_ctx_train.py`

**CLI Args:**
- `--benchmark_workers "0,2,4"`: Comma-separated worker counts to benchmark
- `--benchmark_only`: Only run benchmark, don't train

**Funksjonalitet:**
- Kjører 200 batches (eller 1 epoch med cap) per worker-setting
- Logger samples/sec og wall time per setting
- Velger automatisk beste worker count til videre trening (hvis ikke `--benchmark_only`)

**Output:**
```
[BENCHMARK SUMMARY]
  num_workers | samples/sec | wall_time_sec
  ---------------------------------------------
            0 |      123.45 |         123.45
            2 |      234.56 |          67.89 ← BEST
            4 |      212.34 |          78.90
```

**Krav:**
- Deterministisk seed for benchmark
- Ikke endre trening ellers
- Logger samples/sec per epoch (allerede implementert)

---

## 6. Gate-Stability Sanity etter Epoch 1

**File:** `gx1/models/entry_v10/entry_v10_ctx_train.py`

**Funksjonalitet:**
- Etter epoch 1: Sjekk `gate_std < 0.05` OG `max(|corrs|) < 0.05`
- Hvis begge er sanne: Print warning og anbefal å redusere `--gate_stability_weight`

**Output:**
```
⚠️  [GATE SANITY] GATE MAY BE OVER-REGULARIZED
  gate_std: 0.0234 (expected > 0.05)
  max_correlation: 0.0123 (expected > 0.05)
  Recommendation: Reduce --gate_stability_weight
    Current: 0.1
    Suggested: 0.03 (e.g., 0.1 → 0.03)
  Note: This is a warning, not an error. Training will continue.
```

**Krav:**
- Ikke endre automatisk, kun logg
- Training fortsetter (ikke hard fail)

---

## Files Changed

1. `gx1/models/entry_v10/calibration_paths.py` (NEW) - SSoT for calibrator paths
2. `gx1/scripts/build_entry_v10_ctx_training_dataset.py` (UPDATED)
   - `--start/--end/--max_rows/--dry_run` args
   - `validate_built_dataset()` function
   - `write_manifest()` function
   - Calibrator usage stats tracking
3. `gx1/models/entry_v10/entry_v10_ctx_train.py` (UPDATED)
   - `--benchmark_workers` and `--benchmark_only` args
   - Worker benchmark functionality
   - Gate-stability sanity after epoch 1
4. `gx1/models/entry_v10/xgb_calibration.py` (UPDATED)
   - Uses SSoT `get_calibrator_path()` for logging
   - Enhanced error messages with exact paths

---

## Usage Examples

### Build Dataset (Mini)
```bash
export GX1_REQUIRE_XGB_CALIBRATION=1

python gx1/scripts/build_entry_v10_ctx_training_dataset.py \
    --data data/entry_v9/full_2025.parquet \
    --output data/entry_v10_ctx/FULLYEAR_2025_mini.parquet \
    --start 2025-01-01T00:00:00Z \
    --end 2025-01-07T23:59:59Z \
    --max_rows 1000 \
    --policy_config gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml \
    --feature_meta_path gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json \
    --seq_scaler_path gx1/models/entry_v9/nextgen_2020_2025_clean/seq_scaler.joblib \
    --snap_scaler_path gx1/models/entry_v9/nextgen_2020_2025_clean/snap_scaler.joblib \
    --calibration_dir models/xgb_calibration \
    --calibration_method platt \
    --time_split
```

### Train with Worker Benchmark
```bash
export GX1_GATED_FUSION_ENABLED=1
export GX1_REQUIRE_XGB_CALIBRATION=1

python -m gx1.models.entry_v10.entry_v10_ctx_train \
    --data data/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION_train.parquet \
    --out_dir models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION \
    --feature_meta_path gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json \
    --seq_scaler_path gx1/models/entry_v9/nextgen_2020_2025_clean/seq_scaler.joblib \
    --snap_scaler_path gx1/models/entry_v9/nextgen_2020_2025_clean/snap_scaler.joblib \
    --seq_len 30 \
    --batch_size 32 \
    --epochs 10 \
    --learning_rate 1e-4 \
    --seed 1337 \
    --num_workers 2 \
    --benchmark_workers "0,2,4" \
    --enable_gate_stability_loss \
    --gate_stability_weight 0.1 \
    --policy_config gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml
```

---

## Benefits

1. **Robustness:**
   - Hard fail på alle kritiske feil (calibration, NaN/inf, range)
   - Soft warnings for potensielle problemer
   - SSoT for calibrator paths (ingen duplikasjon)

2. **Observability:**
   - Manifest med all metadata
   - Calibrator usage stats
   - Validation statistics
   - Worker benchmark results
   - Gate sanity warnings

3. **Flexibility:**
   - Mini-builds for testing (`--start/--end/--max_rows`)
   - Dry run for verifisering (`--dry_run`)
   - Worker benchmark for optimalisering (`--benchmark_workers`)

4. **Reproducibility:**
   - Manifest inkluderer git commit, build command, alle paths
   - Deterministic sampling (`--max_rows`)
   - All metadata skrevet til filer

---

**End of Document**
