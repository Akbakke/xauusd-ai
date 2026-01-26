# FULLYEAR_2025 Training Pipeline Fix

**Date:** 2026-01-07  
**Status:** ✅ Implemented

---

## Problem

FULLYEAR training brukte raw XGB fordi calibrators ikke fantes. Dette undergraver hele Phase 1+2 (gate lærer feil).

---

## Solution

### 1. Hard Fail hvis Calibrators Mangler

**File:** `gx1/scripts/build_entry_v10_ctx_training_dataset.py`

- `verify_calibrators_exist()` funksjon som:
  - Logger eksakt path som søkes: `{calibration_dir}/{policy_id}/{session}/calibrator_{method}.joblib`
  - Logger policy_id, session, og bucket
  - Hard fail hvis `GX1_REQUIRE_XGB_CALIBRATION=1` og calibrators mangler
  - Ingen fallback til raw når require=1

**File:** `gx1/models/entry_v10/xgb_calibration.py`

- `apply_xgb_calibration()` oppdatert til å hard fail hvis:
  - `GX1_REQUIRE_XGB_CALIBRATION=1` og calibrator mangler
  - Logger eksakt session og regime_bucket som mangler

---

### 2. Del Pipeline: Offline Build Script

**File:** `gx1/scripts/build_entry_v10_ctx_training_dataset.py`

**Funksjonalitet:**
- Bygger FULLYEAR_2025 train/val/test parquet med:
  - Base features (V9)
  - HTF features (H1/H4)
  - XGB inference (per session)
  - XGB calibration (Platt/Isotonic)
  - Uncertainty signals (entropy, uncertainty_score)
  - Sequence/snapshot/context feature packing
- Setter `calibration_applied=true` i metadata
- Støtter time-based split (train/val/test)

**Usage:**
```bash
python gx1/scripts/build_entry_v10_ctx_training_dataset.py \
    --data data/entry_v9/full_2025.parquet \
    --output data/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION.parquet \
    --policy_config gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml \
    --feature_meta_path gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json \
    --seq_scaler_path gx1/models/entry_v9/nextgen_2020_2025_clean/seq_scaler.joblib \
    --snap_scaler_path gx1/models/entry_v9/nextgen_2020_2025_clean/snap_scaler.joblib \
    --calibration_dir models/xgb_calibration \
    --calibration_method platt \
    --time_split
```

---

### 3. Training Script: Bruker Prebuilt Dataset

**File:** `gx1/models/entry_v10/entry_v10_ctx_train.py`

**Endringer:**
- Verifiserer at dataset har `calibration_applied=true` metadata
- Hard fail hvis `GX1_REQUIRE_XGB_CALIBRATION=1` og dataset mangler calibration
- Dataset bruker prebuilt features (ikke bygger features i training loop)
- Verifiserer at dataset har XGB kolonner: `p_cal`, `margin`, `p_hat`, `uncertainty_score`

**File:** `gx1/models/entry_v10/entry_v10_ctx_dataset.py`

**Endringer:**
- `__getitem__()` oppdatert til å bruke prebuilt features fra DataFrame
- Ikke lenger bygger features via `build_v9_runtime_features()`
- Leser XGB features direkte fra DataFrame kolonner

---

### 4. Out Dir Hygiene

**File:** `gx1/models/entry_v10/entry_v10_ctx_train.py`

**Funksjonalitet:**
- Oppretter `out_dir` ved start
- Skriver metadata før første epoch:
  - `train_config.json`: Training konfigurasjon
  - `env_dump.json`: Environment variabler
  - `git_commit.txt`: Git commit hash
- Logger til `out_dir/training.log` (unbuffered, line buffered)

**Helper Functions:**
- `write_training_metadata()`: Skriver alle metadata filer
- `get_git_commit()`: Henter git commit hash
- `Tee` class: Tee output til multiple streams (stdout + log file)

---

### 5. DataLoader Throughput

**File:** `gx1/models/entry_v10/entry_v10_ctx_train.py`

**Endringer:**
- `--num_workers` default endret fra 0 til 2
- `persistent_workers=True` når `num_workers > 0`
- Logger samples/sec per epoch for å måle effekt

**DataLoader Configuration:**
```python
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=device.type != "cpu",
    persistent_workers=args.num_workers > 0,  # PHASE 2
    collate_fn=collate_entry_v10_ctx_batch,
)
```

---

## Workflow

### Step 1: Build Dataset (Offline)

```bash
export GX1_REQUIRE_XGB_CALIBRATION=1

python gx1/scripts/build_entry_v10_ctx_training_dataset.py \
    --data data/entry_v9/full_2025.parquet \
    --output data/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION.parquet \
    --policy_config gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml \
    --feature_meta_path gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json \
    --seq_scaler_path gx1/models/entry_v9/nextgen_2020_2025_clean/seq_scaler.joblib \
    --snap_scaler_path gx1/models/entry_v9/nextgen_2020_2025_clean/snap_scaler.joblib \
    --calibration_dir models/xgb_calibration \
    --calibration_method platt \
    --time_split
```

**Output:**
- `FULLYEAR_2025_GATED_FUSION_train.parquet`
- `FULLYEAR_2025_GATED_FUSION_val.parquet`
- `FULLYEAR_2025_GATED_FUSION_test.parquet`

### Step 2: Train Model

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
    --enable_gate_stability_loss \
    --gate_stability_weight 0.1 \
    --policy_config gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml
```

**Output:**
- `models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION/train_config.json`
- `models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION/env_dump.json`
- `models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION/git_commit.txt`
- `models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION/training.log`
- `models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION/model_state_dict.pt`
- `models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION/bundle_metadata.json`

---

## Benefits

1. **No Feature Building in Training Loop:**
   - Throughput er IO/loader-bound, ikke feature-build-bound
   - Kan bruke `num_workers > 0` for parallel loading
   - `persistent_workers=True` for bedre performance

2. **Calibration Guaranteed:**
   - Hard fail hvis calibrators mangler når `GX1_REQUIRE_XGB_CALIBRATION=1`
   - Ingen silent fallback til raw XGB
   - Gate lærer på kalibrerte features, ikke raw

3. **Reproducibility:**
   - `train_config.json`: Alle training parametere
   - `env_dump.json`: Environment variabler
   - `git_commit.txt`: Git commit hash
   - `training.log`: Full training log

4. **Better Throughput:**
   - Prebuilt dataset = raskere loading
   - `num_workers=2` + `persistent_workers=True` = parallel loading
   - Samples/sec logging per epoch

---

## Files Changed

1. `gx1/scripts/build_entry_v10_ctx_training_dataset.py` (NEW)
2. `gx1/models/entry_v10/entry_v10_ctx_train.py` (UPDATED)
3. `gx1/models/entry_v10/entry_v10_ctx_dataset.py` (UPDATED)
4. `gx1/models/entry_v10/xgb_calibration.py` (UPDATED - hard fail logic)

---

**End of Document**
