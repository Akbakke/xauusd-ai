# FULLYEAR_2025 Training Robust Setup

**Date:** 2026-01-07  
**Status:** ✅ Ready for Training

---

## Overview

Robust and falsifiable training setup for ENTRY_V10_CTX + Gated Fusion on FULLYEAR_2025 dataset.

**Key Requirements:**
- ✅ TRAIN policy config (not VERIFY)
- ✅ Gate collapse detection (including "stuck at 0.5")
- ✅ Gate responsiveness metrics (correlations) per epoch
- ✅ Dataset calibration verification at startup
- ✅ Time-based split (not random)

---

## 1. TRAIN Policy Config

**File:** `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml`

**Key Settings:**
- `entry_models.v10_ctx.enabled: true`
- `require_v9_for_entry: false`
- `GX1_GATED_FUSION_ENABLED: true` (via environment variable)
- `GX1_REQUIRE_XGB_CALIBRATION: 1` (via environment variable)

**Post-Veto Gates (Only Safety Kill-Switch):**
- `HARD_WARMUP`: Insufficient warmup bars (< 288)
- `HARD_SESSION_BLOCK`: ASIA session blocked (SNIPER requirement)
- `HARD_SPREAD_CAP`: Spread > 100 bps (hard cap)
- `KILL_SWITCH`: System kill-switch active (not used in training)

**Note:** No post-model veto gates (thresholds, regime gates, etc.). All filtering happens via eligibility checks before model inference.

---

## 2. Gate Collapse Detection

### Detection Rules

**Hard Fail (Immediate Stop):**
- `gate_std < 0.01` AND `gate_mean < 0.05` → **GATE_COLLAPSE: ≈0**
- `gate_std < 0.01` AND `gate_mean > 0.95` → **GATE_COLLAPSE: ≈1**

**Warning (Stop After 2-3 Epochs):**
- `gate_std < 0.02` AND `0.45 < gate_mean < 0.55` → **GATE_STUCK_NEUTRAL**

**Debug Output (When Stuck):**
- Sample batch dump: `uncertainty_score`, `margin`, `entropy`, `gate`
- Input variance check: `uncertainty_score std`, `margin std`, `entropy std`

**Action:**
- First occurrence: Warning + debug dump
- After 2-3 epochs: Hard fail with clear error message

---

## 3. Gate Responsiveness Metrics

### Per Epoch Logging

**Spearman Correlations:**
- `corr(gate, uncertainty_score)`: Expected < 0 (high uncertainty → low gate)
- `corr(gate, entropy)`: Expected < 0 (high entropy → low gate)
- `corr(gate, abs(margin))`: Expected > 0 (high margin → high gate)

**Bucketed Means:**
- `gate_by_uncertainty_decile`: Gate mean for 10 uncertainty deciles
  - Decile 1: Lowest uncertainty (0-10th percentile)
  - Decile 10: Highest uncertainty (90-100th percentile)
  - Expected: Gate decreases as uncertainty increases

**Example Output:**
```
[GATE_RESPONSIVENESS EPOCH 1]
  ✅ corr(gate, uncertainty_score): -0.3421 (expected < 0)
  ✅ corr(gate, entropy): -0.3123 (expected < 0)
  ✅ corr(gate, abs(margin)): 0.2345 (expected > 0)
  gate_by_uncertainty_decile:
    decile_1: uncertainty=[0.000, 0.123], gate_mean=0.6234, n=1234
    decile_2: uncertainty=[0.123, 0.234], gate_mean=0.6123, n=1234
    ...
    decile_10: uncertainty=[0.876, 1.000], gate_mean=0.4123, n=1234
```

---

## 4. Dataset Calibration Verification

### Startup Checks

**Check 1: Dataset Metadata**
- Look for `calibration_applied` flag in dataset metadata
- If found: Verify flag is `True`

**Check 2: Distribution Comparison**
- If `p_cal` and `p_raw` columns exist:
  - Compare distributions (mean, std)
  - If distributions differ significantly: Calibration verified
  - If distributions similar: Warning (calibration may not be applied)

**Check 3: Calibration Directory**
- Check if `models/xgb_calibration/<policy_id>` exists
- If exists: Assume calibration will be applied during feature building

**Check 4: Sample Verification**
- Extract sample from dataset
- Verify `snap_x[85:88]` values are in valid ranges:
  - `p_cal` ∈ [0, 1]
  - `margin` ∈ [0, 1]
  - `p_hat` ∈ [0, 1]
- Verify `seq_x[:, 15]` (uncertainty_score) ∈ [0, 1]

**Fail-Fast:**
- If none of the above checks pass: Hard fail with clear error message
- No silent assumptions about calibration

---

## 5. Time-Based Split

### Split Configuration

**Train:** 2025-01-01 → 2025-09-30 (9 months)
**Val:** 2025-10-01 → 2025-11-30 (2 months)
**Test:** 2025-12-01 → 2025-12-31 (1 month, hold-out for eval)

**Implementation:**
- Split based on `ts` column (if available)
- If no `ts` column: Warning + use full dataset

**Logging:**
```
[TIME_SPLIT]
  Train: 2025-01-01 → 2025-09-30 (123456 rows)
  Val: 2025-10-01 → 2025-11-30 (12345 rows)
  Test: 2025-12-01 → 2025-12-31 (1234 rows)
  Total: 137035 rows
```

**Note:** Training uses train split only. Val/test splits can be used for evaluation later.

---

## Updated Training Command

```bash
# Set environment variables
export GX1_GATED_FUSION_ENABLED=1
export GX1_REQUIRE_XGB_CALIBRATION=1
export GX1_ALLOW_UNCALIBRATED_XGB=0

# Run training
python -m gx1.models.entry_v10.entry_v10_ctx_train \
    --data data/entry_v10/entry_v10_dataset.parquet \
    --out_dir models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION \
    --feature_meta_path gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json \
    --seq_scaler_path gx1/models/entry_v9/nextgen_2020_2025_clean/seq_scaler.joblib \
    --snap_scaler_path gx1/models/entry_v9/nextgen_2020_2025_clean/snap_scaler.joblib \
    --seq_len 30 \
    --batch_size 64 \
    --epochs 10 \
    --learning_rate 1e-4 \
    --seed 1337 \
    --enable_gate_stability_loss \
    --gate_stability_weight 0.1 \
    --policy_config gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml
```

---

## Expected Output

### Per Epoch

**Loss Components:**
```
[EPOCH 1 SUMMARY]
  loss: 0.5234
  loss_direction: 0.4123
  loss_early_move: 0.2345
  loss_quality: 0.1234
  loss_gate_stability: 0.0123
  wall_clock_sec: 1234.5
  throughput: 123.45 samples/sec
```

**Gate Statistics:**
```
[GATE_STATS EPOCH 1]
  gate_mean: 0.5234
  gate_std: 0.1234
  gate_variance: 0.0152
  gate_per_regime:
    vol_0: mean=0.5123, std=0.1123, n=1234
    ...
```

**Gate Responsiveness:**
```
[GATE_RESPONSIVENESS EPOCH 1]
  ✅ corr(gate, uncertainty_score): -0.3421 (expected < 0)
  ✅ corr(gate, entropy): -0.3123 (expected < 0)
  ✅ corr(gate, abs(margin)): 0.2345 (expected > 0)
  gate_by_uncertainty_decile:
    decile_1: uncertainty=[0.000, 0.123], gate_mean=0.6234, n=1234
    ...
    decile_10: uncertainty=[0.876, 1.000], gate_mean=0.4123, n=1234
```

---

## Fail-Fast Conditions

**Hard Fail (Training Stops):**
1. `GATE_COLLAPSE: ≈0` (gate collapsed to 0)
2. `GATE_COLLAPSE: ≈1` (gate collapsed to 1)
3. `GATE_STUCK_NEUTRAL` (after 2-3 epochs)
4. `CALIBRATION_VERIFY_FAILED` (cannot prove calibration)
5. `CALIBRATION_VERIFY_FAILED` (sample values invalid)

**Warnings (Continue Training):**
1. `GATE_STUCK_NEUTRAL` (first occurrence)
2. `p_raw` and `p_cal` distributions similar
3. No `ts` column (no time split)

---

## Metrics JSON Output

**Per Epoch:**
```json
{
  "epoch": 1,
  "loss": 0.5234,
  "loss_direction": 0.4123,
  "loss_early_move": 0.2345,
  "loss_quality": 0.1234,
  "loss_gate_stability": 0.0123,
  "gate_stats": {
    "gate_mean": 0.5234,
    "gate_std": 0.1234,
    "gate_responsiveness": {
      "corr_gate_uncertainty": -0.3421,
      "corr_gate_entropy": -0.3123,
      "corr_gate_margin": 0.2345,
      "gate_by_uncertainty_decile": {
        "decile_1": {"uncertainty_range": [0.0, 0.123], "gate_mean": 0.6234, "n_samples": 1234},
        ...
      }
    }
  }
}
```

---

## Summary

**Files Created/Modified:**
- ✅ `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml`
- ✅ `gx1/models/entry_v10/entry_v10_ctx_train.py` (updated)
- ✅ `docs/FULLYEAR_2025_TRAINING_ROBUST_SETUP.md`

**Features:**
- ✅ TRAIN policy config (not VERIFY)
- ✅ Gate collapse detection (including "stuck at 0.5")
- ✅ Gate responsiveness metrics (correlations + deciles)
- ✅ Dataset calibration verification
- ✅ Time-based split (train/val/test)

**No Model Architecture Changes:**
- All changes are hygiene/telemetry/config
- Model architecture unchanged

---

**End of Document**
