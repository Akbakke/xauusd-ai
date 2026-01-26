# FULLYEAR_2025 Training Setup

**Date:** 2026-01-07  
**Status:** ✅ Ready for Training

---

## Overview

Training setup for ENTRY_V10_CTX + Gated Fusion on FULLYEAR_2025 dataset.

**Key Requirements:**
- ✅ Kalibrerte XGB-inputs (from Phase 1)
- ✅ uncertainty_score in features
- ✅ raw snapshot backup
- ✅ Gate statistics logging per epoch
- ✅ Loss components (decision vs gate-stability)
- ✅ Fail-fast if gate collapses (≈0 or ≈1 constant)

---

## Training Command

```bash
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
    --policy_config gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml
```

---

## Features

### 1. Kalibrerte XGB-Inputs

**Source:** Phase 1 calibration outputs

**In Dataset:**
- `snap_x[:, 85]`: `p_cal` (calibrated probability)
- `snap_x[:, 86]`: `margin` (from calibrated)
- `snap_x[:, 87]`: `p_hat` (from calibrated)

**In Sequence:**
- `seq_x[:, 13]`: `p_cal` (calibrated)
- `seq_x[:, 14]`: `margin` (from calibrated)
- `seq_x[:, 15]`: `uncertainty_score` (normalized entropy)

**Note:** Dataset must be built with calibrated XGB outputs. If using raw dataset, calibration must be applied during feature building.

### 2. Uncertainty Score

**Location:** `seq_x[:, 15]`

**Computation:**
```python
entropy = prob_entropy(p_cal)
uncertainty_score = entropy / log(2.0)  # Normalize to [0, 1]
```

**Usage:** Input to Gated Fusion (affects gate value)

### 3. Raw Snapshot Backup

**Location:** `snap_x[:, 0:85]` (base snapshot features, no XGB channels)

**Usage:** Encoded separately via `snap_raw_encoder` and used as backup in Gated Fusion

**Purpose:** Provides fallback representation when XGB state is uncertain

---

## Gate Statistics Logging

### Per Epoch

**Logged Metrics:**
- `gate_mean`: Mean gate value
- `gate_std`: Standard deviation
- `gate_min`: Minimum gate value
- `gate_max`: Maximum gate value
- `gate_p5`: 5th percentile
- `gate_p95`: 95th percentile
- `gate_variance`: Variance

**Per Regime:**
- `gate_per_regime[regime_key]`: Statistics per vol regime (LOW/MEDIUM/HIGH/EXTREME)

**Example Output:**
```
[GATE_STATS EPOCH 1]
  gate_mean: 0.5234
  gate_std: 0.1234
  gate_min: 0.1234
  gate_max: 0.8765
  gate_p5: 0.2345
  gate_p95: 0.7890
  gate_variance: 0.0152
  gate_per_regime:
    vol_0: mean=0.5123, std=0.1123, n=1234
    vol_1: mean=0.5234, std=0.1234, n=2345
    vol_2: mean=0.5345, std=0.1345, n=3456
    vol_3: mean=0.5456, std=0.1456, n=4567
```

---

## Loss Components

### Primary Losses

1. **Direction Loss:** `BCEWithLogitsLoss(direction_logit, y_direction)`
   - Weight: `1.0` (LOSS_WEIGHT_DIRECTION)

2. **Early Move Loss:** `BCEWithLogitsLoss(early_move_logit, y_early_move)`
   - Weight: `0.5` (LOSS_WEIGHT_EARLY_MOVE)

3. **Quality Loss:** `SmoothL1Loss(quality_score, y_quality_score)`
   - Weight: `0.25` (LOSS_WEIGHT_QUALITY)

### Gate Stability Loss (Optional)

**Enabled:** `--enable_gate_stability_loss`

**Computation:**
```python
# Encourage gate stability in stable regimes (LOW=0, MEDIUM=1)
stable_regime_mask = (vol_regime_id == 0) | (vol_regime_id == 1)
gate_stable = gate[stable_regime_mask]
gate_variance = torch.var(gate_stable)
loss_gate_stability = gate_variance * stable_regime_mask.float().mean()
```

**Weight:** `--gate_stability_weight` (default: `0.1`)

**Total Loss:**
```python
total_loss = (
    1.0 * loss_direction +
    0.5 * loss_early_move +
    0.25 * loss_quality +
    0.1 * loss_gate_stability  # If enabled
)
```

---

## Fail-Fast: Gate Collapse Detection

**Trigger:** Gate collapses to ≈0 or ≈1 (constant)

**Check:**
```python
if gate_std < 0.01:  # Very low variance
    if gate_mean < 0.05:  # Gate ≈ 0
        raise RuntimeError("GATE_COLLAPSE: Gate collapsed to ≈0")
    elif gate_mean > 0.95:  # Gate ≈ 1
        raise RuntimeError("GATE_COLLAPSE: Gate collapsed to ≈1")
```

**Action:** Training stops immediately with clear error message

**Why:** If gate is constant, Gated Fusion is not learning. This indicates:
- Model initialization issue
- Gate MLP not receiving gradients
- Learning rate too high/low
- Data issue (all samples have same uncertainty)

---

## Output Files

### Model Artifacts

- `model_state_dict.pt`: Model weights
- `bundle_metadata.json`: Bundle metadata (includes gated fusion info)
- `feature_contract_hash.txt`: Feature contract hash
- `train_config.json`: Training configuration

### Metrics

- `metrics.json`: Training metrics including:
  - First/last step losses
  - Mean gradient norm
  - Eligibility stats
  - Epoch metrics (with gate stats)
  - Wall clock time
  - Throughput

---

## Expected Gate Behavior

### Healthy Gate

- **Mean:** ~0.4-0.6 (no strong bias)
- **Std:** > 0.1 (varies based on uncertainty)
- **Distribution:** Spread across [0, 1] range
- **Per Regime:** Varies by regime (HIGH vol may have different gate than LOW vol)

### Warning Signs

- **Gate ≈ 0:** Model always trusts Transformer (ignores XGB)
- **Gate ≈ 1:** Model always trusts XGB (ignores Transformer)
- **Low Variance:** Gate not learning (constant)
- **Extreme Values:** All gates near 0 or 1

### Fail-Fast Triggers

- `gate_std < 0.01` AND `gate_mean < 0.05` → **GATE_COLLAPSE: ≈0**
- `gate_std < 0.01` AND `gate_mean > 0.95` → **GATE_COLLAPSE: ≈1**

---

## Next Steps

1. **Train Model:**
   ```bash
   # Run training command above
   ```

2. **Monitor Gate Stats:**
   - Check per-epoch gate statistics
   - Verify gate responds to uncertainty
   - Confirm no collapse

3. **Evaluate:**
   ```bash
   python gx1/analysis/eval_gated_fusion.py \
       --replay_dir runs/replay_shadow/... \
       --output_dir reports/fusion
   ```

4. **GO/NO-GO:**
   - Gate statistics healthy
   - No collapse detected
   - Performance metrics acceptable

---

## Troubleshooting

### Gate Collapses to 0

**Possible Causes:**
- Gate MLP initialization too negative
- Learning rate too high
- XGB state not informative
- Transformer always better

**Fix:**
- Check gate MLP initialization (should be bias=0.0)
- Reduce learning rate
- Verify XGB calibration is working
- Check if Transformer is dominating

### Gate Collapses to 1

**Possible Causes:**
- Gate MLP initialization too positive
- Learning rate too high
- XGB always better
- Transformer not learning

**Fix:**
- Check gate MLP initialization
- Reduce learning rate
- Verify Transformer is learning
- Check if XGB is dominating

### Gate Not Varying

**Possible Causes:**
- Gate MLP not receiving gradients
- All samples have same uncertainty
- Learning rate too low

**Fix:**
- Check gradient flow to gate MLP
- Verify uncertainty_score varies
- Increase learning rate

---

**End of Document**
