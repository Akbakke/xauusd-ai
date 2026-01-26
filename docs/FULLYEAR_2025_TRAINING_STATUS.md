# FULLYEAR_2025 Training Status Report

**Date:** 2026-01-07  
**Status:** 🟡 Training in Progress

---

## Summary

FULLYEAR_2025 training for ENTRY_V10_CTX + Gated Fusion is currently running.

**Training Process:**
- PID: 8910
- Runtime: ~5+ minutes (as of last check)
- Status: Active (CPU: 84.6%, Memory: 5.4%)

---

## Training Configuration

**Dataset:**
- Source: `data/entry_v9/full_2025.parquet`
- Total rows: 70,217
- Time split:
  - Train: 2025-01-01 → 2025-09-30 (53,093 rows)
  - Val: 2025-10-01 → 2025-11-30 (11,814 rows)
  - Test: 2025-12-01 → 2025-12-31 (5,310 rows)

**Model Configuration:**
- Sequence length: 30
- Batch size: 32
- Epochs: 3
- Learning rate: 1e-4
- Seed: 1337
- Gate stability loss: Enabled (weight: 0.1)

**XGB Features:**
- ✅ XGB models loaded (EU, US, OVERLAP)
- ⚠️ Calibration: Using raw XGB features (calibrators not found)
- ✅ XGB channels added to seq_x (16 features) and snap_x (88 features)

**Policy:**
- Config: `GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml`
- Gated Fusion: Enabled
- XGB Calibration: Not required (using raw)

---

## Expected Outputs (Per Epoch)

### 1. Loss Components
- `loss`: Total loss
- `loss_direction`: Direction prediction loss
- `loss_early_move`: Early move prediction loss
- `loss_quality`: Quality score loss
- `loss_gate_stability`: Gate stability loss (if enabled)

### 2. Gate Statistics
- `gate_mean`: Mean gate value (expected: ~0.4–0.6)
- `gate_std`: Gate standard deviation (expected: > 0.1)
- `gate_min`: Minimum gate value
- `gate_max`: Maximum gate value
- `gate_p5`: 5th percentile
- `gate_p95`: 95th percentile
- `gate_variance`: Gate variance
- `gate_per_regime`: Gate statistics per regime

### 3. Gate Responsiveness
- `corr_gate_uncertainty`: Spearman correlation (expected: < 0)
- `corr_gate_entropy`: Spearman correlation (expected: < 0)
- `corr_gate_margin`: Spearman correlation (expected: > 0)
- `gate_by_uncertainty_decile`: Gate mean for 10 uncertainty deciles

### 4. Validation Metrics
- Val loss (should be stable, not diverging from train loss)
- Val accuracy (if computed)

---

## Success Criteria

### Gate Behavior ✅
- [ ] `gate_mean` ~0.4–0.6 (not stuck at 0, 0.5, or 1)
- [ ] `gate_std` > 0.1 (gate is responsive, not constant)
- [ ] Clear response to uncertainty (correlations as expected)

### Correlations ✅
- [ ] `corr_gate_uncertainty` < 0 (high uncertainty → low gate)
- [ ] `corr_gate_entropy` < 0 (high entropy → low gate)
- [ ] `corr_gate_margin` > 0 (high margin → high gate)

### Validation ✅
- [ ] No gate collapse (gate not stuck at 0 or 1)
- [ ] No calibration errors (using raw XGB is acceptable)
- [ ] Stable val-loss (not diverging from train loss)

---

## Known Issues

1. **Log Output:** Training log not being written to file (output may be going to stderr or buffered)
2. **Calibration:** Using raw XGB features (calibrators not trained yet - acceptable for initial training)

---

## Next Steps

1. **Wait for Training Completion:**
   - Monitor process: `ps aux | grep entry_v10_ctx_train`
   - Check output directory: `models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION/`
   - Look for: `model_state_dict.pt`, `metrics.json`, `train_config.json`

2. **Extract Results:**
   - Gate statistics per epoch
   - Gate responsiveness metrics
   - Loss curves (train vs val)
   - Final model checkpoint

3. **Evaluate:**
   - Check if gate behavior meets success criteria
   - Verify correlations are as expected
   - Confirm no gate collapse occurred

---

## How to Check Results

```bash
# Check if training is still running
ps aux | grep entry_v10_ctx_train

# Check output directory
ls -lh models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION/

# Check metrics (if available)
cat models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION/metrics.json

# Check training config
cat models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION/train_config.json
```

---

**Note:** Training is resource-intensive and may take 30-60+ minutes depending on dataset size and hardware. Please wait for completion before extracting final results.

---

**End of Report**
