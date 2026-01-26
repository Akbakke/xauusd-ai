# STEP 3C: FULLYEAR Training Plan

**Status:** Ready for execution  
**Baseline:** Smoke-run (SMOKE_20260106_ctxfusion)  
**Date:** 2026-01-06

---

## 1) FULLYEAR Training Command

```bash
export ENTRY_CONTEXT_FEATURES_ENABLED=true
python3 -m gx1.models.entry_v10.entry_v10_ctx_train \
  --data data/entry_v9/full_2025.parquet \
  --out_dir models/entry_v10_ctx/FULLYEAR_2025_ctxfusion \
  --feature_meta_path gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json \
  --seq_scaler_path gx1/models/entry_v9/nextgen_2020_2025_clean/seq_scaler.joblib \
  --snap_scaler_path gx1/models/entry_v9/nextgen_2020_2025_clean/snap_scaler.joblib \
  --seq_len 30 \
  --batch_size 64 \
  --epochs 5 \
  --num_workers 0 \
  --device cpu \
  --seed 1337 \
  --log_every 100 \
  --learning_rate 1e-4 \
  2>&1 | tee models/entry_v10_ctx/FULLYEAR_2025_ctxfusion/run_log.txt
```

**Key differences from smoke-run:**
- `--limit_samples` removed (full dataset)
- `--epochs 5` (instead of 1)
- `--log_every 100` (instead of 50, for less verbose output)
- `--seq_scaler_path` and `--snap_scaler_path` added (for consistency with runtime)

---

## 2) Expected Runtime and Memory Usage

### Runtime Estimate
- **Total rows:** 70,217
- **Estimated eligible (94.3%):** ~66,214 samples
- **Smoke-run baseline:** 190 seconds for 4,713 samples (1 epoch)
- **Estimated per epoch:** ~44 minutes
- **Estimated total (5 epochs):** ~222 minutes (~3.7 hours)

**Note:** Actual runtime may vary based on:
- CPU load
- Memory pressure
- Dataset filtering overhead

### Memory Usage
- **Model size:** ~3.1 MB (model_state_dict.pt)
- **Batch size:** 64 samples
- **Estimated peak memory:** ~2-4 GB (CPU, depends on feature computation)

**Recommendation:** Monitor memory during first epoch. If OOM occurs, reduce `--batch_size` to 32.

---

## 3) Risks Before Start

### Low Risk
- ✅ **Determinism:** seed=1337, num_workers=0 ensures reproducibility
- ✅ **Data quality:** Same data source as smoke-run (validated)
- ✅ **Feature contract:** Same hash as smoke-run (93ab1d31534f2dc7)

### Medium Risk
- ⚠️ **Memory:** Full dataset may cause OOM on systems with <8GB RAM
  - **Mitigation:** Monitor first epoch, reduce batch_size if needed
- ⚠️ **Runtime:** ~3.7 hours may be interrupted
  - **Mitigation:** Training script saves checkpoint after each epoch

### High Risk
- ❌ **None identified** (smoke-run validated all critical paths)

---

## 4) Smoke-Run Bundle Reference

**Baseline bundle:** `models/entry_v10_ctx/SMOKE_20260106_ctxfusion/`

**Key artifacts:**
- `feature_contract_hash.txt`: `93ab1d31534f2dc7`
- `train_config.json`: seq_len=30, batch_size=64, learning_rate=1e-4
- `bundle_metadata.json`: supports_context_features=true, ctx_cat_dim=5, ctx_cont_dim=2

**Verification:**
```bash
# Verify smoke-run bundle
python3 scripts/verify_entry_v10_ctx_bundle.py \
  --bundle_dir models/entry_v10_ctx/SMOKE_20260106_ctxfusion \
  --expected_hash 93ab1d31534f2dc7
```

---

## 5) Post-Train Verification Plan

### A) Bundle Verification
```bash
# Verify FULLYEAR bundle
python3 scripts/verify_entry_v10_ctx_bundle.py \
  --bundle_dir models/entry_v10_ctx/FULLYEAR_2025_ctxfusion \
  --expected_hash 93ab1d31534f2dc7
```

**Must pass:**
- ✅ `supports_context_features == true`
- ✅ `expected_ctx_cat_dim == 5`
- ✅ `expected_ctx_cont_dim == 2`
- ✅ `feature_contract_hash == 93ab1d31534f2dc7`
- ✅ `model_variant == "v10_ctx"`

### B) 1-Week Replay Comparison

**Baseline (smoke-run bundle):**
```bash
export ENTRY_CONTEXT_FEATURES_ENABLED=true
# Run 1-week replay with smoke-run bundle
# (baseline metrics)
```

**FULLYEAR (new bundle):**
```bash
export ENTRY_CONTEXT_FEATURES_ENABLED=true
# Run 1-week replay with FULLYEAR bundle
# (compare metrics)
```

**Compare:**
- `n_candidates` (should be similar or lower)
- `n_trades_created` (should be similar)
- `entry_quality` (MAE/MFE in first 6 bars)
- `goes_against_us_rate` (should be similar or better)

---

## 6) Telemetry & Reporting

Training script will output:
- **Per epoch:**
  - Loss curves (total, direction, early_move, quality)
  - Grad norm stats
  - Class balance (y_dir_mean)
  - Wall clock time
  - Throughput (samples/sec)

- **Final metrics.json:**
  - First/last step losses
  - Mean grad norm
  - Dataset len
  - Eligibility rate
  - Total wall clock time
  - Average throughput
  - Per-epoch metrics history

---

## 7) Success Criteria

✅ **Training completes** without errors  
✅ **Bundle verification passes** all checks  
✅ **Feature contract hash matches** smoke-run  
✅ **Loss decreases** across epochs (or stabilizes)  
✅ **No NaN/Inf** in losses or gradients  
✅ **Eligibility rate** is reasonable (0.1 < rate < 0.99)  

---

## 8) Next Steps (After Training)

1. ✅ Verify bundle
2. ✅ Run 1-week replay comparison
3. ⏸️ **STOP** - Do not proceed to:
   - Threshold sweep
   - Exit tuning
   - Hyperparameter tuning
   - Architecture changes

**Wait for explicit approval before proceeding to STEP 4.**



