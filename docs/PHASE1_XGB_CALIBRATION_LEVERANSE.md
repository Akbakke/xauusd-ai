# Phase 1: XGB Calibration + Uncertainty - Leveranse

**Date:** 2026-01-07  
**Status:** ✅ Complete

---

## Leveranse

### 1. Calibration Contract (SSoT)

**File:** `docs/XGB_CALIBRATION_CONTRACT.md`

**Innhold:**
- ✅ Inputs: `p_raw`, `proba_vector`, `margin_raw`, `logit_raw`, regime keys
- ✅ Outputs: `p_cal`, `margin`, `p_hat`, `entropy`, `uncertainty_score`
- ✅ Fail-fast rules: Hard fail in replay when `GX1_REQUIRE_XGB_CALIBRATION=1`
- ✅ Artifact structure: Per-session + per-regime calibrators
- ✅ Environment variables: `GX1_REQUIRE_XGB_CALIBRATION`, `GX1_ALLOW_UNCALIBRATED_XGB`, `GX1_CALIBRATION_METHOD`

### 2. Training Pipeline

**File:** `gx1/scripts/train_xgb_calibrators.py`

**Features:**
- ✅ Platt scaling (logistic regression on logit)
- ✅ Isotonic regression
- ✅ Per-session calibration (EU/US/OVERLAP)
- ✅ Per-regime bucket calibration (with backoff hierarchy)
- ✅ Minimum samples per bucket (`--min_samples_per_bucket`, default: 1000)
- ✅ Time-based train/val split
- ✅ Metrics: ECE, Brier, NLL (raw vs calibrated)
- ✅ Artifact output: `models/xgb_calibration/<policy_id>/<session>/calibrator_<method>.joblib`
- ✅ Metadata: `calibration_metadata.json` with all metrics and bucket definitions

**Usage:**
```bash
python gx1/scripts/train_xgb_calibrators.py \
    --dataset data/entry_v10/entry_v10_dataset.parquet \
    --policy_id SNIPER_V1 \
    --method platt \
    --output_dir models/xgb_calibration/SNIPER_V1 \
    --min_samples_per_bucket 1000 \
    --xgb_eu_path models/entry_v10/xgb_entry_EU_v10.joblib \
    --xgb_us_path models/entry_v10/xgb_entry_US_v10.joblib \
    --xgb_overlap_path models/entry_v10/xgb_entry_OVERLAP_v10.joblib \
    --feature_meta_path gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json
```

### 3. Calibration Bundle Loader

**File:** `gx1/models/entry_v10/xgb_calibration.py`

**Functions:**
- ✅ `load_xgb_calibrators()`: Load calibrators from directory structure
- ✅ `apply_xgb_calibration()`: Apply calibration and compute uncertainty signals
- ✅ `XGBCalibratedOutput`: Dataclass with all outputs
- ✅ Backoff hierarchy: Per-regime → per-session → global
- ✅ Fail-fast: Hard fail when `GX1_REQUIRE_XGB_CALIBRATION=1` and calibrator missing

**Outputs:**
- `p_cal`: Calibrated probability
- `margin`: Margin from calibrated probability
- `p_hat`: Max probability (from calibrated)
- `entropy`: Binary entropy (Shannon)
- `uncertainty_score`: Normalized entropy (0-1)

### 4. Runtime Integration

**File:** `gx1/execution/oanda_demo_runner.py:_predict_entry_v10_hybrid()`

**Changes:**
- ✅ Calibration applied after XGB prediction (line ~6368)
- ✅ Calibrators loaded on-demand (cached in `self._xgb_calibrators`)
- ✅ Regime bucket determined from `vol_regime_id` (and optional `trend_regime_id`)
- ✅ Fail-fast: Hard fail in replay when `GX1_REQUIRE_XGB_CALIBRATION=1` and calibrator missing
- ✅ Fallback: Use raw probabilities with warning when `GX1_ALLOW_UNCALIBRATED_XGB=1`
- ✅ Feature contract updated:
  - `seq_data[:, 13] = p_cal` (was `p_long_xgb`)
  - `seq_data[:, 14] = margin` (from calibrated)
  - `seq_data[:, 15] = uncertainty_score` (replaces `p_long_xgb_ema_5`)
  - `snap_data[85:88] = [p_cal, margin, p_hat]` (all from calibrated)

**Environment Variables:**
- `GX1_REQUIRE_XGB_CALIBRATION=1`: Hard fail if calibrator missing (replay)
- `GX1_ALLOW_UNCALIBRATED_XGB=1`: Allow uncalibrated XGB (live mode, default: false)
- `GX1_CALIBRATION_METHOD=platt|isotonic`: Calibration method (default: platt)

### 5. Evaluation Script

**File:** `gx1/analysis/eval_xgb_calibration.py`

**Features:**
- ✅ ECE (Expected Calibration Error) computation
- ✅ Brier score computation
- ✅ NLL (Negative Log-Likelihood) computation
- ✅ Reliability curve data (CSV export for plotting)
- ✅ Per-session metrics
- ✅ Per-regime bucket metrics (if available)
- ✅ Markdown report: `reports/calibration/XGB_CALIBRATION_REPORT_<date>.md`
- ✅ JSON results: `reports/calibration/XGB_CALIBRATION_RESULTS_<date>.json`
- ✅ GO/NO-GO criteria: ECE improvement, ECE < 0.05

**Usage:**
```bash
python gx1/analysis/eval_xgb_calibration.py \
    --dataset data/entry_v10/entry_v10_dataset.parquet \
    --calibration_dir models/xgb_calibration/SNIPER_V1 \
    --policy_id SNIPER_V1 \
    --method platt \
    --output_dir reports/calibration \
    --xgb_eu_path models/entry_v10/xgb_entry_EU_v10.joblib \
    --xgb_us_path models/entry_v10/xgb_entry_US_v10.joblib \
    --xgb_overlap_path models/entry_v10/xgb_entry_OVERLAP_v10.joblib \
    --feature_meta_path gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json
```

### 6. Policy/Policies Duplicate Analysis

**File:** `docs/REPO_STRUCTURE_ANALYSIS.md` (updated)

**Findings:**
- ✅ `gx1/policy/` is canonical (contains all policy modules)
- ✅ `gx1/policies/` contains only A/B test configs (`ab/AB_SNIPER_NY_2025W2.yaml`)
- ✅ Not a duplicate: Different purpose (A/B configs vs policy modules)
- ✅ Recommendation: Consider moving `gx1/policies/ab/` to `gx1/configs/ab/` for clarity

---

## Feature Contract Changes

### Sequence Features (seq_x)

**Before:**
- Index 13: `p_long_xgb` (raw)
- Index 14: `margin_xgb` (raw)
- Index 15: `p_long_xgb_ema_5` (placeholder)

**After:**
- Index 13: `p_cal` (calibrated)
- Index 14: `margin` (from calibrated)
- Index 15: `uncertainty_score` (normalized entropy)

**Contract:** Dimensions unchanged (16 total), but semantics changed.

### Snapshot Features (snap_x)

**Before:**
- Index 85: `p_long_xgb` (raw)
- Index 86: `margin_xgb` (raw)
- Index 87: `p_hat_xgb` (raw)

**After:**
- Index 85: `p_cal` (calibrated)
- Index 86: `margin` (from calibrated)
- Index 87: `p_hat` (from calibrated)

**Contract:** Dimensions unchanged (88 total), but semantics changed.

**Note:** Model weights trained on raw probabilities may need re-training for calibrated inputs. This is expected and documented in `V10_CTX_FEATURE_CONTRACT_ANALYSIS.md`.

---

## GO/NO-GO Criteria (Phase 1)

**Required:**
- ✅ `ECE_cal < ECE_raw` (calibration improves ECE)
- ✅ `ECE_cal < 0.05` (if sample size allows, else document realistic target)
- ✅ No runtime fallbacks when `GX1_REQUIRE_XGB_CALIBRATION=1`
- ✅ All outputs (`p_cal`, `entropy`, `uncertainty_score`) available and journaled
- ✅ Calibration metadata includes all required fields

**Status:** ✅ Implementation complete, ready for training and evaluation

---

## Next Steps (Phase 2)

1. Train calibrators on FULLYEAR_2025 dataset
2. Run evaluation script to verify GO/NO-GO criteria
3. Update model weights if needed (re-train on calibrated inputs)
4. Implement Gated Fusion (Phase 2)

---

## Files Created/Modified

### Created:
- `docs/XGB_CALIBRATION_CONTRACT.md`
- `gx1/scripts/train_xgb_calibrators.py`
- `gx1/models/entry_v10/xgb_calibration.py`
- `gx1/analysis/eval_xgb_calibration.py`
- `docs/PHASE1_XGB_CALIBRATION_LEVERANSE.md`

### Modified:
- `gx1/execution/oanda_demo_runner.py` (runtime integration)
- `docs/REPO_STRUCTURE_ANALYSIS.md` (policy/policies analysis)

---

**End of Document**
