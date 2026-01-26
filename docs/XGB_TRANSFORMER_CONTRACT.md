# XGB → Transformer Contract

**Version:** 1.0  
**Updated:** 2026-01-25

---

## Overview

This document defines the contract between XGBoost outputs and Transformer inputs
in the V10 Hybrid Entry pipeline.

---

## What XGB Represents

XGBoost models in V10 Hybrid serve as **feature extractors**, not decision makers.

| Output | Description | Range | Semantics |
|--------|-------------|-------|-----------|
| `p_long_xgb` | Raw probability of profitable long entry | [0, 1] | Higher = more likely profitable |
| `p_hat_xgb` | Calibrated probability | [0, 1] | After Platt/Isotonic scaling |
| `uncertainty_score` | Model uncertainty estimate | [0, 1] | Higher = less confident |

**Important:** XGB outputs are **signals to the Transformer**, not entry decisions.

---

## Allowed Transformer Inputs

### Sequence (SEQ) Input

| Feature | Source | Notes |
|---------|--------|-------|
| Base SEQ features (12) | Prebuilt parquet | z-score normalized |
| `p_long_xgb` | XGB output | Calibrated if available |
| `uncertainty_score` | XGB output | Signal, not veto |
| Padding (1) | Constant zeros | |

**Total:** 15 features, shape `[T=48, 15]`

### Snapshot (SNAP) Input

| Feature | Source | Notes |
|---------|--------|-------|
| Base SNAP features (85) | Prebuilt parquet | z-score normalized |
| `p_long_xgb` | XGB output | Calibrated if available |
| `p_hat_xgb` | XGB output | Calibrated probability |

**Total:** 87 features, shape `[87]`

---

## OOD Handling

Out-of-distribution (OOD) XGB outputs can destabilize Transformer predictions.

### Quantile Clipping (Recommended)

All XGB outputs are clipped to training quantiles:

```
p_long_xgb → clip(p_long_xgb, p1_train, p99_train)
```

This prevents extreme values from affecting downstream predictions.

### Calibration (Recommended)

Platt scaling or Isotonic regression is applied to stabilize probabilities
across different years/regimes:

```
p_calibrated = calibrator.transform(p_raw)
```

Calibrator is trained on multiyear data (2020-2025) and stored as artifact.

---

## uncertainty_score Semantics

**Previous behavior (deprecated):**
- `uncertainty_score` could implicitly block entries via thresholds
- This caused silent trade reduction in uncertain regimes

**Current behavior:**
- `uncertainty_score` is a **signal channel** to the Transformer
- Transformer learns how to weight uncertainty
- No implicit entry blocking based on uncertainty
- Telemetry logs: `uncertainty_used_as = "signal"`

---

## Regression Guards

The following guards are enforced in truth runs:

1. **Raw XGB outputs must not be used directly**
   - Calibration or at minimum quantile clipping is required
   - Hard-fail if `GX1_XGB_CALIBRATOR_PATH` is not set in truth mode

2. **margin_xgb must not be present**
   - Removed in Jan 2026 cleanup
   - Hard-fail if detected in channel lists

3. **feature_build_call_count must be 0**
   - PREBUILT-only enforcement
   - Hard-fail if features are built at runtime

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GX1_XGB_CALIBRATOR_PATH` | Path to calibrator .pkl | None (disabled) |
| `GX1_XGB_CLIPPER_PATH` | Path to quantile clipper .pkl | None (disabled) |
| `GX1_XGB_CHANNEL_MASK` | Channels to mask (comma-sep) | "" |
| `GX1_XGB_REQUIRE_CALIBRATION` | Hard-fail if no calibrator | 0 |

### Artifacts

| Artifact | Location | SHA Logged |
|----------|----------|------------|
| Calibrator | `GX1_DATA/models/calibrators/xgb_calibrator_*.pkl` | Yes |
| Clipper | `GX1_DATA/models/calibrators/xgb_clipper_*.pkl` | Yes |

---

## Telemetry

The following fields are logged in `ENTRY_FEATURES_USED.json`:

```json
{
  "xgb_used_as": "pre",
  "xgb_calibration_applied": true,
  "xgb_calibrator_sha": "a1b2c3d4...",
  "xgb_clipping_applied": true,
  "xgb_clipper_sha": "e5f6g7h8...",
  "uncertainty_used_as": "signal",
  "p_long_raw_mean": 0.48,
  "p_long_calibrated_mean": 0.50
}
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-25 | Initial contract, calibration support |
