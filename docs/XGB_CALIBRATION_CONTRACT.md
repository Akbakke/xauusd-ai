# XGB Calibration Contract

**Date:** 2026-01-07  
**Purpose:** Single Source of Truth (SSoT) for XGB calibration inputs, outputs, and fail-fast rules  
**Status:** Phase 1 - Calibration + Uncertainty

---

## Overview

XGB models produce raw probabilities that need calibration to match empirical frequencies. This contract defines:
- Inputs to calibration (raw XGB outputs)
- Outputs from calibration (calibrated probabilities + uncertainty signals)
- Fail-fast rules (no silent fallbacks)

---

## A) Inputs

### Per XGB Model (EU/US/OVERLAP)

**File:** `gx1/execution/oanda_demo_runner.py:_predict_entry_v10_hybrid()` (line 6368)

**Raw XGB Outputs:**
- `p_raw` (float): Raw probability for LONG class
  - Source: `xgb_model.predict_proba(xgb_features)[0]`
  - Mapping: Determined by `xgb_model.classes_` (0=SHORT, 1=LONG)
  - Range: [0.0, 1.0]
- `proba_vector` (np.ndarray, optional): Full probability vector for all classes
  - Source: `xgb_model.predict_proba(xgb_features)[0]`
  - Shape: `[n_classes]` (typically `[2]` for binary classification)
  - Usage: For entropy calculation (if multi-class) or binary entropy

**Derived Raw Metrics:**
- `margin_raw` (float): `abs(p_raw - (1.0 - p_raw))` = `abs(2 * p_raw - 1.0)`
  - Range: [0.0, 1.0]
  - Meaning: Confidence in prediction (0 = uncertain, 1 = very confident)
- `logit_raw` (float, optional): Logit transformation of `p_raw`
  - Formula: `logit = log(p_raw / (1.0 - p_raw))`
  - Usage: Required for Platt scaling (logistic regression on logit)

**Regime Keys (for per-regime calibration):**

| Key | Type | Source | Values | Stability |
|-----|------|--------|--------|-----------|
| `session_id` | int | `policy_state["session"]` | 0=ASIA, 1=EU, 2=US, 3=OVERLAP | ✅ Stable |
| `trend_regime_id` | int | `ctx_cat[1]` or `policy_state` | 0=UP, 1=DOWN, 2=NEUTRAL | ✅ Stable |
| `vol_regime_id` | int | `ctx_cat[2]` or `atr_regime_id` | 0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME | ✅ Stable |
| `atr_bucket` | int | `ctx_cat[3]` | 0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME | ✅ Stable |
| `spread_bucket` | int | `ctx_cat[4]` | 0=LOW, 1=MEDIUM, 2=HIGH | ⚠️ Less stable |

**Regime Bucket Definition:**
- **Primary:** `(session_id, vol_regime_id)` - Most stable combination
- **Secondary:** `(session_id, trend_regime_id, vol_regime_id)` - More granular
- **Fallback:** `session_id` only - If insufficient samples per bucket

**Minimum Samples per Bucket:**
- `N_min = 1000` (configurable via `CALIBRATION_MIN_SAMPLES_PER_BUCKET`)
- If `count < N_min`: Backoff to parent bucket (e.g., `(session, vol)` → `session`)

---

## B) Outputs (Runtime API)

### Standardized Outputs per Bar

**File:** `gx1/execution/oanda_demo_runner.py:_predict_entry_v10_hybrid()` (after calibration)

| Output | Type | Formula | Range | Usage |
|--------|------|---------|-------|-------|
| `p_raw` | float | Raw XGB probability | [0.0, 1.0] | Baseline comparison |
| `p_cal` | float | Calibrated probability | [0.0, 1.0] | **Primary output** |
| `margin` | float | `abs(p_cal - (1.0 - p_cal))` | [0.0, 1.0] | Confidence metric |
| `p_hat` | float | `max(p_cal, 1.0 - p_cal)` | [0.0, 1.0] | Max probability |
| `entropy` | float | Shannon entropy | [0.0, log(2)] | Uncertainty metric |
| `uncertainty_score` | float | Normalized uncertainty | [0.0, 1.0] | Gating signal |

### Detailed Output Definitions

#### 1. `p_cal` (Calibrated Probability)

**Method A: Platt Scaling**
```python
# Training: Fit logistic regression on logit
logit_raw = log(p_raw / (1.0 - p_raw))
p_cal = sigmoid(a * logit_raw + b)  # a, b from Platt scaler
```

**Method B: Isotonic Regression**
```python
# Training: Fit isotonic regressor on p_raw
p_cal = isotonic_regressor.transform([p_raw])[0]
```

**Selection:** Via config `calibration_method: "platt" | "isotonic"`

#### 2. `entropy` (Uncertainty Metric)

**Binary Classification:**
```python
p = p_cal
entropy = -(p * log(p) + (1.0 - p) * log(1.0 - p))
# Handle edge cases: p = 0 or p = 1 → entropy = 0
```

**Multi-class (if available):**
```python
proba_vector = [p_cal, 1.0 - p_cal]  # or full vector if multi-class
entropy = -sum(p_i * log(p_i) for p_i in proba_vector if p_i > 0)
```

**Range:** [0.0, log(2)] for binary, [0.0, log(n_classes)] for multi-class

#### 3. `uncertainty_score` (Normalized Gating Signal)

**Formula:**
```python
# Normalize entropy to [0.0, 1.0] for binary classification
entropy_max = log(2)  # Maximum entropy for binary
uncertainty_score = entropy / entropy_max

# Alternative: Combine entropy + margin
# uncertainty_score = 1.0 - margin  # Higher margin = lower uncertainty
# Or: uncertainty_score = (1.0 - margin) * (entropy / entropy_max)
```

**Default:** `uncertainty_score = entropy / log(2)` (simple normalization)

**Range:** [0.0, 1.0]
- `0.0` = Very certain (low entropy, high margin)
- `1.0` = Very uncertain (high entropy, low margin)

#### 4. `leaf_id` / `leaf_vec` (Optional, Phase 1 Hook)

**Purpose:** XGB leaf indices as uncertainty proxy (OOD detection)

**Implementation (Phase 1):**
- Placeholder: `leaf_id = None`, `leaf_vec = None`
- Hook: `xgb_model.apply(xgb_features)` if available
- Future: Extract leaf indices and use as embedding

**Status:** Optional in Phase 1, required in Phase 2 (Gated Fusion)

---

## C) Fail-Fast Rules

### Rule 1: Missing Calibrator (Replay Mode)

**Condition:** `GX1_REQUIRE_XGB_CALIBRATION=1` AND calibrator not found

**Action:** Hard fail with `RuntimeError`

**Error Message:**
```
XGB_CALIBRATION_REQUIRED: Calibrator missing for session={session}, regime={regime_bucket}.
Expected path: {expected_path}
Set GX1_REQUIRE_XGB_CALIBRATION=0 to allow uncalibrated XGB (not recommended).
```

**Location:** `gx1/execution/oanda_demo_runner.py:_predict_entry_v10_hybrid()`

### Rule 2: Missing Calibrator (Live Mode)

**Condition:** Calibrator not found AND `GX1_ALLOW_UNCALIBRATED_XGB=1` (default: false)

**Action:** Warning + use raw probabilities

**Warning Message:**
```
[WARNING] XGB_CALIBRATION_MISSING: Calibrator missing for session={session}, regime={regime_bucket}.
Using raw probabilities (p_cal = p_raw). This is not recommended for production.
Set GX1_REQUIRE_XGB_CALIBRATION=1 to hard-fail on missing calibrators.
```

**Fallback:** `p_cal = p_raw`, `entropy` and `uncertainty_score` computed from `p_raw`

### Rule 3: Calibration Method Mismatch

**Condition:** Calibrator method does not match expected method

**Action:** Hard fail in replay, warning in live

**Error Message:**
```
XGB_CALIBRATION_METHOD_MISMATCH: Expected method={expected}, got={actual}.
Calibrator path: {calibrator_path}
```

### Rule 4: Dimension Mismatch

**Condition:** Calibrator input dimension does not match XGB output

**Action:** Hard fail

**Error Message:**
```
XGB_CALIBRATION_DIM_MISMATCH: Calibrator expects {expected_dim} classes, got {actual_dim}.
XGB model classes: {xgb_model.classes_}
```

### Rule 5: Invalid Probability Range

**Condition:** `p_cal` outside [0.0, 1.0] after calibration

**Action:** Hard fail

**Error Message:**
```
XGB_CALIBRATION_INVALID_OUTPUT: p_cal={p_cal} outside [0.0, 1.0].
This indicates a corrupted calibrator. Calibrator path: {calibrator_path}
```

---

## D) Calibration Artifact Structure

### Directory Structure

```
models/xgb_calibration/
  <policy_id>/              # e.g., "SNIPER_V1", "FARM_V2B"
    <session>/              # "EU", "US", "OVERLAP"
      calibrator_platt.joblib
      calibrator_isotonic.joblib
      calibration_metadata.json
      <regime_bucket>/      # Optional: per-regime calibrators
        calibrator_platt.joblib
        calibrator_isotonic.joblib
        calibration_metadata.json
```

### Calibration Metadata JSON

**File:** `calibration_metadata.json`

```json
{
  "method": "platt" | "isotonic",
  "session": "EU" | "US" | "OVERLAP",
  "regime_bucket": "EU_LOW" | "EU_MEDIUM" | null,
  "training_window": {
    "start": "2025-01-01T00:00:00Z",
    "end": "2025-12-31T23:55:00Z"
  },
  "metrics": {
    "ece_raw": 0.123,
    "ece_cal": 0.045,
    "brier_raw": 0.234,
    "brier_cal": 0.198,
    "nll_raw": 0.456,
    "nll_cal": 0.412,
    "improvement_ece": 0.078,
    "improvement_brier": 0.036
  },
  "bucket_definition": {
    "keys": ["session_id", "vol_regime_id"],
    "values": {
      "EU_LOW": {"session_id": 1, "vol_regime_id": 0},
      "EU_MEDIUM": {"session_id": 1, "vol_regime_id": 1}
    }
  },
  "counts": {
    "total": 50000,
    "per_bucket": {
      "EU_LOW": 12000,
      "EU_MEDIUM": 15000,
      "EU_HIGH": 8000
    }
  },
  "backoff_hierarchy": {
    "EU_LOW": "EU_LOW",
    "EU_MEDIUM": "EU_MEDIUM",
    "EU_HIGH": "EU"  // Backed off to session-only
  },
  "feature_contract_hash": "abc123...",
  "xgb_model_hash": "def456..."
}
```

---

## E) Runtime Integration Points

### 1. Calibrator Loading

**Location:** `gx1/execution/oanda_demo_runner.py:__init__()` (after XGB models loaded)

**Function:** `load_xgb_calibrators(policy_id, sessions, calibration_method)`

**Returns:** `Dict[str, Dict[str, Any]]` = `{session: {regime_bucket: calibrator}}`

### 2. Calibration Application

**Location:** `gx1/execution/oanda_demo_runner.py:_predict_entry_v10_hybrid()` (after XGB prediction, line 6368)

**Function:** `apply_xgb_calibration(p_raw, session, regime_bucket, calibrators)`

**Returns:** `XGBCalibratedOutput(p_raw, p_cal, margin, p_hat, entropy, uncertainty_score)`

### 3. Feature Contract Update

**Location:** `gx1/execution/oanda_demo_runner.py:_predict_entry_v10_hybrid()` (after calibration)

**Current:** `seq_data[:, 13:16] = [p_long_xgb, margin_xgb, p_long_xgb_ema_5]`

**Updated:** `seq_data[:, 13:16] = [p_cal, margin, uncertainty_score]` (or more channels if needed)

**Contract Change:** Document in `bundle_metadata.json` if dimensions change

---

## F) Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GX1_REQUIRE_XGB_CALIBRATION` | `0` | Hard fail if calibrator missing (replay) |
| `GX1_ALLOW_UNCALIBRATED_XGB` | `0` | Allow uncalibrated XGB in live mode |
| `GX1_CALIBRATION_METHOD` | `"platt"` | Calibration method: "platt" or "isotonic" |
| `GX1_CALIBRATION_MIN_SAMPLES` | `1000` | Minimum samples per bucket for per-regime calibration |

---

## G) GO/NO-GO Criteria (Phase 1)

**Required:**
- ✅ `ECE_cal < ECE_raw` (calibration improves ECE)
- ✅ `ECE_cal < 0.05` (if sample size allows, else document realistic target)
- ✅ No runtime fallbacks when `GX1_REQUIRE_XGB_CALIBRATION=1`
- ✅ All outputs (`p_cal`, `entropy`, `uncertainty_score`) available and journaled
- ✅ Calibration metadata includes all required fields

**Fail-Fast:**
- Hard fail in replay if any GO/NO-GO criterion fails
- Warning in live mode (degraded operation)

---

## Summary

**Inputs:** `p_raw`, `proba_vector`, `margin_raw`, `logit_raw`, regime keys  
**Outputs:** `p_cal`, `margin`, `p_hat`, `entropy`, `uncertainty_score`  
**Fail-Fast:** No silent fallbacks, hard fail in replay when required  
**Artifacts:** Per-session + per-regime calibrators with metadata  
**Contract:** Explicit dimension tracking, no implicit changes

---

**End of Document**
