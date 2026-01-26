# V10_CTX Feature Contract Analysis

**Date:** 2026-01-07  
**Status:** CONTRACT MISMATCH IDENTIFIED - MODEL NEEDS RE-TRAINING

## Summary

Runtime produces **16 seq features** and **88 snap features**, but the loaded V10_CTX model (`SMOKE_20260106_ctxfusion`) was trained on **13 seq features** and **85 snap features**.

## Feature Dimensions

### Runtime (Actual)
- **seq_input_dim: 16**
  - 13 base seq features from metadata
  - 3 XGB channels: `p_long_xgb`, `margin_xgb`, `p_long_xgb_ema_5` (indices 13-15)
- **snap_input_dim: 88**
  - 85 base snap features from metadata
  - 3 XGB channels: `p_long_xgb`, `margin_xgb`, `p_hat_xgb` (indices 85-87)

### Bundle Metadata (Expected by Model)
- **seq_input_dim: 13** (original, now updated to 16)
- **snap_input_dim: 85** (original, now updated to 88)

## Feature Names

### Seq Features (13 base, from metadata)
1. `atr50`
2. `atr_regime_id`
3. `atr_z`
4. `body_pct`
5. `ema100_slope`
6. `ema20_slope`
7. `pos_vs_ema200`
8. `roc100`
9. `roc20`
10. `session_id`
11. `std50`
12. `trend_regime_tf24h`
13. `wick_asym`

**Extra 3 channels (indices 13-15):**
14. `p_long_xgb` (XGB prediction for current bar)
15. `margin_xgb` (XGB margin for current bar)
16. `p_long_xgb_ema_5` (XGB EMA-5, currently placeholder)

### Snap Features (85 base, from metadata)
See `gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json` for full list.

**Extra 3 channels (indices 85-87):**
86. `p_long_xgb` (XGB prediction for current bar)
87. `margin_xgb` (XGB margin for current bar)
88. `p_hat_xgb` (XGB max probability)

## Root Cause

The V10 hybrid design includes XGB channels in both seq and snap tensors. This is intentional and part of the hybrid architecture. However, the `SMOKE_20260106_ctxfusion` bundle was trained before this was standardized, so it expects 13/85 instead of 16/88.

## Single Source of Truth (SSoT)

**Chosen: A (Runtime = 16/88)**

**Rationale:**
- Runtime is the authoritative source for feature production
- XGB channels are part of the V10 hybrid design (not a bug)
- Model architecture already supports 16/88 (default parameters)
- Bundle metadata has been updated to 16/88

## Solution

### Option 1: Re-train Bundle (Recommended)
Re-train the `SMOKE_20260106_ctxfusion` bundle with:
- `seq_input_dim=16`
- `snap_input_dim=88`

This ensures the model weights match the runtime contract.

### Option 2: Use Different Bundle
If another bundle exists that was trained on 16/88, use that instead.

### Current Status
- ✅ Bundle metadata updated to 16/88
- ✅ Contract check implemented (hard fail on mismatch)
- ❌ Model weights still expect 13/85 (needs re-training)

## Contract Check Implementation

Added explicit contract check in `_predict_entry_v10_hybrid()`:
- Verifies `seq_data.shape[-1] == bundle_metadata.seq_input_dim`
- Verifies `snap_data.shape[-1] == bundle_metadata.snap_input_dim`
- Hard fails in replay mode with clear error message
- Logs feature names and dimensions for debugging

## Next Steps

1. Re-train `SMOKE_20260106_ctxfusion` bundle with 16/88 dimensions
2. Verify bundle loads and runs without shape errors
3. Re-run 1-week replay to confirm `n_ctx_model_calls == n_v10_calls`
