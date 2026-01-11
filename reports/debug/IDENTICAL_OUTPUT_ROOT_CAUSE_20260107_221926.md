# Identical Output Root Cause Investigation

**Date:** 2026-01-07 22:19:26

## Git Information

- **Commit:** `2a79bcdfee56cdc6c92586d4b069bb2b15fb758b`
- **Dirty State:** `dirty`

## Policy Configuration

- **Policy Path:** `/Users/andrekildalbakke/Desktop/GX1 XAUUSD/gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml`

## Environment Variables

| Variable | Value |
|----------|-------|
| `GX1_ALLOW_UNCALIBRATED_XGB` | `0` |
| `GX1_GATED_FUSION_ENABLED` | `1` |
| `GX1_REQUIRE_XGB_CALIBRATION` | `1` |

## Artifact Verification

| Artifact | Path | Hash (SHA256) | Size (bytes) |
|----------|------|---------------|--------------|
| `checkpoint_model` | `/Users/andrekildalbakke/Desktop/GX1 XAUUSD/models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION/model_state_dict.pt` | `97d90be83aa980cd...` | 3718036 |
| `v10_ctx_metadata` | `/Users/andrekildalbakke/Desktop/GX1 XAUUSD/models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION/bundle_metadata.json` | `ebaaa9a663e76cb0...` | 317 |
| `v10_ctx_model` | `/Users/andrekildalbakke/Desktop/GX1 XAUUSD/models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION/model_state_dict.pt` | `97d90be83aa980cd...` | 3718036 |

## Input Tensor Proof

### Shapes

- `seq_x`: [64, 30, 16]
- `snap_x`: [64, 88]
- `ctx_cat`: [64, 5]
- `ctx_cont`: [64, 2]

### XGB Channels (seq_x[:, :, 13:16])

- **p_cal**: min=0.0703, max=0.2179, mean=0.1383, std=0.0364, constant=False
- **margin**: min=0.5642, max=0.8594, mean=0.7234, std=0.0728, constant=False
- **uncertainty_score**: min=0.3670, max=0.7563, mean=0.5717, std=0.0961, constant=False

### XGB Channels (snap_x[:, 85:88])

- **p_cal**: min=0.0703, max=0.2179, mean=0.1383, std=0.0364, constant=False
- **margin**: min=0.5642, max=0.8594, mean=0.7234, std=0.0728, constant=False
- **p_hat**: min=0.7821, max=0.9297, mean=0.8617, std=0.0364, constant=False

### Context Features

- `ctx_cat`: unique_values=[0, 1]
- `ctx_cont`: min=1.1648, max=12.2292, constant=False

## Intervention Tests

| Test | Max Abs Delta | Mean Abs Delta | P95 Abs Delta | Decision Flips |
|------|---------------|----------------|---------------|----------------|
| zero_xgb | 1.435550 | 0.528655 | 1.116665 | 6 |
| randomize_xgb | 1.629989 | 0.426526 | 1.260217 | 1 |
| zero_ctx | 1.099918 | 0.280602 | 0.554042 | 0 |
| randomize_seq | 4.174471 | 0.926334 | 2.363570 | 5 |
| force_gate_0 | 1.565808 | 0.756277 | 1.404974 | 7 |
| force_gate_1 | 6.218034 | 2.032428 | 4.533087 | 31 |

## Verdict

**ROOT CAUSE: Threshold/post-gates dominating decisions.**

**Fix:** Compare raw logits distributions, not thresholded decisions. Consider adjusting evaluation methodology.

## Minimal Fix

(To be determined based on root cause analysis)

