# Depth Ladder Baseline: ENTRY_V10_CTX Architecture

**Status:** LOCKED  
**Date:** 2026-01-19  
**Baseline:** Trial160 (2020-2025, 29,710 trades)

---

## Baseline Architecture (LOCKED)

### Transformer Configuration

**Model Variant:** `v10_ctx` (ENTRY_V10_CTX)

**Sequence Encoder:**
- `num_layers`: **3** (baseline, LOCKED)
- `d_model`: 128
- `n_heads`: 4
- `dim_feedforward`: 512 (d_model * 4, auto-calculated)
- `dropout`: 0.05
- `max_seq_len`: 30
- `pooling`: "mean"
- `use_positional_encoding`: True
- `causal`: True

**Snapshot Encoder:**
- `hidden_dims`: (256, 128)
- `use_layernorm`: True
- `dropout`: 0.0

**Fusion Layer:**
- `fusion_hidden_dim`: 128
- `fusion_dropout`: 0.1

**Context Features:**
- `ctx_cat_dim`: 5
- `ctx_cont_dim`: 2
- `ctx_emb_dim`: 42
- `ctx_embedding_dim`: 8

**Input Dimensions:**
- `seq_input_dim`: 16 (13 base + 3 XGB channels)
- `snap_input_dim`: 88 (85 base + 3 XGB channels)

---

## Baseline Bundle

**Bundle Path:** `models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION/`

**Metadata:**
- `model_variant`: "v10_ctx"
- `supports_context_features`: true
- `feature_contract_hash`: "93ab1d31534f2dc7"
- `schema_version`: "v10_ctx"
- `seq_input_dim`: 13 (base, +3 XGB = 16 total)
- `snap_input_dim`: 85 (base, +3 XGB = 88 total)
- `max_seq_len`: 30

**Note:** Baseline bundle metadata does not explicitly store `transformer_layers`, but code defaults to `num_layers=3` for variant="v10_ctx".

---

## XGBoost Channels

**Session Models:**
- EU: `models/entry_v10/xgb_entry_EU_v10.joblib`
- US: `models/entry_v10/xgb_entry_US_v10.joblib`
- OVERLAP: `models/entry_v10/xgb_entry_OVERLAP_v10.joblib`
- ASIA: `models/entry_v10/xgb_entry_ASIA_v10.joblib`

**Channels (3 per input):**
- `p_cal`: Calibrated probability
- `margin`: Margin from calibrated
- `p_hat` / `uncertainty_score`: Uncertainty (for sequence)

---

## Training Configuration (Baseline)

**Dataset:**
- Source: FULLYEAR 2020-2025 canonical splits
- Train: 2020-2024 (or 2025-01-01 → 2025-09-30 for FULLYEAR_2025)
- Val: 2025-10-01 → 2025-11-30
- Test: 2025-12-01 → 2025-12-31

**Training Hyperparameters:**
- `epochs`: 10 (baseline)
- `batch_size`: 64
- `learning_rate`: 1e-4
- `seed`: 1337
- `seq_len`: 30
- `device`: auto (cuda/mps/cpu)

**Features:**
- `feature_meta_path`: `gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json`
- `seq_scaler_path`: `gx1/models/entry_v9/nextgen_2020_2025_clean/seq_scaler.joblib`
- `snap_scaler_path`: `gx1/models/entry_v9/nextgen_2020_2025_clean/snap_scaler.joblib`

**Policy:**
- `policy_config`: `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml`
- `GX1_GATED_FUSION_ENABLED`: 1
- `GX1_REQUIRE_XGB_CALIBRATION`: 1

---

## Depth Ladder Variant: LPLUS1

**Target:** `num_layers = baseline_layers + 1 = 3 + 1 = 4`

**All Other Parameters:** IDENTICAL to baseline

**Invariant Check:**
- If `depth_ladder_mode=1`, ALL hyperparameters must match baseline except `n_layers`
- FATAL if any other parameter differs

---

## Code References

**Model Definition:**
- `gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py` (line 285-296)

**Training Script:**
- `gx1/models/entry_v10/entry_v10_ctx_train.py`

**Bundle Loading:**
- `gx1/models/entry_v10/entry_v10_bundle.py` (`load_entry_v10_ctx_bundle`)

**Runtime Usage:**
- `gx1/execution/oanda_demo_runner.py` (line 2336-2400)

---

## ⚠️  DO NOT MODIFY BASELINE

This baseline is LOCKED for Depth Ladder experiments. Any changes to baseline architecture must be explicitly documented and approved.

See: `docs/TRUTH_BASELINE_LOCK.md`
