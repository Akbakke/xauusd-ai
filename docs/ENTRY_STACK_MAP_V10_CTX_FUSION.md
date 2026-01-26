# ENTRY Stack Map: V10_CTX Fusion Architecture

**Date:** 2026-01-07  
**Purpose:** Complete SSoT mapping of ENTRY stack before implementing "Gated Fusion + calibrated XGB + uncertainty signals"  
**Status:** Phase 0 - Mapping Only (No Logic Changes)

---

## Table of Contents

1. [Runtime Call Graph](#runtime-call-graph)
2. [Data Contracts](#data-contracts)
3. [Decision Contract](#decision-contract)
4. [Feature Truth Sources](#feature-truth-sources)
5. [Dead Code / Deprecated Paths](#dead-code--deprecated-paths)
6. [Calibration & Uncertainty](#calibration--uncertainty)
7. [Implementation Plan (Phase 1-3)](#implementation-plan-phase-1-3)

---

## Runtime Call Graph

### Entry Point

**File:** `gx1/execution/oanda_demo_runner.py`  
**Method:** `_run_replay_impl()` (line ~7652)  
**Function:** Main replay loop that iterates through M5 bars chronologically

**Flow:**
```
_run_replay_impl()
  └─> For each bar i in [first_valid_eval_idx, len(df)):
        ├─> candles_history = df.iloc[start_idx_for_history : i + 1]
        ├─> evaluate_entry(candles_history)  [line 6970]
        │     └─> EntryManager.evaluate_entry()  [gx1/execution/entry_manager.py:751]
        └─> evaluate_and_close_trades(candles_history)
```

### Entry Evaluation Flow

**File:** `gx1/execution/entry_manager.py`  
**Method:** `evaluate_entry(candles: pd.DataFrame)` (line 751)

**Detailed Flow:**

```
EntryManager.evaluate_entry(candles)
  │
  ├─> [1] Hard Eligibility Check (BEFORE feature build)
  │     └─> _check_hard_eligibility(candles, policy_state)  [line 246]
  │           ├─> Session check (ASIA blocked for SNIPER)
  │           ├─> Warmup check (min_bars_for_features)
  │           ├─> Spread cap check
  │           └─> Kill-switch check
  │           Returns: (eligible: bool, reason: str)
  │           If not eligible: return None (STOP - no feature build)
  │
  ├─> [2] Soft Eligibility Check (AFTER hard, BEFORE feature build)
  │     └─> _check_soft_eligibility(candles, policy_state)  [line 453]
  │           ├─> Vol regime check (EXTREME blocked)
  │           └─> ATR proxy computation (for context features)
  │           Returns: (eligible: bool, reason: str)
  │           If not eligible: return None (STOP - no feature build)
  │
  ├─> [3] Context Features Build (AFTER soft eligibility)
  │     └─> build_entry_context_features()  [gx1/execution/entry_context_features.py]
  │           ├─> Input: candles, policy_state, atr_proxy, spread_bps
  │           ├─> Output: EntryContextFeatures object
  │           │     - ctx_cat: [5] int64 (session_id, trend_regime_id, vol_regime_id, atr_bucket, spread_bucket)
  │           │     - ctx_cont: [2] float32 (atr_bps, spread_bps)
  │           └─> Only if ENTRY_CONTEXT_FEATURES_ENABLED=true
  │
  ├─> [4] Feature Building (AFTER eligibility checks)
  │     └─> build_live_entry_features(candles)  [gx1/execution/live_features.py:build_live_entry_features]
  │           ├─> build_basic_v1(candles)  [gx1/features/basic_v1.py:387]
  │           │     └─> NumPy-only hot-path (pandas forbidden)
  │           │     └─> Returns: DataFrame with ~200+ features
  │           ├─> build_v9_runtime_features(df_raw, feature_meta_path, ...)  [gx1/features/runtime_v9.py]
  │           │     ├─> _load_feature_meta(meta_path)  [line 66]
  │           │     │     └─> Returns: (seq_features: List[str], snap_features: List[str])
  │           │     ├─> build_v9_live_base_features(df_raw)  [line 211]
  │           │     │     └─> Ensures required columns exist (atr, mid, range, ret_*, rvol_*, etc.)
  │           │     ├─> _apply_scalers(df_features, seq_features, snap_features, ...)  [line 156]
  │           │     │     └─> Applies RobustScaler to seq and snap features
  │           │     └─> Returns: (df_v9_feats: DataFrame, seq_feat_names: List[str], snap_feat_names: List[str])
  │           └─> Returns: EntryFeatureBundle
  │                 - features: DataFrame (V9 features)
  │                 - atr_bps: float
  │                 - spread_bps: float
  │
  ├─> [5] Big Brain V1 / FARM Regime Inference (OBSERVE-ONLY)
  │     └─> Big Brain V1 inference (if enabled)  [line 924]
  │           └─> Adds to policy_state: brain_trend_regime, brain_vol_regime, brain_risk_score
  │     OR
  │     └─> FARM regime inference (if FARM_V2B mode)  [line 1033]
  │           └─> infer_farm_regime(session, atr_regime_id)  [gx1/regime/farm_regime.py]
  │
  ├─> [6] Stage-0 Opportunitetsfilter (BEFORE model inference)
  │     └─> should_consider_entry(policy_state, ...)  [gx1/policy/entry_v9_policy_sniper.py]
  │           ├─> Session check (EU/US/OVERLAP allowed for SNIPER)
  │           ├─> Trend/vol regime check (blocks certain combinations)
  │           └─> Returns: (should_consider: bool, reason: str)
  │           If not should_consider: return None (STOP - no model inference)
  │
  ├─> [7] Model Inference (V10 Hybrid)
  │     └─> _predict_entry_v10_hybrid(entry_bundle, candles, policy_state, entry_context_features)  [oanda_demo_runner.py:6145]
  │           │
  │           ├─> [7a] Build V9 Features (if not already built)
  │           │     └─> build_v9_runtime_features(df_raw, feature_meta_path, seq_scaler_path, snap_scaler_path)
  │           │           └─> Returns: (df_v9_feats, seq_feat_names, snap_feat_names)
  │           │
  │           ├─> [7b] XGB Inference (per session)
  │           │     ├─> Get XGB model: xgb_model = entry_v10_bundle.xgb_models_by_session[current_session]
  │           │     ├─> Prepare XGB features: xgb_features = current_row[snap_feat_names].values  [line 6364]
  │           │     ├─> XGB prediction: xgb_proba = xgb_model.predict_proba(xgb_features)[0]  [line 6368]
  │           │     ├─> Extract probabilities:
  │           │     │     - p_long_xgb = float(xgb_proba[1])  [line 6373]
  │           │     │     - p_short_xgb = 1.0 - p_long_xgb  [line 6380]
  │           │     │     - margin_xgb = abs(p_long_xgb - p_short_xgb)  [line 6381]
  │           │     │     - p_hat_xgb = max(p_long_xgb, p_short_xgb)  [line 6382]
  │           │     └─> Compute p_long_xgb_ema_5 (placeholder - uses current p_long_xgb)  [line 6389]
  │           │
  │           ├─> [7c] Build Sequence Tensor [1, seq_len, 16]
  │           │     ├─> seq_data = np.zeros((seq_len, 16), dtype=np.float32)  [line 6395]
  │           │     ├─> Fill seq features (13 from seq_feat_names): seq_data[:, 0:13]  [line 6404]
  │           │     └─> Fill XGB channels (3): seq_data[:, 13:16] = [p_long_xgb, margin_xgb, p_long_xgb_ema_5]  [line 6422-6424]
  │           │
  │           ├─> [7d] Build Snapshot Tensor [1, 88]
  │           │     ├─> snap_data = np.zeros(88, dtype=np.float32)  [line 6428]
  │           │     ├─> Fill snap features (85 from snap_feat_names): snap_data[0:85]  [line 6431]
  │           │     └─> Fill XGB channels (3): snap_data[85:88] = [p_long_xgb, margin_xgb, p_hat_xgb]  [line 6436-6438]
  │           │
  │           ├─> [7e] Contract Check (hard fail on mismatch)
  │           │     ├─> Verify seq_data.shape[-1] == bundle_metadata.seq_input_dim  [line 6461]
  │           │     └─> Verify snap_data.shape[-1] == bundle_metadata.snap_input_dim  [line 6475]
  │           │
  │           ├─> [7f] Context Features Preparation
  │           │     ├─> If ctx enabled: ctx_cat_t = [1, 5], ctx_cont_t = [1, 2]  [line 6582-6583]
  │           │     └─> Else: session_id_t, vol_regime_id_t, trend_regime_id_t (legacy)  [line 6597-6599]
  │           │
  │           ├─> [7g] Transformer Inference
  │           │     └─> transformer_model(seq_x, snap_x, ctx_cat, ctx_cont, ...)  [line 6624]
  │           │           ├─> EntryV10CtxHybridTransformer.forward()  [gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:437]
  │           │           │     ├─> seq_encoder(seq_x)  → seq_emb [B, d_model]
  │           │           │     ├─> snap_encoder(snap_x)  → snap_emb [B, d_model]
  │           │           │     ├─> context_encoder(ctx_cat, ctx_cont)  → ctx_emb [B, 42]
  │           │           │     └─> fusion_layer(seq_emb, snap_emb, ctx_emb)  → outputs
  │           │           └─> Returns: {"direction_logit": ..., "early_move_logit": ..., "quality_logit": ...}
  │           │
  │           └─> Returns: EntryPrediction
  │                 - prob_long: float (from sigmoid(direction_logit))
  │                 - prob_short: float (1.0 - prob_long)
  │                 - margin: float
  │                 - p_hat: float
  │
  ├─> [8] Decision Gates (POST-model inference)
  │     └─> should_enter_trade(prediction, entry_params, entry_gating, ...)  [gx1/execution/oanda_demo_runner.py:764]
  │           ├─> p_side_min check (asymmetric: LONG vs SHORT thresholds)
  │           ├─> margin_min check (asymmetric)
  │           ├─> side_ratio_min check (p_side / p_other >= ratio)
  │           └─> sticky-side logic (don't flip side if ratio < threshold)
  │           Returns: 'long', 'short', or None
  │
  ├─> [9] Trade Creation (if all gates pass)
  │     └─> LiveTrade(...)  [gx1/execution/live_trade.py]
  │           ├─> trade_id, timestamp, side, entry_price, units
  │           ├─> prediction, policy_state, entry_context_features
  │           └─> exit_profile (from exit_config)
  │
  └─> [10] Journal / Telemetry
        ├─> _record_entry_diag(trade, ...)  [oanda_demo_runner.py]
        └─> trade_journal.log_entry_snapshot(...)  [gx1/execution/trade_journal.py]
```

### Key Functions Summary

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `evaluate_entry()` | `entry_manager.py` | 751 | Main entry evaluation orchestrator |
| `_check_hard_eligibility()` | `entry_manager.py` | 246 | Hard gates (session, warmup, spread, kill-switch) |
| `_check_soft_eligibility()` | `entry_manager.py` | 453 | Soft gates (vol regime) |
| `build_entry_context_features()` | `entry_context_features.py` | - | Build ctx_cat [5] and ctx_cont [2] |
| `build_live_entry_features()` | `live_features.py` | - | Build EntryFeatureBundle |
| `build_basic_v1()` | `basic_v1.py` | 387 | Build base features (NumPy-only) |
| `build_v9_runtime_features()` | `runtime_v9.py` | - | Build V9 seq/snap features + apply scalers |
| `_predict_entry_v10_hybrid()` | `oanda_demo_runner.py` | 6145 | V10 hybrid prediction (XGB + Transformer) |
| `should_enter_trade()` | `oanda_demo_runner.py` | 764 | Post-model decision gates |
| `should_consider_entry()` | `entry_v9_policy_sniper.py` | - | Stage-0 opportunitetsfilter |

---

## Data Contracts

### Input Tensors

| Tensor | Shape | Source | Feature Names Source | Used by Model |
|--------|-------|--------|---------------------|---------------|
| `seq_x` | `[1, 30, 16]` | Runtime | `feature_meta.json` (13 base) + XGB channels (3) | ✅ Yes |
| `snap_x` | `[1, 88]` | Runtime | `feature_meta.json` (85 base) + XGB channels (3) | ✅ Yes |
| `ctx_cat` | `[1, 5]` | Runtime | `build_entry_context_features()` | ✅ Yes (if ctx enabled) |
| `ctx_cont` | `[1, 2]` | Runtime | `build_entry_context_features()` | ✅ Yes (if ctx enabled) |

### Sequence Features (16 total)

**Base Features (13):**
1. `atr50` - ATR over 50 bars
2. `atr_regime_id` - ATR regime bucket (0-3)
3. `atr_z` - ATR z-score
4. `body_pct` - Body percentage of range
5. `ema100_slope` - EMA100 slope
6. `ema20_slope` - EMA20 slope
7. `pos_vs_ema200` - Position vs EMA200
8. `roc100` - Rate of change (100 bars)
9. `roc20` - Rate of change (20 bars)
10. `session_id` - Session identifier
11. `std50` - Standard deviation (50 bars)
12. `trend_regime_tf24h` - Trend regime (24h timeframe)
13. `wick_asym` - Wick asymmetry

**XGB Channels (3, indices 13-15):**
14. `p_long_xgb` (index 13) - XGB probability for LONG
15. `margin_xgb` (index 14) - XGB margin (|p_long - p_short|)
16. `p_long_xgb_ema_5` (index 15) - XGB EMA-5 (currently placeholder)

**Source:** `gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json` (seq_features)

### Snapshot Features (88 total)

**Base Features (85):**
See `gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json` (snap_features) for complete list.

**XGB Channels (3, indices 85-87):**
86. `p_long_xgb` (index 85) - XGB probability for LONG
87. `margin_xgb` (index 86) - XGB margin
88. `p_hat_xgb` (index 87) - XGB max probability (max(p_long, p_short))

**Source:** `gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json` (snap_features)

### Context Features

**Categorical (ctx_cat, [5] int64):**
1. `session_id` (0=ASIA, 1=EU, 2=US, 3=OVERLAP)
2. `trend_regime_id` (0=UP, 1=DOWN, 2=NEUTRAL)
3. `vol_regime_id` (0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME)
4. `atr_bucket` (0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME)
5. `spread_bucket` (0=LOW, 1=MEDIUM, 2=HIGH)

**Continuous (ctx_cont, [2] float32):**
1. `atr_bps` - ATR in basis points
2. `spread_bps` - Spread in basis points

**Source:** `gx1/execution/entry_context_features.py` → `build_entry_context_features()`

### Bundle Metadata Contract

**File:** `models/entry_v10_ctx/SMOKE_20260106_ctxfusion/bundle_metadata.json`

```json
{
  "seq_input_dim": 16,        // Must match runtime seq_x.shape[-1]
  "snap_input_dim": 88,       // Must match runtime snap_x.shape[-1]
  "expected_ctx_cat_dim": 5,   // Must match runtime ctx_cat.shape[-1]
  "expected_ctx_cont_dim": 2, // Must match runtime ctx_cont.shape[-1]
  "max_seq_len": 30,          // Sequence length
  "supports_context_features": true
}
```

**Contract Check:** Implemented in `_predict_entry_v10_hybrid()` (line 6440-6487)
- Hard fail in replay if mismatch
- Warning in live mode

---

## Decision Contract

### Gate Locations

| Gate | Location | Type | File | Function |
|------|----------|------|------|----------|
| **Hard Eligibility** | Pre-feature-build | Pre-model | `entry_manager.py` | `_check_hard_eligibility()` |
| **Soft Eligibility** | Pre-feature-build | Pre-model | `entry_manager.py` | `_check_soft_eligibility()` |
| **Stage-0** | Pre-model inference | Pre-model | `entry_v9_policy_sniper.py` | `should_consider_entry()` |
| **Decision Gates** | Post-model inference | Post-model | `oanda_demo_runner.py` | `should_enter_trade()` |
| **Safety Kill-Switch** | Any time | Safety | `entry_manager.py` | Kill-switch check in hard eligibility |

### Hard Eligibility Gates

**File:** `gx1/execution/entry_manager.py:246`

1. **Session Gate:**
   - SNIPER: Blocks ASIA session
   - Allowed: EU, US, OVERLAP
   - Reason: `HARD_ELIGIBILITY_SESSION_BLOCK`

2. **Warmup Gate:**
   - Requires: `min_bars_for_features` (typically 288 bars)
   - Reason: `HARD_ELIGIBILITY_WARMUP`

3. **Spread Cap:**
   - Blocks if spread > threshold
   - Reason: `HARD_ELIGIBILITY_SPREAD_CAP`

4. **Kill-Switch:**
   - Global kill-switch check
   - Reason: `HARD_ELIGIBILITY_KILLSWITCH`

**Action:** If any gate fails → return None (STOP - no feature build, no model call)

### Soft Eligibility Gates

**File:** `gx1/execution/entry_manager.py:453`

1. **Vol Regime Gate:**
   - Blocks EXTREME vol regime
   - Reason: `SOFT_ELIGIBILITY_VOL_REGIME_EXTREME`

**Action:** If gate fails → return None (STOP - no feature build, no model call)

### Stage-0 Opportunitetsfilter

**File:** `gx1/policy/entry_v9_policy_sniper.py`

**Function:** `should_consider_entry(policy_state, ...)`

**Gates:**
1. Session check (EU/US/OVERLAP allowed)
2. Trend/vol regime combination check
   - Blocks: TREND_UP + HIGH vol in EU
   - Allows: MR + LOW/MID vol, TREND_DOWN + MID/HIGH vol

**Action:** If gate fails → return None (STOP - no model inference)

### Decision Gates (Post-Model)

**File:** `gx1/execution/oanda_demo_runner.py:764`

**Function:** `should_enter_trade(prediction, entry_params, entry_gating, ...)`

**Gates:**
1. **p_side_min** (asymmetric):
   - LONG: default 0.55
   - SHORT: default 0.55 (can be higher)
   - Action: If `p_side < p_side_min` → return None

2. **margin_min** (asymmetric):
   - LONG: default 0.08
   - SHORT: default 0.08 (can be higher)
   - Action: If `margin < margin_min` → return None

3. **side_ratio_min**:
   - Default: 1.25
   - Action: If `p_side / p_other < side_ratio_min` → return None

4. **Sticky-Side Logic:**
   - If `last_side != current_side` and `bars_since_last_side <= sticky_bars`:
     - If `ratio < side_ratio_min` → block side flip

**Action:** If any gate fails → return None (no trade)

### Safety Kill-Switch

**Location:** Hard eligibility check (line 246)

**Purpose:** Global emergency stop (allows manual intervention)

**Action:** If active → return None (STOP - no feature build, no model call)

### Current Gate Summary

| Gate | Input to Model | Post-Model Veto | Safety Kill-Switch |
|------|---------------|-----------------|-------------------|
| Hard Eligibility | ❌ No (blocks before) | N/A | ✅ Yes (kill-switch) |
| Soft Eligibility | ❌ No (blocks before) | N/A | ❌ No |
| Stage-0 | ❌ No (blocks before) | N/A | ❌ No |
| Decision Gates | N/A | ✅ Yes | ❌ No |

**Target State (after Phase 1-3):**
- Only safety kill-switch as post-veto
- All other gates → input to model (gated fusion)

---

## Feature Truth Sources

### Feature Metadata

**File:** `gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json`

**Structure:**
```json
{
  "seq_features": [...],      // 13 base features
  "snap_features": [...],     // 85 base features
  "seq_input_dim": 13,        // Base only (runtime adds 3 XGB channels)
  "snap_input_dim": 85        // Base only (runtime adds 3 XGB channels)
}
```

**Usage:**
- Loaded by `_load_feature_meta()` in `runtime_v9.py:66`
- Used to extract feature names for V9 feature building
- **Note:** Runtime adds 3 XGB channels to both seq (→16) and snap (→88)

### Scalers

**Seq Scaler:**
- **Path:** `entry_v10_cfg.get("seq_scaler_path")`
- **Type:** `RobustScaler` (sklearn)
- **Usage:** Applied in `_apply_scalers()` in `runtime_v9.py:156`
- **Applied to:** seq_features (13 base features)

**Snap Scaler:**
- **Path:** `entry_v10_cfg.get("snap_scaler_path")`
- **Type:** `RobustScaler` (sklearn)
- **Usage:** Applied in `_apply_scalers()` in `runtime_v9.py:156`
- **Applied to:** snap_features (85 base features)

**Status:** ✅ Used (applied before XGB inference)

### XGB Models

**Path:** `entry_v10_bundle.xgb_models_by_session`

**Structure:**
```python
{
  "EU": XGBClassifier,
  "US": XGBClassifier,
  "OVERLAP": XGBClassifier
}
```

**Session Routing:**
- Determined by `policy_state.get("session")` (EU/US/OVERLAP)
- ASIA has no XGB model (expected, not an error)

**Usage:**
- Called in `_predict_entry_v10_hybrid()` (line 6353)
- Input: `snap_features` (85 base features, already scaled)
- Output: `xgb_proba` → `p_long_xgb`, `margin_xgb`, `p_hat_xgb`

**Status:** ✅ Used (per-session routing)

### Transformer Bundle

**Path:** `entry_v10_bundle.transformer_model`

**Type:** `EntryV10CtxHybridTransformer` (if ctx enabled) or `EntryV10HybridTransformer` (legacy)

**Expected Dimensions:**
- `seq_input_dim`: 16 (from bundle_metadata.json)
- `snap_input_dim`: 88 (from bundle_metadata.json)
- `expected_ctx_cat_dim`: 5 (if ctx enabled)
- `expected_ctx_cont_dim`: 2 (if ctx enabled)

**Status:** ✅ Used (with contract check)

### Mismatch Status

| Component | Metadata | Runtime Output | Model Weights | Status |
|-----------|----------|----------------|---------------|--------|
| seq_input_dim | 13 (base) | 16 (base + 3 XGB) | 13 (needs re-training) | ⚠️ Model mismatch |
| snap_input_dim | 85 (base) | 88 (base + 3 XGB) | 85 (needs re-training) | ⚠️ Model mismatch |
| ctx_cat_dim | 5 | 5 | 5 | ✅ Match |
| ctx_cont_dim | 2 | 2 | 2 | ✅ Match |

**Action Required:** Re-train V10_CTX bundle with `seq_input_dim=16, snap_input_dim=88`

---

## Dead Code / Deprecated Paths

### Candidates for Deletion

#### 1. Legacy V9 Entry Path

**File:** `gx1/execution/oanda_demo_runner.py:6051`

**Function:** `_predict_entry_v9(...)`

**Status:** ⚠️ Potentially unused (V10 is primary path)

**Evidence:**
- V9 path only called if `entry_v9_enabled=True` and `entry_v10_enabled=False`
- SNIPER uses V10, not V9

**Risk:** Low (guarded by config)

**Recommendation:** Keep for backward compatibility, but mark as deprecated

#### 2. FARM_V2B Entry Path

**File:** `gx1/execution/entry_manager.py:932`

**Function:** FARM_V2B regime inference

**Status:** ⚠️ Unused for SNIPER (SNIPER uses Big Brain V1 or replay tags)

**Evidence:**
- `is_farm_v2b = False` for SNIPER policies
- Code path only active if `entry_v9_policy_farm_v2b.enabled=True`

**Risk:** Low (guarded by config)

**Recommendation:** Keep for FARM policies, but document as SNIPER-unused

#### 3. Temperature Scaling (Legacy)

**File:** `gx1/execution/oanda_demo_runner.py:5762`

**Function:** `_get_temperature_map()`, `_apply_temperature()`

**Status:** ⚠️ Present but not used in V10 path

**Evidence:**
- Temperature scaling applied in `_predict_entry_v9()` (line 5937)
- V10 path does not apply temperature scaling
- Comment: "TEMPORARY TEST: Force T=1.0 for all sessions (disable temperature scaling)"

**Risk:** Medium (dead code in V10 path)

**Recommendation:** Remove from V10 path, or implement calibration-based temperature

#### 4. Legacy Regime Embeddings

**File:** `gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py:332`

**Component:** `RegimeEmbeddings` (in EntryV10CtxHybridTransformer)

**Status:** ⚠️ Present but not used when ctx enabled

**Evidence:**
- `RegimeEmbeddings` initialized but not used in forward() when ctx_cat/ctx_cont provided
- Only used for backward compatibility with legacy V10 bundles

**Risk:** Low (backward compatibility)

**Recommendation:** Keep for backward compatibility, document as legacy-only

#### 5. Duplicate Feature Building

**Location:** `_predict_entry_v10_hybrid()` calls `build_v9_runtime_features()` even though `build_live_entry_features()` already built features

**Status:** ⚠️ Redundant call

**Evidence:**
- `build_live_entry_features()` already calls `build_v9_runtime_features()` internally
- `_predict_entry_v10_hybrid()` calls it again (line 6306)

**Risk:** Medium (performance overhead)

**Recommendation:** Reuse features from `entry_bundle` instead of rebuilding

#### 6. p_long_xgb_ema_5 Placeholder

**File:** `gx1/execution/oanda_demo_runner.py:6389`

**Status:** ⚠️ Placeholder (not implemented)

**Evidence:**
```python
p_long_xgb_ema_5 = p_long_xgb  # Placeholder - would need to track XGB history
```

**Risk:** Low (uses current value, not historical EMA)

**Recommendation:** Implement proper EMA-5 tracking or remove if not needed

### Toggles / Flags

| Flag | Default | Used? | Purpose |
|------|---------|-------|---------|
| `ENTRY_CONTEXT_FEATURES_ENABLED` | `false` | ✅ Yes | Enable ctx features |
| `GX1_ASSERT_NO_PANDAS` | `0` | ✅ Yes | Runtime pandas guard |
| `GX1_REPLAY_EXPECT_V10_CTX` | `0` | ✅ Yes | Replay safety assert |
| `GX1_CTX_NULL_BASELINE` | `0` | ⚠️ Test only | Force null ctx for A/B |
| `GX1_CTX_CONSUMPTION_PROOF` | `0` | ⚠️ Test only | Verify ctx is consumed |

### Scripts

| Script | Status | Recommendation |
|--------|--------|---------------|
| `scripts/run_sniper_1w_perf_guards.sh` | ✅ Active | Keep |
| `scripts/verify_entry_v10_ctx_bundle.py` | ✅ Active | Keep |
| `scripts/run_mini_replay_perf.py` | ✅ Active | Keep |

---

## Calibration & Uncertainty

### Current State

#### ✅ Temperature Scaling (Legacy)

**File:** `gx1/execution/oanda_demo_runner.py:5762`

**Method:** `_apply_temperature(p: float, T: float) -> float`

**Formula:**
```python
logit = np.log(p / (1.0 - p))
scaled_logit = logit / T
scaled_p = 1.0 / (1.0 + np.exp(-scaled_logit))
```

**Status:** ⚠️ Present but disabled in V10 path (T=1.0 forced)

**Usage:** Only in V9 path (not V10)

#### ✅ ECE (Expected Calibration Error)

**File:** `gx1/execution/telemetry.py:42`

**Function:** `ece_bin(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float`

**Status:** ✅ Implemented (for evaluation, not runtime)

**Usage:** Telemetry/metrics only

#### ✅ Entropy

**File:** `gx1/execution/telemetry.py:24`

**Function:** `prob_entropy(p: float) -> float`

**Status:** ✅ Implemented (for telemetry, not runtime)

**Usage:** Telemetry/metrics only

### Missing Components

#### ❌ Platt Scaling

**Status:** Not implemented

**Location:** Should be in `gx1/models/entry_v10/` or `gx1/execution/`

**Purpose:** Calibrate XGB probabilities to match empirical frequencies

**Required:**
- Training: Fit Platt scaler on validation set (per session/regime)
- Runtime: Apply `platt_scaler.transform(p_xgb)` → `p_calibrated`

#### ❌ Isotonic Regression

**Status:** Not implemented

**Location:** Should be in `gx1/models/entry_v10/` or `gx1/execution/`

**Purpose:** Non-parametric calibration (more flexible than Platt)

**Required:**
- Training: Fit isotonic regressor on validation set (per session/regime)
- Runtime: Apply `isotonic_regressor.transform(p_xgb)` → `p_calibrated`

#### ❌ Per-Regime Calibration

**Status:** Not implemented

**Purpose:** Different calibration curves for different regimes (LOW/MEDIUM/HIGH vol, TREND_UP/DOWN/NEUTRAL)

**Required:**
- Training: Fit separate calibrators per regime combination
- Runtime: Route to appropriate calibrator based on `ctx_cat`

#### ❌ Uncertainty Metrics (Runtime)

**Status:** Partially implemented (entropy in telemetry, not in runtime)

**Missing:**
- **Entropy:** `-p*log(p) - (1-p)*log(1-p)` (for XGB predictions)
- **Margin Distribution:** Track `margin_xgb` distribution for OOD detection
- **Leaf Embedding:** XGB leaf indices as uncertainty proxy (if available)

**Required:**
- Runtime: Compute entropy, margin, leaf indices for each XGB prediction
- Pass to Transformer as additional signals (gated fusion)

#### ❌ Reliability Curves

**Status:** Not implemented

**Purpose:** Visualize calibration quality (predicted vs actual frequency)

**Location:** Should be in `gx1/analysis/` or evaluation scripts

**Required:**
- Evaluation: Plot reliability curves per session/regime
- Report ECE, Brier score, calibration slope

### Summary

| Component | Status | Location | Purpose |
|-----------|--------|----------|---------|
| Temperature Scaling | ⚠️ Legacy (V9 only) | `oanda_demo_runner.py` | Calibration (disabled in V10) |
| ECE | ✅ Eval only | `telemetry.py` | Calibration metrics |
| Entropy | ✅ Eval only | `telemetry.py` | Uncertainty metrics |
| Platt Scaling | ❌ Missing | - | Calibration |
| Isotonic Regression | ❌ Missing | - | Calibration |
| Per-Regime Calibration | ❌ Missing | - | Calibration |
| Uncertainty (Runtime) | ❌ Missing | - | Gated fusion inputs |
| Reliability Curves | ❌ Missing | - | Evaluation |

---

## Implementation Plan (Phase 1-3)

### Phase 1: XGB Calibration + Uncertainty Outputs

#### 1.1 Training: Calibration Model

**Location:** `gx1/models/entry_v10/calibration/`

**Components:**
- `train_xgb_calibration.py`: Fit Platt/Isotonic calibrators
- `calibration_bundle.py`: Save/load calibration models
- `calibration_artifacts/`: Per-session/regime calibrators

**Artifacts:**
- `platt_scaler_EU.joblib`, `platt_scaler_US.joblib`, `platt_scaler_OVERLAP.joblib`
- `platt_scaler_EU_LOW.joblib`, `platt_scaler_EU_MEDIUM.joblib`, ... (per-regime)
- `calibration_metadata.json`: Maps session/regime → calibrator path

**Training Data:**
- Validation set from V10 training
- Per session: `(p_xgb, y_true)` pairs
- Per regime: `(p_xgb, y_true)` pairs filtered by `ctx_cat`

**Output:**
- Calibrated probabilities: `p_calibrated = platt_scaler.transform(p_xgb)`
- Calibration metrics: ECE, Brier score, reliability curves

#### 1.2 Runtime: Calibration Application

**Location:** `gx1/execution/oanda_demo_runner.py:_predict_entry_v10_hybrid()`

**Changes:**
```python
# After XGB prediction (line 6368)
p_long_xgb_raw = float(xgb_proba[1])

# Load calibrator (per session/regime)
calibrator = load_calibrator(session=current_session, regime=vol_regime_id)
p_long_xgb_calibrated = calibrator.transform([p_long_xgb_raw])[0]

# Compute uncertainty metrics
entropy_xgb = prob_entropy(p_long_xgb_calibrated)
margin_xgb = abs(p_long_xgb_calibrated - (1.0 - p_long_xgb_calibrated))
```

**Outputs:**
- `p_long_xgb_calibrated`: Calibrated probability
- `entropy_xgb`: Uncertainty metric
- `margin_xgb`: Margin (updated with calibrated prob)
- `p_hat_xgb`: Max probability (updated)

#### 1.3 Uncertainty Signals

**Location:** `gx1/execution/oanda_demo_runner.py:_predict_entry_v10_hybrid()`

**Compute:**
- **Entropy:** `-p*log(p) - (1-p)*log(1-p)` (for `p_long_xgb_calibrated`)
- **Margin Distribution:** Track `margin_xgb` over time (for OOD detection)
- **Leaf Embedding:** Extract XGB leaf indices (if available) as uncertainty proxy

**Pass to Transformer:**
- Add to `seq_data` or `snap_data` as additional channels
- Or: Pass as separate `uncertainty` tensor to gated fusion

### Phase 2: Gated Fusion Contract

#### 2.1 XGB Signals to Transformer

**Current:** XGB channels in seq/snap tensors (indices 13-15, 85-87)

**Enhanced Signals:**
- `p_long_xgb_calibrated` (replaces `p_long_xgb`)
- `margin_xgb` (updated with calibrated prob)
- `entropy_xgb` (NEW)
- `p_hat_xgb` (updated)
- `leaf_embedding` (NEW, if available)

**Location:** `gx1/execution/oanda_demo_runner.py:_predict_entry_v10_hybrid()`

**Changes:**
- Update `seq_data[:, 13] = p_long_xgb_calibrated`
- Update `seq_data[:, 14] = margin_xgb` (calibrated)
- Add `seq_data[:, 15] = entropy_xgb` (or keep `p_long_xgb_ema_5` if needed)
- Update `snap_data[85:88] = [p_long_xgb_calibrated, margin_xgb, p_hat_xgb]`

#### 2.2 Raw Snapshot Backup

**Purpose:** Provide "raw" snapshot features (without XGB signals) as backup for gated fusion

**Options:**
- **Option A:** Separate `snap_raw` tensor (85 features, no XGB channels)
- **Option B:** Embedding of raw snapshot (via `SnapshotEncoder` without XGB channels)
- **Option C:** Use existing `snap_x` but gate XGB channels separately

**Recommendation:** Option B (embedding of raw snapshot)

**Location:** `gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py`

**Changes:**
- Add `snap_raw_encoder` (separate encoder for 85 base features)
- Or: Add `gate_xgb_channels` flag to `SnapshotEncoder`

#### 2.3 Gate Representation

**Options:**
- **Option A:** Scalar gate per bar (single gate value for all XGB channels)
- **Option B:** Per-head gate (different gate per Transformer head)
- **Option C:** Per-token gate (different gate per sequence position)

**Recommendation:** Option A (scalar gate per bar) for simplicity

**Gate Computation:**
```python
gate = sigmoid(uncertainty_mlp(entropy_xgb, margin_xgb, leaf_embedding))
```

**Location:** `gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py`

**Changes:**
- Add `GatedFusionLayer` that takes `snap_emb`, `xgb_signals`, `gate`
- Apply gate: `fused = gate * xgb_signals + (1 - gate) * snap_raw_emb`

### Phase 3: Training/Eval + GO/NO-GO Metrics

#### 3.1 Training Changes

**Location:** `gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py`

**Changes:**
- Add `GatedFusionLayer` to model architecture
- Train with gated fusion enabled
- Loss: Standard V10 loss (direction + early_move + quality)

**Evaluation:**
- Compare gated vs non-gated performance
- Measure calibration improvement (ECE, Brier score)
- Measure robustness (tail risk, drawdown, regime stability)

#### 3.2 Eval Metrics

**Location:** `gx1/analysis/` or evaluation scripts

**Metrics:**
- **Calibration Error:** ECE, Brier score, reliability curves (per session/regime)
- **Robustness:**
  - Tail risk: Max drawdown, VaR (95th percentile loss)
  - Regime stability: Performance across LOW/MEDIUM/HIGH vol regimes
  - OOD detection: Entropy distribution, margin distribution
- **Gating Behavior:**
  - Gate value distribution (when does gate activate?)
  - Gate vs performance correlation

#### 3.3 GO/NO-GO Criteria

**Required:**
- ✅ `n_ctx_model_calls == n_v10_calls` (ctx model called for all V10 calls)
- ✅ `ctx_proof_fail_count == 0` (ctx is consumed by model)
- ✅ `n_pandas_ops_detected == 0` (no pandas in hot-path)
- ✅ `timeout_count == 0` (no feature build timeouts)
- ✅ `fast_path_enabled == True` (fast path active)
- ✅ **NEW:** `calibration_ece < 0.05` (calibration error < 5%)
- ✅ **NEW:** `tail_risk_dd < -200 bps` (max drawdown < 200 bps)
- ✅ **NEW:** `regime_stability_score > 0.8` (performance stable across regimes)

**Fail-Fast:**
- Hard fail in replay if any GO/NO-GO criterion fails
- Warning in live mode (degraded operation)

---

## Summary

### Current State
- ✅ Runtime call graph mapped
- ✅ Data contracts documented
- ✅ Decision gates identified
- ✅ Feature truth sources mapped
- ⚠️ Calibration missing (only legacy temperature scaling)
- ⚠️ Uncertainty metrics missing (only in telemetry)
- ⚠️ Gated fusion not implemented

### Next Steps
1. **Phase 1:** Implement XGB calibration + uncertainty outputs
2. **Phase 2:** Implement gated fusion contract
3. **Phase 3:** Train/eval + GO/NO-GO metrics

### Key Findings
- Model needs re-training (13/85 → 16/88)
- Temperature scaling disabled in V10 path
- No calibration for XGB predictions
- No uncertainty signals passed to Transformer
- Decision gates are post-model (should be pre-model or gated fusion)

---

**End of Document**
