# XGB → Transformer Signal Contract (KANONISK)

> **Status:** FROZEN - Read-only truth documentation  
> **Last Updated:** 2026-01-30  
> **Purpose:** Authoritative documentation of XGBoost → Transformer signal injection contract

## Executive Summary

**Transformer mottar N=2 komprimerte beslutningsspor fra XGB per scope (SEQ/SNAP).**

Disse representerer XGB som **opinionated prior**, ikke rå features. XGB er en komprimert ekspert som leverer kalibrerte sannsynligheter og usikkerhetsscore til Transformer.

---

## 1. XGB Architecture (Truth)

### 1.1 Model Structure

**Type:** Universal Multihead Model (`xgb_universal_multihead_v2`)

- **Single artifact** containing multiple session heads
- **One head per session** (EU, OVERLAP, US)
- **Shared feature space** across all heads
- **Separate 3-class classifiers** per session head

**Model File:**
- `xgb_universal_multihead_v2.joblib` (single file with all heads)
- `xgb_universal_multihead_v2_meta.json` (metadata: feature_list, schema_hash, etc.)

**Implementation:**
- `gx1/xgb/multihead/xgb_multihead_model_v1.py::XGBMultiheadModel`

### 1.2 Session Heads

| Head Name | Session | Enabled | Mode | Rationale |
|-----------|---------|---------|------|-----------|
| `EU` | EU | ✅ `true` | `normal` | Active, produces calibrated probabilities |
| `OVERLAP` | OVERLAP | ✅ `true` | `normal` | Active, produces calibrated probabilities |
| `US` | US | ❌ `false` | `neutral_probs` | Disabled due to drift (KS=0.227619, PSI=0.277938) |

**Policy Configuration:**
- File: `gx1/configs/xgb_session_policy.json`
- Loaded at runtime via `load_session_policy()`

**Disabled Head Behavior:**
- Returns neutral probabilities: `(1/3, 1/3, 1/3)` for `(p_long, p_short, p_flat)`
- Returns maximum uncertainty: `uncertainty = 1.0`
- Signals Transformer to take full responsibility for US decisions

### 1.3 Head Outputs (Per Session)

Each head produces **4 outputs**:

1. **`p_long`**: Probability of LONG class (3-class classifier, class 0)
2. **`p_short`**: Probability of SHORT class (3-class classifier, class 1)
3. **`p_flat`**: Probability of FLAT class (3-class classifier, class 2)
4. **`uncertainty`**: Normalized entropy = `entropy / log(3)`

**Invariants:**
- `p_long + p_short + p_flat = 1.0` (within tolerance)
- `uncertainty ∈ [0.0, 1.0]` (0 = certain, 1 = maximum uncertainty)

---

## 2. XGB → Transformer Signal Contract

### 2.1 Channel Count

**Total XGB Channels:** `XGB_CHANNELS = 3` (for model compatibility)

**Active Channels:** `2` per scope (SEQ and SNAP use different channels)

**Note:** `margin_xgb` was **REMOVED** as of 2026-01-25 (ablation showed it's harmful). The model dimension `XGB_CHANNELS=3` is kept for compatibility, but only 2 channels are actively used.

### 2.2 Sequence (SEQ) Channels

**Scope:** `SEQ` (sequence input to Transformer)

**Channel Count:** `2` active channels

**Indices:** `[13, 14]` (BASE_SEQ_FEATURES=13, XGB_CHANNELS=3, but only 2 used)

| Index | Channel Name | Datatype | Semantics | Universal? |
|-------|--------------|----------|-----------|------------|
| 13 | `p_long_xgb` | `float` [0.0, 1.0] | XGB probability of LONG | ✅ Yes |
| 14 | `uncertainty_score` | `float` [0.0, 1.0] | Normalized entropy (uncertainty) | ✅ Yes |
| 15 | *(unused)* | - | Reserved for compatibility | - |

**Injection:**
- Injected at **all timesteps** in sequence (same value for all timesteps in window)
- Uses **current session's XGB head output** (EU/OVERLAP/US)
- For disabled heads (US): uses neutral constants

### 2.3 Snapshot (SNAP) Channels

**Scope:** `SNAP` (snapshot input to Transformer)

**Channel Count:** `2` active channels

**Indices:** `[85, 86]` (BASE_SNAP_FEATURES=85, XGB_CHANNELS=3, but only 2 used)

| Index | Channel Name | Datatype | Semantics | Universal? |
|-------|--------------|----------|-----------|------------|
| 85 | `p_long_xgb` | `float` [0.0, 1.0] | XGB probability of LONG | ✅ Yes |
| 86 | `p_hat_xgb` | `float` [0.0, 1.0] | `max(p_long, p_short)` (calibrated) | ✅ Yes |
| 87 | *(unused)* | - | Reserved for compatibility | - |

**Injection:**
- Injected at **current timestep** only (snapshot)
- Uses **current session's XGB head output** (EU/OVERLAP/US)
- For disabled heads (US): uses neutral constants

### 2.4 Channel Universality

**All XGB channels are UNIVERSAL** (same semantikk for all session heads):

- `p_long_xgb`: Same meaning for EU, OVERLAP, US (probability of LONG)
- `uncertainty_score`: Same meaning for all sessions (normalized entropy)
- `p_hat_xgb`: Same meaning for all sessions (`max(p_long, p_short)`)

**No per-head semantic differences.** The only difference is:
- **EU/OVERLAP**: Channels contain **informative values** (vary, std > 0)
- **US**: Channels contain **neutral constants** (1/3, 1.0) when disabled

---

## 3. Runtime Behavior (Truth)

### 3.1 Session Routing

**Current Session Detection:**
- Determined by `current_session` (EU/OVERLAP/US) based on time-of-day
- Only **current session's head** is used (not all heads simultaneously)

**Policy Check:**
- Loaded from `gx1/configs/xgb_session_policy.json`
- Checked before XGB prediction: `session_config.get("enabled")`

### 3.2 Enabled Sessions (EU, OVERLAP)

**Behavior:**
1. Load XGB multihead model
2. Call `model.predict_proba(df, session="EU"|"OVERLAP")`
3. Extract outputs: `p_long`, `p_short`, `p_flat`, `uncertainty`
4. Compute `p_hat = max(p_long, p_short)`
5. Inject into Transformer:
   - SEQ: `[p_long_xgb, uncertainty_score]` (all timesteps)
   - SNAP: `[p_long_xgb, p_hat_xgb]` (current timestep)

**Expected Values:**
- `p_long_xgb`: Varies (std > 0), typically in range [0.2, 0.8]
- `uncertainty_score`: Varies (std > 0), typically in range [0.3, 0.9]
- `p_hat_xgb`: Varies (std > 0), typically in range [0.3, 0.8]

### 3.3 Disabled Sessions (US)

**Behavior:**
1. Policy check: `US.enabled = false`
2. **Skip XGB model call** (no prediction)
3. Use neutral constants:
   - `p_long_xgb = 1/3`
   - `p_short_xgb = 1/3`
   - `p_flat_xgb = 1/3`
   - `uncertainty_score = 1.0`
   - `p_hat_xgb = max(1/3, 1/3) = 1/3`
4. Inject into Transformer:
   - SEQ: `[1/3, 1.0]` (all timesteps)
   - SNAP: `[1/3, 1/3]` (current timestep)

**Expected Values:**
- `p_long_xgb`: Constant `0.333...` (std = 0)
- `uncertainty_score`: Constant `1.0` (std = 0)
- `p_hat_xgb`: Constant `0.333...` (std = 0)

**Rationale:**
- US head disabled due to drift (KS=0.227619, PSI=0.277938)
- Neutral constants signal Transformer to take full responsibility
- Maximum uncertainty (1.0) indicates XGB has no opinion

---

## 4. Implementation Details

### 4.1 Code Paths

**XGB Model Loading:**
- `gx1/xgb/multihead/xgb_multihead_model_v1.py::XGBMultiheadModel.load()`
- Loaded via `entry_v10_bundle.py` (same codepath as runtime)

**XGB Prediction:**
- `gx1/execution/oanda_demo_runner.py::_get_xgb_outputs_for_session()`
- Calls `model.predict_proba(df, session=current_session)`

**Policy Loading:**
- `gx1/xgb/multihead/xgb_multihead_model_v1.py::load_session_policy()`
- Cached per process (loaded once)

**Transformer Injection:**
- `gx1/execution/oanda_demo_runner.py` (lines ~7685-7738)
- Uses `feature_contract_v10_ctx.py` constants:
  - `SEQ_XGB_CHANNEL_START = 13`
  - `SNAP_XGB_CHANNEL_START = 85`
  - `XGB_CHANNELS = 3` (but only 2 used)

### 4.2 Feature Contract

**Contract Module:**
- `gx1/features/feature_contract_v10_ctx.py`

**Constants:**
```python
BASE_SEQ_FEATURES = 13
BASE_SNAP_FEATURES = 85
XGB_CHANNELS = 3  # For compatibility, but only 2 used
TOTAL_SEQ_FEATURES = 16  # 13 + 3
TOTAL_SNAP_FEATURES = 88  # 85 + 3

SEQ_XGB_CHANNEL_START = 13
SNAP_XGB_CHANNEL_START = 85

SEQ_XGB_CHANNEL_NAMES = ["p_long_xgb", "uncertainty_score"]
SNAP_XGB_CHANNEL_NAMES = ["p_long_xgb", "p_hat_xgb"]
```

---

## 5. Verification

### 5.1 Runtime Verification

**Script:** `gx1/scripts/inspect_xgb_transformer_contract.py` (read-only)

**Checks:**
1. ✅ XGB model loads correctly
2. ✅ Policy loads correctly (EU/OVERLAP enabled, US disabled)
3. ✅ EU/OVERLAP channels vary (std > 0)
4. ✅ US channels are constant (std = 0, values = 1/3, 1.0)
5. ✅ No extra XGB-derived signals leak into Transformer
6. ✅ Channel indices match contract (SEQ: 13-14, SNAP: 85-86)

### 5.2 Expected Runtime Behavior

**EU Session:**
- `p_long_xgb`: std > 0.05, mean ≈ 0.4-0.6
- `uncertainty_score`: std > 0.05, mean ≈ 0.5-0.7
- `p_hat_xgb`: std > 0.05, mean ≈ 0.4-0.6

**OVERLAP Session:**
- `p_long_xgb`: std > 0.05, mean ≈ 0.4-0.6
- `uncertainty_score`: std > 0.05, mean ≈ 0.5-0.7
- `p_hat_xgb`: std > 0.05, mean ≈ 0.4-0.6

**US Session:**
- `p_long_xgb`: std = 0.0, value = 0.333...
- `uncertainty_score`: std = 0.0, value = 1.0
- `p_hat_xgb`: std = 0.0, value = 0.333...

---

## 6. Summary Statement

**Transformer mottar N=2 komprimerte beslutningsspor fra XGB per scope (SEQ/SNAP).**

- **SEQ scope:** `p_long_xgb`, `uncertainty_score` (2 channels)
- **SNAP scope:** `p_long_xgb`, `p_hat_xgb` (2 channels)

**Total active channels:** 2 (not 3, despite `XGB_CHANNELS=3` for compatibility)

**All channels are UNIVERSAL** (same semantikk for all session heads).

**These represent XGB as opinionated prior, not raw features.**

**For disabled heads (US):** Channels contain neutral constants (1/3, 1.0) signaling Transformer to take full responsibility.

---

## 7. Transformer Input Contract (Complete)

### 7.1 Sequence (SEQ) Input

**Dimensions:** `[seq_len, 16]`

**Composition:**
- Base features: 13 (from feature_meta.json)
- XGB channels: 3 (but only 2 active: `p_long_xgb`, `uncertainty_score`)

### 7.2 Snapshot (SNAP) Input

**Dimensions (baseline):** `[88]` (when `GX1_TRANSFORMER_SESSION_TOKEN=0`)  
**Dimensions (with tokens):** `[92]` (when `GX1_TRANSFORMER_SESSION_TOKEN=1`)

**Composition (baseline):**
- Base features: 85 (from feature_meta.json)
- XGB channels: 3 (but only 2 active: `p_long_xgb`, `p_hat_xgb`)

**Composition (with session tokens):**
- Base features: 85
- XGB channels: 3 (but only 2 active)
- Session tokens: 4 (one-hot: `session_is_asia`, `session_is_eu`, `session_is_overlap`, `session_is_us`)

### 7.3 Session Tokens (A/B Test Feature)

**Status:** A/B test feature (controlled by `GX1_TRANSFORMER_SESSION_TOKEN` env var)

**When Enabled (`GX1_TRANSFORMER_SESSION_TOKEN=1`):**
- 4 one-hot features added to SNAP input (indices 88-91, total SNAP dim = 92)
- `session_is_asia`: 1.0 if current session is ASIA, else 0.0
- `session_is_eu`: 1.0 if current session is EU, else 0.0
- `session_is_overlap`: 1.0 if current session is OVERLAP, else 0.0
- `session_is_us`: 1.0 if current session is US, else 0.0

**Session Classification:**
- Determined by UTC hour via `infer_session_tag()`:
  - ASIA: 22:00-07:00 UTC (hours 22-06:59)
  - EU: 07:00-12:00 UTC (hours 7-11)
  - OVERLAP: 12:00-16:00 UTC (hours 12-15)
  - US: 16:00-22:00 UTC (hours 16-21)
- See `gx1/docs/SESSION_CLASSIFICATION_TRUTH.md` for complete documentation

**Invariant:**
- Exactly one token = 1.0 per bar (one-hot encoding)
- Sum of all tokens = 1.0 (within tolerance)

**Semantics:**
- These are **identity features** (regime/session metadata), not XGB outputs
- Deterministic per bar based on session routing
- Do not affect XGB model or routing
- Pure session identity signal for Transformer

**ASIA Token Note:**
- ASIA-token er kun identitet; ASIA trading enable/disable er separat beslutning
- Token eksisterer for kompletthet, men ASIA trading/routing er uendret

**Default:** Disabled (`GX1_TRANSFORMER_SESSION_TOKEN=0`) - baseline behavior unchanged

---

## 8. References

- **XGB Model:** `gx1/xgb/multihead/xgb_multihead_model_v1.py`
- **Policy Config:** `gx1/configs/xgb_session_policy.json`
- **Feature Contract:** `gx1/features/feature_contract_v10_ctx.py`
- **Injection Code:** `gx1/execution/oanda_demo_runner.py` (lines ~7685-7740)
- **Verification Script:** `gx1/scripts/inspect_xgb_transformer_contract.py`
- **Session Token Inspection:** `gx1/scripts/inspect_transformer_session_token.py`