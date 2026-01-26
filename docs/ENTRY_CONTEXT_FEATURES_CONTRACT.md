# Entry Context Features Contract

**Date:** 2026-01-06  
**Status:** STEP 2 - Contract Definition  
**Purpose:** Define context features as MODEL INPUT (not gates)

---

## 1. Context Features List (7 features)

### 1.1 `session_id` (categorical)

- **Type:** Categorical (integer ID)
- **Values:** `0=ASIA, 1=EU, 2=US, 3=OVERLAP`
- **Source:** `infer_session_tag(timestamp)` from `gx1.execution.live_features`
- **Normalization:** None (categorical)
- **Default/Fallback:** `0` (ASIA) if timestamp invalid
- **Availability:** ✅ Available without feature build (time-based only)
- **Embedding Dim:** 8 (configurable)

### 1.2 `trend_regime_id` (categorical)

- **Type:** Categorical (integer ID)
- **Values:** `0=TREND_DOWN, 1=TREND_NEUTRAL, 2=TREND_UP`
- **Source:** 
  - Primary: `policy_state.get("brain_trend_regime")` (from Big Brain V1 or FARM)
  - Fallback: Computed from candles using `ensure_replay_tags()` (requires minimal compute)
- **Normalization:** None (categorical)
- **Default/Fallback:** `1` (TREND_NEUTRAL) if UNKNOWN
- **Availability:** ⚠️ Requires minimal compute (ATR/trend from candles, but cheaper than full feature build)
- **Embedding Dim:** 8 (configurable)

### 1.3 `vol_regime_id` (categorical)

- **Type:** Categorical (integer ID)
- **Values:** `0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME`
- **Source:**
  - Primary: `policy_state.get("brain_vol_regime")` (from Big Brain V1 or FARM)
  - Fallback: Computed from cheap ATR proxy (soft eligibility already does this)
- **Normalization:** None (categorical)
- **Default/Fallback:** `1` (MEDIUM) if UNKNOWN (UNKNOWN is NOT a valid regime ID)
- **Availability:** ⚠️ Requires minimal compute (ATR proxy, already computed in soft eligibility)
- **Embedding Dim:** 8 (configurable)
- **Note:** EXTREME is blocked by soft eligibility, so model should never see `vol_regime_id=3` in practice

### 1.4 `atr_bps` (continuous)

- **Type:** Continuous (float, basis points)
- **Range:** `[0.0, 1000.0]` bps (clipped)
- **Source:**
  - Primary: `entry_bundle.atr_bps` (from feature build)
  - Fallback: Cheap ATR proxy (from soft eligibility)
- **Normalization:** Z-score (computed from training data distribution)
- **Default/Fallback:** `50.0` bps (median estimate) if unavailable
- **Availability:** ⚠️ Requires minimal compute (ATR proxy available from soft eligibility)
- **Units:** Basis points (1 bps = 0.01%)

### 1.5 `atr_bucket` (categorical)

- **Type:** Categorical (integer ID)
- **Values:** `0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME`
- **Source:** Derived from `atr_bps` using percentile buckets:
  - `LOW`: 0-33rd percentile
  - `MEDIUM`: 33rd-67th percentile
  - `HIGH`: 67th-95th percentile
  - `EXTREME`: >95th percentile (should be blocked by soft eligibility)
- **Normalization:** None (categorical)
- **Default/Fallback:** `1` (MEDIUM) if `atr_bps` unavailable
- **Availability:** ⚠️ Requires `atr_bps` (same as above)
- **Embedding Dim:** 8 (configurable)

### 1.6 `spread_bps` (continuous)

- **Type:** Continuous (float, basis points)
- **Range:** `[0.0, 500.0]` bps (clipped)
- **Source:**
  - Primary: `entry_bundle.spread_bps` (from feature build)
  - Fallback: `_get_spread_bps_before_features(candles)` (from hard eligibility)
- **Normalization:** Z-score (computed from training data distribution)
- **Default/Fallback:** `10.0` bps (typical spread) if unavailable
- **Availability:** ✅ Available without feature build (from bid/ask quotes)
- **Units:** Basis points (1 bps = 0.01%)

### 1.7 `spread_bucket` (categorical)

- **Type:** Categorical (integer ID)
- **Values:** `0=LOW, 1=MEDIUM, 2=HIGH`
- **Source:** Derived from `spread_bps` using percentile buckets:
  - `LOW`: 0-33rd percentile
  - `MEDIUM`: 33rd-67th percentile
  - `HIGH`: 67th-100th percentile
- **Normalization:** None (categorical)
- **Default/Fallback:** `1` (MEDIUM) if `spread_bps` unavailable
- **Availability:** ✅ Available without feature build (from `spread_bps`)
- **Embedding Dim:** 8 (configurable)

---

## 2. Availability Contract

### 2.1 Without Feature Build (Hard Eligibility)

**Available:**
- ✅ `session_id` (time-based only)
- ✅ `spread_bps` (from bid/ask quotes)
- ✅ `spread_bucket` (derived from `spread_bps`)

**Not Available:**
- ❌ `trend_regime_id` (requires ATR/trend computation)
- ❌ `vol_regime_id` (requires ATR computation)
- ❌ `atr_bps` (requires ATR computation)
- ❌ `atr_bucket` (requires `atr_bps`)

### 2.2 After Minimal Cheap Computation (Soft Eligibility)

**Available:**
- ✅ `session_id` (already available)
- ✅ `spread_bps` (already available)
- ✅ `spread_bucket` (already available)
- ✅ `atr_bps` (from cheap ATR proxy, computed in soft eligibility)
- ✅ `atr_bucket` (derived from `atr_bps`)
- ✅ `vol_regime_id` (derived from `atr_bps` percentile)
- ⚠️ `trend_regime_id` (requires trend computation, may need full feature build)

**Note:** `trend_regime_id` may require full feature build for accurate computation. For now, we'll use a fallback (TREND_NEUTRAL) if not available from policy_state.

---

## 3. Embedding Design

### 3.1 Categorical Embeddings

**Features:**
- `session_id` → embedding dim = 8
- `trend_regime_id` → embedding dim = 8
- `vol_regime_id` → embedding dim = 8
- `atr_bucket` → embedding dim = 8
- `spread_bucket` → embedding dim = 8

**Total Categorical Embeddings:** 5 features × 8 dims = 40 dims

**Configuration:**
- Embedding dimensions are **configurable** via bundle metadata
- Default: 8 dims per categorical feature
- Can be overridden per feature in bundle config

### 3.2 Continuous Features

**Features:**
- `atr_bps` → normalized (Z-score) → 1 dim
- `spread_bps` → normalized (Z-score) → 1 dim

**Total Continuous Features:** 2 dims

**Normalization:**
- Z-score computed from training data distribution
- Mean/std stored in bundle metadata
- Applied at runtime before model input

---

## 4. Range & Clipping

### 4.1 `atr_bps`

- **Range:** `[0.0, 1000.0]` bps
- **Clipping:** Values outside range are clipped to bounds
- **Rationale:** 
  - 0 bps = no volatility (theoretical minimum)
  - 1000 bps = 10% ATR (extreme volatility, should be blocked by soft eligibility)
  - Empirical: Typical XAUUSD ATR is 20-200 bps

### 4.2 `spread_bps`

- **Range:** `[0.0, 500.0]` bps
- **Clipping:** Values outside range are clipped to bounds
- **Rationale:**
  - 0 bps = no spread (theoretical minimum)
  - 500 bps = 5% spread (extreme, should be blocked by hard eligibility)
  - Empirical: Typical XAUUSD spread is 5-50 bps

### 4.3 Categorical Features

- **No clipping needed** (integer IDs are already bounded by value set)
- **Validation:** IDs must be in valid range (0..N-1 for N categories)

---

## 5. Validation Rules

### 5.1 Replay Mode (Hard Fail)

**Contract:**
- If `ENTRY_CONTEXT_FEATURES_ENABLED=true`:
  - All context features **MUST** be present and valid
  - Missing or invalid features → `RuntimeError("CONTEXT_FEATURE_MISSING", details)`
  - Fail-fast before model inference

**Validation Checks:**
1. All categorical IDs in valid range
2. All continuous features are finite (not NaN/Inf)
3. All features are present (not None)

**Error Format:**
```python
RuntimeError(
    "CONTEXT_FEATURE_MISSING: "
    f"feature={feature_name} "
    f"value={value} "
    f"reason={reason} "
    f"cycle={n_cycles}"
)
```

### 5.2 Live Mode (Warning + Fallback)

**Contract:**
- If `ENTRY_CONTEXT_FEATURES_ENABLED=true`:
  - Missing features → warning + fallback to default
  - Invalid features → warning + fallback to default
  - Log `"data_integrity_degraded"` flag

**Fallback Strategy:**
- Categorical: Use default ID (middle value, e.g., MEDIUM)
- Continuous: Use default value (median estimate)

**Logging:**
```python
log.warning(
    "[CONTEXT_FEATURES] data_integrity_degraded: "
    f"feature={feature_name} "
    f"value={value} "
    f"fallback={fallback_value} "
    f"reason={reason}"
)
```

### 5.3 Backward Compatibility

**Contract:**
- If `ENTRY_CONTEXT_FEATURES_ENABLED=false`:
  - Context features are **NOT** built
  - Model runs without context (old behavior)
  - No validation errors

**Model Compatibility:**
- Old bundles (`supports_context_features=false`):
  - Context features ignored (not passed to model)
  - Model runs with existing seq_x + snap_x only
- New bundles (`supports_context_features=true`):
  - Context features **REQUIRED** (hard fail if missing)
  - Model expects ctx_cat + ctx_cont tensors

---

## 6. Feature Contract Hash

**Purpose:** Ensure model and runtime use same feature contract

**Computation:**
```python
import hashlib
import json

contract_dict = {
    "session_id": {"type": "categorical", "values": [0, 1, 2, 3], "embedding_dim": 8},
    "trend_regime_id": {"type": "categorical", "values": [0, 1, 2], "embedding_dim": 8},
    "vol_regime_id": {"type": "categorical", "values": [0, 1, 2, 3], "embedding_dim": 8},
    "atr_bps": {"type": "continuous", "range": [0.0, 1000.0], "normalization": "zscore"},
    "atr_bucket": {"type": "categorical", "values": [0, 1, 2, 3], "embedding_dim": 8},
    "spread_bps": {"type": "continuous", "range": [0.0, 500.0], "normalization": "zscore"},
    "spread_bucket": {"type": "categorical", "values": [0, 1, 2], "embedding_dim": 8},
}

contract_json = json.dumps(contract_dict, sort_keys=True)
contract_hash = hashlib.sha256(contract_json.encode()).hexdigest()[:16]
```

**Storage:**
- Bundle metadata: `feature_contract_hash`
- Runtime validation: Compare bundle hash with runtime contract hash
- Mismatch → hard fail in replay, warning in live

---

## 7. Summary

| Feature | Type | Availability | Embedding Dim | Range/Values |
|---------|------|--------------|---------------|--------------|
| `session_id` | Categorical | ✅ Hard eligibility | 8 | 0-3 |
| `trend_regime_id` | Categorical | ⚠️ Soft eligibility | 8 | 0-2 |
| `vol_regime_id` | Categorical | ⚠️ Soft eligibility | 8 | 0-3 |
| `atr_bps` | Continuous | ⚠️ Soft eligibility | 1 (normalized) | [0, 1000] bps |
| `atr_bucket` | Categorical | ⚠️ Soft eligibility | 8 | 0-3 |
| `spread_bps` | Continuous | ✅ Hard eligibility | 1 (normalized) | [0, 500] bps |
| `spread_bucket` | Categorical | ✅ Hard eligibility | 8 | 0-2 |

**Total Context Dimensions:**
- Categorical embeddings: 5 × 8 = 40 dims
- Continuous features: 2 dims
- **Total: 42 dims**

---

## 8. Implementation Notes

### 8.1 Build Order

1. **Hard Eligibility** → blocks before any computation
2. **Soft Eligibility** → computes cheap ATR proxy (needed for `atr_bps`, `atr_bucket`, `vol_regime_id`)
3. **Context Features Build** → builds all 7 features (cheaper than full V9 feature build)
4. **Full Feature Build** → builds V9 features (seq + snap)
5. **Model Inference** → passes context + features to model

### 8.2 Performance Target

**Context Features Build Cost:**
- Target: < 10% of full V9 feature build time
- Uses: Cheap ATR proxy (already computed), session inference (O(1)), spread from quotes (O(1))
- Avoids: Full feature pipeline, heavy rolling operations

### 8.3 Testing Strategy

1. **Regression Test:** Context flag OFF → identical behavior
2. **Compatibility Test:** Context flag ON + old bundle → hard fail in replay
3. **Unit Tests:** Contract validation, tensor shapes, determinism

---

**Next Steps:**
- Implement `EntryContextFeatures` dataclass
- Implement `build_entry_context_features()` method
- Integrate into `evaluate_entry()` flow
- Update model interface (with fallback)



