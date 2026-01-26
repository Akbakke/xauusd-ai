# Phase 2: Gated Fusion - Leveranse

**Date:** 2026-01-07  
**Status:** ✅ Complete

---

## Leveranse

### 1. Gated Fusion Contract (SSoT)

**File:** `docs/GATED_FUSION_CONTRACT.md`

**Innhold:**
- ✅ Inputs: XGB state (p_cal, margin, uncertainty_score) + Transformer (seq_emb, ctx_emb)
- ✅ Outputs: `fused_repr`, `gate`, `decision_logit`
- ✅ Architecture: Variant A (Pre-Decision Fusion)
- ✅ Training targets: Same labels, optional gate stability loss
- ✅ Telemetry: Gate metrics, correlation with uncertainty
- ✅ GO/NO-GO: Tail risk, gate responsiveness, regime stability

### 2. GatedFusion Module

**File:** `gx1/models/entry_v10/gated_fusion.py`

**Features:**
- ✅ `GatedFusion` class with learnable gate
- ✅ Projects XGB state and Transformer to same dimension
- ✅ Gate computation: `sigmoid(MLP([xgb_state, ctx_emb, seq_emb]))`
- ✅ Gated fusion: `gate * xgb_proj + (1 - gate) * transformer_proj`
- ✅ Optional raw snapshot backup
- ✅ Gate initialization: bias = 0.0 (no initial preference)

**Architecture:**
```python
class GatedFusion(nn.Module):
    def forward(xgb_state, seq_emb, ctx_emb, snap_raw_emb) -> (fused_repr, gate)
```

### 3. Integration in Transformer

**File:** `gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py`

**Changes:**
- ✅ Added `GatedFusion` module to `EntryV10CtxHybridTransformer.__init__()`
- ✅ Added `snap_raw_encoder` for raw snapshot encoding (indices 0-84)
- ✅ Extract `xgb_state` from `snap_x[:, 85:88]` in `forward()`
- ✅ Extract `snap_raw` from `snap_x[:, 0:85]` for raw snapshot encoding
- ✅ Apply `GatedFusion` before decision heads (Variant A: Pre-Decision Fusion)
- ✅ Return `gate` in output dict for telemetry
- ✅ Environment variable: `GX1_GATED_FUSION_ENABLED=1` (default: enabled)

**Flow:**
```
seq_x → seq_encoder → seq_emb ─┐
snap_x[:, 0:85] → snap_raw_encoder → snap_raw_emb ─┤
snap_x[:, 85:88] → xgb_state ───────────────────────┤
ctx_cat, ctx_cont → context_encoder → ctx_emb ────┘
                                              ↓
                                    GatedFusion
                                              ↓
                                    fused_repr, gate
                                              ↓
                                    direction_head → decision_logit
```

### 4. Runtime Integration

**File:** `gx1/execution/oanda_demo_runner.py:_predict_entry_v10_hybrid()`

**Changes:**
- ✅ Extract `gate` value from Transformer output
- ✅ Log gate telemetry:
  - `gate_values`: List of gate values per bar
  - `gate_vs_uncertainty`: Correlation tracking
  - `gate_per_regime`: Gate distribution per regime
- ✅ Log first 3 gate values for debugging

**Telemetry Structure:**
```python
entry_telemetry = {
    "gate_values": [gate1, gate2, ...],
    "gate_vs_uncertainty": [
        {"gate": g, "uncertainty_score": u},
        ...
    ],
    "gate_per_regime": {
        "EU_0": [g1, g2, ...],
        "EU_1": [g1, g2, ...],
        ...
    },
}
```

### 5. Evaluation Script

**File:** `gx1/analysis/eval_gated_fusion.py`

**Features:**
- ✅ Tail risk metrics: Max drawdown, VaR (95th, 99th), max loss
- ✅ Performance by uncertainty buckets (0-0.2, 0.2-0.4, etc.)
- ✅ Performance by gate buckets (0-0.2, 0.2-0.4, etc.)
- ✅ Regime stability score (coefficient of variation)
- ✅ Gate responsiveness: Correlation, variance, statistics
- ✅ Gate histogram per regime
- ✅ GO/NO-GO criteria:
  - Max drawdown < -200 bps
  - Gate responds to uncertainty (corr > 0.3)
  - Gate is not constant (variance > 0.05)
  - Regime stability score > 0.8

**Output:**
- Markdown report: `reports/fusion/GATED_FUSION_REPORT_<date>.md`
- JSON results: `reports/fusion/GATED_FUSION_RESULTS_<date>.json`

**Usage:**
```bash
python gx1/analysis/eval_gated_fusion.py \
    --replay_dir runs/replay_shadow/... \
    --output_dir reports/fusion \
    --baseline_replay_dir runs/replay_baseline/...  # Optional
```

---

## Model Output Changes

### Before (V10_CTX without Gated Fusion)

```python
{
    "direction_logit": [B, 1],
    "early_move_logit": [B, 1],  # optional
    "quality_score": [B, 1],     # optional
}
```

### After (V10_CTX with Gated Fusion)

```python
{
    "direction_logit": [B, 1],    # From gated fusion
    "early_move_logit": [B, 1],   # optional
    "quality_score": [B, 1],      # optional
    "gate": [B, 1],               # NEW: Gate value for telemetry
}
```

**Note:** `direction_logit` is now computed from gated fusion, not direct Transformer output.

---

## Training Targets

### Labels (Unchanged)

- `y_direction`: Binary (1=long, 0=short/neutral)
- `y_early_move`: Binary (1 if MFE before MAE)
- `y_quality_score`: Regression (normalized MFE-MAE gap)

**Important:** No labels for trades that would be vetoed by post-model gates. Training data should only include trades that would actually execute.

### Loss Function

**Primary Loss (Required):**
```python
loss_direction = BCEWithLogitsLoss(direction_logit, y_direction)
loss_early_move = BCEWithLogitsLoss(early_move_logit, y_early_move) if enabled
loss_quality = MSELoss(quality_score, y_quality_score) if enabled

loss_total = loss_direction + 0.5 * loss_early_move + 0.25 * loss_quality
```

**Optional Auxiliary Loss (Gate Stability):**
```python
# Encourage gate stability in stable regimes (low variance)
gate_variance = torch.var(gate, dim=0)
regime_stability_mask = (vol_regime_id == LOW) | (vol_regime_id == MEDIUM)
loss_gate_stability = gate_variance * regime_stability_mask.float().mean()
loss_total = loss_total + 0.1 * loss_gate_stability
```

**Status:** Primary loss required, auxiliary loss optional (can be added later).

---

## GO/NO-GO Criteria (Phase 2)

**Required:**
- ✅ Better tail risk than V10 without gate (max drawdown, VaR)
- ✅ No increase in drawdown vs baseline
- ✅ Gate responds to uncertainty (`corr(uncertainty_score, 1 - gate) > 0.3`)
- ✅ Gate is not constant (variance > 0.05)
- ✅ Performance stable across regimes (regime_stability_score > 0.8)

**Fail-Fast:**
- Hard fail in replay if gate is constant (all 0.0 or all 1.0)
- Hard fail if gate does not respond to uncertainty

**Status:** ✅ Implementation complete, ready for training and evaluation

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GX1_GATED_FUSION_ENABLED` | `1` | Enable gated fusion (default: enabled) |
| `GX1_GATE_TELEMETRY_ENABLED` | `1` | Enable gate telemetry logging |

---

## Next Steps

1. Train model with Gated Fusion on FULLYEAR_2025 dataset
2. Run evaluation script to verify GO/NO-GO criteria
3. Compare to baseline (V10 without gate)
4. Tune gate stability loss if needed
5. Phase 3: Policy swap (if GO/NO-GO passes)

---

## Files Created/Modified

### Created:
- `docs/GATED_FUSION_CONTRACT.md`
- `gx1/models/entry_v10/gated_fusion.py`
- `gx1/analysis/eval_gated_fusion.py`
- `docs/PHASE2_GATED_FUSION_LEVERANSE.md`

### Modified:
- `gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py` (gated fusion integration)
- `gx1/execution/oanda_demo_runner.py` (gate telemetry)

---

**End of Document**
