# Gated Fusion Contract

**Date:** 2026-01-07  
**Purpose:** Single Source of Truth (SSoT) for Gated Fusion architecture  
**Status:** Phase 2 - Gated Fusion Implementation

---

## Overview

Gated Fusion combines XGB snapshot understanding with Transformer temporal reasoning. The gate learns when to trust XGB vs Transformer based on uncertainty signals.

**Key Principle:** No post-hoc veto (except safety kill-switch). The model makes a unified decision.

---

## A) Inputs to Fusion

### From XGB (Snapshot Understanding)

**Source:** `snap_x` indices 85-87 (from Phase 1 calibration)

| Signal | Index | Type | Range | Meaning |
|--------|-------|------|-------|---------|
| `p_cal` | 85 | float32 | [0.0, 1.0] | Calibrated probability for LONG |
| `margin` | 86 | float32 | [0.0, 1.0] | Confidence margin (from calibrated) |
| `uncertainty_score` | 87 | float32 | [0.0, 1.0] | Normalized entropy (0=certain, 1=uncertain) |

**Optional (Future):**
- `leaf_embedding`: XGB leaf indices as embedding (OOD detection)

**XGB State Vector:**
```python
xgb_state = [p_cal, margin, uncertainty_score]  # [3] or [4] with leaf_embedding
```

### From Transformer (Temporal + Context)

**Sequence Embeddings:**
- `seq_emb`: [B, d_model] - Temporal understanding from sequence encoder
- Source: `seq_encoder(seq_x)` where `seq_x` includes calibrated XGB channels

**Context Features:**
- `ctx_emb`: [B, 42] - Context embeddings (session/trend/vol/ATR/spread)
- Source: `context_encoder(ctx_cat, ctx_cont)`

**Snapshot Embeddings (Raw, without XGB):**
- `snap_raw_emb`: [B, snap_output_dim] - Raw snapshot features (indices 0-84, no XGB channels)
- Source: `snap_encoder(snap_x[:, 0:85])` - Encoded without XGB channels

**Note:** Current `snap_emb` includes XGB channels. For gated fusion, we need raw snapshot as backup.

---

## B) Outputs from Fusion

### Gate Value

**Type:** `gate` ∈ [0, 1] (scalar per sample)

**Semantics:**
- `gate = 0.0`: Trust only Transformer (ignore XGB snapshot)
- `gate = 1.0`: Trust only XGB snapshot (ignore Transformer temporal)
- `gate = 0.5`: Equal weight (no bias)

**Computation:**
```python
gate = sigmoid(MLP([xgb_state, ctx_emb, seq_emb_summary]))
```

**Initialization:** Gate bias ≈ 0.0 (no initial preference)

### Fused Representation

**Type:** `fused_repr` [B, fusion_hidden_dim]

**Computation:**
```python
# Project XGB state to same dimension as Transformer
xgb_proj = MLP_xgb(xgb_state)  # [B, fusion_hidden_dim]

# Project Transformer (seq + ctx) to same dimension
transformer_proj = MLP_transformer(seq_emb, ctx_emb)  # [B, fusion_hidden_dim]

# Gated fusion
fused_repr = gate * xgb_proj + (1 - gate) * transformer_proj
```

**Alternative (with raw snapshot backup):**
```python
# Option: Use raw snapshot as backup
snap_raw_proj = MLP_snap(snap_raw_emb)  # [B, fusion_hidden_dim]
fused_repr = gate * xgb_proj + (1 - gate) * (transformer_proj + snap_raw_proj)
```

### Decision Output

**Type:** `decision_logit` [B, 1]

**Computation:**
```python
decision_logit = direction_head(fused_repr)
p_trade = sigmoid(decision_logit)
```

**Note:** This replaces the current `direction_logit` from Transformer-only path.

---

## C) Architecture Variant

### Variant A: Pre-Decision Fusion (Selected)

**Location:** After fusion MLP, before decision heads

**Flow:**
```
seq_x → seq_encoder → seq_emb ─┐
snap_x → snap_encoder → snap_emb ─┤
ctx_cat, ctx_cont → context_encoder → ctx_emb ─┤
                                              ├→ concat → fusion MLP → fused_base
regime_emb ───────────────────────────────────┘

xgb_state (from snap_x[85:88]) ─→ GatedFusion ─→ fused_repr
fused_base ──────────────────────┘

fused_repr → direction_head → decision_logit
```

**Advantages:**
- Simple and stable
- Easy to debug
- Single fusion point
- Gate affects final decision directly

**Implementation:**
- Extract `xgb_state` from `snap_x[:, 85:88]`
- Extract `snap_raw` from `snap_x[:, 0:85]` (for backup)
- `GatedFusion` takes `xgb_state`, `seq_emb`, `ctx_emb`, `snap_raw_emb`
- Output `fused_repr` replaces current `fused` before decision heads

### Variant B: Token-level Fusion (Not Implemented)

**Location:** Inside Transformer (as additional token)

**Flow:**
```
xgb_state → embed → xgb_token
seq_x + xgb_token → Transformer (with per-head gating)
```

**Status:** Deferred to future phase

---

## D) GatedFusion Module

### Class Definition

```python
class GatedFusion(nn.Module):
    """
    Gated Fusion Layer for XGB + Transformer.
    
    Learns when to trust XGB snapshot vs Transformer temporal reasoning.
    """
    
    def __init__(
        self,
        xgb_state_dim: int = 3,  # p_cal, margin, uncertainty_score
        seq_emb_dim: int = 128,  # d_model from seq_encoder
        ctx_emb_dim: int = 42,   # Context embedding dimension
        snap_raw_emb_dim: int = 128,  # Raw snapshot embedding dimension
        fusion_hidden_dim: int = 256,
        gate_init_bias: float = 0.0,  # No initial bias
    ):
        super().__init__()
        
        # Project XGB state to fusion dimension
        self.xgb_proj = nn.Sequential(
            nn.Linear(xgb_state_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
        )
        
        # Project Transformer (seq + ctx) to fusion dimension
        transformer_input_dim = seq_emb_dim + ctx_emb_dim
        self.transformer_proj = nn.Sequential(
            nn.Linear(transformer_input_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
        )
        
        # Optional: Raw snapshot backup
        self.snap_raw_proj = nn.Sequential(
            nn.Linear(snap_raw_emb_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
        )
        
        # Gate computation MLP
        gate_input_dim = xgb_state_dim + ctx_emb_dim + seq_emb_dim  # Summary of seq_emb
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, fusion_hidden_dim // 2),
            nn.LayerNorm(fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim // 2, 1),
        )
        
        # Initialize gate bias to 0.0 (no initial preference)
        nn.init.zeros_(self.gate_mlp[-1].bias)
        self.gate_mlp[-1].bias.data.fill_(gate_init_bias)
    
    def forward(
        self,
        xgb_state: torch.Tensor,  # [B, 3]
        seq_emb: torch.Tensor,     # [B, seq_emb_dim]
        ctx_emb: torch.Tensor,      # [B, ctx_emb_dim]
        snap_raw_emb: Optional[torch.Tensor] = None,  # [B, snap_raw_emb_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            (fused_repr, gate) where:
            - fused_repr: [B, fusion_hidden_dim]
            - gate: [B, 1] (scalar per sample)
        """
        batch_size = xgb_state.size(0)
        
        # Compute gate
        # Summarize seq_emb (mean pooling if needed, or use as-is if already pooled)
        seq_emb_summary = seq_emb  # Assume already pooled [B, seq_emb_dim]
        
        gate_input = torch.cat([xgb_state, ctx_emb, seq_emb_summary], dim=-1)
        gate_logit = self.gate_mlp(gate_input)  # [B, 1]
        gate = torch.sigmoid(gate_logit)  # [B, 1]
        
        # Project XGB state
        xgb_proj = self.xgb_proj(xgb_state)  # [B, fusion_hidden_dim]
        
        # Project Transformer
        transformer_input = torch.cat([seq_emb, ctx_emb], dim=-1)
        transformer_proj = self.transformer_proj(transformer_input)  # [B, fusion_hidden_dim]
        
        # Optional: Add raw snapshot backup
        if snap_raw_emb is not None:
            snap_raw_proj = self.snap_raw_proj(snap_raw_emb)  # [B, fusion_hidden_dim]
            transformer_proj = transformer_proj + snap_raw_proj
        
        # Gated fusion
        fused_repr = gate * xgb_proj + (1 - gate) * transformer_proj
        
        return fused_repr, gate
```

---

## E) Integration Points

### 1. Model Architecture

**File:** `gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py`

**Changes:**
- Add `GatedFusion` module to `EntryV10CtxHybridTransformer.__init__()`
- Extract `xgb_state` from `snap_x[:, 85:88]` in `forward()`
- Extract `snap_raw` from `snap_x[:, 0:85]` for raw snapshot encoding
- Apply `GatedFusion` before decision heads
- Return `gate` in output dict for telemetry

### 2. Runtime Integration

**File:** `gx1/execution/oanda_demo_runner.py:_predict_entry_v10_hybrid()`

**Changes:**
- No changes needed (XGB state already in `snap_x[85:88]`)
- Gate value logged in telemetry

### 3. Telemetry

**Location:** `gx1/execution/entry_manager.py` or `oanda_demo_runner.py`

**Metrics:**
- `gate_mean`: Mean gate value per bar
- `gate_p95`: 95th percentile gate value
- `gate_std`: Standard deviation of gate values
- `gate_vs_entropy`: Correlation between `uncertainty_score` and `gate`
- `gate_histogram_per_regime`: Gate distribution per regime bucket

---

## F) Training Targets

### Labels

**Same as current V10 training:**
- `y_direction`: Binary (1=long, 0=short/neutral)
- `y_early_move`: Binary (1 if MFE before MAE)
- `y_quality_score`: Regression (normalized MFE-MAE gap)

**Important:** No labels for trades that would be vetoed by post-model gates. Training data should only include trades that would actually execute.

### Loss Function

**Primary Loss:**
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

## G) Telemetry & Proof

### Required Telemetry

**Per Bar:**
- `gate`: Gate value [0, 1]
- `xgb_state`: [p_cal, margin, uncertainty_score]
- `regime_bucket`: For grouping

**Aggregated:**
- `gate_mean`: Mean gate value
- `gate_p95`: 95th percentile
- `gate_std`: Standard deviation
- `gate_vs_entropy_correlation`: Correlation coefficient
- `gate_histogram_per_regime`: Distribution per regime

### Proof Requirements

**Gate Responsiveness:**
- High `uncertainty_score` → `gate` should decrease (trust XGB less)
- Low `uncertainty_score` → `gate` should increase (trust XGB more)
- Correlation: `corr(uncertainty_score, 1 - gate) > 0.3` (positive correlation)

**Gate Stability:**
- In stable regimes (LOW/MEDIUM vol): `gate_std < 0.2` (low variance)
- In unstable regimes (HIGH/EXTREME vol): `gate_std` can be higher

**Gate Distribution:**
- Gate should not be constant (all 0.0 or all 1.0)
- Gate should vary based on uncertainty

---

## H) Evaluation (Phase 2 GO/NO-GO)

### Metrics

**Performance vs Entropy Buckets:**
- Group predictions by `uncertainty_score` buckets (0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)
- Compute performance (PnL, win rate) per bucket
- Expected: Lower performance in high-uncertainty buckets

**Tail Risk vs Gate Values:**
- Group trades by `gate` buckets
- Compute max drawdown, VaR (95th percentile loss) per bucket
- Expected: Lower tail risk when gate is adaptive (not constant)

**Regime Stability Score:**
- Compute performance per regime (LOW/MEDIUM/HIGH vol)
- Compare to V10 without gate
- Expected: Stable or improved performance across regimes

### GO/NO-GO Criteria

**Required:**
- ✅ Better tail risk than V10 without gate (max drawdown, VaR)
- ✅ No increase in drawdown vs baseline
- ✅ Gate responds to uncertainty (`corr(uncertainty_score, 1 - gate) > 0.3`)
- ✅ Gate is not constant (variance > 0.05)
- ✅ Performance stable across regimes (regime_stability_score > 0.8)

**Fail-Fast:**
- Hard fail in replay if gate is constant (all 0.0 or all 1.0)
- Hard fail if gate does not respond to uncertainty

---

## I) Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GX1_GATED_FUSION_ENABLED` | `1` | Enable gated fusion (default: enabled) |
| `GX1_GATE_TELEMETRY_ENABLED` | `1` | Enable gate telemetry logging |

---

## J) Contract Changes

### Model Inputs (Unchanged)

- `seq_x`: [B, 30, 16] (includes XGB channels at indices 13-15)
- `snap_x`: [B, 88] (includes XGB channels at indices 85-87)
- `ctx_cat`: [B, 5]
- `ctx_cont`: [B, 2]

### Model Outputs (Extended)

**Before:**
```python
{
    "direction_logit": [B, 1],
    "early_move_logit": [B, 1],  # optional
    "quality_score": [B, 1],     # optional
}
```

**After:**
```python
{
    "direction_logit": [B, 1],
    "early_move_logit": [B, 1],  # optional
    "quality_score": [B, 1],     # optional
    "gate": [B, 1],              # NEW: Gate value for telemetry
}
```

**Note:** `direction_logit` is now computed from gated fusion, not direct Transformer output.

---

## Summary

**Inputs:** XGB state (p_cal, margin, uncertainty_score) + Transformer (seq_emb, ctx_emb)  
**Outputs:** `fused_repr`, `gate`, `decision_logit`  
**Architecture:** Variant A (Pre-Decision Fusion)  
**Training:** Same labels, optional gate stability loss  
**Telemetry:** Gate metrics, correlation with uncertainty  
**GO/NO-GO:** Tail risk, gate responsiveness, regime stability

---

**End of Document**
