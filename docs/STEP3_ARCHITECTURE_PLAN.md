# STEP 3: Model Training med Context Features - Arkitekturplan

**Date:** 2026-01-06  
**Status:** Design Phase  
**Next:** Implementation

---

## Arkitekturdiagram (Tekst)

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENTRY_V10_CTX Model                          │
└─────────────────────────────────────────────────────────────────┘

INPUT LAYERS:
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   seq_x      │  │   snap_x     │  │   ctx_cat    │  │   ctx_cont   │
│ [B, L, 16]   │  │ [B, 88]      │  │ [B, 5]       │  │ [B, 2]       │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ SeqEncoder   │  │ SnapEncoder   │  │ ContextCat   │  │ ContextCont  │
│ Transformer  │  │ MLP           │  │ Embeddings   │  │ Linear       │
│              │  │               │  │ (5×8 dims)   │  │ + LayerNorm  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                  │                  │                  │
       │ [B, d_model]    │ [B, snap_dim]   │ [B, 40]         │ [B, 2]
       │                  │                  │                  │
       └──────────────────┴──────────────────┴──────────────────┘
                          │
                          ▼
                  ┌──────────────┐
                  │   FUSION      │
                  │   Concat      │
                  │ [B, d_model + │
                  │  snap_dim +   │
                  │  40 + 2]      │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │   FUSION MLP │
                  │   (1-2 lag)  │
                  │ [B, hidden]  │
                  └──────┬───────┘
                         │
                         ▼
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ Direction│   │Early Move│   │ Quality  │
  │   Head    │   │   Head   │   │   Head   │
  └──────────┘   └──────────┘   └──────────┘
```

**Context Injection Point:** Fusion Layer (preferred)

**Begrunnelse:**
1. **Enkel implementering:** Bare endre fusion input dimension
2. **Tilgjengelig for alle heads:** Context påvirker direction, early_move, og quality
3. **Ikke-breaking:** Eksisterende seq_encoder og snap_encoder uendret
4. **Fleksibel:** Kan lett justere context embedding dims uten å røre transformer

**Alternativ (ikke valgt):** Inject i transformer encoder input
- Ulempe: Krever endring av seq_encoder input_dim
- Ulempe: Context er samme for hele sekvensen (redundant)
- Ulempe: Mer komplekst å implementere

---

## Context Features Encoding

### Categorical Features (ctx_cat: [B, 5])
```
session_id      → Embedding(4, 8)  → [B, 8]
trend_regime_id → Embedding(3, 8)  → [B, 8]
vol_regime_id   → Embedding(4, 8)  → [B, 8]
atr_bucket      → Embedding(4, 8)  → [B, 8]
spread_bucket   → Embedding(3, 8)  → [B, 8]
                                ────────────
                                Concat → [B, 40]
```

### Continuous Features (ctx_cont: [B, 2])
```
atr_bps    → LayerNorm → Linear(2, 2) → [B, 2]
spread_bps ──────────────────────────→ [B, 2]
```

**Total Context Dimensions:** 40 (categorical) + 2 (continuous) = 42 dims

---

## Filer som må endres

### LEVERANSE 3A: Modellarkitektur

1. **`gx1/models/entry_v10/entry_v10_hybrid_transformer.py`**
   - Opprett ny klasse: `EntryV10CtxHybridTransformer` (kopi av `EntryV10HybridTransformer`)
   - Legg til `ContextEncoder` klasse:
     - `ContextCatEncoder`: Embeddings for 5 categorical features
     - `ContextContEncoder`: LayerNorm + Linear for 2 continuous features
   - Oppdater `forward()` signature:
     - Legg til `ctx_cat: torch.Tensor` og `ctx_cont: torch.Tensor`
   - Oppdater fusion layer:
     - Concat: `[seq_emb, snap_emb, regime_emb, ctx_cat_emb, ctx_cont_emb]`
     - Oppdater fusion input dimension

2. **`gx1/models/entry_v10/entry_v10_bundle.py`**
   - Oppdater `load_entry_v10_bundle()` til å støtte `variant="v10_ctx"`
   - Valider at bundle metadata har `supports_context_features=true`

### LEVERANSE 3B: Training Pipeline

3. **`gx1/models/entry_v10/entry_v10_dataset.py`** (NY FIL)
   - Opprett ny dataset klasse: `EntryV10CtxDataset`
   - Bygg context features deterministisk (samme kode som runtime)
   - Returner: `seq_x, snap_x, ctx_cat, ctx_cont, labels`
   - Bruk hard/soft eligibility i dataset filtering

4. **`gx1/models/entry_v10/entry_v10_train.py`** (NY FIL eller oppdater eksisterende)
   - Opprett training script for ENTRY_V10_CTX
   - Integrer `EntryV10CtxDataset`
   - Beregn og lagre `feature_contract_hash` i bundle metadata
   - Valider at context features bygges identisk med runtime

5. **`gx1/execution/entry_context_features.py`** (OPPDATER)
   - Eksporter `compute_feature_contract_hash()` funksjon
   - Brukes i training for å lagre hash i bundle metadata

### LEVERANSE 3C: Bundle Metadata & Validation

6. **`gx1/models/entry_v10/entry_v10_bundle.py`** (OPPDATER)
   - Valider `supports_context_features=true` når `variant="v10_ctx"`
   - Valider `feature_contract_hash` match (replay mode)

7. **`gx1/execution/oanda_demo_runner.py`** (OPPDATER)
   - Oppdater `_predict_entry_v10_hybrid()` til å bruke ny model signature
   - Pass `ctx_cat` og `ctx_cont` til model forward()

### LEVERANSE 3D: Telemetry

8. **`gx1/execution/entry_manager.py`** (OPPDATER)
   - Legg til `p_long_by_context` buckets:
     - `p_long_by_session: Dict[str, List[float]]`
     - `p_long_by_vol_regime: Dict[str, List[float]]`
     - `p_long_by_trend_regime: Dict[str, List[float]]`
   - Aggreger per cycle (ikke per trade)

9. **`scripts/run_mini_replay_perf.py`** (OPPDATER)
   - Eksporter `p_long_by_context` buckets i perf summary

---

## Context Injection Design Decision

### Valgt: Fusion Layer Injection

**Implementering:**
```python
# I EntryV10CtxHybridTransformer.forward()
seq_emb = self.seq_encoder(seq_x)  # [B, d_model]
snap_emb = self.snap_encoder(snap_x)  # [B, snap_dim]
regime_emb = self.regime_embeddings(...)  # [B, regime_dim]

# NEW: Context encoding
ctx_cat_emb = self.context_cat_encoder(ctx_cat)  # [B, 40]
ctx_cont_emb = self.context_cont_encoder(ctx_cont)  # [B, 2]

# Fusion: concat all embeddings
fused = torch.cat([
    seq_emb,      # [B, d_model]
    snap_emb,     # [B, snap_dim]
    regime_emb,   # [B, regime_dim]
    ctx_cat_emb,  # [B, 40]
    ctx_cont_emb, # [B, 2]
], dim=-1)  # [B, d_model + snap_dim + regime_dim + 40 + 2]

fused = self.fusion(fused)  # [B, fusion_hidden_dim]
```

**Fordeler:**
- ✅ Enkel implementering (bare endre fusion input dim)
- ✅ Context påvirker alle heads (direction, early_move, quality)
- ✅ Ikke-breaking (seq_encoder og snap_encoder uendret)
- ✅ Fleksibel (kan justere context embedding dims)

**Ulemper:**
- ⚠️ Context kommer sent i pipeline (men dette er OK for første iterasjon)
- ⚠️ Context er ikke tilgjengelig i transformer attention (men kan legges til senere)

**Alternativ (ikke valgt):** Inject i transformer encoder
- Ulempe: Krever endring av seq_encoder input_dim
- Ulempe: Context er samme for hele sekvensen (redundant)
- Ulempe: Mer komplekst å implementere

---

## Training Pipeline Design

### Dataset Building Flow

```
1. Load raw candles (M5 bars)
   ↓
2. Apply Hard Eligibility Filter
   - Warmup check
   - Session check (ASIA blocked)
   - Spread hard cap
   - Kill-switch
   ↓
3. Apply Soft Eligibility Filter
   - Compute cheap ATR proxy
   - Block EXTREME vol regime
   ↓
4. Build Context Features (deterministic)
   - session_id, trend_regime_id, vol_regime_id
   - atr_bps, atr_bucket, spread_bps, spread_bucket
   ↓
5. Build V9 Features (seq + snap)
   - Sequence features (13 seq)
   - Snapshot features (85 snap)
   ↓
6. Generate Labels
   - y_direction, y_early_move, y_quality_score
   ↓
7. Create Dataset
   - seq_x, snap_x, ctx_cat, ctx_cont, labels
```

**Viktig:** Hard/soft eligibility må brukes i training også, slik at modellen kun trenes på bars som ville vært eligible i live.

---

## Feature Contract Hash

**Formål:** Ensure model and runtime use same feature contract

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

## Telemetry: p_long_by_context

**Formål:** Verifisere at modellen faktisk *bruker* context

**Aggregering:**
```python
# Per cycle (not per trade)
p_long_by_session = {
    "EU": [0.65, 0.72, 0.68, ...],
    "US": [0.71, 0.69, 0.73, ...],
    "OVERLAP": [0.70, 0.75, 0.71, ...],
}

p_long_by_vol_regime = {
    "LOW": [0.68, 0.70, 0.69, ...],
    "MEDIUM": [0.71, 0.72, 0.70, ...],
    "HIGH": [0.65, 0.67, 0.66, ...],
}

p_long_by_trend_regime = {
    "UP": [0.75, 0.78, 0.76, ...],
    "DOWN": [0.60, 0.62, 0.61, ...],
    "NEUTRAL": [0.70, 0.72, 0.71, ...],
}
```

**Verification:**
- Hvis modellen bruker context: `p_long_by_session` skal være forskjellig
- Hvis modellen ignorerer context: `p_long_by_session` skal være lik

---

## Neste Steg

1. ✅ Arkitekturdiagram (dette dokument)
2. ⏳ Implementer `EntryV10CtxHybridTransformer` (LEVERANSE 3A)
3. ⏳ Implementer `EntryV10CtxDataset` (LEVERANSE 3B)
4. ⏳ Oppdater bundle metadata (LEVERANSE 3C)
5. ⏳ Legg til telemetry (LEVERANSE 3D)
6. ⏳ FULLYEAR verification plan (LEVERANSE 3E)

---

## Risikoer & Open Questions

1. **Context embedding dims:** 8 dims per categorical feature - er dette optimalt?
   - **Svar:** Start med 8, kan justeres i STEP 4 basert på performance

2. **Context normalization:** Skal ctx_cont normaliseres med Z-score?
   - **Svar:** Ja, samme som seq/snap features (lagres i bundle metadata)

3. **Backward compatibility:** Hvordan håndtere gamle bundles i live?
   - **Svar:** Flag-basert fallback (allerede implementert i STEP 2)

4. **Training data filtering:** Hvor mange bars blir filtrert bort av hard/soft eligibility?
   - **Svar:** Må måles i training pipeline (forventet: 30-50% filtrert)

---

**Status:** Design complete, ready for implementation



