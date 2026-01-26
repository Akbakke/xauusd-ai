# ENTRY_V10 Hybrid Stack – XGBoost + Transformer

**Date:** 2025-12-29  
**Version:** 1.0  
**Status:** Design Document (No Implementation Yet)

## Intro

ENTRY_V10 er en ny entry-arkitektur som eksplisitt kombinerer XGBoost (snapshot-ekspert) og Transformer (sekvens-hjerne) for å produsere en mer robust `p_long` enn dagens ENTRY_V9.

**Nøkkelforskjell fra V9:**
- **V9:** Bruker ENTTEN XGBoost ELLER Transformer (ikke begge samtidig)
- **V10:** Bruker BEGGE samtidig – XGBoost gir lokal snapshot-ekspertise, Transformer ser sekvenser inkludert XGBoost-signaler over tid, og et ensemble-lag kombinerer dem til `p_long_final`

**Mål:**
- Mer robust `p_long` ved å utnytte XGBoost som "lokal ekspert" (M5 snapshot) og Transformer som "global hjernen" (sekvens-kontekst)
- 100% bakoverkompatibel med dagens SNIPER/FARM-setup
- Kan fases inn uten å ødelegge eksisterende V9

---

## Designmål

- **Beholde dagens rike feature-set:** 13 sequence features + 85 snapshot features – INGEN nedkutting
- **XGBoost skal:**
  - Kjøre på M5 snapshot-features (samme 85 som i dag)
  - Gi `p_long_xgb`, `margin_xgb`, `p_hat_xgb` per bar
  - Være session-routed (EU/US/OVERLAP) som i dag
- **Transformer skal:**
  - Se både sekvens-features OG XGBoost-signaler over tid
  - Gi `p_long_tfm` (egen vurdering basert på sekvens + XGB-kontekst)
  - Være regime-conditioned (session, vol_regime, trend_regime)
- **Ensemble på toppen:**
  - `p_long_final = f(p_long_xgb, p_long_tfm, regime)`
  - Vektet kombinering med regime-adaptive vekter
- **100% bakoverkompatibel med:**
  - `entry_manager.py`
  - SNIPER/FARM policies
  - Eksisterende risk guard/setup
  - Kan leve side-by-side med V9

---

## Arkitektur – High Level

### Data Flow

```
Input: M5 candles (+ H1/H4 der vi har det i dag)
  │
  ├─► Feature Pipeline (samme som ENTRY_V9)
  │   ├─► build_v9_live_base_features()
  │   ├─► 13 seq-features (atr50, ema_slope, etc.)
  │   └─► 85 snapshot-features (_v1_*, CLOSE, etc.)
  │
  ├─► XGBoost Lag (M5 Snapshot Ekspert)
  │   ├─► Input: 85 snapshot-features (per bar)
  │   ├─► Session-routing: EU/US/OVERLAP modeller
  │   ├─► Output per bar:
  │   │   ├─► p_long_xgb
  │   │   ├─► p_short_xgb
  │   │   ├─► margin_xgb
  │   │   └─► p_hat_xgb
  │   └─► Historisk buffer: [p_long_xgb_t-29, ..., p_long_xgb_t]
  │
  ├─► Transformer Lag (ENTRY_V10 HYBRID)
  │   ├─► Sequence Input (30 bars lookback):
  │   │   ├─► 13 seq-features (som i dag)
  │   │   └─► 3 nye XGB-kanaler:
  │   │       ├─► p_long_xgb_seq (historisk p_long_xgb)
  │   │       ├─► margin_xgb_seq (historisk margin_xgb)
  │   │       └─► p_long_xgb_ema_seq (EMA-smoothed p_long_xgb)
  │   ├─► Snapshot Input (siste bar):
  │   │   ├─► 85 snapshot-features (som i dag)
  │   │   └─► 3 nye XGB-now features:
  │   │       ├─► p_long_xgb_now
  │   │       ├─► margin_xgb_now
  │   │       └─► p_hat_xgb_now
  │   ├─► Regime Embeddings:
  │   │   ├─► session_id (0=EU, 1=OVERLAP, 2=US)
  │   │   ├─► vol_regime_id (0=LOW, 1=MID, 2=HIGH, 3=EXTREME)
  │   │   └─► trend_regime_id (0=UP, 1=DOWN, 2=RANGE)
  │   └─► Output:
  │       ├─► p_long_tfm (fra direction_logit)
  │       ├─► p_short_tfm (1 - p_long_tfm)
  │       ├─► margin_tfm
  │       └─► quality_score (optional)
  │
  └─► Ensemble Lag
      ├─► Input:
      │   ├─► p_long_xgb (fra XGBoost lag)
      │   ├─► p_long_tfm (fra Transformer lag)
      │   ├─► regime (session, vol_regime, trend_regime)
      │   └─► ev. confidence scores (margin_xgb, margin_tfm)
      ├─► Ensemble Strategy:
      │   ├─► Option A: Konstant vekting
      │   │   └─► p_long_final = w_t * p_long_tfm + w_x * p_long_xgb
      │   │       (f.eks. w_t=0.7, w_x=0.3)
      │   └─► Option B: Regime-adaptive vekting (meta-MLP)
      │       └─► w_t, w_x = meta_mlp(regime_features)
      │           p_long_final = w_t * p_long_tfm + w_x * p_long_xgb
      └─► Output:
          └─► p_long_final (brukes i entry_gating, samme som V9)
```

### Komponenter

1. **Feature Pipeline:** Identisk med ENTRY_V9 (ingen endringer)
2. **XGBoost Lag:** Session-routed snapshot-modeller (samme som i dag, men output brukes eksplisitt)
3. **Transformer Lag:** Ny ENTRY_V10 Hybrid Transformer som ser både sekvens-features og XGBoost-signaler
4. **Ensemble Lag:** Kombinerer XGBoost og Transformer outputs til `p_long_final`

---

## Feature-bruk – Detaljert

### A) Sequence Features (13 stk + 3 nye XGB-kanaler = 16 totalt)

| Name | Type | Beskrivelse | Brukt av V10 |
|------|------|-------------|--------------|
| `ema20_slope` | float | EMA(20) slope (diff per bar) | Transformer (V10) |
| `ema100_slope` | float | EMA(100) slope | Transformer (V10) |
| `pos_vs_ema200` | float | close / EMA200 - 1 | Transformer (V10) |
| `std50` | float | Rolling std(50) | Transformer (V10) |
| `atr50` | float | Rolling ATR(50) | Transformer (V10) |
| `atr_z` | float | ATR Z-score (ATR50 / ATR200 - 1) | Transformer (V10) |
| `roc20` | float | ROC(20) (rate of change) | Transformer (V10) |
| `roc100` | float | ROC(100) | Transformer (V10) |
| `body_pct` | float | (close-open)/(high-low+1e-8) | Transformer (V10) |
| `wick_asym` | float | (upper_wick - lower_wick)/(range+1e-8) | Transformer (V10) |
| `session_id` | int | 0=EU, 1=OVERLAP, 2=US | Transformer (V10) |
| `atr_regime_id` | int | 0=LOW, 1=MID, 2=HIGH, 3=EXTREME | Transformer (V10) |
| `trend_regime_tf24h` | float | EMA100 slope over 24h / ATR100, normalisert | Transformer (V10) |
| **`p_long_xgb_seq`** | **float** | **Historisk p_long_xgb (30 bars)** | **Transformer (V10) - NY** |
| **`margin_xgb_seq`** | **float** | **Historisk margin_xgb (30 bars)** | **Transformer (V10) - NY** |
| **`p_long_xgb_ema_seq`** | **float** | **EMA-smoothed p_long_xgb (30 bars, EMA=5)** | **Transformer (V10) - NY** |

**Total sequence features for V10:** 16 (13 eksisterende + 3 nye XGB-kanaler)

### B) Snapshot Features (85 stk + 3 nye XGB-now = 88 totalt)

| Feature | Kilde | Beskrivelse | XGB_USED | TFM_USED |
|---------|-------|-------------|----------|----------|
| `CLOSE` | Raw candles | Close price | yes | yes |
| `_v1_atr14` | `gx1/features/basic_v1.py::build_basic_v1()` | ATR(14) | yes | yes |
| `_v1_atr_regime_id` | `gx1/features/basic_v1.py::build_basic_v1()` | ATR regime ID (0=LOW, 1=MID, 2=HIGH) | yes | yes |
| `_v1_r1`, `_v1_r3`, `_v1_r5`, `_v1_r8`, `_v1_r12`, `_v1_r24` | `gx1/features/basic_v1.py::build_basic_v1()` | Lagged returns (1, 3, 5, 8, 12, 24 bars) | yes | yes |
| `_v1_r48_z` | `gx1/features/basic_v1.py::build_basic_v1()` | Return Z-score (48-bar window) | yes | yes |
| `_v1_ema_diff` | `gx1/features/basic_v1.py::build_basic_v1()` | EMA(12) - EMA(26) | yes | yes |
| `_v1_rsi14`, `_v1_rsi14_z` | `gx1/features/basic_v1.py::build_basic_v1()` | RSI(14) og RSI Z-score | yes | yes |
| `_v1_rsi2` | `gx1/features/basic_v1.py::build_basic_v1()` | RSI(2) | yes | yes |
| `_v1_rsi2_gt_rsi14` | `gx1/features/basic_v1.py::build_basic_v1()` | RSI2 > RSI14 (bool) | yes | yes |
| `_v1_vwap_drift48` | `gx1/features/basic_v1.py::build_basic_v1()` | Close - VWAP(48) | yes | yes |
| `_v1_pk_sigma20` | `gx1/features/basic_v1.py::build_basic_v1()` | Parkinson volatility estimator | yes | yes |
| `_v1_body_tr`, `_v1_upper_tr`, `_v1_lower_tr` | `gx1/features/basic_v1.py::build_basic_v1()` | Candle body/upper/lower wick ratios | yes | yes |
| `_v1_wick_imbalance` | `gx1/features/basic_v1.py::build_basic_v1()` | Wick imbalance | yes | yes |
| `_v1_session_tag_EU`, `_v1_session_tag_US`, `_v1_session_tag_OVERLAP` | `gx1/features/runtime_v9.py::build_v9_live_base_features()` | Session one-hot encoding | yes | yes |
| `_v1_is_EU`, `_v1_is_US` | `gx1/features/runtime_v9.py::build_v9_live_base_features()` | Session binary flags | yes | yes |
| `_v1_tod_cos`, `_v1_tod_sin` | `gx1/features/basic_v1.py::build_basic_v1()` | Time-of-day (cosine/sine encoding) | yes | yes |
| `_v1_spread_p`, `_v1_spread_z` | `gx1/features/basic_v1.py::build_basic_v1()` | Spread percentage og Z-score | yes | yes |
| `_v1_cost_bps_est`, `_v1_cost_bps_dyn` | `gx1/features/basic_v1.py::build_basic_v1()` | Estimated trading costs | yes | yes |
| `_v1_slip_bps` | `gx1/features/basic_v1.py::build_basic_v1()` | Estimated slippage | yes | yes |
| `_v1_bb_bandwidth_delta_10` | `gx1/features/basic_v1.py::build_basic_v1()` | Bollinger Band width delta | yes | yes |
| `_v1_bb_squeeze_20_2` | `gx1/features/basic_v1.py::build_basic_v1()` | Bollinger Band squeeze indicator | yes | yes |
| `_v1_close_ema_slope_3` | `gx1/features/basic_v1.py::build_basic_v1()` | Close EMA slope (3 vs 6) | yes | yes |
| `_v1_kama_slope_30` | `gx1/features/basic_v1.py::build_basic_v1()` | KAMA slope | yes | yes |
| `_v1_tema_slope_20` | `gx1/features/basic_v1.py::build_basic_v1()` | TEMA slope | yes | yes |
| `_v1_range_adr`, `_v1_range_z` | `gx1/features/basic_v1.py::build_basic_v1()` | Range vs ADR, Range Z-score | yes | yes |
| `_v1_range_comp_20_100` | `gx1/features/basic_v1.py::build_basic_v1()` | Range comparison (20 vs 100) | yes | yes |
| `_v1_ret_ema_diff_2_5` | `gx1/features/basic_v1.py::build_basic_v1()` | Return EMA diff (2 vs 5) | yes | yes |
| `_v1_ret_ema_ratio_5_34` | `gx1/features/basic_v1.py::build_basic_v1()` | Return EMA ratio (5 vs 34) | yes | yes |
| `_v1_int_*` | `gx1/features/basic_v1.py::build_basic_v1()` | Inter-timeframe features (H1, H4) | yes | yes |
| `_v1h1_*`, `_v1h4_*` | `gx1/features/basic_v1.py::build_basic_v1()` | H1/H4 timeframe features | yes | yes |
| `atr`, `mid`, `range`, `ret_1`, `ret_5`, `ret_20` | `gx1/features/runtime_v9.py::build_v9_live_base_features()` | Derived OHLC features | yes | yes |
| `rvol_20`, `rvol_60`, `vol_ratio` | `gx1/features/runtime_v9.py::build_v9_live_base_features()` | Realized volatility features | yes | yes |
| `prob_long`, `prob_short`, `prob_neutral`, `side` | `gx1/features/runtime_v9.py::build_v9_live_base_features()` | Placeholder (defaults to 0.0) | yes | yes |
| `brain_risk_score` | `gx1/features/runtime_v9.py::build_v9_live_base_features()` | Placeholder (defaults to 0.0) | yes | yes |
| **`p_long_xgb_now`** | **XGBoost output** | **p_long_xgb på siste bar** | **N/A** | **yes - NY** |
| **`margin_xgb_now`** | **XGBoost output** | **margin_xgb på siste bar** | **N/A** | **yes - NY** |
| **`p_hat_xgb_now`** | **XGBoost output** | **p_hat_xgb på siste bar** | **N/A** | **yes - NY** |

**Total snapshot features for V10:** 88 (85 eksisterende + 3 nye XGB-now features)

**Note:** Alle 85 eksisterende snapshot-features brukes av både XGBoost (for å generere p_long_xgb) og Transformer (som input til snapshot-encoder).

---

## Treningsoppsett

### Data

- **Universe:** FULLYEAR 2020–2025 (samme som ENTRY_V9) eller minst 2–3 års historikk
- **Targets:** `y = "should_open_long"` (samme label som for V9) med forward-returns/labeling vi allerede bruker
- **Time split:** Train/val/test split basert på datoer (f.eks. 2020-2023 train, 2024 val, 2025 test)

### Stage 1: Tren XGBoost M5-snapshot modeller

**Mål:** Tren session-routed XGBoost-modeller som gir `p_long_xgb` per bar.

**Input:**
- 85 snapshot-features (samme som i dag)
- Session-routing: Separate modeller for EU/US/OVERLAP

**Output:**
- `p_long_xgb`: Sannsynlighet for LONG (fra `predict_proba()`)
- `p_short_xgb`: Sannsynlighet for SHORT
- `margin_xgb`: `abs(p_long_xgb - p_short_xgb)`
- `p_hat_xgb`: `max(p_long_xgb, p_short_xgb)`

**Hyperparameters (forslag):**
- `max_depth`: 6-8
- `n_estimators`: 200-500
- `learning_rate`: 0.01-0.05
- `subsample`: 0.8-0.9
- `colsample_bytree`: 0.8-0.9
- `min_child_weight`: 3-5

**Validering:**
- Per-session metrics (EU/OVERLAP/US)
- Per-regime metrics (trend × vol)
- AUC, precision, recall, F1

### Stage 2: Generer "XGB-annotert" dataset for Transformer

**Mål:** Kjør XGBoost over hele historikken og lagre XGBoost-signaler som nye features.

**Prosess:**
1. **Kjør XGBoost over hele datasettet:**
   - For hver bar, hent snapshot-features
   - Kjør session-routed XGBoost-modell
   - Lagre: `p_long_xgb_t`, `margin_xgb_t`, `p_hat_xgb_t`

2. **Beregn glidende varianter:**
   - `p_long_xgb_ema_5`: EMA(5) over `p_long_xgb`
   - `p_long_xgb_ema_10`: EMA(10) over `p_long_xgb`
   - `margin_xgb_ema_5`: EMA(5) over `margin_xgb`

3. **Bygg sekvenser (30 bars lookback):**
   - **Sequence features:** [13 seq-features + 3 XGB-kanaler]
     - `p_long_xgb_seq`: [p_long_xgb_t-29, ..., p_long_xgb_t]
     - `margin_xgb_seq`: [margin_xgb_t-29, ..., margin_xgb_t]
     - `p_long_xgb_ema_seq`: [p_long_xgb_ema_5_t-29, ..., p_long_xgb_ema_5_t]
   - **Snapshot features:** [85 snap-features + 3 XGB-now features]
     - `p_long_xgb_now`: p_long_xgb_t
     - `margin_xgb_now`: margin_xgb_t
     - `p_hat_xgb_now`: p_hat_xgb_t
   - **Label:** `y = "should_open_long"` (samme som V9)

**Output:**
- Parquet-dataset med kolonner:
  - `seq_features`: [30, 16] (13 seq + 3 XGB-kanaler)
  - `snap_features`: [88] (85 snap + 3 XGB-now)
  - `session_id`, `vol_regime_id`, `trend_regime_id`
  - `y_direction`, `y_early_move`, `y_quality_score` (labels)

### Stage 3: Tren ENTRY_V10 Transformer

**Arkitektur:**
- **d_model:** 128 (default, kan justeres)
- **n_heads:** 4-8 (forslag: start med 4, øk hvis GPU-tilgjengelig)
- **num_layers:** 4-6 (forslag: start med 4, øk hvis GPU-tilgjengelig)
- **dim_feedforward:** 512 (d_model * 4)
- **dropout:** 0.05-0.1 (forslag: start med 0.05)
- **pooling:** "mean" (sequence pooling)
- **causal:** True (causal masking for sequence)

**Input:**
- `seq_x`: `[batch, 30, 16]` (13 seq-features + 3 XGB-kanaler)
- `snap_x`: `[batch, 88]` (85 snap-features + 3 XGB-now features)
- `session_id`: `[batch, 1]` (0=EU, 1=OVERLAP, 2=US)
- `vol_regime_id`: `[batch, 1]` (0=LOW, 1=MID, 2=HIGH, 3=EXTREME)
- `trend_regime_id`: `[batch, 1]` (0=UP, 1=DOWN, 2=RANGE)

**Output:**
- `direction_logit`: `[batch, 1]` → `p_long_tfm = sigmoid(direction_logit)`
- `early_move_logit`: `[batch, 1]` → `p_early = sigmoid(early_move_logit)` (optional)
- `quality_score`: `[batch, 1]` → `quality = tanh(quality_score)` (optional)

**Loss:**
- **Hoved-head:** Binary cross-entropy på `direction_logit` (vekt: 1.0)
- **Auxiliary heads (optional):**
  - `early_move_logit`: Binary cross-entropy (vekt: 0.2)
  - `quality_score`: MSE (vekt: 0.2)

**Validering:**
- Per-session metrics (EU/OVERLAP/US)
- Per-regime metrics (trend × vol)
- Fokus: Avoid regime-collapse (neutral/high osv.)
- AUC, precision, recall, F1 per regime

### Stage 4: Tren Ensemble Meta-MLP (Optional)

**Mål:** Tren en liten MLP som bestemmer vekter `w_t`, `w_x` basert på regime.

**Input:**
- `p_long_xgb`: XGBoost output
- `p_long_tfm`: Transformer output
- `margin_xgb`: XGBoost margin
- `margin_tfm`: Transformer margin
- `session_id`: Session (one-hot)
- `vol_regime_id`: Vol regime (one-hot)
- `trend_regime_id`: Trend regime (one-hot)

**Output:**
- `w_t`: Vekt for Transformer (0-1)
- `w_x`: Vekt for XGBoost (0-1, = 1 - w_t)

**Arkitektur:**
- 2-3 lag MLP
- Hidden dims: [64, 32] eller [32, 16]
- Output: 2 logits → softmax → [w_t, w_x]

**Alternativ:** Konstant vekting (f.eks. w_t=0.7, w_x=0.3) hvis meta-MLP ikke gir forbedring.

---

## Runtime-integrasjon (Plan, ikke kode)

### Ny modul-struktur

**Ny modul:**
- `gx1/models/entry_v10/entry_v10_hybrid_transformer.py`
  - `EntryV10HybridTransformer`: Transformer-klasse som tar seq+snap+XGB-features
  - `EntryV10Ensemble`: Ensemble-lag som kombinerer XGBoost og Transformer outputs

**Ny policy:**
- `gx1/policy/entry_v10_policy_sniper_hybrid.py`
  - Wrapper rundt `entry_v9_policy_sniper` som bruker `p_long_final` i stedet for `p_long`

**Ny config:**
- `gx1/configs/policies/sniper_snapshot/2026_SNIPER_V10/ENTRY_V10_SNIPER_HYBRID.yaml`
  - Entry config for V10 HYBRID
- `gx1/configs/policies/sniper_snapshot/2026_SNIPER_V10/GX1_V12_SNIPER_V10_CANARY.yaml`
  - Main config for V10 CANARY

### Kjørerekkefølge i runtime

**Pseudokode-flyt:**

```python
# 1. Build features (samme pipeline som V9)
entry_bundle = build_live_entry_features(candles)
df_feats, seq_features, snap_features = build_v9_runtime_features(...)

# 2. Hent p_long_xgb fra XGBoost snapshot-modell
xgb_pred = self._predict_entry_xgb(entry_bundle, timestamp)
p_long_xgb = xgb_pred.prob_long
margin_xgb = xgb_pred.margin
p_hat_xgb = xgb_pred.p_hat

# 3. Bygg sekvens med XGB-kanaler
# 3a. Hent historisk XGB-output (fra buffer eller re-compute)
xgb_history = self._get_xgb_history(lookback=30)  # [p_long_xgb_t-29, ..., p_long_xgb_t]
margin_history = self._get_margin_history(lookback=30)
xgb_ema_history = self._compute_ema(xgb_history, window=5)

# 3b. Kombiner seq-features med XGB-kanaler
seq_x_v10 = np.concatenate([
    df_feats[seq_features].values[-30:],  # [30, 13]
    xgb_history.reshape(-1, 1),            # [30, 1]
    margin_history.reshape(-1, 1),         # [30, 1]
    xgb_ema_history.reshape(-1, 1)         # [30, 1]
], axis=1)  # [30, 16]

# 3c. Kombiner snap-features med XGB-now
snap_x_v10 = np.concatenate([
    df_feats[snap_features].values[-1:],   # [1, 85]
    np.array([[p_long_xgb, margin_xgb, p_hat_xgb]])  # [1, 3]
], axis=1)  # [1, 88]

# 4. Kjør V10-transformer → p_long_tfm
v10_pred = self._predict_entry_v10_hybrid(
    seq_x=seq_x_v10,
    snap_x=snap_x_v10,
    session_id=session_id,
    vol_regime_id=vol_regime_id,
    trend_regime_id=trend_regime_id
)
p_long_tfm = v10_pred.prob_long
margin_tfm = v10_pred.margin

# 5. Kombiner til p_long_final i ensemble-laget
p_long_final = self._ensemble_predict(
    p_long_xgb=p_long_xgb,
    p_long_tfm=p_long_tfm,
    margin_xgb=margin_xgb,
    margin_tfm=margin_tfm,
    session_id=session_id,
    vol_regime_id=vol_regime_id,
    trend_regime_id=trend_regime_id
)

# 6. Send p_long_final inn i entry_policy (SNIPER/FARM) på samme måte som V9
entry_prediction = EntryPrediction(
    prob_long=p_long_final,
    prob_short=1.0 - p_long_final,
    p_hat=max(p_long_final, 1.0 - p_long_final),
    margin=abs(p_long_final - (1.0 - p_long_final))
)

# 7. Entry policy evaluering (samme som V9)
df_result = apply_entry_v9_policy_sniper(
    df_signals=current_row,
    config=config,
    meta_model=meta_model,
    meta_feature_cols=meta_feature_cols
)
```

### Bakoverkompatibilitet

**V9 og V10 skal kunne leve side-by-side:**

**Config-styring:**
```yaml
entry_models:
  v9:
    enabled: false  # Deaktiver V9
  v10:
    enabled: true   # Aktiver V10 HYBRID
  xgb:
    enabled: true   # XGBoost må være aktiv for V10
```

**Fallback-strategi:**
1. Hvis V10 ikke er tilgjengelig → fallback til V9
2. Hvis V9 ikke er tilgjengelig → fallback til XGBoost only
3. Hvis XGBoost ikke er tilgjengelig → error (krever minst én modell)

**EntryPrediction interface:**
- V10 returnerer samme `EntryPrediction`-objekt som V9
- Entry policies (SNIPER/FARM) ser ingen forskjell
- `p_long_final` brukes på samme måte som `p_long` i V9

---

## Videre arbeid / TODO-list

### Fase 1: Modell-trening

- [ ] Opprette modell-mapper for ENTRY_V10 (`models/entry_v10/...`)
- [ ] Implementere XGB M5 snapshot-trener (`train_entry_xgb_v10.py`)
  - Session-routed modeller (EU/US/OVERLAP)
  - Hyperparameter tuning (Optuna eller grid search)
  - Validering per session og regime
- [ ] Implementere dataset-builder som lager "XGB-annotert" sekvens-datasett for Transformer
  - Kjør XGBoost over hele historikken
  - Beregn glidende varianter (EMA)
  - Bygg sekvenser med XGB-kanaler
  - Lagre som Parquet-dataset
- [ ] Implementere EntryV10HybridTransformer-klassen
  - Arkitektur: SeqTransformerEncoder + SnapshotEncoder + RegimeEmbeddings
  - Multi-task heads (direction, early_move, quality)
  - Loss-funksjon med vektede auxiliary heads
- [ ] Implementere treningsloop for V10 Transformer
  - DataLoader med EntryV10Dataset
  - Training loop med validation
  - Checkpointing og early stopping
  - Per-session og per-regime metrics

### Fase 2: Ensemble og runtime

- [ ] Implementere ensemble-logikk i runtime
  - Konstant vekting (w_t=0.7, w_x=0.3) som baseline
  - Optional: Regime-adaptive vekting (meta-MLP)
- [ ] Implementere XGBoost history buffer i runtime
  - Buffer for p_long_xgb, margin_xgb over 30 bars
  - EMA-beregning for glidende varianter
- [ ] Integrere V10 i `oanda_demo_runner.py`
  - Ny metode: `_predict_entry_v10_hybrid()`
  - Ensemble-lag: `_ensemble_predict()`
  - Fallback-strategi (V10 → V9 → XGBoost)
- [ ] Oppdatere `entry_manager.py` for V10-støtte
  - Config-lesing for V10
  - Feature-building med XGB-kanaler
  - EntryPrediction med p_long_final

### Fase 3: Configs og testing

- [ ] Lage nye SNIPER V10 configs (canary)
  - `ENTRY_V10_SNIPER_HYBRID.yaml`
  - `GX1_V12_SNIPER_V10_CANARY.yaml`
- [ ] Full-year replay-test av V10 vs V9 vs P4.1
  - FULLYEAR 2025 replay for alle tre
  - Sammenligning: trades, win rate, PnL, regime-distribusjon
  - Tail-risk analyse
- [ ] Regime-spesifikk analyse
  - Per-session performance (EU/OVERLAP/US)
  - Per-regime performance (trend × vol)
  - Identifiser hvor V10 gir størst forbedring

### Fase 4: Dokumentasjon og rollout

- [ ] Oppdatere `ENTRY_STACK_PLONG_DETAILED_OVERVIEW.md` med V10-seksjon
- [ ] Lage trenings-rapport for V10 (hyperparameters, metrics, validering)
- [ ] Lage replay-sammenligningsrapport (V10 vs V9 vs P4.1)
- [ ] Dokumentere ensemble-strategi (konstant vs adaptive vekting)
- [ ] Planlegge canary-rollout for V10 (fase inn gradvis)

---

## Konklusjon

ENTRY_V10 HYBRID er designet for å kombinere styrkene til XGBoost (lokal snapshot-ekspertise) og Transformer (global sekvens-kontekst) i en eksplisitt ensemble-arkitektur. Dette gir en mer robust `p_long` enn dagens V9, som kun bruker én modell om gangen.

**Nøkkelfordeler:**
- **Robusthet:** Ensemble av to uavhengige modeller reduserer risiko for modell-feil
- **Kontekst:** Transformer ser XGBoost-signaler over tid, kan lære når XGBoost er mest pålitelig
- **Bakoverkompatibilitet:** Kan fases inn uten å ødelegge eksisterende V9
- **Fleksibilitet:** Kan justere vekting basert på regime (adaptive ensemble)

**Neste steg:** Implementere Fase 1 (modell-trening) og validere at V10 gir forbedret performance over V9.

---

**End of Document**

