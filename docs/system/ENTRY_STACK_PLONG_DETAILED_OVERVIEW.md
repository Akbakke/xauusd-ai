# ENTRY Stack p_long Detailed Overview

**Date:** 2025-12-29  
**Version:** 1.0

## Intro

ENTRY-stacken i GX1 er ansvarlig for å generere `p_long` (sannsynlighet for LONG-entry) basert på markedsdata. Systemet bruker to parallelle modell-arkitekturer:

1. **XGBoost-modeller** (session-routed): Tradisjonelle gradient boosting-modeller trent per session (EU/US/OVERLAP)
2. **Transformer-modell** (ENTRY_V9): PyTorch-basert transformer-arkitektur med multi-task learning og regime-conditioning

Begge modellene produserer `p_long` som deretter brukes i entry-gating for å bestemme om en trade skal åpnes. Systemet velger automatisk hvilken modell som skal brukes basert på konfigurasjon.

---

## 1. Oversikt over filer & modeller

### 1.1 Entry Policy Moduler

| Fil | Rolle | Modelltype |
|-----|-------|------------|
| `gx1/policy/entry_v9_policy_sniper.py` | SNIPER entry policy (wrapper rundt FARM_V2B) | Policy (bruker XGB/Transformer) |
| `gx1/policy/entry_v9_policy_farm_v2b.py` | FARM entry policy (p_long-driven) | Policy (bruker XGB/Transformer) |
| `gx1/policy/farm_meta_features.py` | Meta-model feature builder (p_profitable) | Feature builder (ikke p_long) |

### 1.2 Entry Models

| Fil/Path | Rolle | Modelltype |
|----------|-------|------------|
| `gx1/models/entry_v9/nextgen_2020_2025_clean/model.pt` | ENTRY_V9 Transformer modell | PyTorch Transformer |
| `gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json` | Feature metadata (seq + snap) | Metadata |
| `gx1/models/entry_v9/nextgen_2020_2025_clean/seq_scaler.joblib` | Sequence feature scaler | Scaler |
| `gx1/models/entry_v9/nextgen_2020_2025_clean/snap_scaler.joblib` | Snapshot feature scaler | Scaler |
| Session-routed XGBoost models (EU/US/OVERLAP) | XGBoost entry models | XGBoost (joblib) |
| `gx1/models/farm_entry_meta/baseline_model.pkl` | Meta-model for p_profitable | Scikit-learn (ikke p_long) |

### 1.3 Feature Building

| Fil | Rolle | Modelltype |
|-----|-------|------------|
| `gx1/features/runtime_v9.py` | V9 runtime feature pipeline | Feature builder |
| `gx1/features/basic_v1.py` | Basic V1 tabular features (_v1_*) | Feature builder |
| `gx1/seq/sequence_features.py` | Sequence features (atr50, ema_slope, etc.) | Feature builder |
| `gx1/execution/live_features.py` | Live feature bundle builder | Feature builder |

### 1.4 Entry Manager & Runner

| Fil | Rolle | Modelltype |
|-----|-------|------------|
| `gx1/execution/entry_manager.py` | Entry decision manager | Policy orchestrator |
| `gx1/execution/oanda_demo_runner.py` | Main runner (model loading, prediction) | Runtime orchestrator |

### 1.5 Configs

| Fil | Rolle | Modelltype |
|-----|-------|------------|
| `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/ENTRY_V9_SNIPER_LONDON_NY.yaml` | SNIPER entry config | Config |
| `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID/ENTRY_V9_FARM_V2B_PROD.yaml` | FARM entry config | Config |

---

## 2. Feature-kart

### 2.1 Rå/Semi-rå Features (Input til feature pipeline)

| Feature | Kilde | Beskrivelse |
|---------|-------|-------------|
| `open`, `high`, `low`, `close` | OHLC candles | Rå prisdata |
| `volume` | Market data | Handelsvolum |
| `spread_bps` | Bid/ask spread | Spread i basispunkter |
| `atr` | Beregnet fra True Range | Average True Range (14-period) |
| `session` | Inferert fra timestamp | Trading session (EU/OVERLAP/US/ASIA) |
| `ts` | Timestamp | Datetime index |

### 2.2 Sequence Features (13 features, brukt av Transformer)

Sequence features er features som beregnes over en sekvens av bars (typisk 30 bars lookback).

| Feature | Kilde | Beskrivelse | Brukt i |
|---------|-------|-------------|---------|
| `ema20_slope` | `gx1/seq/sequence_features.py::build_sequence_features()` | EMA(20) slope (diff per bar) | Transformer |
| `ema100_slope` | `gx1/seq/sequence_features.py::build_sequence_features()` | EMA(100) slope | Transformer |
| `pos_vs_ema200` | `gx1/seq/sequence_features.py::build_sequence_features()` | close / EMA200 - 1 | Transformer |
| `std50` | `gx1/seq/sequence_features.py::build_sequence_features()` | Rolling std(50) | Transformer |
| `atr50` | `gx1/seq/sequence_features.py::build_sequence_features()` | Rolling ATR(50) | Transformer |
| `atr_z` | `gx1/seq/sequence_features.py::build_sequence_features()` | ATR Z-score (ATR50 / ATR200 - 1) | Transformer |
| `roc20` | `gx1/seq/sequence_features.py::build_sequence_features()` | ROC(20) (rate of change) | Transformer |
| `roc100` | `gx1/seq/sequence_features.py::build_sequence_features()` | ROC(100) | Transformer |
| `body_pct` | `gx1/seq/sequence_features.py::build_sequence_features()` | (close-open)/(high-low+1e-8) | Transformer |
| `wick_asym` | `gx1/seq/sequence_features.py::build_sequence_features()` | (upper_wick - lower_wick)/(range+1e-8) | Transformer |
| `session_id` | `gx1/seq/sequence_features.py::build_sequence_features()` | int (0=EU, 1=OVERLAP, 2=US) | Transformer |
| `atr_regime_id` | `gx1/seq/sequence_features.py::build_sequence_features()` | int (0=LOW, 1=MID, 2=HIGH, 3=EXTREME) | Transformer |
| `trend_regime_tf24h` | `gx1/seq/sequence_features.py::build_sequence_features()` | EMA100 slope over 24h / ATR100, normalisert | Transformer |

### 2.3 Snapshot Features (85 features, brukt av både XGBoost og Transformer)

Snapshot features er tabulære features beregnet på siste bar (ikke sekvens).

| Feature | Kilde | Beskrivelse | Brukt i |
|---------|-------|-------------|---------|
| `CLOSE` | Raw candles | Close price | XGBoost, Transformer |
| `_v1_atr14` | `gx1/features/basic_v1.py::build_basic_v1()` | ATR(14) | XGBoost, Transformer |
| `_v1_atr_regime_id` | `gx1/features/basic_v1.py::build_basic_v1()` | ATR regime ID (0=LOW, 1=MID, 2=HIGH) | XGBoost, Transformer |
| `_v1_r1`, `_v1_r3`, `_v1_r5`, `_v1_r8`, `_v1_r12`, `_v1_r24` | `gx1/features/basic_v1.py::build_basic_v1()` | Lagged returns (1, 3, 5, 8, 12, 24 bars) | XGBoost, Transformer |
| `_v1_r48_z` | `gx1/features/basic_v1.py::build_basic_v1()` | Return Z-score (48-bar window) | XGBoost, Transformer |
| `_v1_ema_diff` | `gx1/features/basic_v1.py::build_basic_v1()` | EMA(12) - EMA(26) | XGBoost, Transformer |
| `_v1_rsi14`, `_v1_rsi14_z` | `gx1/features/basic_v1.py::build_basic_v1()` | RSI(14) og RSI Z-score | XGBoost, Transformer |
| `_v1_rsi2` | `gx1/features/basic_v1.py::build_basic_v1()` | RSI(2) | XGBoost, Transformer |
| `_v1_rsi2_gt_rsi14` | `gx1/features/basic_v1.py::build_basic_v1()` | RSI2 > RSI14 (bool) | XGBoost, Transformer |
| `_v1_vwap_drift48` | `gx1/features/basic_v1.py::build_basic_v1()` | Close - VWAP(48) | XGBoost, Transformer |
| `_v1_pk_sigma20` | `gx1/features/basic_v1.py::build_basic_v1()` | Parkinson volatility estimator | XGBoost, Transformer |
| `_v1_body_tr`, `_v1_upper_tr`, `_v1_lower_tr` | `gx1/features/basic_v1.py::build_basic_v1()` | Candle body/upper/lower wick ratios | XGBoost, Transformer |
| `_v1_wick_imbalance` | `gx1/features/basic_v1.py::build_basic_v1()` | Wick imbalance | XGBoost, Transformer |
| `_v1_session_tag_EU`, `_v1_session_tag_US`, `_v1_session_tag_OVERLAP` | `gx1/features/runtime_v9.py::build_v9_live_base_features()` | Session one-hot encoding | XGBoost, Transformer |
| `_v1_is_EU`, `_v1_is_US` | `gx1/features/runtime_v9.py::build_v9_live_base_features()` | Session binary flags | XGBoost, Transformer |
| `_v1_tod_cos`, `_v1_tod_sin` | `gx1/features/basic_v1.py::build_basic_v1()` | Time-of-day (cosine/sine encoding) | XGBoost, Transformer |
| `_v1_spread_p`, `_v1_spread_z` | `gx1/features/basic_v1.py::build_basic_v1()` | Spread percentage og Z-score | XGBoost, Transformer |
| `_v1_cost_bps_est`, `_v1_cost_bps_dyn` | `gx1/features/basic_v1.py::build_basic_v1()` | Estimated trading costs | XGBoost, Transformer |
| `_v1_slip_bps` | `gx1/features/basic_v1.py::build_basic_v1()` | Estimated slippage | XGBoost, Transformer |
| `_v1_bb_bandwidth_delta_10` | `gx1/features/basic_v1.py::build_basic_v1()` | Bollinger Band width delta | XGBoost, Transformer |
| `_v1_bb_squeeze_20_2` | `gx1/features/basic_v1.py::build_basic_v1()` | Bollinger Band squeeze indicator | XGBoost, Transformer |
| `_v1_close_ema_slope_3` | `gx1/features/basic_v1.py::build_basic_v1()` | Close EMA slope (3 vs 6) | XGBoost, Transformer |
| `_v1_kama_slope_30` | `gx1/features/basic_v1.py::build_basic_v1()` | KAMA slope | XGBoost, Transformer |
| `_v1_tema_slope_20` | `gx1/features/basic_v1.py::build_basic_v1()` | TEMA slope | XGBoost, Transformer |
| `_v1_range_adr`, `_v1_range_z` | `gx1/features/basic_v1.py::build_basic_v1()` | Range vs ADR, Range Z-score | XGBoost, Transformer |
| `_v1_range_comp_20_100` | `gx1/features/basic_v1.py::build_basic_v1()` | Range comparison (20 vs 100) | XGBoost, Transformer |
| `_v1_ret_ema_diff_2_5` | `gx1/features/basic_v1.py::build_basic_v1()` | Return EMA diff (2 vs 5) | XGBoost, Transformer |
| `_v1_ret_ema_ratio_5_34` | `gx1/features/basic_v1.py::build_basic_v1()` | Return EMA ratio (5 vs 34) | XGBoost, Transformer |
| `_v1_int_*` | `gx1/features/basic_v1.py::build_basic_v1()` | Inter-timeframe features (H1, H4) | XGBoost, Transformer |
| `_v1h1_*`, `_v1h4_*` | `gx1/features/basic_v1.py::build_basic_v1()` | H1/H4 timeframe features | XGBoost, Transformer |
| `atr`, `mid`, `range`, `ret_1`, `ret_5`, `ret_20` | `gx1/features/runtime_v9.py::build_v9_live_base_features()` | Derived OHLC features | XGBoost, Transformer |
| `rvol_20`, `rvol_60`, `vol_ratio` | `gx1/features/runtime_v9.py::build_v9_live_base_features()` | Realized volatility features | XGBoost, Transformer |
| `prob_long`, `prob_short`, `prob_neutral`, `side` | `gx1/features/runtime_v9.py::build_v9_live_base_features()` | Placeholder (defaults to 0.0) | XGBoost, Transformer |
| `brain_risk_score` | `gx1/features/runtime_v9.py::build_v9_live_base_features()` | Placeholder (defaults to 0.0) | XGBoost, Transformer |

**Total:** 13 sequence features + 85 snapshot features = 98 features

---

## 3. XGBoost p_long-pipeline

### 3.1 Modell-lokasjon og lasting

**Modell-filer:**
- Session-routed XGBoost models (EU/US/OVERLAP) lastes via `gx1/execution/oanda_demo_runner.py::load_entry_models()`
- Modellene er joblib-filer (XGBoost-klassifiserere)
- Metadata (feature columns) lastes fra JSON-fil

**Kode-lokasjon:**
```python
# gx1/execution/oanda_demo_runner.py::load_entry_models()
# Linje 445-523
```

### 3.2 Feature-preprosessering før XGBoost

**Pipeline:**
1. **Build base features:** `build_v9_live_base_features()` → `build_basic_v1()` + `build_sequence_features()`
2. **Subset til snapshot features:** Kun snapshot features (85 features) brukes av XGBoost
3. **Scaling:** Snapshot scaler (`snap_scaler.joblib`) brukes hvis tilgjengelig
4. **NaN-håndtering:** `np.nan_to_num()` med `fillna_value=0.0`
5. **Type conversion:** Alle features konverteres til `np.float32`

**Kode-lokasjon:**
```python
# gx1/features/runtime_v9.py::build_v9_runtime_features()
# Linje 364-500
```

### 3.3 XGBoost Prediction

**Input:**
- Snapshot features (85 features) i riktig rekkefølge (fra metadata)
- Session-routing: Modell velges basert på `session_tag` (EU/US/OVERLAP)

**Prediction:**
```python
# gx1/execution/oanda_demo_runner.py::_predict_entry()
# Linje 5520-5697

# 1. Resolve session key
session_key = self._resolve_session_key(session_tag)  # "EU", "US", eller "OVERLAP"

# 2. Get model for session
model = self.entry_model_bundle.models.get(session_key)

# 3. Align features to model's expected columns
aligned = feat_df.reindex(columns=feature_cols, fill_value=0.0)

# 4. Predict probabilities
probs = model.predict_proba(row)  # Shape: (1, n_classes)

# 5. Extract probabilities using model.classes_
prediction = extract_entry_probabilities(probs, classes)
```

**Output:**
- `prob_long`: Sannsynlighet for LONG (fra `predict_proba()`)
- `prob_short`: Sannsynlighet for SHORT (fra `predict_proba()`)
- `prob_neutral`: Sannsynlighet for NEUTRAL (hvis 3-class, ellers 0.0)
- `p_hat`: `max(prob_long, prob_short)`
- `margin`: `abs(prob_long - prob_short)`

**Kode-lokasjon:**
```python
# gx1/execution/oanda_demo_runner.py::extract_entry_probabilities()
# Linje 526-621
```

### 3.4 Temperature Scaling (valgfritt)

XGBoost-predictions kan skaleres med temperature:
```python
# gx1/execution/oanda_demo_runner.py::_predict_entry()
# Linje 5650-5695

T = float(temp_map.get(session_key, 1.0))  # Default: T=1.0 (no scaling)
if T != 1.0:
    prediction.prob_long = self._apply_temperature(prediction.prob_long, T)
    prediction.prob_short = self._apply_temperature(prediction.prob_short, T)
```

**Note:** Temperature scaling er for øyeblikket deaktivert (T=1.0 hardcoded).

---

## 4. Transformer p_long-pipeline

### 4.1 Modell-lokasjon og lasting

**Modell-filer:**
- `gx1/models/entry_v9/nextgen_2020_2025_clean/model.pt` (PyTorch checkpoint)
- `gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json` (feature metadata)
- `gx1/models/entry_v9/nextgen_2020_2025_clean/seq_scaler.joblib` (sequence scaler)
- `gx1/models/entry_v9/nextgen_2020_2025_clean/snap_scaler.joblib` (snapshot scaler)

**Arkitektur:**
- **SeqTransformerEncoder:** Transformer encoder for sequence features (13 features, 30 bars lookback)
- **SnapshotEncoder:** MLP encoder for snapshot features (85 features)
- **Regime embeddings:** Session, vol_regime, trend_regime embeddings
- **Multi-task heads:**
  - `direction_logit`: Binary classification (LONG vs not LONG)
  - `early_move_logit`: Early momentum prediction
  - `quality_score`: Trade quality score

**Kode-lokasjon:**
```python
# gx1/models/entry_v9/entry_v9_transformer.py
# Linje 1-453
```

### 4.2 Feature-preprosessering før Transformer

**Pipeline:**
1. **Build base features:** `build_v9_live_base_features()` → `build_basic_v1()` + `build_sequence_features()`
2. **Split seq/snap:** Sequence features (13) og snapshot features (85) separeres
3. **Scaling:**
   - Sequence features: `seq_scaler.joblib` (RobustScaler)
   - Snapshot features: `snap_scaler.joblib` (RobustScaler)
4. **Sequence extraction:** Last 30 bars (lookback) for sequence features
5. **Snapshot extraction:** Last bar only for snapshot features

**Kode-lokasjon:**
```python
# gx1/execution/oanda_demo_runner.py::_predict_entry_v9()
# Linje 5699-5848

# 1. Build V9 runtime features
df_feats, seq_features, snap_features = build_v9_runtime_features(
    df_raw=candles,
    feature_meta_path=self.entry_v9_feature_meta_path,
    seq_scaler_path=seq_scaler_path,
    snap_scaler_path=snap_scaler_path,
)

# 2. Extract sequence features (last 30 bars)
seq_X = df_feats[seq_features].values[-lookback:].astype(np.float32)  # [30, 13]

# 3. Extract snapshot features (last bar)
snap_X = df_feats[snap_features].values[-1:].astype(np.float32)  # [1, 85]

# 4. Extract regime IDs
session_id = int(current_bar["session_id"])  # 0=EU, 1=OVERLAP, 2=US
vol_regime_id = int(current_bar["atr_regime_id"])  # 0=LOW, 1=MID, 2=HIGH, 3=EXTREME
trend_regime_id = map_trend_regime(current_bar["trend_regime_tf24h"])  # 0=UP, 1=DOWN, 2=RANGE
```

### 4.3 Transformer Prediction

**Input:**
- `seq_x`: `[1, 30, 13]` (batch, seq_len, n_seq_features)
- `snap_x`: `[1, 85]` (batch, n_snap_features)
- `session_id_t`: `[1]` (batch, session ID)
- `vol_regime_id_t`: `[1]` (batch, vol regime ID)
- `trend_regime_id_t`: `[1]` (batch, trend regime ID)

**Forward pass:**
```python
# gx1/models/entry_v9/entry_v9_transformer.py::EntryV9Transformer.forward()
# Linje 200-300 (ca.)

# 1. Encode sequence
seq_emb = self.seq_encoder(seq_x)  # [1, d_model]

# 2. Encode snapshot
snap_emb = self.snap_encoder(snap_x)  # [1, d_model]

# 3. Encode regime embeddings
session_emb = self.session_embedding(session_id_t)  # [1, d_model]
vol_emb = self.vol_regime_embedding(vol_regime_id_t)  # [1, d_model]
trend_emb = self.trend_regime_embedding(trend_regime_id_t)  # [1, d_model]

# 4. Fuse embeddings
fused = seq_emb + snap_emb + session_emb + vol_emb + trend_emb  # [1, d_model]

# 5. Multi-task heads
direction_logit = self.direction_head(fused)  # [1, 1]
early_move_logit = self.early_move_head(fused)  # [1, 1]
quality_score = self.quality_head(fused)  # [1, 1]
```

**Output:**
```python
# gx1/execution/oanda_demo_runner.py::_predict_entry_v9()
# Linje 5818-5843

# Convert logits to probabilities
prob_direction = torch.sigmoid(direction_logit).cpu().item()  # p_long
prob_early = torch.sigmoid(early_move_logit).cpu().item()  # p_early_momentum
quality = quality_score.cpu().item()  # quality_score

# Map to EntryPrediction
prob_long = float(prob_direction)
prob_short = float(1.0 - prob_direction)
margin = abs(prob_long - prob_short)
p_hat = max(prob_long, prob_short)
```

**Arkitektur-detaljer:**
- **d_model:** 128 (default)
- **n_heads:** 4 (default)
- **num_layers:** 3 (default)
- **dim_feedforward:** 512 (d_model * 4)
- **dropout:** 0.05
- **pooling:** "mean" (sequence pooling)
- **causal:** True (causal masking for sequence)

---

## 5. Samspill og endelig p_long

### 5.1 Modell-valg (XGBoost vs Transformer)

**Systemet bruker ENTTEN XGBoost ELLER Transformer, ikke begge samtidig.**

**Valg-logikk:**
```python
# gx1/execution/oanda_demo_runner.py::_predict_entry()
# Linje 5520-5697

# 1. Try Transformer first (if enabled)
if self.entry_v9_model is not None:
    v9_pred = self._predict_entry_v9(entry_bundle, candles, policy_state)
    if v9_pred is not None:
        return v9_pred  # Use Transformer

# 2. Fallback to XGBoost (if Transformer not available)
if self.entry_model_bundle is not None:
    xgb_pred = self._predict_entry(entry_bundle, timestamp)
    if xgb_pred is not None:
        return xgb_pred  # Use XGBoost
```

**Konfigurasjon:**
- **Transformer:** Aktiveres via `entry_models.v9.enabled: true` i config
- **XGBoost:** Aktiveres via session-routed models (EU/US/OVERLAP)

**Aktuell bruk:**
- **SNIPER:** Bruker Transformer (ENTRY_V9) som primær modell
- **FARM:** Bruker Transformer (ENTRY_V9) som primær modell
- **XGBoost:** Brukes som fallback hvis Transformer ikke er tilgjengelig

### 5.2 Endelig p_long (runtime)

**Beregningsflyt:**
1. **Feature building:** `build_live_entry_features()` → `build_v9_runtime_features()`
2. **Model prediction:**
   - Transformer: `_predict_entry_v9()` → `prob_long = sigmoid(direction_logit)`
   - XGBoost: `_predict_entry()` → `prob_long = extract_entry_probabilities().prob_long`
3. **EntryPrediction object:** `EntryPrediction(prob_long=..., prob_short=..., p_hat=..., margin=...)`
4. **Entry gating:** `p_long` brukes i `entry_v9_policy_sniper` eller `entry_v9_policy_farm_v2b`

**Kode-lokasjon:**
```python
# gx1/execution/entry_manager.py::evaluate_entry()
# Linje 472-2000 (ca.)

# 1. Build features
entry_bundle = build_live_entry_features(candles)

# 2. Get prediction (Transformer or XGBoost)
v9_pred = self._runner._predict_entry_v9(entry_bundle, candles, policy_state)
# OR
v9_pred = self._runner._predict_entry(entry_bundle, timestamp)

# 3. Extract p_long
p_long = v9_pred.prob_long  # This is the final p_long used for gating
```

### 5.3 Meta-model (p_profitable) - IKKE p_long

**Viktig:** Meta-modellen (`gx1/models/farm_entry_meta/baseline_model.pkl`) produserer `p_profitable`, ikke `p_long`.

**Bruksområde:**
- `p_profitable` beregnes og logges, men brukes **IKKE** for entry-gating i FARM_V2B/SNIPER
- `p_profitable` er kun for logging/analyse

**Feature-set for meta-model:**
- `side_sign`, `p_long`, `entry_prob_long`, `entry_prob_short`, `atr_bps`, `is_long`, `is_short`
- Se `gx1/policy/farm_meta_features.py::build_meta_feature_vector()` for detaljer

---

## 6. Entry-gating & policy-regler

### 6.1 Entry Gating Thresholds

**SNIPER (baseline):**
- `min_prob_long: 0.67`
- `p_side_min.long: 0.67`
- `p_side_min.short: 1.0` (blocks shorts)
- `margin_min.long: 0.50`
- `side_ratio_min: 1.25`
- `sticky_bars: 1`

**SNIPER P4.1:**
- `min_prob_long: 0.80`
- `p_side_min.long: 0.80`
- `p_side_min.short: 1.0`
- `margin_min.long: 0.50`
- `side_ratio_min: 1.25`
- `sticky_bars: 1`

**FARM_V2B:**
- `min_prob_long: 0.68`
- `p_side_min.long: 0.68`
- `p_side_min.short: 1.0`
- `margin_min.long: 0.50`
- `side_ratio_min: 1.25`
- `sticky_bars: 1`

### 6.2 Entry Gating Pipeline

**Kode-lokasjon:**
```python
# gx1/policy/entry_v9_policy_sniper.py::apply_entry_v9_policy_sniper()
# Linje 29-336

# STEP 0: SNIPER GUARD (session + vol regime)
guard_passed_mask = sniper_guard_v1(df, allow_high_vol=True, allow_extreme_vol=False)

# STEP 1: Optional trend filter (soft gate only)
if require_trend_up:
    # Log but don't hard-filter (soft gate)

# STEP 2: Side-aware probability thresholds
long_mask = df["p_long"] >= min_prob_long  # 0.67 (baseline) or 0.80 (P4.1)
short_mask = df["p_short"] >= min_prob_short if allow_short else False
prob_mask = long_mask | short_mask

# STEP 3: Compute p_profitable (for logging only, NOT for filtering)
p_profitable = meta_model.predict_proba(X_meta)[:, 1]  # Logged but not used

# STEP 4: Final mask
final_mask = prob_mask  # p_long threshold is the ONLY filter (p_profitable not used)
```

### 6.3 Hvordan p_long brukes i entry-beslutning

**Pipeline:**
1. **p_long genereres:** Transformer eller XGBoost → `EntryPrediction.prob_long`
2. **Entry policy evalueres:** `apply_entry_v9_policy_sniper()` eller `apply_entry_v9_policy_farm_v2b()`
3. **Gating sjekkes:**
   - `p_long >= min_prob_long` (hard threshold)
   - `p_side >= p_side_min.long` (side-aware threshold)
   - `margin >= margin_min.long` (margin threshold)
   - `side_ratio >= side_ratio_min` (side ratio threshold)
4. **Trade opprettes:** Hvis alle gates passerer → `LiveTrade` opprettes

**Kode-lokasjon:**
```python
# gx1/execution/entry_manager.py::evaluate_entry()
# Linje 1637-2000 (ca.)

# 1. Get p_long from prediction
v9_pred = self._runner._predict_entry_v9(...)
p_long = v9_pred.prob_long

# 2. Build DataFrame for policy evaluation
current_row["prob_long"] = v9_pred.prob_long
current_row["prob_short"] = v9_pred.prob_short

# 3. Apply entry policy
df_result = apply_entry_v9_policy_sniper(current_row, config, meta_model, meta_feature_cols)

# 4. Check if entry passed
if df_result["entry_v9_policy_sniper"].iloc[0]:
    # Create trade
    trade = LiveTrade(...)
```

### 6.4 Kobling til SNIPER/FARM-regimekartet

**SNIPER:**
- **Sessions:** EU, OVERLAP, US
- **Vol regimes:** LOW, MEDIUM, HIGH (EXTREME blocked)
- **Trend regimes:** Allowed (soft gate only)
- **Regime blocks:** `TREND_NEUTRAL × HIGH`, `TREND_DOWN × HIGH` (P4.1: også `TREND_UP × LOW`)

**FARM:**
- **Sessions:** ASIA only
- **Vol regimes:** LOW, MEDIUM (HIGH/EXTREME blocked)
- **Trend regimes:** Allowed (no hard filter)
- **Regime blocks:** None (brutal guard handles session+vol)

**Entry gating skjer etter regime-filtering:**
1. **Stage-0:** Regime/session filter (blokkerer før modell-evaluering)
2. **Model prediction:** p_long genereres (Transformer eller XGBoost)
3. **Entry policy:** p_long threshold + side-aware gating
4. **Risk guard:** Spread/ATR/cooldown checks (før trade opprettelse)

---

## 7. Konklusjon

ENTRY-stacken i GX1 bruker to parallelle modell-arkitekturer (XGBoost og Transformer) for å generere `p_long`. Systemet velger automatisk hvilken modell som skal brukes basert på konfigurasjon, med Transformer som primær modell og XGBoost som fallback.

**Nøkkelpunkter:**
1. **Feature pipeline:** 13 sequence features + 85 snapshot features = 98 total features
2. **XGBoost:** Session-routed models (EU/US/OVERLAP), bruker kun snapshot features
3. **Transformer:** ENTRY_V9 model, bruker både sequence og snapshot features med regime-conditioning
4. **p_long:** Genereres av ENTTEN XGBoost ELLER Transformer (ikke begge)
5. **Entry gating:** `p_long >= min_prob_long` er hovedfilter (p_profitable brukes ikke for filtering)
6. **Regime-integrasjon:** Entry gating skjer etter regime-filtering (Stage-0)

**Fremtidige forbedringer:**
- Ensemble av XGBoost + Transformer (vektet kombinering)
- Adaptive modell-valg basert på regime
- Temperature scaling tuning per session
- Meta-model som hard gate (hvis p_profitable viser seg å være prediktiv)

---

**End of Document**

