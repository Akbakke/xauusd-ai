# Label V2 Design Note

**Status:** ✅ IMPLEMENTED  
**Design Date:** 2026-01-26  
**Implementation Date:** 2026-01-27  
**Location:** `gx1/scripts/train_xgb_universal_multihead_v2.py`

---

## IMPLEMENTED AS OF 2026-01-27

### What was implemented

| Component | Implementation | File |
|-----------|----------------|------|
| `create_3class_labels_v2()` | Margin-based labeling (T, M) | `train_xgb_universal_multihead_v2.py` |
| `get_label_distribution()` | Label distribution logging | `train_xgb_universal_multihead_v2.py` |
| `--label-version v1\|v2` | Flag to select label version | `train_xgb_universal_multihead_v2.py` |
| `--threshold-v2` | Symmetric threshold T (default 0.4) | `train_xgb_universal_multihead_v2.py` |
| `--min-margin-v2` | Minimum margin M (default 0.25) | `train_xgb_universal_multihead_v2.py` |
| Fail-fast validation | T ∈ (0, 2.0], M ∈ [0, 2.0] | `train_xgb_universal_multihead_v2.py` |
| Label version in metadata | Stored in model meta | `train_xgb_universal_multihead_v2.py` |
| Eval script label display | Reads label version from model | `eval_xgb_multihead_v2_multiyear.py` |
| Temporal split args | `--train-years`, `--val-years` | `eval_xgb_multihead_v2_multiyear.py` |
| Smoke test | `test_label_v2_smoke.py` | `gx1/scripts/test_label_v2_smoke.py` |

### V2 Definition (as implemented)

```python
# Parameters (global, not session-specific)
T = 0.4   # threshold_v2: symmetric threshold (ATR multiple)
M = 0.25  # min_margin_v2: minimum margin for directional label

# Margin calculation
margin = long_payoff - short_payoff

# Labels
LONG:  long_payoff >= T AND margin >= M
SHORT: short_payoff >= T AND margin <= -M
FLAT:  otherwise
```

### Usage

**Training with V2 labels:**
```bash
/home/andre2/venvs/gx1/bin/python gx1/scripts/train_xgb_universal_multihead_v2.py \
    --years 2020 2021 2022 2023 2024 2025 \
    --sessions EU US OVERLAP \
    --label-version v2 \
    --threshold-v2 0.4 \
    --min-margin-v2 0.25
```

**Evaluation with temporal split:**
```bash
/home/andre2/venvs/gx1/bin/python gx1/scripts/eval_xgb_multihead_v2_multiyear.py \
    --years 2020 2021 2022 2023 2024 2025 \
    --reference-year 2025 \
    --train-years 2020 2021 2022 2023 \
    --val-years 2024 2025
```

**Smoke test:**
```bash
/home/andre2/venvs/gx1/bin/python gx1/scripts/test_label_v2_smoke.py
```

### What was NOT changed

- V1 remains default (`--label-version v1`)
- No changes to runtime/trading/Transformer pipeline
- No changes to payoff calculation (`compute_payoffs()`)
- No changes to drift thresholds (KS/PSI)
- No asymmetric thresholds (T_long, T_short removed from design)
- No payoff ratio (replaced with margin)

### Next Steps

1. Run smoke test: `/home/andre2/venvs/gx1/bin/python gx1/scripts/test_label_v2_smoke.py`
2. Train V2 model on small slice to verify no errors
3. Run eval with `--train-years 2020 2021 2022 2023 --val-years 2024 2025`
4. Compare KS/PSI metrics vs V1 baseline
5. If drift improves → Promote XGB to GO
6. THEN proceed with Transformer integration

---

## NÅVÆRENDE DEFINISJON (V1)

### Funksjon: `create_3class_labels()`
**Location:** Lines 149-219 in `train_xgb_universal_multihead_v2.py`

### Payoff Beregning
```python
# ATR-normalized payoffs (from compute_payoffs(), lines 100-146)
long_payoff = (future_close - close - spread_cost) / atr
short_payoff = (close - future_close - spread_cost) / atr

# Where:
#   future_close = np.roll(close, -lookahead_bars)  # 12 bars ahead
#   spread_cost = close * (spread_bps / 10000)      # Default: 2.0 bps
#   atr = max(_v1_atr or atr, close * 0.0001)       # Min 1 bps
```

### Label Klassifisering (Hard Threshold)
```python
# Single threshold (calibrated per head, default 0.5)
threshold = threshold_atr_mult  # Per-head calibrated (0.10-0.50 range)

# LONG: long_payoff >= T AND short_payoff < T
long_mask = (long_payoff >= threshold) & (short_payoff < threshold)
labels[long_mask] = 0

# SHORT: short_payoff >= T AND long_payoff < T
short_mask = (short_payoff >= threshold) & (long_payoff < threshold)
labels[short_mask] = 1

# FLAT: otherwise (including when both payoffs >= T or both < T)
labels[~long_mask & ~short_mask] = 2
```

### Problemer med V1
1. **Binær logikk**: Hard cutoff ved threshold, ingen gradient
2. **Symmetrisk threshold**: Samme threshold for LONG og SHORT (kan være suboptimalt)
3. **Ikke session-spesifikk**: Threshold kalibreres per head, men payoff-beregning er identisk
4. **Ignorerer payoff ratio**: Hvis både long_payoff og short_payoff er høye, blir det FLAT (kan være LONG hvis long_payoff >> short_payoff)

---

## FORESLÅTT V2 DESIGN

### Core Ideer
1. **Asymmetric thresholds**: Separate thresholds for LONG vs SHORT
2. **Payoff ratio consideration**: Vurder forholdet mellom long_payoff og short_payoff
3. **Session-specific parameters**: Tillat session-spesifikke parametre i payoff-beregning
4. **Soft boundaries**: Gradvis overgang mellom klasser (optional, for future)

### V2 Formel (Hard Labels, Asymmetric)

```python
# Session-specific thresholds
threshold_long = threshold_long_atr_mult[session]   # e.g., 0.5 for EU, 0.6 for US
threshold_short = threshold_short_atr_mult[session] # e.g., 0.4 for EU, 0.5 for US

# Payoff ratio (for edge cases)
payoff_ratio = long_payoff / (short_payoff + 1e-8)  # Avoid division by zero

# LONG: long_payoff >= T_long AND (short_payoff < T_short OR payoff_ratio > ratio_threshold)
long_mask = (
    (long_payoff >= threshold_long) & 
    (short_payoff < threshold_short) &
    (payoff_ratio > 1.2)  # Long must be at least 20% better
)
labels[long_mask] = 0

# SHORT: short_payoff >= T_short AND (long_payoff < T_long OR payoff_ratio < 1/ratio_threshold)
short_mask = (
    (short_payoff >= threshold_short) & 
    (long_payoff < threshold_long) &
    (payoff_ratio < 0.83)  # Short must be at least 20% better (1/1.2)
)
labels[short_mask] = 1

# FLAT: otherwise
labels[~long_mask & ~short_mask] = 2
```

### Session-Spesifikke Parametre

| Parameter | Default | Session-Specific? | Beskrivelse |
|-----------|---------|-------------------|-------------|
| `threshold_long_atr_mult` | 0.5 | ✅ Yes | LONG threshold (ATR multiple) |
| `threshold_short_atr_mult` | 0.4 | ✅ Yes | SHORT threshold (ATR multiple) |
| `payoff_ratio_threshold` | 1.2 | ✅ Yes | Minimum ratio for LONG vs SHORT dominance |
| `lookahead_bars` | 12 | ❌ No (global) | Bars to look ahead (1 hour) |
| `spread_bps` | 2.0 | ❌ No (global) | Spread cost in basis points |
| `atr_min_bps` | 1.0 | ❌ No (global) | Minimum ATR in bps (for normalization) |

### Kalibrering (V2)

**Per-session calibration:**
```python
def calibrate_threshold_v2(
    long_payoffs: np.ndarray,
    short_payoffs: np.ndarray,
    session: str,
    target_flat_min: float = 0.70,
    target_flat_max: float = 0.90,
    min_class_rate: float = 0.02,
) -> Tuple[float, float, float]:
    """
    Calibrate asymmetric thresholds + ratio threshold per session.
    
    Returns:
        (threshold_long, threshold_short, payoff_ratio_threshold)
    """
    # Grid search over:
    #   threshold_long: [0.3, 0.4, 0.5, 0.6, 0.7]
    #   threshold_short: [0.2, 0.3, 0.4, 0.5, 0.6]
    #   payoff_ratio_threshold: [1.1, 1.2, 1.3, 1.5]
    
    # Score: balance LONG/SHORT, FLAT in target range, min class rates
    # Return best combination
```

### Eksempel: Session-Spesifikke Verdier

**EU Session:**
- `threshold_long = 0.5` (moderate)
- `threshold_short = 0.4` (slightly lower for SHORT)
- `payoff_ratio_threshold = 1.2` (20% dominance required)

**US Session:**
- `threshold_long = 0.6` (higher, more conservative)
- `threshold_short = 0.5` (higher, more conservative)
- `payoff_ratio_threshold = 1.3` (30% dominance required, more strict)

**OVERLAP Session:**
- `threshold_long = 0.45` (slightly lower)
- `threshold_short = 0.35` (lower for SHORT)
- `payoff_ratio_threshold = 1.15` (15% dominance, more lenient)

---

## IMPLEMENTASJON PLAN

### Phase 1: Design + Sanity (Current)
- ✅ Design note (this document)
- ⏳ Sanity check: Verifiser at asymmetric thresholds gir bedre class balance

### Phase 2: Implementation (Behind Flag)
- Add `--label-version v2` flag to `train_xgb_universal_multihead_v2.py`
- Implement `create_3class_labels_v2()` function
- Implement `calibrate_threshold_v2()` function
- Store session-specific thresholds in model meta

### Phase 3: Evaluation Only
- Train model with `--label-version v2`
- Run `eval_xgb_multihead_v2_multiyear.py` (KS/PSI drift)
- Compare vs V1 baseline
- **DO NOT** touch Transformer or other components

### Phase 4: Promotion
- If V2 improves drift (KS/PSI < thresholds), promote to GO
- Update MASTER_MODEL_LOCK.json
- **Then** proceed with Transformer integration

---

## SANITY CHECKS

### Pre-Implementation
1. **Class Balance**: Asymmetric thresholds skal gi bedre LONG/SHORT balance per session
2. **Payoff Ratio**: Ratio threshold skal redusere "confused" labels (hvor både long og short er høye)
3. **Session Divergence**: Session-specific thresholds skal gi mer divergente heads

### Post-Implementation
1. **Drift Metrics**: KS/PSI skal være < 0.15/0.10 for alle heads
2. **Class Distribution**: Min 2% per class, FLAT 70-90%
3. **Head Signatures**: Heads skal ha forskjellige p_long/p_short/p_flat means

---

## NOTATER

- **Ikke legg til flere features** - Kun endre label-definisjon
- **Ikke juster Transformer** - Kun XGB training/eval
- **Ikke Optuna** - Manuell threshold calibration er nok
- **Ikke "bare teste litt"** - Full multiyear eval før promotering

---

# FAGLIG SANITY REVIEW

**Reviewer:** Claude  
**Date:** 2026-01-27  
**Status:** CONDITIONAL GO (med justeringer)

---

## 1. WHAT V2 FIXES VS V1

### Trades som flyttes fra LONG/SHORT → FLAT

| Scenario | V1 Label | V2 Label | Kommentar |
|----------|----------|----------|-----------|
| long_payoff=0.55, short_payoff=0.45 (T=0.5) | LONG | FLAT | ✅ Korrekt: Edge er marginal, ratio≈1.2 men ikke klart dominerende |
| long_payoff=0.7, short_payoff=0.6 (T=0.5) | FLAT¹ | FLAT | ✅ V1 gir FLAT fordi begge>=T, V2 også FLAT pga ratio<1.2 |
| long_payoff=0.3, short_payoff=0.1 (T=0.5) | FLAT | FLAT | ✅ Begge under T, ingen endring |

¹ V1 gir FLAT når begge payoffs >= T (AND-logikken feiler for begge retninger)

### Edge-typer som bevares

| Scenario | V1 Label | V2 Label | Kommentar |
|----------|----------|----------|-----------|
| long_payoff=0.8, short_payoff=0.2 (T=0.5) | LONG | LONG | ✅ Klar edge, ratio=4.0 >> 1.2 |
| long_payoff=1.2, short_payoff=-0.5 (T=0.5) | LONG | LONG | ✅ Sterk edge, men **ratio-beregningen er problematisk** (se nedenfor) |
| short_payoff=0.9, long_payoff=0.1 (T=0.5) | SHORT | SHORT | ✅ Klar SHORT edge |

### Hvor reduseres regime-/volatilitet-lekkasje

**V2 forbedrer:**
1. **High-vol perioder**: Når ATR er høy, er det lettere å få begge payoffs over threshold. V2s ratio-krav filtrer ut "begge-høye" cases.
2. **Ambiguous moves**: Konsolidering der pris oscillerer gir lignende long/short payoffs. V2 → FLAT.

**V2 forbedrer IKKE:**
1. **ATR-drift over år**: Kjerneproblemet er at ATR i 2021 ≠ ATR i 2025. ATR-normalized payoffs shifter systematisk.
2. **Session-spesifikk volatilitet**: Hvis US har 30% høyere ATR enn EU, vil payoff-distribusjoner være forskjellige uansett threshold.

---

## 2. KRITISK ANALYSE AV V2 KOMPONENTER

### 2.1 Asymmetriske Thresholds (Pro/Contra)

**Pro:**
- Kan fange session-spesifikk directional bias (f.eks. US har historisk mer short-momentum)
- Tillater finere kontroll over class balance per head

**Contra:**
- ⚠️ **Overfitting-risiko**: 2 thresholds × 3 sessions = 6 parametre som kalibreres per-year-data
- ⚠️ **Ustabil under regime-skifte**: Threshold kalibrert på 2020-2024 kan være feil for 2025
- ⚠️ **Drift-forsterkende**: Hvis vi kalibrerer T_long≠T_short for å matche historisk class balance, låser vi inn historisk bias

**Konklusjon:** Asymmetriske thresholds er **høy-risiko** for drift-stabilitet. Foretrekk symmetrisk T.

### 2.2 Payoff Ratio (kritisk feil i design)

**Problem:**
```python
payoff_ratio = long_payoff / (short_payoff + 1e-8)
```

Denne formelen er **matematisk ustabil**:

| long_payoff | short_payoff | ratio | Tolkning |
|-------------|--------------|-------|----------|
| 0.6 | 0.3 | 2.0 | ✅ OK, LONG er 2x bedre |
| 0.6 | -0.3 | -2.0 | ❌ Negativ ratio, bryter logikken |
| 0.6 | 0.0 | 6e7 | ❌ Ekstrem ratio, numerisk ustabil |
| -0.2 | -0.5 | 0.4 | ❌ Begge negative, ratio gir feil bilde |

**Foreslått fix:** Bruk **payoff margin** i stedet for ratio:
```python
margin = long_payoff - short_payoff
# LONG: long_payoff >= T AND margin >= min_margin
# SHORT: short_payoff >= T AND (-margin) >= min_margin
```

**Margin-fordeler:**
- Alltid numerisk stabil
- Enkel tolkning: "hvor mye bedre er LONG enn SHORT?"
- Ingen divisjon, ingen edge cases

### 2.3 calibrate_threshold(): Forenklingsforslag

**Nåværende V2 kalibrering:**
- Grid search over: T_long × T_short × ratio_threshold = 5 × 5 × 4 = 100 kombinasjoner
- Per session, per year-range
- Scorer på: LONG/SHORT balance, FLAT target, min class rates

**Problemer:**
1. **For mange frihetsgrader** → Overfitting
2. **Implisitt temporal leakage**: Kalibrerer på all data, inkludert test-år
3. **Manglende stability-constraint**: Ingen straff for parametre som gir høy mellom-år varians

**Foreslått forenkling:**
```python
def calibrate_threshold_v2_simplified(
    long_payoffs: np.ndarray,
    short_payoffs: np.ndarray,
) -> Tuple[float, float]:
    """
    Returns: (threshold, min_margin)
    
    - Single symmetric threshold T
    - Single margin M (global, ikke session-specific)
    - Calibrate on oldest 3 years, validate on newest 2
    """
    # Grid: T ∈ [0.3, 0.4, 0.5, 0.6], M ∈ [0.1, 0.2, 0.3, 0.4]
    # Score: minimize(variance_of_class_rates_across_years) + balance_penalty
```

---

## 3. US-HEAD FOKUS

### Hvorfor V2 kan være mer stabil i 2021/2023

**Hypotese om US drift:**
- 2021: Post-COVID recovery, gull hadde høy volatilitet, ATR var ~40% høyere enn 2025
- 2023: Banking crisis (SVB), gull-rally, ATR-spikes
- Begge år har **episodisk høy volatilitet** som gir mange "begge payoffs høye" labels i V1

**V2 forbedring:**
- Margin-krav filtrerer ut "begge høye" cases → Færre LONG/SHORT labels i high-vol perioder
- Dette gjør label-distribusjonen mer stabil på tvers av volatilitets-regimer

**Kvantitativ forventning:**
- V1: US head har ~25% LONG i 2025, ~35% LONG i 2021 (pga høyere ATR → lettere å nå threshold)
- V2: Margin-krav skal redusere 2021 LONG-rate til ~28%, tettere på 2025

### Failure modes som gjenstår

1. **Strukturell ATR-drift**: Hvis gull-volatilitet fortsetter å synke, vil fremtidige år ha enda lavere ATR. Labels blir mer FLAT-dominert.
2. **Session-spesifikk regime**: Hvis US-session går fra "trend" til "range" (eller omvendt), vil payoff-karakteristikken endre seg.
3. **Lookahead mismatch**: 12 bars (1 time) kan være for kort/langt for ulike volatilitets-regimer.

---

## 4. PARAMETER-KLASSIFISERING

### MÅ være globale (for å unngå overfitting)

| Parameter | Verdi | Begrunnelse |
|-----------|-------|-------------|
| `lookahead_bars` | 12 | Definerer hva vi predikerer. Session-spesifikk lookahead = leakage. |
| `spread_bps` | 2.0 | Faktisk spread er global, ikke session-avhengig. |
| `atr_min_bps` | 1.0 | Numerisk stabilitet, ingen grunn til session-variasjon. |
| `min_margin` | **0.2-0.3** | ⚠️ BØR være global. Session-spesifikk margin = overfitting. |

### KAN være session-spesifikke (med forsiktighet)

| Parameter | Anbefaling | Begrunnelse |
|-----------|------------|-------------|
| `threshold` (T) | **Global foretrukket** | Session-spesifikk T er overfitting-risiko. Hvis nødvendig: maks 0.1 forskjell mellom sessions. |
| `target_flat_rate` | Session-spesifikk OK | EU/US/OVERLAP kan ha naturlig forskjellig FLAT-rate. |

### IKKE session-spesifikke (fjern fra V2)

| Parameter | Hvorfor ikke |
|-----------|--------------|
| `threshold_long` vs `threshold_short` | Asymmetri uten klar markedsteori. Legg til kun hvis data tydelig viser bias. |
| `payoff_ratio_threshold` | Ratio er ustabil, bruk margin i stedet. |

---

## 5. ANBEFALT MINIMAL V2 DEFINISJON

### Pseudokode

```python
# ==== GLOBAL PARAMETERS (ikke session-spesifikke) ====
T = 0.4                # Symmetric threshold (ATR multiple)
M = 0.25               # Minimum margin (long_payoff - short_payoff)
lookahead_bars = 12    # 1 hour
spread_bps = 2.0       # Transaction cost

# ==== PAYOFF BEREGNING (uendret fra V1) ====
long_payoff = (future_close - close - spread_cost) / atr
short_payoff = (close - future_close - spread_cost) / atr
margin = long_payoff - short_payoff

# ==== LABEL KLASSIFISERING (V2 med margin) ====

# LONG: payoff over threshold AND significantly better than SHORT
long_mask = (long_payoff >= T) & (margin >= M)

# SHORT: payoff over threshold AND significantly better than LONG
short_mask = (short_payoff >= T) & (margin <= -M)

# FLAT: otherwise
flat_mask = ~long_mask & ~short_mask

labels[long_mask] = 0   # LONG
labels[short_mask] = 1  # SHORT
labels[flat_mask] = 2   # FLAT
```

### Forskjeller fra opprinnelig V2

| Aspekt | Opprinnelig V2 | Anbefalt V2 |
|--------|----------------|-------------|
| Thresholds | Asymmetrisk (T_long, T_short) | Symmetrisk (T) |
| Session-spesifikk | T_long, T_short, ratio per session | Kun T kan variere (maks ±0.1) |
| Edge-filter | Payoff ratio | Payoff margin (M) |
| Kalibrering | Grid 100 kombinasjoner | Grid 16 kombinasjoner (4×4) |
| Parametre | 3 per session = 9 totalt | 2 globale (T, M) |

### Hvorfor margin > ratio

1. **Numerisk stabil**: Ingen divisjon, ingen edge cases
2. **Tolkbar**: "LONG må være minst 0.25 ATR bedre enn SHORT"
3. **Drift-robust**: Margin i ATR-enheter er relativt stabil på tvers av volatilitets-regimer
4. **Enklere kalibrering**: 2 parametre vs 3

---

## 6. GREEN-LIGHT / NO-GO KRITERIER

### Before Implementation

| Kriterie | Status | Handling hvis fail |
|----------|--------|-------------------|
| Margin-logikk er matematisk korrekt | ✅ GO | - |
| Ingen session-spesifikk asymmetri i første versjon | ✅ GO | - |
| Lookahead/spread forblir global | ✅ GO | - |
| Grid-søk har ≤25 kombinasjoner | ✅ GO (4×4=16) | Reduser grid |
| Kalibrering splittes: train on 2020-2023, validate on 2024-2025 | ⚠️ MÅ IMPLEMENTERES | Legg til temporal split |

### After Implementation (Before Promotion)

| Kriterie | Threshold | Handling hvis fail |
|----------|-----------|-------------------|
| KS drift (vs 2025) | < 0.15 | Juster T eller M |
| PSI drift (vs 2025) | < 0.10 | Juster T eller M |
| Min class rate (LONG, SHORT) | ≥ 2% | Reduser T |
| Max class rate (FLAT) | ≤ 92% | Øk M eller T |
| US head 2021 vs 2025 KS | < 0.12 | Hvis fail, vurder session-spesifikk T |
| US head 2023 vs 2025 KS | < 0.12 | Hvis fail, vurder session-spesifikk T |

### Promotion Gate

```
IF all_heads_KS < 0.15 AND all_heads_PSI < 0.10:
    PROMOTE XGB to GO
    THEN proceed with Transformer integration
ELSE:
    ITERATE on T, M values
    DO NOT touch Transformer
```

---

## 7. REMAINING RISKS

### Risiko 1: ATR-drift er strukturell
**Beskrivelse:** Hvis gull-volatilitet fortsetter å falle, vil fremtidige år ha systematisk lavere ATR. ATR-normalized payoffs blir større (samme price move / lavere ATR), og label-distribusjonen shifter mot mer LONG/SHORT.

**Mitigering:** Ikke løsbart med label-design alene. Krever enten:
- ATR-regime-normalisering (kompliserer pipeline)
- Re-training med nyere data (operasjonell løsning)

### Risiko 2: Margin-parameter er også kalibrert
**Beskrivelse:** Selv om M er global, er den fortsatt kalibrert på historiske data. Hvis markedsstruktur endres, kan M være feil.

**Mitigering:** Velg M fra "stabil" zone (0.2-0.3) der class-balance er robust. Ikke optimer for eksakt class-rate.

### Risiko 3: Session-karakteristikk kan endre seg
**Beskrivelse:** US session i 2025 kan ha annen karakteristikk enn 2021. Session-spesifikke parametre kan gi falsk trygghet.

**Mitigering:** Start med globale parametre. Tillat session-spesifikt kun hvis KS-gapet er > 0.05 mellom sessions.

---

## 8. SUMMARY

| Aspekt | Opprinnelig V2 Design | Anbefalt Justering |
|--------|----------------------|-------------------|
| Asymmetriske T | Session-spesifikk T_long, T_short | ❌ Fjern. Bruk global symmetrisk T. |
| Payoff ratio | long/short ratio | ❌ Fjern. Bruk margin = long - short. |
| Session-parametre | 3 per session | 0 per session (globale T, M) |
| Kalibrering | 100 kombinasjoner | 16 kombinasjoner (4×4) |
| Temporal split | Ikke spesifisert | Train 2020-2023, validate 2024-2025 |

### Final Verdict

**CONDITIONAL GO** for implementering med følgende justeringer:
1. Erstatt payoff ratio med payoff margin
2. Bruk symmetrisk threshold (ikke T_long/T_short)
3. Start med globale parametre (ingen session-spesifikk kalibrering)
4. Implementer temporal split i kalibrering

Designet er klart for implementering når justeringene er akseptert.
