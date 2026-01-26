# SYSTEM SPEC: SNIPER & FARM Unified Architecture

**Generert:** 2025-12-28  
**Status:** Konsolidert kunnskapsgrunnlag for videreutvikling  
**Formål:** Blueprint for FARM/SNIPER-fordeling som to spesialiserte motorer

---

## Executive Summary

**FARM:** Low-frequency, high-precision ASIA-range sniper (0.44 trades/dag, 84.6% win rate)  
**SNIPER:** High-frequency, EU/US high-vol strateg (132.9 trades/dag, 62.45% win rate)

**Designmål:** To komplementære motorer som ikke overlapper - FARM for rolig ASIA-market, SNIPER for volatile EU/US markets.

---

## A) Hvordan SNIPER Fungerer Nå

### A1. Sessions Aktivert

**Aktive Sessions:** EU, OVERLAP, US  
**Config:** `allowed_sessions: ["EU", "OVERLAP", "US"]`

**Trade-distribusjon (FULLYEAR 2025):**
- **US:** 22,264 trades (66.5%)
- **EU:** 11,225 trades (33.5%)
- **OVERLAP:** 0 trades (0%)

**Observasjon:** OVERLAP har 0 trades - enten kort tidsvindu eller clock-gating/logikk skiller OVERLAP fra EU/US.

**Hvorfor høy tradefrekvens:**
- 2x bredere trading window enn FARM (EU/US ~16 timer vs ASIA ~8 timer)
- US session dominerer med 66.5% av alle trades
- US session er lengst og mest volatile

---

### A2. ACTIVE Filters

#### Hard Filters (Entry Blocking)

1. **Session Gate (SNIPER Guard V1)**
   - Tillater kun: EU, OVERLAP, US
   - Implementert i: `gx1/policy/farm_guards.py::sniper_guard_v1()`
   - Status: ✅ Aktiv

2. **Volatility Regime Gate**
   - Tillater: LOW, MEDIUM, HIGH
   - Blokkerer: EXTREME
   - Config: `allow_high_vol: true`, `allow_extreme_vol: false`
   - Status: ✅ Aktiv

3. **p_long Threshold**
   - `min_prob_long: 0.67`
   - Entry gating: `p_side_min.long: 0.67`
   - Status: ✅ Aktiv

4. **Entry Gating (p_side_min, margin_min, side_ratio)**
   - `p_side_min.long: 0.67`
   - `margin_min.long: 0.50`
   - `side_ratio_min: 1.25`
   - `sticky_bars: 1`
   - Status: ✅ Aktiv

5. **SNIPER Risk Guard V1**
   - Cooldown: `cooldown_bars_after_entry: 1` (5 min)
   - Spread block: `block_if_spread_bps_gte: 3500` (35 bps)
   - ATR block: `block_if_atr_bps_gte: 25.0` (0.25%)
   - EXTREME vol block: `block_if_vol_regime_in: ["EXTREME"]`
   - US clamp: `extra_min_prob_long: 0.02`
   - OVERLAP clamp: `extra_min_prob_long: 0.01`
   - Status: ✅ Aktiv

6. **EXTREME Vol Block**
   - Verifisert: 0 EXTREME vol trades i datasett
   - Status: ✅ Fungerer korrekt

#### Soft Filters / Size Overlays

1. **Q4 × C_CHOP Size Overlay**
   - Reduserer size til 50% i US session for Q4 × C_CHOP
   - Config: `sniper_q4_cchop_overlay.enabled: true`
   - Status: ✅ Aktiv (size-reduksjon, ikke entry block)

---

### A3. INACTIVE Filters

1. **Trend × Vol Combo Blocks**
   - Config: `block_trend_vol_combos: []` (tom liste)
   - Konsekvens: Tillater alle kombinasjoner inkludert problematiske
   - Status: ❌ Ikke aktiv

2. **require_trend_up Filter**
   - Config: `require_trend_up: false` (soft gate only)
   - Konsekvens: Tillater TREND_NEUTRAL og TREND_DOWN (63% av trades)
   - Status: ❌ Ikke aktiv (hard-filter)

3. **Q4 × B_MIXED Overlay**
   - Config mangler i baseline
   - Status: ❌ Ikke aktiv

4. **Q4 × A_TREND Overlay**
   - Config mangler i baseline
   - Status: ❌ Ikke aktiv

5. **Max Open Trades Override**
   - Config: `max_open_trades_override: 1` (i risk guard config)
   - Status: ❌ Ikke implementert i kode (bruker 10 fra execution config)

6. **min_time_between_trades_sec**
   - Ikke konfigurert i SNIPER config
   - Status: ❌ Ikke aktiv

---

### A4. Hvor Trades Kommer Fra (Andel per Session/Vol/Trend)

**Datasett:** FULLYEAR 2025 (33,489 trades)

#### Per Session:
- **US:** 22,264 (66.5%)
- **EU:** 11,225 (33.5%)
- **OVERLAP:** 0 (0%)

#### Per Vol Regime:
- **HIGH:** 12,957 (38.7%)
- **MEDIUM:** 8,917 (26.6%)
- **LOW:** 11,615 (34.7%)
- **EXTREME:** 0 (0%)

#### Per Trend × Vol Kombinasjon:
| Regime | Trades | % |
|--------|--------|---|
| TREND_NEUTRAL × LOW | 11,286 | 33.7% |
| **TREND_NEUTRAL × HIGH** | **10,140** | **30.3%** |
| TREND_NEUTRAL × MEDIUM | 8,114 | 24.2% |
| TREND_DOWN × HIGH | 1,606 | 4.8% |
| TREND_UP × HIGH | 1,211 | 3.6% |
| TREND_UP × MEDIUM | 458 | 1.4% |
| TREND_DOWN × MEDIUM | 303 | 0.9% |
| TREND_UP × LOW | 296 | 0.9% |
| TREND_DOWN × LOW | 75 | 0.2% |

#### p_long Distribution:
- **Mean:** 0.856
- **Median:** 0.872
- **>= 0.80:** 25,327 trades (75.6%)
- **0.67-0.80:** 8,162 trades (24.4%)
- **< 0.67:** 0 trades (0%)

---

### A5. Hvor Svakheter Ligger

#### Problemkombinasjoner (Høy Andel, Potensielt Risikabelt):

1. **TREND_NEUTRAL × HIGH: 10,140 trades (30.3%)**
   - Høyest enkeltstående kombinasjon
   - Choppy marked + høy volatilitet = høy risiko
   - Ville vært blokkert i FARM (HIGH vol ikke tillatt)

2. **TREND_DOWN × HIGH: 1,606 trades (4.8%)**
   - Nedtrend + høy volatilitet = mot trend
   - Ville vært blokkert i FARM (HIGH vol ikke tillatt)

3. **TREND_NEUTRAL × MEDIUM: 8,114 trades (24.2%)**
   - Choppy marked, men lavere risiko enn HIGH vol

**Total problematiske trades (NEUTRAL/DOWN × HIGH):** ~11,746 trades (35.1%)

---

### A6. p_long Distribution og Effekten av Policy B (0.80)

**Baseline Performance (p_long >= 0.67):**
- **33,489 trades**
- **Win rate: 62.45%**
- **Avg reward: 350.14 bps**
- **Reward/1k: 350,144.06**

**Policy B (p_long >= 0.80):**
- **25,327 trades** (24.4% reduksjon)
- **Win rate: 66.23%** (+3.78%)
- **Avg reward: 384.77 bps** (+34.63 bps)
- **Reward/1k: 384,767.94** (+9.9% forbedring)
- **P05 reward: -58.26** (bedre enn -76.32)

**Konklusjon:** Policy B gir umiddelbar forbedring:
- Færre trades (25k vs 33k)
- Høyere win rate (66% vs 62%)
- Bedre reward/1k (+9.9%)
- Bedre risiko-profil

**Trades som filtreres ut (p_long 0.67-0.80):** 8,162 trades (24.4%)

---

## B) Hvordan FARM Fungerer Nå

### B1. Session og Vol Regime

**Sessions:** Kun ASIA  
**Config:** `allowed_sessions: ["ASIA"]`

**Vol Regimes:** Kun LOW + MEDIUM  
**Config:** `allowed_vol_regimes: ["LOW", "MEDIUM"]`  
**Config:** `allow_high_vol: false`, `allow_extreme_vol: false`

**p_long Threshold:** 0.68  
**Config:** `min_prob_long: 0.68`

**Guard:** `farm_brutal_guard_v2()` i `gx1/policy/farm_guards.py`

---

### B2. Hvorfor Lavfrekvent/Høy-Precision

**Performance (FULLYEAR 2025):**
- **162 trades** totalt
- **0.44 trades/dag** (162 / 365 dager)
- **Win rate: 84.6%**
- **EV/trade: 117.67 bps**
- **Trades/dag:** ~0.44 (lignende 3/dag-målet med markedsstopp)

**Design-filosofi:**
- Strikt session-filtering: Kun ASIA (~8 timer/dag)
- Strikt vol-filtering: Kun LOW/MEDIUM (ikke HIGH)
- Høyere p_long threshold: 0.68
- Fokus på kvalitet over kvantitet

---

### B3. Hvilke Regimefiltre Gjør Den Sterk på Rolig Marked

1. **ASIA Session Only**
   - ASIA session er typisk roligere enn EU/US
   - Mindre volatilitet = bedre for range-strategi
   - Fokuserer på best time-of-day for range-trading

2. **LOW/MEDIUM Vol Only**
   - Blokkerer HIGH vol = mindre tail risk
   - Fokuserer på stabile, forutsigbare markeder
   - Range-strategi fungerer bedre i lavere volatilitet

3. **Range-Aware Exit Router (V3_RANGE)**
   - ML-basert exit routing med range features
   - `range_pos`, `distance_to_range`, `range_edge_dist_atr`
   - RULE6A kun aktivert når `range_edge_dist_atr < 1.0` (edge-spesialisert)

4. **p_long Threshold 0.68**
   - Høyere enn SNIPER's 0.67
   - Fokuserer på høyeste-kvalitet signals

**Konklusjon:** FARM er optimalisert for rolige, range-baserte markeder med lav volatilitet og høy predictability.

---

## C) Designmål

### C1. FARM: "Low/Medium-Vol ASIA Sniper"

**Scope:**
- Sessions: ASIA only
- Vol regimes: LOW, MEDIUM only
- Trend: Ingen hard-filter (men fokuserer på range/neutral markets)
- Exit: Range-aware (V3_RANGE router)

**Mål:**
- Lav frekvens (~0.5 trades/dag)
- Høy win rate (>80%)
- Range-spesialisert
- Konservativ, høy-precision

**Kvalitetsindikatorer:**
- Win rate > 80%
- EV/trade > 100 bps
- Lav tail risk (P90 loss < 50 bps)

---

### C2. SNIPER: "High-Vol EU/US Strateg"

**Scope:**
- Sessions: EU, OVERLAP, US
- Vol regimes: LOW, MEDIUM, HIGH (ikke EXTREME)
- Trend: Ingen hard-filter (men kan blokkere spesifikke kombinasjoner)
- Exit: Standard SNIPER exits

**Mål:**
- Høyere frekvens (10-100 trades/dag, avhengig av filtering)
- Moderat win rate (>60%)
- Volatil-marked spesialisert
- Smart risiko (ikke gambler)

**Kvalitetsindikatorer:**
- Win rate > 60%
- EV/trade > 300 bps
- Kontrollert tail risk (P90 loss < 100 bps)

---

### C3. Konservativ men Smart Risiko

**Prinsipper:**
1. Hard blocks for ekstreme tilstander (EXTREME vol, ekstrem spread)
2. Regime-aware filtering (blokker problematiske kombinasjoner)
3. Quality over quantity (høyere p_long threshold = bedre selektivitet)
4. Cooldown og spread guards for å unngå overtrading

**Ikke Gambler:**
- Ikke trade i EXTREME vol
- Ikke trade mot sterk trend i høy vol (TREND_DOWN × HIGH)
- Ikke trade i choppy + høy vol (TREND_NEUTRAL × HIGH) uten god grunn
- Cooldown mellom trades for å unngå clustering

---

### C4. Begge Motorer Skal Utfylle Hverandre, Ikke Overlappe

**FARM Ownership:**
- ASIA session
- LOW/MEDIUM vol regimes
- Range-baserte markeder
- Lav frekvens, høy precision

**SNIPER Ownership:**
- EU/OVERLAP/US sessions
- HIGH vol regimes (i tillegg til LOW/MEDIUM)
- Volatil-marked strategi
- Høyere frekvens, moderat precision

**Overlapp (Tillatt):**
- Begge kan trade i LOW/MEDIUM vol (men forskjellige sessions)
- Begge kan trade i TREND_NEUTRAL/LOW (men forskjellige sessions)

**Overlapp (Unngå):**
- SNIPER skal ikke trade i ASIA session (reservert for FARM)
- FARM skal ikke trade i HIGH vol (reservert for SNIPER)

---

## D) Regimekart (Trend × Vol × Session → Hvem Eier Hva?)

### D1. Session Ownership

| Session | FARM | SNIPER | Kommentar |
|---------|------|--------|-----------|
| ASIA | ✅ Eier | ❌ Blokkert | FARM's eksklusive domene |
| EU | ❌ Blokkert | ✅ Eier | SNIPER's domene |
| OVERLAP | ❌ Blokkert | ✅ Eier | SNIPER's domene (men 0 trades i praksis) |
| US | ❌ Blokkert | ✅ Eier | SNIPER's domene (66.5% av trades) |

---

### D2. Vol Regime Ownership

| Vol Regime | FARM | SNIPER | Kommentar |
|------------|------|--------|-----------|
| LOW | ✅ Tillatt | ✅ Tillatt | Begge, men forskjellige sessions |
| MEDIUM | ✅ Tillatt | ✅ Tillatt | Begge, men forskjellige sessions |
| HIGH | ❌ Blokkert | ✅ Tillatt | SNIPER's eksklusive domene |
| EXTREME | ❌ Blokkert | ❌ Blokkert | Ingen (for farlig) |

---

### D3. Trend × Vol × Session Kombinasjoner

| Session | Trend | Vol | FARM | SNIPER | Kommentar |
|---------|-------|-----|------|--------|-----------|
| ASIA | Any | LOW | ✅ | ❌ | FARM only |
| ASIA | Any | MEDIUM | ✅ | ❌ | FARM only |
| ASIA | Any | HIGH | ❌ | ❌ | Ingen (FARM blokkerer, SNIPER ikke i ASIA) |
| EU/US | Any | LOW | ❌ | ✅ | SNIPER only |
| EU/US | Any | MEDIUM | ❌ | ✅ | SNIPER only |
| EU/US | Any | HIGH | ❌ | ✅ | SNIPER only (men problematisk kombinasjoner) |
| Any | Any | EXTREME | ❌ | ❌ | Ingen (hard block) |

---

### D4. Problematiske Kombinasjoner for SNIPER

**Kombinasjoner som bør blokkeres eller nedvektes:**

1. **TREND_NEUTRAL × HIGH** (10,140 trades = 30.3%)
   - Choppy marked + høy volatilitet = høy risiko
   - Anbefaling: Blokker eller strengere filtering

2. **TREND_DOWN × HIGH** (1,606 trades = 4.8%)
   - Mot trend + høy volatilitet = høy risiko
   - Anbefaling: Blokker

3. **TREND_NEUTRAL × MEDIUM** (8,114 trades = 24.2%)
   - Mindre problematisk, men kan nedvektes

---

## E) Entry-Regler per Motor

### E1. FARM Entry-Regler

#### Hard Filters:
1. **Session:** ASIA only (`farm_brutal_guard_v2`)
2. **Vol Regime:** LOW or MEDIUM only
3. **p_long Threshold:** >= 0.68
4. **EXTREME Vol:** Blokkert

#### Soft Filters:
- `require_trend_up: false` (soft gate only)
- `enable_profitable_filter: false`

#### Config Path:
`gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID/ENTRY_V9_FARM_V2B_PROD.yaml`

---

### E2. SNIPER Entry-Regler

#### Hard Filters:
1. **Session:** EU, OVERLAP, US (`sniper_guard_v1`)
2. **Vol Regime:** LOW, MEDIUM, HIGH (ikke EXTREME)
3. **p_long Threshold:** >= 0.67
4. **EXTREME Vol:** Blokkert (risk guard)
5. **Spread Block:** >= 35 bps (global), >= 30 bps (US), >= 32 bps (OVERLAP)
6. **ATR Block:** >= 25 bps
7. **Cooldown:** 1 bar (5 min) etter entry

#### Soft Filters / Size Overlays:
- `require_trend_up: false` (soft gate only)
- `block_trend_vol_combos: []` (tom = ingen block)
- Q4 × C_CHOP size overlay (50% size i US)

#### Session Clamps:
- US: `extra_min_prob_long: +0.02` (effektivt 0.69)
- OVERLAP: `extra_min_prob_long: +0.01` (effektivt 0.68)

#### Config Path:
`gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_REPLAY_SHADOW_2025.yaml`

---

## F) Candidate Policies

### F1. P1: p_long ≥ 0.80 (Policy B Baseline-Mod)

**Beskrivelse:** Øk p_long threshold fra 0.67 til 0.80

**Forventet Effekt:**
- **Trades:** 25,327 (24.4% reduksjon fra 33,489)
- **Win rate:** 66.23% (vs 62.45% baseline)
- **Reward/1k:** 384,768 (vs 350,144 baseline = +9.9%)
- **P05 reward:** -58.26 (bedre enn -76.32)

**Implementasjon:**
- Endre `min_prob_long: 0.80` i `ENTRY_V9_SNIPER_LONDON_NY.yaml`
- Endre `p_side_min.long: 0.80` i `entry_gating`

**Risiko:** Lav - allerede testet og verifisert i policy evaluation

**Klar for:** CANARY test → PROD

---

### F2. P2: Regime-Blocks (NEUTRAL×HIGH + DOWN×HIGH)

**Beskrivelse:** Blokker problematiske trend × vol kombinasjoner

**Forventet Effekt:**
- **TREND_NEUTRAL × HIGH:** -10,140 trades (30.3%)
- **TREND_DOWN × HIGH:** -1,606 trades (4.8%)
- **Total reduksjon:** ~11,746 trades (35.1%)

**Implementasjon:**
```yaml
entry_gating:
  stage0:
    block_trend_vol_combos:
      - trend: "TREND_NEUTRAL"
        vol: "HIGH"
      - trend: "TREND_DOWN"
        vol: "HIGH"
```

**Risiko:** Moderat - trenger replay-test for å verifisere PnL-effekt

**Klar for:** REPLAY test → CANARY → PROD

---

### F3. P3: Hybrid Sniper (HIGH Vol Kun i TREND_UP)

**Beskrivelse:** Tillat HIGH vol kun når trend er TREND_UP

**Forventet Effekt:**
- Blokker TREND_NEUTRAL × HIGH og TREND_DOWN × HIGH
- Tillat TREND_UP × HIGH (1,211 trades)
- Reduksjon: ~10,535 trades (31.5%)

**Implementasjon:**
```yaml
entry_v9_policy_sniper:
  allow_high_vol: true  # Men begrens via stage0
entry_gating:
  stage0:
    block_trend_vol_combos:
      - trend: "TREND_NEUTRAL"
        vol: "HIGH"
      - trend: "TREND_DOWN"
        vol: "HIGH"
```

**Risiko:** Moderat - mer aggressiv enn P2

**Klar for:** REPLAY test → CANARY → PROD

---

### F4. P4: Kombinasjon (P1 + P2)

**Beskrivelse:** Kombiner p_long 0.80 + regime-blocks

**Forventet Effekt:**
- P1 reduksjon: ~8,162 trades (p_long 0.67-0.80)
- P2 reduksjon: ~11,746 trades (problematiske kombinasjoner)
- Overlapp: Estimat ~15,000-18,000 trades totalt
- Ny total: ~15,000-18,000 trades (50-55% reduksjon)

**Implementasjon:**
- Kombiner P1 + P2 konfigurasjoner

**Risiko:** Høy - trenger grundig replay-test

**Klar for:** REPLAY test → CANARY

---

## G) Data som Mangler for Neste Forbedring

### G1. Performance Data per Regime

**Mangler:**
- PnL per trend × vol kombinasjon for SNIPER
- Win rate per trend × vol kombinasjon
- EV/trade per kombinasjon

**Eksisterer:**
- Trade count per kombinasjon ✅
- p_long distribusjon per kombinasjon ✅

**Trenger:**
- Replay med PnL-beregning per regime
- Sammenligning: NEUTRAL×HIGH vs UP×HIGH performance

---

### G2. Timing Quality Data

**Eksisterer:**
- TimingCritic modell trent ✅
- AVOID_TRADE / IMMEDIATE_OK / DELAY_BETTER labels ✅

**Mangler:**
- Performance forbedring hvis TimingCritic brukes som filter
- Optimal threshold for TimingCritic (nå >= 0.60 ga 0 trades)

**Trenger:**
- Sweep over TimingCritic thresholds (0.40, 0.45, 0.50, 0.55)
- Analyse av AVOID_TRADE trades (221 trades, 0% win rate, -178 bps avg)

---

### G3. Entry Critic Data

**Eksisterer:**
- EntryCritic modell trent ✅
- Policy C (EntryCritic >= 0.60) ga 0 trades ❌

**Mangler:**
- Optimal threshold for EntryCritic
- Performance forbedring med EntryCritic filter

**Trenger:**
- Sweep over EntryCritic thresholds (0.40, 0.45, 0.50, 0.55)
- Kombinasjon: p_long 0.80 + EntryCritic filter

---

### G4. Q4 Overlay Data

**Eksisterer:**
- Q4 × C_CHOP overlay aktiv ✅
- Q4 × B_MIXED overlay config mangler ❌
- Q4 × A_TREND overlay config mangler ❌

**Mangler:**
- Performance-effekt av Q4 × B_MIXED overlay
- Performance-effekt av Q4 × A_TREND overlay
- Optimal multipliers for hver overlay

**Trenger:**
- Replay med Q4 × B_MIXED overlay aktivert
- Replay med Q4 × A_TREND overlay aktivert
- AB-test av ulike multiplier-verdier

---

## H) Forslag til CANARY Rollout-Strategi

### H1. Phase 1: Policy B (p_long 0.80) - LOW RISK

**Rationale:** Allerede testet og verifisert i policy evaluation

**Steps:**
1. ✅ Policy evaluation allerede fullført
2. Opprett CANARY config: `GX1_V11_SNIPER_CANARY_P1_PLONG_080.yaml`
3. Deploy CANARY med parallell logging
4. Monitor over 1-2 uker:
   - Trade count vs baseline
   - Win rate vs baseline
   - PnL vs baseline
5. Hvis OK → PROD

**Expected Timeline:** 1-2 uker

---

### H2. Phase 2: Regime Blocks (P2) - MODERATE RISK

**Rationale:** Trenger replay-test først, men lav risiko for negative effekter

**Steps:**
1. Replay-test P2 på FULLYEAR 2025
2. Verifiser PnL-effekt (forventet: bedre win rate, færre trades)
3. Hvis OK → Opprett CANARY config
4. Deploy CANARY med parallell logging
5. Monitor over 2-3 uker
6. Hvis OK → PROD

**Expected Timeline:** 2-3 uker (inkl. replay-test)

---

### H3. Phase 3: Kombinasjon (P1 + P2) - HIGH RISK

**Rationale:** Større endring, trenger grundig testing

**Steps:**
1. Replay-test P1 + P2 kombinasjon på FULLYEAR 2025
2. Verifiser at reduksjon er forventet (~50-55%)
3. Verifiser at win rate og PnL forbedres
4. Hvis OK → Opprett CANARY config
5. Deploy CANARY med parallell logging
6. Monitor over 3-4 uker
7. Hvis OK → PROD

**Expected Timeline:** 3-4 uker (inkl. replay-test)

---

### H4. CANARY Deployment Pattern

**Config Structure:**
```yaml
meta:
  role: CANARY
  description: "Policy B: p_long >= 0.80"

entry_config: gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/ENTRY_V9_SNIPER_LONDON_NY_P1.yaml
```

**Monitoring:**
- Parallell logging: CANARY + BASELINE
- Compare metrics: trades/dag, win rate, PnL
- Alert hvis CANARY underpresterer signifikant

**Rollback:**
- Hvis CANARY underpresterer → deaktiver CANARY
- Keep baseline aktiv
- Analyser årsak før neste forsøk

---

## I) CLI-Kommandoer for Replay & Eval

### I1. Replay Full Year 2025

```bash
# SNIPER Baseline
bash scripts/run_replay.sh \
  gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_REPLAY_SHADOW_2025.yaml \
  2025-01-01 2025-12-31 \
  7 \
  runs/replay_shadow/SNIPER_BASELINE_FULLYEAR

# FARM Baseline
bash scripts/run_replay.sh \
  gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml \
  2025-01-01 2025-12-31 \
  7 \
  runs/replay_shadow/FARM_BASELINE_FULLYEAR
```

---

### I2. Policy Candidate Replay

```bash
# P1: p_long 0.80
# (Først opprett config med min_prob_long: 0.80)
bash scripts/run_replay.sh \
  gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_REPLAY_SHADOW_2025_P1.yaml \
  2025-01-01 2025-12-31 \
  7 \
  runs/replay_shadow/SNIPER_P1_PLONG_080

# P2: Regime blocks
# (Først opprett config med block_trend_vol_combos)
bash scripts/run_replay.sh \
  gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_REPLAY_SHADOW_2025_P2.yaml \
  2025-01-01 2025-12-31 \
  7 \
  runs/replay_shadow/SNIPER_P2_REGIME_BLOCKS

# P4: Kombinasjon (P1 + P2)
bash scripts/run_replay.sh \
  gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_REPLAY_SHADOW_2025_P4.yaml \
  2025-01-01 2025-12-31 \
  7 \
  runs/replay_shadow/SNIPER_P4_COMBINED
```

---

### I3. Policy Evaluation (Entry Policy Eval)

```bash
# Evaluer policy-kandidater på eksisterende RL dataset
python3 gx1/scripts/eval_entry_policies_v1.py \
  --dataset data/rl/sniper_shadow_rl_dataset_FULLYEAR_2025_RECAP.parquet \
  --output reports/rl/ENTRY_POLICY_EVAL_V2_FULLYEAR_2025_RECAP.md
```

---

### I4. Trade Analysis per Regime

```bash
# Analyser trades per regime (trend × vol)
python3 gx1/scripts/analyze_trades_by_regime.py \
  --trade_log runs/replay_shadow/SNIPER_P1_PLONG_080/trade_log_*.csv \
  --output reports/sniper/SNIPER_P1_REGIME_ANALYSIS.md
```

---

## J) Handlingspunkter (Prioritert)

### J1. Kort Sikt (Neste 1-2 Uker)

1. ✅ **Konsolider kunnskap** (dette dokumentet)
2. **Implementer P1 (p_long 0.80)** i CANARY config
3. **Deploy CANARY P1** og monitor
4. **Replay-test P2 (regime blocks)** på FULLYEAR 2025

---

### J2. Mellomlang Sikt (Neste 2-4 Uker)

5. **Analyser P2 replay-results** (PnL per regime)
6. **Implementer P2** i CANARY config hvis OK
7. **Deploy CANARY P2** og monitor
8. **Replay-test P4 (P1 + P2 kombinasjon)** hvis P1 og P2 er OK

---

### J3. Lang Sikt (Neste 1-3 Måneder)

9. **Analyser TimingCritic thresholds** (sweep 0.40-0.55)
10. **Analyser EntryCritic thresholds** (sweep 0.40-0.55)
11. **Implementer Q4 overlays** (B_MIXED, A_TREND) hvis nødvendig
12. **Optimaliser regime-block kombinasjoner** basert på PnL-data

---

## K) Nøkkelstatistikk (Sammendrag)

### K1. SNIPER Baseline (FULLYEAR 2025)

- **Trades:** 33,489
- **Trades/dag:** 132.9
- **Win rate:** 62.45%
- **Avg reward:** 350.14 bps
- **Reward/1k:** 350,144.06
- **P05 reward:** -76.32 bps
- **P95 reward:** 1,076.34 bps

**Distribution:**
- US session: 66.5%
- HIGH vol: 38.7%
- TREND_NEUTRAL × HIGH: 30.3%
- p_long >= 0.80: 75.6%

---

### K2. FARM Baseline (FULLYEAR 2025)

- **Trades:** 162
- **Trades/dag:** 0.44
- **Win rate:** 84.6%
- **EV/trade:** 117.67 bps
- **ASIA session:** 100%
- **LOW/MEDIUM vol:** 100%

---

### K3. Policy B (p_long >= 0.80) - Simulert

- **Trades:** 25,327 (24.4% reduksjon)
- **Win rate:** 66.23% (+3.78%)
- **Avg reward:** 384.77 bps (+34.63 bps)
- **Reward/1k:** 384,767.94 (+9.9%)
- **P05 reward:** -58.26 (bedre)

---

## L) Referanser

### L1. Rapporter

- `reports/rl/SNIPER_2025_FULLYEAR_RECAP_COMPLETE.md` - Hovedrapport
- `reports/rl/ENTRY_POLICY_EVAL_V1_FULLYEAR_2025_RECAP.md` - Policy evaluation
- `reports/rl/SHADOW_COUNTERFACTUAL_REPORT_FULLYEAR_2025_RECAP_V2.md` - Shadow analysis
- `reports/sniper/debug/SNIPER_2025_TRADE_FREQUENCY_DIAGNOSIS.md` - Frequency analysis
- `reports/sniper/debug/SNIPER_SAFETY_LAYERS_IN_CODE.md` - Safety layers mapping
- `reports/sniper/debug/SNIPER_OLD_VS_NEW_SAFETY_LAYERS.md` - FARM vs SNIPER comparison

### L2. Configs

- SNIPER: `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_REPLAY_SHADOW_2025.yaml`
- FARM: `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`

### L3. Datasett

- `data/rl/sniper_shadow_rl_dataset_FULLYEAR_2025_RECAP.parquet` (33,489 trades, 42,465 rows)

---

**Dokument Status:** ✅ Komplett  
**Neste Steg:** Implementer P1 (p_long 0.80) CANARY config og deploy

