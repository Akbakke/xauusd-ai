# SNIPER & FARM Knowledge Capsule

**Generert:** 2025-12-28  
**Status:** Konsolidert quick-reference for AI-pair programming  
**Formål:** Kort oppsummering av FARM/SNIPER-konfigurasjon og policy-kandidater

**Autoritativ kilde:** `docs/system/SYSTEM_SPEC_SNIPER_FARM_UNIFIED.md`

---

## FARM: Hvordan den er konfigurert (1 side)

### Core Configuration

**Config Path:** `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`  
**Entry Config:** `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID/ENTRY_V9_FARM_V2B_PROD.yaml`

### Sessions & Vol Regimes

- **Sessions:** ASIA only (`allowed_sessions: ["ASIA"]`)
- **Vol Regimes:** LOW + MEDIUM only (`allowed_vol_regimes: ["LOW", "MEDIUM"]`)
- **HIGH vol:** Blokkert (`allow_high_vol: false`)
- **EXTREME vol:** Blokkert (`allow_extreme_vol: false`)

**Guard:** `farm_brutal_guard_v2()` i `gx1/policy/farm_guards.py` (hard filter)

### Entry Parameters

- **p_long threshold:** `min_prob_long: 0.68`
- **require_trend_up:** `false` (soft gate only)
- **enable_profitable_filter:** `false`
- **Long-only:** `allow_short: false`

### Exit Strategy

- **Router:** HYBRID_ROUTER_V3_RANGE (ML decision tree)
- **Exit Policies:** RULE5 + RULE6A (ML-routed)
- **Guardrail:** RULE6A kun aktivert når `range_edge_dist_atr < 1.0`
- **Features:** `range_pos`, `distance_to_range`, `range_edge_dist_atr`

### Performance (FULLYEAR 2025)

- **Trades:** 162 totalt
- **Trades/dag:** 0.44
- **Win rate:** 84.6%
- **EV/trade:** 117.67 bps
- **Max open trades:** 5

**Design-filosofi:** Low-frequency, high-precision ASIA-range sniper. Fokuserer på rolige, range-baserte markeder med lav volatilitet.

---

## SNIPER: Hvordan den er konfigurert (1 side)

### Core Configuration

**Config Path:** `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_REPLAY_SHADOW_2025.yaml`  
**Entry Config:** `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/ENTRY_V9_SNIPER_LONDON_NY.yaml`

### Sessions & Vol Regimes

- **Sessions:** EU, OVERLAP, US (`allowed_sessions: ["EU", "OVERLAP", "US"]`)
- **Vol Regimes:** LOW + MEDIUM + HIGH (`allowed_vol_regimes: ["LOW", "MEDIUM", "HIGH"]`)
- **HIGH vol:** Tillatt (`allow_high_vol: true`)
- **EXTREME vol:** Blokkert (`allow_extreme_vol: false`)

**Guard:** `sniper_guard_v1()` i `gx1/policy/farm_guards.py` (hard filter)

### Entry Parameters

- **p_long threshold:** `min_prob_long: 0.67`
- **Entry gating:** `p_side_min.long: 0.67`, `margin_min.long: 0.50`, `side_ratio_min: 1.25`
- **require_trend_up:** `false` (soft gate only)
- **Long-only:** `allow_short: false`

### Risk Guard (SNIPER Risk Guard V1)

- **Cooldown:** `cooldown_bars_after_entry: 1` (5 min)
- **Spread block:** `block_if_spread_bps_gte: 3500` (35 bps global, 30 bps US, 32 bps OVERLAP)
- **ATR block:** `block_if_atr_bps_gte: 25.0` (0.25%)
- **EXTREME vol:** Blokkert (`block_if_vol_regime_in: ["EXTREME"]`)
- **US clamp:** `extra_min_prob_long: 0.02` (effektivt 0.69)
- **OVERLAP clamp:** `extra_min_prob_long: 0.01` (effektivt 0.68)

### Size Overlays

- **Q4 × C_CHOP:** Aktiv (`sniper_q4_cchop_overlay.enabled: true`), 50% size i US
- **Q4 × B_MIXED:** Ikke aktiv (config mangler)
- **Q4 × A_TREND:** Ikke aktiv (config mangler)

### INACTIVE Filters (Problematisk)

- **Trend × Vol combo blocks:** `block_trend_vol_combos: []` (tom = ingen blokkering)
- **require_trend_up hard-filter:** Ikke aktiv (kun soft gate)
- **Max open trades override:** Config-verdi `max_open_trades_override: 1` ikke implementert (bruker 10)

### Performance (FULLYEAR 2025 - Policy A Baseline)

- **Trades:** 33,489 totalt
- **Trades/dag:** 132.9
- **Win rate:** 62.45%
- **Avg reward:** 350.14 bps
- **Reward/1k:** 350,144.06
- **Max open trades:** 10 (fra execution config)

**Design-filosofi:** High-frequency, EU/US high-vol strateg. Bredere trading window og høyere toleranse for volatilitet enn FARM.

---

## Regimekart: Trend × Vol × Session → Hvem Eier Hva?

| Session | Trend | Vol | FARM | SNIPER | Kommentar |
|---------|-------|-----|------|--------|-----------|
| ASIA | Any | LOW | ✅ | ❌ | FARM only |
| ASIA | Any | MEDIUM | ✅ | ❌ | FARM only |
| ASIA | Any | HIGH | ❌ | ❌ | Ingen (FARM blokkerer, SNIPER ikke i ASIA) |
| ASIA | Any | EXTREME | ❌ | ❌ | Hard block (ingen) |
| EU | Any | LOW | ❌ | ✅ | SNIPER only |
| EU | Any | MEDIUM | ❌ | ✅ | SNIPER only |
| EU | Any | HIGH | ❌ | ✅ | SNIPER only |
| EU | Any | EXTREME | ❌ | ❌ | Hard block (ingen) |
| OVERLAP | Any | LOW | ❌ | ✅ | SNIPER only (men 0 trades i praksis) |
| OVERLAP | Any | MEDIUM | ❌ | ✅ | SNIPER only (men 0 trades i praksis) |
| OVERLAP | Any | HIGH | ❌ | ✅ | SNIPER only (men 0 trades i praksis) |
| OVERLAP | Any | EXTREME | ❌ | ❌ | Hard block (ingen) |
| US | Any | LOW | ❌ | ✅ | SNIPER only (66.5% av alle trades) |
| US | Any | MEDIUM | ❌ | ✅ | SNIPER only |
| US | Any | HIGH | ❌ | ✅ | SNIPER only (38.7% av alle trades) |
| US | Any | EXTREME | ❌ | ❌ | Hard block (ingen) |

### SNIPER Trade Distribution (FULLYEAR 2025)

**Per Session:**
- US: 22,264 trades (66.5%)
- EU: 11,225 trades (33.5%)
- OVERLAP: 0 trades (0%)

**Per Vol Regime:**
- HIGH: 12,957 trades (38.7%)
- MEDIUM: 8,917 trades (26.6%)
- LOW: 11,615 trades (34.7%)
- EXTREME: 0 trades (0%) ✅

**Per Trend × Vol (Top 3):**
- TREND_NEUTRAL × LOW: 11,286 trades (33.7%)
- **TREND_NEUTRAL × HIGH: 10,140 trades (30.3%)** ⚠️
- TREND_NEUTRAL × MEDIUM: 8,114 trades (24.2%)
- TREND_DOWN × HIGH: 1,606 trades (4.8%) ⚠️

---

## De Tre Viktigste Svakhets-Punktene i Dagens SNIPER Baseline

### 1. TREND_NEUTRAL × HIGH: 10,140 trades (30.3%)

**Problem:** Choppy marked + høy volatilitet = høy risiko. Dette er den største enkeltstående kombinasjonen.

**Konsekvens:** Ville vært blokkert i FARM (HIGH vol ikke tillatt). Ingen eksplisitt block i SNIPER (`block_trend_vol_combos: []` er tom).

**Potensial:** -10,140 trades hvis blokkert.

---

### 2. TREND_DOWN × HIGH: 1,606 trades (4.8%)

**Problem:** Nedtrend + høy volatilitet = mot trend = høy risiko.

**Konsekvens:** Ville vært blokkert i FARM (HIGH vol ikke tillatt). Ingen eksplisitt block i SNIPER.

**Potensial:** -1,606 trades hvis blokkert.

**Total problematiske trades (NEUTRAL/DOWN × HIGH):** ~11,746 trades (35.1%)

---

### 3. p_long 0.67-0.80: 8,162 trades (24.4%)

**Problem:** Lavere p_long threshold (0.67) tillater trades som ville blitt filtrert ut med 0.80.

**Konsekvens:** Policy B (p_long >= 0.80) gir bedre performance:
- 25,327 trades (vs 33,489 baseline)
- 66.23% win rate (vs 62.45% baseline)
- 384,768 reward/1k (vs 350,144 baseline = +9.9%)
- Bedre risiko-profil (P05: -58.26 vs -76.32)

**Potensial:** -8,162 trades hvis threshold økes til 0.80, med forbedret win rate og reward.

---

## Policy-Kandidater: Kort Definisjon

### P1 (Policy B): p_long ≥ 0.80

**Beskrivelse:** Øk p_long threshold fra 0.67 til 0.80. Allerede testet i policy evaluation med umiddelbar forbedring.

**Forventet:** 25,327 trades (24.4% reduksjon), 66.23% win rate, +9.9% reward/1k.

**Risiko:** Lav (allerede verifisert).

---

### P2: Regime Blocks (TREND_NEUTRAL × HIGH + TREND_DOWN × HIGH)

**Beskrivelse:** Aktiver `block_trend_vol_combos` for å blokkere problematiske trend × vol kombinasjoner.

**Forventet:** ~11,746 færre trades (35.1% reduksjon).

**Risiko:** Moderat (trenger replay-test for å verifisere PnL-effekt).

---

### P3: Hybrid Sniper (HIGH Vol Kun i TREND_UP)

**Beskrivelse:** Tillat HIGH vol kun når trend er TREND_UP. Blokker TREND_NEUTRAL × HIGH og TREND_DOWN × HIGH.

**Forventet:** ~10,535 færre trades (31.5% reduksjon), beholder TREND_UP × HIGH (1,211 trades).

**Risiko:** Moderat (mer aggressiv enn P2).

---

### P4: Kombinasjon (P1 + P2)

**Beskrivelse:** Kombiner p_long 0.80 + regime blocks (TREND_NEUTRAL × HIGH, TREND_DOWN × HIGH).

**Forventet:** ~15,000-18,000 trades (50-55% reduksjon fra baseline).

**Risiko:** Høy (større endring, trenger grundig testing).

---

## TODO: Canary-Politikker Implementation

### Phase 1: P1 (p_long 0.80) - LOW RISK

- [ ] Opprett SNIPER CANARY-config for P1
  - **Fil:** `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/ENTRY_V9_SNIPER_LONDON_NY_P1.yaml`
  - **Endring:** `min_prob_long: 0.80`, `p_side_min.long: 0.80`
  - **Basert på:** `ENTRY_V9_SNIPER_LONDON_NY.yaml` (baseline)
  - **Rationale:** Policy B allerede testet i evaluation, lav risiko

- [ ] Opprett CANARY main config for P1
  - **Fil:** `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_SNIPER_CANARY_P1_PLONG_080.yaml`
  - **Basert på:** `GX1_V11_REPLAY_SHADOW_2025.yaml` (baseline)
  - **Endring:** `entry_config` peker til `ENTRY_V9_SNIPER_LONDON_NY_P1.yaml`
  - **Meta:** `role: CANARY`, `description: "Policy B: p_long >= 0.80"`

- [ ] Lag script for replay & eval av P1
  - **Fil:** `gx1/scripts/replay_sniper_canary_p1.sh`
  - **Funksjon:** Kjør replay på FULLYEAR 2025 med P1 config
  - **Output:** `runs/replay_shadow/SNIPER_P1_PLONG_080/`

---

### Phase 2: P2 (Regime Blocks) - MODERATE RISK

- [ ] Opprett SNIPER entry config for P2
  - **Fil:** `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/ENTRY_V9_SNIPER_LONDON_NY_P2.yaml`
  - **Endring:** Legg til `block_trend_vol_combos` i `entry_gating.stage0`:
    ```yaml
    entry_gating:
      stage0:
        block_trend_vol_combos:
          - trend: "TREND_NEUTRAL"
            vol: "HIGH"
          - trend: "TREND_DOWN"
            vol: "HIGH"
    ```
  - **Rationale:** Blokker problematiske kombinasjoner, moderat risiko

- [ ] Opprett CANARY main config for P2
  - **Fil:** `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_SNIPER_CANARY_P2_REGIME_BLOCKS.yaml`
  - **Basert på:** `GX1_V11_REPLAY_SHADOW_2025.yaml` (baseline)
  - **Endring:** `entry_config` peker til `ENTRY_V9_SNIPER_LONDON_NY_P2.yaml`
  - **Meta:** `role: CANARY`, `description: "P2: Regime blocks (NEUTRAL×HIGH, DOWN×HIGH)"`

- [ ] Lag script for replay & eval av P2
  - **Fil:** `gx1/scripts/replay_sniper_canary_p2.sh`
  - **Funksjon:** Kjør replay på FULLYEAR 2025 med P2 config
  - **Output:** `runs/replay_shadow/SNIPER_P2_REGIME_BLOCKS/`

---

### Phase 3: P4 (P1 + P2 Kombinasjon) - HIGH RISK

- [ ] Opprett SNIPER entry config for P4
  - **Fil:** `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/ENTRY_V9_SNIPER_LONDON_NY_P4.yaml`
  - **Endring:** Kombiner P1 (p_long 0.80) + P2 (regime blocks)
  - **Rationale:** Større endring, trenger grundig testing

- [ ] Opprett CANARY main config for P4
  - **Fil:** `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_SNIPER_CANARY_P4_COMBINED.yaml`
  - **Basert på:** `GX1_V11_REPLAY_SHADOW_2025.yaml` (baseline)
  - **Endring:** `entry_config` peker til `ENTRY_V9_SNIPER_LONDON_NY_P4.yaml`
  - **Meta:** `role: CANARY`, `description: "P4: Combined P1 (p_long 0.80) + P2 (regime blocks)"`

- [ ] Lag script for replay & eval av P4
  - **Fil:** `gx1/scripts/replay_sniper_canary_p4.sh`
  - **Funksjon:** Kjør replay på FULLYEAR 2025 med P4 config
  - **Output:** `runs/replay_shadow/SNIPER_P4_COMBINED/`

---

### Phase 4: Scripts & Tooling

- [ ] Lag script for replay & eval av alle P1/P2/P4
  - **Fil:** `gx1/scripts/replay_sniper_canary_all.sh`
  - **Funksjon:** Kjør alle CANARY-kandidater på FULLYEAR 2025
  - **Output:** Sammenlignende rapport

- [ ] Lag script for live CANARY-logging (baseline + canary side-by-side)
  - **Fil:** `gx1/scripts/monitor_sniper_canary.sh`
  - **Funksjon:** Monitor live CANARY vs baseline side-by-side
  - **Output:** Live logging av trades, win rate, PnL for begge

- [ ] Lag rapport-mal: SNIPER CANARY Weekly Report
  - **Fil:** `reports/sniper/SNIPER_CANARY_WEEKLY_REPORT_TEMPLATE.md`
  - **Innhold:**
    - Trade count (baseline vs canary)
    - Win rate (baseline vs canary)
    - PnL (baseline vs canary)
    - Regime distribution
    - Anomalies / alerts
    - Recommendation (continue/rollback)

---

### Phase 5: Testing & Validation

- [ ] Replay-test P2 på FULLYEAR 2025
  - **Verifiser:** PnL-effekt, trade count reduksjon, win rate
  - **Rapport:** `reports/sniper/SNIPER_P2_REPLAY_RESULTS.md`

- [ ] Replay-test P4 på FULLYEAR 2025
  - **Verifiser:** Kombinert effekt, trade count reduksjon, win rate
  - **Rapport:** `reports/sniper/SNIPER_P4_REPLAY_RESULTS.md`

- [ ] Sammenlign alle kandidater (Baseline, P1, P2, P4)
  - **Rapport:** `reports/sniper/SNIPER_CANARY_COMPARISON.md`
  - **Innhold:** Side-by-side sammenligning av alle metrics

---

## Nøkkelbegreper (Nomenklatur)

**Policy A:** Dagens SNIPER-baseline (GX1_V11_REPLAY_SHADOW_2025.yaml, p_long 0.67, 33,489 trades)

**Policy B / P1:** p_long ≥ 0.80 (25,327 trades, 66.23% win rate)

**P2:** Regime blocks (TREND_NEUTRAL × HIGH, TREND_DOWN × HIGH)

**P3:** Hybrid sniper (HIGH vol kun i TREND_UP)

**P4:** Kombinasjon P1 + P2 (p_long 0.80 + regime blocks)

**FARM PROD_BASELINE:** FARM_V2B_HYBRID_V3_RANGE_PROD (162 trades, 84.6% win rate, ASIA only)

**SNIPER Baseline:** GX1_V11_REPLAY_SHADOW_2025 (33,489 trades, 62.45% win rate, EU/US)

---

**Dokument Status:** ✅ Komplett  
**Neste Steg:** Implementer P1 CANARY config (Phase 1)

