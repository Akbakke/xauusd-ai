# Exit Strategy Inventory

**SSoT for exit strategies in GX1_ENGINE.**  
Kun statisk analyse, ingen runtime-endringer. Bruk konfig som fasit.

---

## 1. Current Canonical Exit (aktiv i dag)

### Policy SSoT
- **Policy:** `GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml` (TRUTH/model_compare default)
- **exit_config:** `SNIPER_EXIT_RULES_A.yaml` → `type: FARM_V2_RULES`
- **exit_policies:** rule5=SNIPER_EXIT_RULES_A, rule6a=SNIPER_EXIT_RULES_ADAPTIVE
- **hybrid_exit_router:** HYBRID_ROUTER_V3 (model: `gx1/models/exit_router/exit_router_v3_tree.pkl`)

### Beslutningspunkt
- **Fil:** `gx1/execution/exit_manager.py`
- **Metode:** `evaluate_and_close_trades(candles)` (linje ~168)
- **Rekkefølge:**
  1. EXIT_POLICY_V2 (ML, hvis enabled) – pre-empts alt
  2. EXIT_FARM_V2_RULES (hovedpath)
  3. EXIT_FIXED_BAR_CLOSE (hvis enabled)
  4. EXIT_FARM_V1 (hvis enabled)

### Aktive komponenter (FARM_V2_RULES-path)
- **Implementasjon:** `gx1/policy/exit_farm_v2_rules.py` – `ExitFarmV2Rules`, `get_exit_policy_farm_v2_rules()`
- **Konfig:** `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_A.yaml`

### Parametre (SNIPER_EXIT_RULES_A)
| Parameter | Verdi | Beskrivelse |
|-----------|-------|-------------|
| enable_rule_a | true | Profit-capture |
| enable_rule_b | false | Fast loss-cut |
| enable_rule_c | false | Timeout-abandonment |
| rule_a_profit_min_bps | 5.0 | Min profit for exit |
| rule_a_profit_max_bps | 9.0 | Max profit target |
| rule_a_adaptive_threshold_bps | 4.0 | Aktivering av trailing |
| rule_a_trailing_stop_bps | 2.0 | Avstand på trailing stop |
| rule_a_adaptive_bars | 3 | Bars for adaptive sjekk |

### Inputs / signaler
- **Pris:** `current_bid`, `current_ask` fra `candles["bid_close"]`, `candles["ask_close"]`
- **Entry:** `entry_bid`, `entry_ask`, `entry_time`, `side`
- **PnL:** Beregnes via `compute_pnl_bps()`
- **State:** `bars_held`, `mae_bps`, `mfe_bps` (oppdatert per bar)
- **ATR:** `runtime_atr_bps` fra `_compute_runtime_atr_bps(candles)`
- **Regime:** Fra `trade.extra` / policy state

### Hybrid router (når aktiv)
- **Entry-punkt:** `entry_manager.py` – `exit_mode_selector.choose_exit_profile()` (linje ~5214)
- **Router:** `gx1/core/hybrid_exit_router.py` – `hybrid_exit_router_v3()`
- **Output:** `FARM_EXIT_V2_RULES_A` (RULE5) eller `FARM_EXIT_V2_RULES_ADAPTIVE_v1` (RULE6A)
- **Guardrail:** `range_edge_dist_atr >= 1.0` → override til RULE5

### Linjenivå – hvor exit avgjøres
- **Exit avgjøres:** `exit_manager.py` linje ~364–382 – `policy.on_bar()` returnerer `ExitDecision`
- **request_close kalles:** `exit_manager.py` linje ~456
- **Policy on_bar:** `exit_farm_v2_rules.py` – `ExitFarmV2Rules.on_bar(price_bid, price_ask, ts, atr_bps)`

### Type
- **Regelbasert** (Rule A: profit 5–9 bps + adaptive trailing 2 bps)
- **Hybrid:** Hvis router aktiv, velges RULE5 vs RULE6A per trade (ML decision tree)

---

## 2. Oversikt over eksisterende exit-strategier

| Strategy | Fil/Klasse | Type | Inputs | Kort beskrivelse | Status | Hvordan aktivere |
|----------|------------|------|--------|------------------|--------|-------------------|
| **RULE5 (FARM_V2_RULES)** | `gx1/policy/exit_farm_v2_rules.py` | Rules | bid/ask, bars, PnL, MAE/MFE | Profit 5–9 bps, trailing 2 bps, Rule B/C valgfrie | Aktiv (canonical) | exit_config type: FARM_V2_RULES |
| **RULE6A (ADAPTIVE)** | `gx1/policy/exit_farm_v2_rules_adaptive.py` | Rules+ATR | + ATR, MFE/MAE | TP1/TP2, breakeven, MFE-basert trailing | Aktiv (via router) | hybrid_exit_router + exit_policies.rule6a |
| **EXIT_POLICY_V2** | `gx1/exits/exit_policy_v2.py` | ML | ATR, PnL, bars, state | XGB-basert exit-modell | Deaktivert default | GX1_EXIT_POLICY_V2=1 + YAML |
| **FARM_V1** | `gx1/policy/exit_farm_v1_policy.py` | Rules | SL/TP/timeout | Frossen baseline, SL=-20, TP=8 | Eksperimentell | exit type: FARM_V1 |
| **FIXED_BAR_CLOSE** | `gx1/policy/exit_fixed_bar.py` | Rules (time) | bars | Fast eller tilfeldig N bars | Sanity-test | exit type: FIXED_BAR_CLOSE |
| **ExitCritic** | `gx1/execution/exit_critic_controller.py` | ML hybrid | exit snapshot | EXIT_NOW / SCALP_PROFIT override | Valgfri overlay | exit_critic i policy |
| **TickWatcher** | TickWatcher (live) | Tick | tick streams | Tick-basert exit | Av when FARM_V2 | tick_exit.enabled |
| **Hybrid Router V1/V2/V2B/V3** | `gx1/core/hybrid_exit_router.py` | ML | atr_pct, spread_pct, regime, range_* | Velger RULE5 vs RULE6A per trade | Aktiv (V3) | hybrid_exit_router i policy |

### Pros/cons (konseptuell)

| Strategy | Pros | Cons |
|----------|------|------|
| **RULE5** | Enkel, deterministisk, lav MAE-potensial | Faste nivåer, lite adaptivt |
| **RULE6A** | ATR-adaptiv, bedre i chop | Mer kompleks, kan gi mer close_to_MFE |
| **EXIT_POLICY_V2** | Full ML-flexibilitet | Krever modell, GX1_EXIT_POLICY_V2 |
| **FARM_V1** | Frossen baseline | Kun for FARM_V1 mode |
| **FIXED_BAR** | Enkel sanity | Ikke ekonomisk |
| **ExitCritic** | Kan kutte tap raskt | Overlay, ekstra modell |

---

## 3. Konseptuell sammenligning

### Høy-momentum (Q2-type)
- **RULE5:** Fast profit target 5–9 bps kan forlate penger på bordet.
- **RULE6A:** ATR-skalert TP/trailing kan fange mer i trends.
- **ExitCritic EXIT_NOW:** Kan kutte tap tidlig i falske breakouts.

### Chop/mean-revert (Q3-type)
- **RULE5:** 5–9 bps capture passer bra for chop.
- **RULE6A:** BE + trailing kan beskytte gevinster i volatil chop.

### Funn fra excursions_summary (BASE28, threshold 0.48)
- **Q2:** Høy close_to_MFE (p95 116.7 bps) → vi eksiterer ofte for tidlig.
- **Q3:** Lav MAE (median 6.8 bps) → RULE5 fungerer bra.

### Lavest risiko å teste først
1. **RULE6A pure** – Bytt exit_config til EXIT_FARM_V2_RULES_ADAPTIVE (alle trades RULE6A). Minst semantikk-endring, kun parameterbytte.
2. **RULE5 + strammere trailing** – Øk `rule_a_trailing_stop_bps` fra 2 til 2.5–3 for å redusere close_to_MFE.
3. **Hybrid V3 med lavere cutoff** – Senk `v3_range_edge_cutoff` for å øke RULE6A-andel i gunstig regime.

---

## 4. Shortlist for exit-sammenligning

| Prioritet | Strategi | Hva å teste | Risiko |
|-----------|----------|-------------|--------|
| 1 | **RULE6A pure** | exit_config → SNIPER_EXIT_RULES_ADAPTIVE (type EXIT_FARM_V2_RULES_ADAPTIVE) | Lav |
| 2 | **RULE5 trailing 3 bps** | rule_a_trailing_stop_bps: 3.0 i SNIPER_EXIT_RULES_A | Lav |
| 3 | **Hybrid V3 cutoff 0.8** | v3_range_edge_cutoff: 0.8 (mer RULE6A) | Medium |

### Anbefalt rekkefølge
1. Kjør quarters med **RULE6A pure** – sammenlign mae_bps og close_to_mfe_bps vs baseline.
2. Hvis RULE6A forverrer Q3: Test **RULE5 trailing 3 bps**.
3. Hvis hybrid ønskes: Test **cutoff 0.8** og sammenlign RULE5/RULE6A-fordeling.

---

## 5. Konfig-referanse (SSoT)

- **exit_config:** `policy.exit_config` → peker til én exit-YAML
- **exit.type:** FARM_V2_RULES | EXIT_FARM_V2_RULES_ADAPTIVE | FARM_V1 | FIXED_BAR_CLOSE
- **exit.params:** Regelspesifikke parametre
- **exit_policies.rule5 / rule6a:** Brukes av hybrid router
- **hybrid_exit_router.version:** HYBRID_ROUTER_V1 | V2 | V2B | V3
- **hybrid_exit_router.v3_range_edge_cutoff:** Guardrail (default 1.0)

---

## 6. Ferdig-definisjon

- [x] Vi vet eksakt hvilken exit vi bruker nå (FARM_V2_RULES/RULE5, evt. RULE6A via hybrid)
- [x] Vi har full oversikt over eksisterende alternativer
- [x] Vi kan velge 2–3 exits å sammenligne videre uten å gjette
