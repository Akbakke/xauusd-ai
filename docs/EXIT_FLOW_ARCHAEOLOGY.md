# Exit-flow archaeology — dagens exit slik den er (read-only)

**Referanserun:** `GX1_DATA/reports/truth_e2e_sanity/E2E_SANITY_20260218_165729`  
**Exit i run:** `exit_type=FARM_V2_RULES`, `exit_profile=SNIPER_EXIT_RULES_A`, 391 trades, alle lukket via Rule A.

---

## 1. Exit call-graph

### Steder exit blir besluttet (fil / klasse / funksjon / ca. linje)

| Sted | Fil | Klasse | Funksjon | Linje |
|------|-----|--------|----------|--------|
| Replay bar → exit per bar | `gx1/execution/oanda_demo_runner.py` | GX1DemoRunner | `_simulate_tick_exits_for_bar_impl` | 16454 |
| Kalles fra replay-loop | `gx1/execution/oanda_demo_runner.py` | GX1DemoRunner | (replay bar loop) | 14430 |
| Samlet exit-evaluering (live/replay) | `gx1/execution/exit_manager.py` | ExitManager | `evaluate_and_close_trades` | 168 |
| FARM_V2_RULES (RULE5) i ExitManager | `gx1/execution/exit_manager.py` | ExitManager | `evaluate_and_close_trades` (blokk FARM_V2_RULES) | 310–590 |
| RULE5 on_bar-kall (ExitManager) | `gx1/execution/exit_manager.py` | ExitManager | `evaluate_and_close_trades` | 450 |
| Replay-spesifikk FARM_V2_RULES | `gx1/execution/oanda_demo_runner.py` | GX1DemoRunner | `_simulate_tick_exits_for_bar_impl` | 16784–16867 |
| Regelmotoren (Rule A/B/C) | `gx1/policy/exit_farm_v2_rules.py` | ExitFarmV2Rules | `on_bar` | 172–331 |

### Kallkjede: «ny bar i replay» → «trade lukkes»

1. Replay leser en bar; for hver bar kalles `_simulate_tick_exits_for_bar(ts, candle_row)` — **oanda_demo_runner.py ~14430** (inne i bar-loop).
2. `_simulate_tick_exits_for_bar` → `_simulate_tick_exits_for_bar_impl(bar_ts, candle_row)` — **~17044–17049**.
3. I `_simulate_tick_exits_for_bar_impl`: for hver åpen trade hentes `exit_profile` fra `trade.extra` (**~16662–16665**). Hvis ikke FIXED_BAR_CLOSE og ikke RULE6A/ADAPTIVE, brukes **FARM_V2_RULES**-grenen (**~16784–16867**).
4. For FARM_V2_RULES: state hentes/opprettes via `_init_farm_v2_rules_state`; deretter `decision = policy.on_bar(price_bid, price_ask, ts, atr_bps)` — **~16844** (policy = `ExitFarmV2Rules`-instans).
5. **exit_farm_v2_rules.py**: `ExitFarmV2Rules.on_bar(price_bid, price_ask, ts, atr_bps)` (**~172**) beregner pnl_bps, oppdaterer mae/mfe og trailing-state, evaluerer Rule A/B/C; returnerer `ExitDecision` eller None.
6. Hvis `decision is not None`: runner kaller `request_close(...)` (**~16848**), fjerner trade fra `open_trades`, tear-down av exit-state; deretter `_log_trade_close_with_metrics` og trade_journal/exit_audit skrives (exit_manager tilsvarer **~534–573** for live-path).

Alternativ live-path: `evaluate_and_close_trades(candles)` kalles (**oanda_demo_runner ~11927–11928, 17010–17011**); **exit_manager.py** `evaluate_and_close_trades` (**~168**) gjør det samme: henter bid/ask fra `candles`, for hver trade med FARM_V2_RULES kalles `policy.on_bar(...)` (**~450**).

---

## 2. Inputs (kun det som faktisk leses i kode)

| Input | Hvor den kommer fra | Trade vs market | Causal |
|-------|---------------------|------------------|--------|
| **price_bid, price_ask** | Siste rad i `candles` (exit_manager ~230–231) eller `candle_row` i replay (~16840). Fra bar-data. | Market (nåværende bar) | Ja |
| **entry_bid, entry_ask** | Satt på trade ved `reset_on_entry()` (exit_farm_v2_rules.py ~151–154); leses i `on_bar` for `compute_pnl_bps`. | Trade | Ja |
| **side** | Trade.side; satt i `reset_on_entry` (~155). | Trade | Ja |
| **bars_held** | Telles i `on_bar` (self.bars_held += 1, ~186). | Trade (state) | Ja |
| **mae_bps, mfe_bps** | Oppdateres i `on_bar` fra pnl_bps (~192–195). | Trade (state) | Ja |
| **rule_a_trailing_active, rule_a_trailing_high** | Oppdateres i `on_bar` (~201–202, 209–214). | Trade (state) | Ja |
| **ts** | Bar-timestamp; brukes kun i logging (_maybe_log_state). | Market | Ja |
| **atr_bps** | Sendes inn til `on_bar` (signatur ~172); **brukes ikke** i metodens body (verifisert: ingen referanse 183–331). | — | — |

Ingen model logits, transformer-output eller policy_decisions leses i `on_bar`. **on_bar bruker kun price + intern state.**

---

## 3. Exit-state (per trade)

| State | Beskrivelse | Initieres | Oppdateres | Slettes |
|-------|-------------|-----------|------------|--------|
| **entry_bid, entry_ask, entry_price, entry_ts, side, trade_id** | Entry-info for PnL og exit-pris. | `reset_on_entry()` — exit_farm_v2_rules.py ~151–164 | Ikke i on_bar | Ved ny `reset_on_entry` |
| **bars_held** | Antall bar siden entry. | reset_on_entry (~157) | on_bar: += 1 (~186) | reset_on_entry |
| **mae_bps, mfe_bps** | Worst/best PnL så langt. | reset_on_entry (~158–159) | on_bar (~192–195) | reset_on_entry |
| **rule_a_trailing_active** | Om Rule A trailing er aktiv. | reset_on_entry (~161) | on_bar: True når pnl >= adaptive_threshold i de første adaptive_bars (~209–214) | reset_on_entry |
| **rule_a_trailing_high** | Høyeste PnL mens trailing er aktiv. | reset_on_entry (~162) | on_bar (~201–202, 213) | reset_on_entry |
| **_last_logged_bar** | For verbose logging. | reset_on_entry (~164) | _maybe_log_state | reset_on_entry |

**Hvor state holdes:** I `ExitFarmV2Rules`-instansen per trade. Runner holder instanser i `exit_farm_v2_rules_states[trade.trade_id]` (exit_manager ~434–435; oanda_demo_runner ~16822).  
**Initieres:** Ved første exit-eval for trade ved `_init_farm_v2_rules_state` → `policy.reset_on_entry(entry_bid, entry_ask, ...)` (oanda_demo_runner ~3881–3926; exit_manager ~437–439).  
**Slettes:** `_teardown_exit_state(trade.trade_id)` fjerner state fra `exit_farm_v2_rules_states` etter lukking (oanda_demo_runner ~16856; exit_manager ~556).

---

## 4. Exit-reasons

### Alle reason-strenger (fra kode)

| Reason | Fil | Linje | Regel |
|--------|-----|--------|--------|
| RULE_A_TRAILING | exit_farm_v2_rules.py | 236 | Rule A trailing stop |
| RULE_A_PROFIT | exit_farm_v2_rules.py | 256 | Rule A profit target |
| RULE_B_FAST_LOSS | exit_farm_v2_rules.py | 279 | Rule B |
| RULE_C_TIMEOUT | exit_farm_v2_rules.py | 301 | Rule C |
| RULE_FORCE_TIMEOUT | exit_farm_v2_rules.py | 318 | force_exit_bars |
| EXIT_CRITIC_EXIT_NOW | exit_manager.py | 383 | ExitCritic (valgfri) |

### Hvor de logges

- **exit_manager**: Ved exit kalles `_log_trade_close_with_metrics` → `trade_journal.log_exit_summary(..., exit_reason=...)` og `trade_journal.log(EVENT_TRADE_CLOSED, {..., "exit_reason": exit_reason})` — **exit_manager.py ~1163–1194**.  
- **Replay FARM_V2_RULES**: `request_close(..., reason=exit_reason)` deretter exit_audit og trade_journal fra runner (tilsvarende exit_manager-path).  
- **exits_*.jsonl**: `_log_exit_audit(trade_id, source, reason, pnl_bps, ...)` — **oanda_demo_runner.py ~5742, 5768**; `source` er f.eks. "EXIT_FARM_V2_RULES", `reason` er exit_reason-strengen.

### I referanserunnen (E2E_SANITY_20260218_165729)

Fra `chunk_0/logs/exits/exits_20260218.jsonl`:  
- **reason:** RULE_A_TRAILING (214), RULE_A_PROFIT (159), TP_TICK (15), SL_TICK (3).  
- **source:** EXIT_FARM_V2_RULES (373), TICK (18).

Altså: Rule A (trailing + profit) og noen TICK (TP/SL) er brukt. **Rule B, Rule C og RULE_FORCE_TIMEOUT** er ikke brukt i denne runnen (SNIPER_EXIT_RULES_A har kun Rule A enabled; B/C er disabled i YAML). De er ikke «døde» i kode, men inaktive for denne config.

---

## 5. Artefakter (denne runnen)

| Artefakt | Eksakt sti | Finnes? | Merknad |
|----------|-------------|--------|--------|
| chunk_footer.json | `replay/chunk_0/chunk_footer.json` | Ja | `exit_type`, `exit_profile`, `router_enabled`, `exit_critic_enabled` (replay_chunk.py ~841–853). |
| policy_decisions_*_MERGED.parquet | `replay/policy_decisions_E2E_SANITY_*_MERGED.parquet` | Ja | `exit_policy_id`: SNIPER_EXIT_RULES_A (392), None (42319). |
| logs/exits/*.jsonl | `replay/chunk_0/logs/exits/exits_20260218.jsonl` | Ja | Én linje per close: ts, trade_id, source, reason, pnl_bps, accepted, bars_in_trade. |
| trade_journal_index.csv | `replay/chunk_0/trade_journal/trade_journal_index.csv` | Ja | I TRUTH genereres index post-merge fra trade_outcomes (SSoT); runtime skriver ikke CSV i TRUTH (trade_journal.py ~900–906). |
| trade_journal.jsonl | `replay/chunk_0/trade_journal/trade_journal.jsonl` | **Nei** | Definisjon: trade_journal.py ~139 `self.journal_path = self.journal_dir / "trade_journal.jsonl"`. I TRUTH/replay er event-stream (log()) ikke nødvendigvis skrivet til denne filen; referanserun har den ikke. |
| Per-trade JSON | `replay/chunk_0/trade_journal/{key}.json` | (ikke verifisert) | _write_trade_json skriver til trade_json_dir / f"{key}.json" (~353). |

**Hvorfor trade_journal.jsonl ikke finnes i TRUTH:** I TRUTH modus skippes CSV-skriving til index (trade_journal.py ~202, 900–906). Journal-klassen har `journal_path` for jsonl, men den samlede event-stream (log()-kall som skriver til jsonl) er enten ikke brukt i replay/TRUTH eller skrives et annet sted. Referanserunnen har ingen `trade_journal.jsonl`; exit-hendelser er i `logs/exits/exits_*.jsonl` og i trade_journal per-trade-struktur (exit_summary i minnet og skrevet til per-trade JSON ved _write_trade_json).

---

## 6. Router: faktisk rolle

- **Blir router brukt til å velge exit i denne runnen?**  
  **Nei, i praksis.** Alle 392 enter-beslutninger har `exit_policy_id=SNIPER_EXIT_RULES_A` (policy_decisions parquet). Det betyr at hver trade fikk én exit-profil; ingen RULE6A. Enten er `exit_hybrid_enabled`/`exit_mode_selector` ikke satt (default False/None) i replay, eller router returnerer konsekvent et profilnavn som mappes til samme exit_config (SNIPER_EXIT_RULES_A). Kode: `exit_profile` settes enten av `choose_exit_profile` (entry_manager ~5227) kun hvis `getattr(self, "exit_hybrid_enabled", False)` og `getattr(self, "exit_mode_selector", None)` — ellers settes den av `_ensure_exit_profile(trade, ...)` fra `exit_config_name` (oanda_demo_runner ~3986–4005).

- **Hva bestemmer exit_profile i praksis?**  
  **exit_config_name** (stem av exit-config YAML, f.eks. SNIPER_EXIT_RULES_A). Når hybrid ikke er aktiv, sørger `_ensure_exit_profile` (oanda_demo_runner ~3986) for at `trade.extra["exit_profile"] = exit_config_name` (eller stem fra policy.get("exit_config")).

- **Er router en beslutning, eller bare et label ved entry?**  
  **Kun et label ved entry.** Router (`ExitModeSelector.choose_exit_profile` i gx1/policy/exit_hybrid_controller.py ~53) kalles fra entry_manager ~5227 én gang ved trade-opprettelse og setter `trade.extra["exit_profile"]`. Selve exit-beslutningen tas senere i `on_bar`; ingen ny router-kall per bar. I denne runnen er alle trades merket med samme profil (SNIPER_EXIT_RULES_A), så det er ingen «switching» mellom RULE5 og RULE6A underveis.

---

*Ren lesing/arkæologi; ingen kodeendringer.*
