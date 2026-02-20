# EXIT_BASELINE_SPEC — Exit strategy as built (read-only documentation)

**Scope:** Exit strategy in GX1 as it runs today. No new code; exact file paths, function names, config keys, and runtime flags.

**Reference run:** E2E full-year 2025 GO (`E2E_SANITY_20260218_165729`) reported `exit_type: FARM_V2_RULES`, `exit_profile: SNIPER_EXIT_RULES_A`, `router_enabled: false`, `exit_critic_enabled: false`.

**Engine root:** `/home/andre2/src/GX1_ENGINE`

---

## A) Policy wiring — what config drives exit and where it is loaded

| What | Where |
|------|--------|
| **Main policy (replay)** | `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml` |
| **Policy loader** | `gx1/execution/oanda_demo_runner.py`: `load_yaml_config(policy_path)`; `self.policy_path` / `self.policy` set in `__init__` (e.g. ~1527–1528, 1776–1850). |
| **Exit config key** | Policy YAML: `exit_config: gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_A.yaml` |
| **Resolution** | `oanda_demo_runner.py` ~1776–1850: `exit_cfg_path = workspace_root / exit_config_raw`; `exit_cfg = load_yaml_config(exit_cfg_path)`; `exit_config_name = exit_cfg_path.stem` → **SNIPER_EXIT_RULES_A**. |
| **Exit YAML (RULE5)** | `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_A.yaml` — top-level `exit.type: FARM_V2_RULES`, `exit.params`: `enable_rule_a`, `rule_a_*`, etc. |
| **Optional router** | Policy YAML: `hybrid_exit_router` (version, model_path, v3_range_edge_cutoff). If present, `ExitModeSelector` can choose RULE5 vs RULE6A per trade. |
| **Optional exit_policies** | Policy YAML: `exit_policies.rule5` / `exit_policies.rule6a` — paths to SNIPER_EXIT_RULES_A and SNIPER_EXIT_RULES_ADAPTIVE. |
| **RunIdentity / observability** | `exit_type` and `exit_config_name` set on runner in `oanda_demo_runner.py` ~1936–1937, 3487; chunk_footer reads `runner.exit_type`, `runner.exit_config_name`, `runner.policy` for `exit_profile` / `router_enabled` / `exit_critic_enabled`. |

---

## B) Runtime call chain — runner loop → exit evaluator → rule set → trade close

1. **Replay bar loop**  
   `gx1/execution/oanda_demo_runner.py`: Replay invokes `_simulate_tick_exits_for_bar_impl(bar_ts, candle_row)` per bar (~14430, 16454). That uses `candle_row` (bid/ask columns) and calls exit policies.

2. **Unified exit entry point**  
   Live/replay both go through **ExitManager.evaluate_and_close_trades(candles)**:
   - **Live:** `GX1DemoRunner.evaluate_and_close_trades` → `_evaluate_and_close_trades_impl` → `exit_manager.evaluate_and_close_trades(candles)` (~11927–11928, 17010–17011).
   - **Replay:** Inside `_simulate_tick_exits_for_bar_impl`, for each bar the runner calls exit logic that ends up in the same ExitManager paths (FARM_V2_RULES on_bar, etc.).

3. **ExitManager.evaluate_and_close_trades**  
   `gx1/execution/exit_manager.py` ~168–590:
   - Gets `current_bid` / `current_ask` from `candles` last row; `runtime_atr_bps = _compute_runtime_atr_bps(candles)`.
   - Order of evaluation: Exit Policy V2 (if enabled) → EXIT_FARM_V2_RULES_ADAPTIVE (RULE6A) → **EXIT_FARM_V2_RULES (RULE5)**.
   - For **FARM_V2_RULES**: per open trade with `exit_profile` not RULE6A/ADAPTIVE, gets or creates per-trade state from `exit_farm_v2_rules_factory`, then calls **policy.on_bar(price_bid, price_ask, ts, atr_bps)** (~450–454).
   - If `exit_decision` is not None: `request_close(...)`, remove from open_trades, `_teardown_exit_state`, `record_realized_pnl`, **`_log_trade_close_with_metrics`** (~556–573).

4. **Replay-specific FARM_V2_RULES path**  
   `gx1/execution/oanda_demo_runner.py` ~16662–16867: gets `exit_profile` from `trade.extra`; if FIXED_BAR_CLOSE uses fixed-bar policy; if RULE6A uses RULE6A policy; else **FARM_V2_RULES**: ensures state, then `decision = policy.on_bar(price_bid, price_ask, ts, atr_bps)` (~16844). On decision: request_close, remove trade, log.

5. **Rule implementation (RULE5)**  
   `gx1/policy/exit_farm_v2_rules.py`: **ExitFarmV2Rules.on_bar(price_bid, price_ask, ts, atr_bps)** (~172). Uses only **price and internal state** (entry_bid/entry_ask, bars_held, mae_bps, mfe_bps, rule_a_trailing_high, etc.). No model logits. Returns **ExitDecision(exit_price, reason, bars_held, pnl_bps, mae_bps, mfe_bps)** or None. Reasons: e.g. `RULE_A_PROFIT`, `RULE_A_TRAILING`, `RULE_B_FAST_LOSS`, `RULE_C_TIMEOUT`.

6. **Factory / boot**  
   `oanda_demo_runner.py` ~1947–1973: when `exit.type == "FARM_V2_RULES"`, calls `get_exit_policy_farm_v2_rules(...)` from `gx1.policy.exit_farm_v2_rules` with params from `exit_cfg["exit"]["params"]`; stores in `self.exit_farm_v2_rules_policy` and `self.exit_farm_v2_rules_factory`; per-trade state in `self.exit_farm_v2_rules_states`.

---

## C) Inputs — what exit actually uses

| Input | Source | Used by |
|-------|--------|--------|
| **price_bid, price_ask** | Current bar: `candles["bid_close"].iloc[-1]`, `candles["ask_close"].iloc[-1]` (exit_manager ~230–231); replay passes bar bid/ask. | FARM_V2_RULES.on_bar: PnL and exit price. |
| **ts (timestamp)** | Bar time `candles.index[-1]`. | FARM_V2_RULES: logging / state only; no time-based rule in SNIPER_EXIT_RULES_A (rule_c is bar-count based). |
| **atr_bps** | `_compute_runtime_atr_bps(candles)` in exit_manager. | Passed to on_bar; RULE5 (SNIPER_EXIT_RULES_A) does not use it in the enabled rules (rule_a only). RULE6A uses it. |
| **entry_bid, entry_ask** | From trade at `reset_on_entry()` (set when trade opens). | PnL in on_bar: `compute_pnl_bps(entry_bid, entry_ask, price_bid, price_ask, side)`. |
| **Per-trade state** | bars_held, mae_bps, mfe_bps, rule_a_trailing_high, rule_a_trailing_active. | Rule A/B/C logic. |
| **Model / Transformer** | Not used. Exit is rule-based (RULE5/RULE6A). | — |
| **Regime / session** | Used only by **ExitModeSelector** (hybrid_exit_router) to choose exit_profile at entry time. Not passed into on_bar. | Router input (atr_pct, spread, session, regime, range_*). |

**Conclusion:** Exit is **rule-based**. RULE5 (SNIPER_EXIT_RULES_A) uses: current bid/ask, entry bid/ask, bars_held, and internal MAE/MFE/trailing state. No ML exit logits; no Transformer exit output.

---

## D) Outputs / artefacts — what confirms exit in a run-dir

| Artefact | Path / location | Set by |
|----------|------------------|--------|
| **chunk_footer.json** | `run_dir/chunk_0/chunk_footer.json` | `gx1/execution/replay_chunk.py` ~841–853: `exit_profile` (runner.exit_config_name or "RULE6A_PURE"), `exit_type` (runner.exit_type), `router_enabled` (bool(policy.get("hybrid_exit_router"))), `exit_critic_enabled` (bool(policy.get("exit_critic",{}).get("enabled", False))). |
| **trade_journal** | `run_dir/chunk_0/trade_journal/` (index + events) | `exit_manager._log_trade_close_with_metrics` → `trade_journal.log_exit_summary(...)` and `trade_journal.log(EVENT_TRADE_CLOSED, {exit_time, exit_price, exit_profile, exit_reason, bars_held, pnl_bps, ...})` — `gx1/execution/exit_manager.py` ~1163–1194; `gx1/monitoring/trade_journal.py` log_exit_summary. |
| **EXIT_TRIGGERED events** | Same trade_journal | `exit_manager.py` ~476–497: `trade_journal.log_exit_event(...)` and `trade_journal.log(EVENT_EXIT_TRIGGERED, {...})` when FARM_V2_RULES triggers. |
| **exits_*.jsonl** | Live: `log_dir/exits/exits_YYYYMMDD.jsonl`. Replay: `log_dir/exits/exits_YYYYMMDD_YYYYMMDD.jsonl` (used by report generator). | `oanda_demo_runner.py`: `exit_audit_path` ~3234–3236; `_log_exit_audit(...)` appends one JSONL line per close (~5742, 5768). |
| **EXIT_COVERAGE_SUMMARY.json** | `output_dir/EXIT_COVERAGE_SUMMARY.json` | `gx1/scripts/replay_eval_gated_parallel.py` ~3627: `atomic_write_json(json_path, summary)`. Summary includes totals, per-chunk stats, `truth_exit_journal_ok`. |
| **EXIT_COVERAGE_SUMMARY.md** | `output_dir/EXIT_COVERAGE_SUMMARY.md` | Same script ~3655: markdown table. |
| **RUN_COMPLETED.json** | `run_dir/RUN_COMPLETED.json` | Written by replay completion contract; no exit-specific fields. |

---

## E) Flags / env vars that affect exit

| Env / config | Effect |
|--------------|--------|
| **Policy YAML: exit_config** | Path to exit YAML; determines exit.type (FARM_V2_RULES, EXIT_FARM_V2_RULES_ADAPTIVE, FIXED_BAR_CLOSE, FARM_V1) and params. |
| **Policy YAML: hybrid_exit_router** | If set, router_enabled=True in chunk_footer; EntryManager uses ExitModeSelector.choose_exit_profile(...) at entry to set trade.extra["exit_profile"] (RULE5 vs RULE6A). |
| **Policy YAML: exit_critic** | If exit_critic.enabled true, exit_critic_enabled=True in chunk_footer; ExitManager may call exit_critic_controller.maybe_apply_exit_critic(...) before FARM_V2_RULES.on_bar; can force EXIT_NOW or SCALP_PROFIT. |
| **GX1_EXIT_POLICY_V2** | If "1", Exit Policy V2 (e.g. ExitPolicyV2 from exit_policy_v2.py) is loaded and evaluated first in evaluate_and_close_trades; can pre-empt other exits. |
| **exit.params in exit YAML** | For FARM_V2_RULES: enable_rule_a/b/c, rule_a_profit_min_bps, rule_a_profit_max_bps, rule_a_trailing_stop_bps, rule_a_adaptive_threshold_bps, rule_a_adaptive_bars, rule_b_mae_threshold_bps, rule_b_max_bars, rule_c_timeout_bars, rule_c_min_profit_bps, force_exit_bars, verbose_logging, debug_trade_ids. |
| **exit_control (policy)** | allowed_loss_closers, require_trade_open — used by ExitArbiter when request_close is called. |

No GX1_* env vars are required solely for exit in the baseline (SNIPER_EXIT_RULES_A + FARM_V2_RULES); exit is driven by policy YAML and runner state.

---

## Relevante filer (prioritert)

1. **gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml** — Policy med exit_config, hybrid_exit_router, exit_control.
2. **gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_A.yaml** — Exit type FARM_V2_RULES og params (rule_a/b/c).
3. **gx1/execution/oanda_demo_runner.py** — Policy load, exit_cfg_path, exit_type/exit_config_name, FARM_V2_RULES boot (~1934–2000), replay exit branch (~16642–16867), exit_audit path (~3234–3236, 16415).
4. **gx1/execution/exit_manager.py** — evaluate_and_close_trades (~168), RULE6A block (~232–308), FARM_V2_RULES block (~310–590), _log_trade_close_with_metrics (~1130–1195).
5. **gx1/policy/exit_farm_v2_rules.py** — ExitFarmV2Rules, on_bar (~172), get_exit_policy_farm_v2_rules (~390).
6. **gx1/execution/replay_chunk.py** — chunk_footer payload exit_profile, exit_type, router_enabled, exit_critic_enabled (~840–853).
7. **gx1/scripts/replay_eval_gated_parallel.py** — EXIT_COVERAGE_SUMMARY.json/.md write (~3626–3658).
8. **gx1/execution/entry_manager.py** — choose_exit_profile (~5227), _ensure_exit_profile (~3986), exit_profile on trade.extra.
9. **gx1/policy/exit_hybrid_controller.py** — hybrid_exit_router_v3 / ExitModeSelector.
10. **gx1/core/hybrid_exit_router.py** — hybrid_exit_router_v3(ctx), model load (exit_router_v3_tree.pkl).
11. **gx1/monitoring/trade_journal.py** — log_exit_summary, EVENT_TRADE_CLOSED, EVENT_EXIT_TRIGGERED.
12. **gx1/execution/exit_critic_controller.py** — maybe_apply_exit_critic (optional overlay).
13. **gx1/policy/exit_farm_v2_rules_adaptive.py** — RULE6A (on_bar, get_exit_policy_farm_v2_rules_adaptive).
14. **gx1/utils/pnl.py** — compute_pnl_bps(entry_bid, entry_ask, exit_bid, exit_ask, side).
15. **docs/EXIT_STRATEGY_INVENTORY.md** — Overordnet oversikt over exit-variantene.

---

*Dokumentet beskriver kun eksisterende oppførsel; ingen kodeendringer.*
