# FARM Pipeline Integrity Overview

This note documents how the current FARM (FARM_V2B) stack is wired inside the
runner so we can reason about the entire “bar → entry → exit → log” flow.

## Entry pipeline

1. **Bar ingestion** – `_run_replay_impl` (or `run_once`) fetches the next M5
   candle set and hands it to `EntryManager.evaluate_entry`.
2. **Feature build** – `build_live_entry_features` (runtime_v9) and the TCN
   helpers create the feature bundle consumed by ENTRY_V9.
3. **Guards + regimes**
   * Big Brain / FARM regime detection (`farm_regime.infer_farm_regime`,
     `farm_guards`, `entry_v9_policy_farm_v2b`) decide whether the bar is even
     eligible (ASIA + LOW/MEDIUM ATR).
   * Stage‑0, brutal guard, telemetry kill‑switches, max concurrent trades,
     daily loss guard, and min time since last entry all live in
     `EntryManager.evaluate_entry`.
4. **Model inference** – ENTRY_V9 transformer (`entry_v9_policy_farm_v2b`)
   produces `prediction` (`p_long`, `p_short`, margin). Optional meta‑model
   scores (FARM_V2) are attached to `policy_state`.
5. **Trade creation** – when all gates pass:
   * `LiveTrade` is instantiated with bid/ask, ATR bucket, probabilities,
     metadata (session, regime, FARM extras).
   * `_ensure_exit_profile` on the runner stamps an `exit_profile` (from the
     active `exit_config`) and initialises the exit policy state.
   * `_record_entry_diag` logs the trade (id, timestamp, session, regime,
     `p_long`, spread, ATR bucket, exit_profile) and updates counters.
6. **State storage** – `LiveTrade` objects live in `GX1DemoRunner.open_trades`
   until they are closed. All per-trade exit state lives in `trade.extra`.

### Entry invariants

* If a policy defines an `exit_config`, every `LiveTrade` must have
  `trade.extra["exit_profile"]`. `_ensure_exit_profile` now raises a
  `RuntimeError` if it cannot set this.
* FARM entries must originate from ASIA session + LOW/MEDIUM ATR. We log every
  trade at creation time so mis‑routed entries are immediately visible.

## Exit pipeline

1. **Bar evaluation** – `_run_replay_impl` (or `run_once`) calls
   `ExitManager.evaluate_and_close_trades` after entries for the bar are
   processed.
2. **Tick/TP watchers disabled** – in FARM_V2B mode the TickWatcher and broker
   TP/SL are disabled; only `ExitManager` drives exits.
3. **ExitManager flow**
   * Snapshot `open_trades` and log `[EXIT] Evaluating {n}`.
   * For every trade, compute current bid/ask PnL, bars in trade, and choose
     the exit path based on `trade.extra["exit_profile"]`.
   * For `FARM_EXIT_V2_RULES_*`, call the per-runner singleton
     `ExitFarmV2Rules` instance (`exit_farm_v2_rules_policy.on_bar`), passing
     bid/ask close. The policy owns trade-specific state (bars held, MAE/MFE,
     trailing stop flags). When it returns an `ExitDecision`, ExitManager calls
     `request_close`.
   * `request_close` routes through the central ExitArbiter (slippage checks,
     allowed loss closers, etc.) and, on success, updates `open_trades`,
     telemetry, and `trade_log`.
   * The exit loop logs every requested close and produces a summary
     `[EXIT] Close summary: requested=X accepted=Y remaining=Z`.
4. **Exit policy state**
   * `ExitFarmV2Rules` stores per-trade state (bars held, MAE/MFE, trailing
     stop). It is reset for every trade in `_ensure_exit_profile` when the
     trade opens.
   * `exit_profile` is the key routing hint – if it is missing, a trade would
     completely skip exit evaluation, which is why we now fail fast when a
     policy defines an exit config.

### Exit invariants

* Every trade with `exit_profile` is evaluated on every bar. We log the number
  of open trades per bar and every time the policy signals a close.
* Exit decisions must result in `exit_time`, `exit_price`, and `pnl_bps`
  entries in the trade log; the diagnostic check script verifies this.

## Diagnostic tooling

* `gx1/execution/oanda_demo_runner.py` now maintains `entry_diag` counters and
  logs per-trade diagnostics (session/regime/p_long/spread/exit_profile).
* `exit_manager` logs `[EXIT]` messages per bar and per close decision so we
  can verify that rules are firing.
* A diagnostic exit config (`FARM_EXIT_V2_RULES_DIAG`) uses looser thresholds
  plus a max-hold failsafe to prove the replay plumbing closes trades.
* `scripts/check_trade_log.py` reads any replay trade log and reports:
  total trades, missing `exit_profile`, open vs closed counts, and a basic PnL
  distribution for closed trades.

## Q1 2025 replay baselines (bid/ask)

We now run every parity replay through the parallel harness:

```bash
export M5_DATA="data/raw/xauusd_m5_2025_bid_ask.parquet"
bash scripts/active/run_replay.sh \
  gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_DIAG.yaml \
  2025-01-02 2025-03-31 6 gx1/wf_runs/FARM_V2B_EXIT_DIAG_Q1_FULL
```

The default worker count is 6 (pass an explicit value to override).

### FARM_V2B + EXIT_DIAG (Q1 2025)

* Total trades: **674** (ASIA session only)
* Closed trades: **674 (100%)**
* Missing `exit_profile`: **0**
* Closed PnL (bps): **avg 3.83**, **median -3.91**, **min -30.90**, **max 209.17**, **win rate 7.27%**
* Trades/day ≈ **8.3**, EV/day ≈ **31.8 bps**
* Exit mix: ~10% TIMEOUT, 90% RULE_A trailing (no SL/SL_BE/TP2 hits in this diag config)
* Logs: `[EXIT] Evaluating …` on every bar, `[REPLAY] FARM_V2_RULES exit triggered …` per close,
  plus per-trade entry diagnostics in `gx1/wf_runs/FARM_V2B_EXIT_DIAG_Q1_FULL/logs/replay.log`.

### FARM_V2B + EXIT_A (Q1 2025 production profile)

Command (same window, 6 workers):

```bash
export M5_DATA="data/raw/xauusd_m5_2025_bid_ask.parquet"
bash scripts/active/run_replay.sh \
  gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_Q1.yaml \
  2025-01-02 2025-03-31 6 gx1/wf_runs/FARM_V2B_EXIT_A_Q1
```

Aggregates:

* Total trades: **36** (ASIA + LOW/MED regime only)
* Closed trades: **36 (100%)**
* Missing `exit_profile`: **0**
* Closed PnL (bps): **avg 119.65**, **median 188.26**, **min -234.45**, **max 203.78**, **win rate 91.67%**
* Trades/day ≈ **0.47**, EV/day ≈ **55.96 bps** (from `gx1/wf_runs/FARM_V2B_EXIT_A_Q1/results.json`)
* Exit mix: all closes via FARM_EXIT_V2_RULES_A trailing logic; no TIMEOUT/SL events in this window
* Entry diagnostics file (`farm_entry_diag_*.json`) shows 100% exit_profile coverage and regime stats.

These paired runs give us a clean Q1 bid/ask baseline:

| Policy | Trades | Closed | Miss exit_profile | Avg bps | EV/day |
| ------ | ------ | ------ | ----------------- | ------- | ------ |
| EXIT_DIAG | 674 | 674 (100%) | 0 | 3.83 | 31.8 |
| EXIT_A_Q1 | 36 | 36 (100%) | 0 | 119.65 | 55.96 |

Future production replays should match these invariants (exit_profile coverage,
per-bar `[EXIT]` logs, merged trade log + `results.json`), then we can safely
extend to longer windows.
