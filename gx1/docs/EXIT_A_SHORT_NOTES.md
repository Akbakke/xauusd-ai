# EXIT_A Short-Path Notes

These bullets summarise what the current FARM_V2B + EXIT_A stack does (and does
not) do for short trades so we have a baseline before wiring any short-specific
policy.

## Entry side (ENTRY_V9 + FARM_V2B)

- ENTRY_V9 produces `prob_long`, `prob_short`, and `prob_neutral`. The downstream
  `EntryPrediction` object keeps both probabilities and `EntryManager` will
  always choose the higher of the two when `should_enter_trade(...)` is called.
- The FARM_V2B policy (`entry_v9_policy_farm_v2b.py`) filters candidates solely
  on `p_long >= min_prob_long`. There is *no* `min_prob_short` hook and setting
  `allow_short: true` only toggles a TODO warning – shorts are still discarded
  during the p_long filter.
- Stage-0 / brutal guard V2 currently enforces ASIA session + (LOW ∪ MEDIUM)
  ATR buckets for every signal; there is no session/vol override specific to
  short trades.
- Entry gating (`entry_gating.p_side_min`) is symmetric, so after the p_long
  filter a short trade *could* pass if `prob_short > prob_long`, but again the
  brutal p_long filter removes almost all short-heavy bars before they reach
  that stage. Guard, telemetry, daily-loss, and cooldown checks are side-agnostic.

## Exit side (EXIT_FARM_V2_RULES / RULE5)

- `ExitFarmV2Rules.reset_on_entry` explicitly raises `ValueError` for any
  `side != "long"`. All internal state (PnL tracking, trailing logic) assumes a
  long position and uses `entry_ask`/`entry_bid` accordingly. No short handling
  exists today.
- `compute_pnl_bps` *can* evaluate short PnL if we passed `side="short"`, but the
  exit policy never reaches that code path because it stops during `reset_on_entry`.
- TickWatcher / broker TP/SL integration is tied to long behaviour (BE, TP, SL
  thresholds are symmetric but not side-aware in config), so even if shorts were
  created they would either be rejected or stuck in open_trades forever.

## Missing pieces for SHORT support

1. **Entry filter** – FARM_V2B needs a side-aware probability gate (e.g.
   `min_prob_short`) and should not throw away bars with weak `p_long`. Simply
   setting `min_prob_long` ≈ 0 allows shorts through, but that also lets very
   weak long bars reach the guard.
2. **Guard metadata** – `farm_brutal_guard_v2` and `get_farm_entry_metadata`
   always stamp `session_entry`/`vol_regime_entry` for longs. They should work
   for shorts but we should double-check telemetry dashboards expect a `side`
   column when analysing FARM segments.
3. **Exit policy** – RULE5 must support `side="short"`: update
   `reset_on_entry`, MAE/MFE tracking, and trailing logic to use bid/ask flipped
   when holding a short. Without this change any short entry triggers a runtime
   error.
4. **Telemetry / diagnostics** – the `[ENTRY_DIAG]` logging already prints the
   chosen side, but we need to ensure scripts like `scripts/check_trade_log.py`
   and dashboards treat shorts correctly (e.g. EV/day calculations, drawdown).

Until these gaps are closed, a “short-only” config can be authored, but it will
yield zero trades because the policy never allows them (and the exit layer would
break if it did). This document should be updated once the entry policy,
guards, and exit controller understand `side="short"` end-to-end.
