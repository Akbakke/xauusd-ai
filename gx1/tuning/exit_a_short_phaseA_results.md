# ⚠️ EXPERIMENTAL / NOT USED IN PRODUCTION ⚠️
# This document is archived for experimental purposes only.
# Production pipelines use long-only strategies.
#
# EXIT_A Short Sniper – Phase-A probe (Q1 2025)

| Window | Trades (open) | Closed trades | Trades/day | Win rate | EV/trade (bps) | EV/day (bps) | Missing exit_profile | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025‑01‑02 → 2025‑03‑31 | 10 | 0 | 0.11 | N/A | N/A | N/A | 0 | EXIT_FARM_V2_RULES rejects `side=\"short\"` and never initialises state |

### Interpretation

1. ENTRY_V9 happily emitted 10 short candidates (all ASIA+LOW regime), but the
   exit policy raised `ValueError: EXIT_FARM_V2_RULES only supports LONG
   positions` during `reset_on_entry`, so every short remained open with no
   `exit_time`/`pnl_bps`.
2. The `[EXIT]` loop spammed `reset_on_entry() must be called before on_bar()`
   which is why the parallel replay stalled after a couple of minutes – chunks
   keep evaluating trades that never initialise the exit state.
3. Until EXIT_A (RULE5) is updated to support `side="short"` (bid/ask inversion,
   MAE/MFE tracking, trailing stop logic), long-only wiring guarantees the short
   sniper profile cannot produce actionable metrics.

See `gx1/docs/EXIT_A_SHORT_NOTES.md` for the full list of blockers and the
artefacts from the failed replay under
`gx1/wf_runs/FARM_V2B_EXIT_A_SHORT_SNIPER_Q1/`.
