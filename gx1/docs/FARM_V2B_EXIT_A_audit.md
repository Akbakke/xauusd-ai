# FARM_V2B + EXIT_RULES_A – Leakage & PnL Audit (2025-12-06)

## Execution Path
1. `scripts/run_farm_v2b_exitA_full.sh` loads M5 parquet, slices `[START_DATE, END_DATE]`, and calls `scripts/replay_entry_exit_parallel.py` with policy `GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_FULL.yaml`.
2. `scripts/replay_entry_exit_parallel.py` (`gx1/scripts/replay_entry_exit_parallel.py:1-120,350-420`) splits bars into chunks and spins `GX1DemoRunner` instances (replay mode).
3. `GX1DemoRunner` boot sequence `gx1/execution/oanda_demo_runner.py:1400-1550`:
   - Loads entry config `ENTRY_V9_FARM_V2B.yaml` and exit config `FARM_EXIT_V2_RULES_A.yaml`.
   - Builds runtime features via `gx1/features/runtime_v9.py` (seq + snap features), referencing `basic_v1` + `sequence_features`.
   - Entry decision: `evaluate_entry` uses V9 transformer (`gx1/execution/oanda_demo_runner.py:5840-6260`), brutal guard V2, BigBrain metadata, and writes `trade.extra`.
   - Exit decision: `evaluate_and_close_trades` wires either `ExitFarmV2Rules` (`gx1/execution/oanda_demo_runner.py:8230-8335`) or default exit arbiter.
4. Trade log writing happens in `_update_trade_log_on_close` and `append_trade_log` (`gx1/execution/oanda_demo_runner.py:4007-4075, 8940-8980`). Replay summary uses `gx1/execution/oanda_demo_runner.py:10008-10052`.

## Feature Leakage Checks
- `gx1/features/basic_v1.py` shifts all predictive columns by ≥1 bar (`.shift(1)` on returns, ATR, RSI, etc.), ensuring only history ≤ current bar is used. ATR regime uses rolling windows without `center=True`.
- `gx1/seq/sequence_features.py` builds EMA/ATR/ROC features with `.shift` and `.rolling` backward windows only; no centered windows or future lookups. Session + regime IDs rely on timestamps/features already lagged.
- `gx1/features/runtime_v9.py` enforces a DatetimeIndex, rebuilds features each bar, and drops any column containing leakage substrings (`LEAKAGE_SUBSTRINGS` guard). Labels / `mfe/mae/pnl` columns are removed before inference.
- Result: **Ingen future-baserte features funnet** – all inspected features rely on historical OHLCV data at or before the active bar.

## PnL Calculation
- Centralised helper `gx1/utils/pnl.py:1-40` (`compute_pnl_bps`) now takes `(entry_bid, entry_ask, exit_bid, exit_ask, side)` so spread is enforced explicitly. Every runtime caller (TickWatcher, replay flush, `ExitManager`, `ExitFarmV2Rules`) routes through this helper.
- Unit tests in `tests/test_pnl_utils.py` cover long and short scenarios with non-zero spread to catch regressions.
- Bid/ask sourced directly from the 2025 candle set (or live ticks) are mandatory; the runner raises if the columns are missing. Commission remains zero, but spread cost is always paid on entry/exit.
- Trade log PnL is written exactly when `request_close` succeeds; `exit_price` is the bid for longs (ask for shorts) from either the bar close (`EXIT_FARM_V2_RULES`) or tick snapshot (TickWatcher). No future highs/lows are referenced when logging PnL, only the fill price at exit.

## EXIT_RULES_A Forward-Looking Check
- Config `gx1/configs/exits/FARM_EXIT_V2_RULES_A.yaml` toggles Rule A only (profit capture 6‑9 bps + adaptive trailing). Implementation `gx1/policy/exit_farm_v2_rules.py:1-330`:
  - Maintains only entry price, bars held, MAE/MFE accumulated up to current bar, and trailing stop high.
  - `on_bar` uses current close price only; trailing stop is based on `rule_a_trailing_high` accrued from past bars.
  - No references to future highs/lows; trailing stop triggers only when current `pnl_bps` drops below stored high minus trailing window.
  - Debug logging hook added (`debug_trade_ids` param) to dump `entry_ts`, `exit_price`, `reason`, `pnl`, `MFE`, `MAE` for specified trade IDs.
- Conclusion: **Exit uses only contemporaneous info** – no forward peek.

## Random/Fixed Exit Sanity Mode
- New exit policy `gx1/policy/exit_fixed_bar.py` + config `gx1/configs/exits/FARM_EXIT_FIXED_BARS.yaml` close trades after either a fixed number of bars or a random integer `[min_bars, max_bars]`.
- Policy wiring (`gx1/execution/oanda_demo_runner.py:1465-1487, 8337-8371`) allows running FARM_V2B entry with deterministic/random exits; script `scripts/run_farm_v2b_random_exit.sh` replays a 3‑month window to benchmark entry-only edge.
- Use case: compare EV from random exit vs EXIT_RULES_A to ensure gains aren’t solely from exit heuristics.

## Additional Logging / Tests
- Debug logging for specific trades via `debug_trade_ids` (set in exit config) prints entry/exit timestamps, prices, PnL, MFE/MAE for manual verification against raw candles.
- `tests/test_pnl_utils.py` safeguards against regressions in PnL calculations.

## Findings Summary
| Area | Status | Notes |
| --- | --- | --- |
| Execution path | OK | Pipeline traced from shell script → replay → runner; data loaded once and sliced. |
| Features | OK | All predictive features lagged (`basic_v1`, `sequence_features`, runtime guard). No labels in inference view. |
| PnL | OK (bid/ask enforced) | All closes use bid/ask fills via `compute_pnl_bps`; spread is paid each trade (commission still zero). |
| Exit rules | OK | Rule A uses only current price, MAE/MFE up to now, trailing from historical peak. Debug logging available. |
| Random exit sanity | Available | `scripts/run_farm_v2b_random_exit.sh` runs entry edge vs random/fixed exits for baseline comparison. |

### Suggested Follow-ups
1. Consider explicit spread/slippage model in replay (deduct e.g. 1 pip round trip) for conservative EV estimates.
2. Automate random-exit EV comparison inside CI to detect regressions (entry losing standalone edge).
3. When enabling debug logging, cross-check a couple of trades against raw candles stored in `gx1/wf_runs/.../price_data_filtered.parquet`.
