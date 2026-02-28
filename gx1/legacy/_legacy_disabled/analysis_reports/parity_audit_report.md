# Parity Audit Report

**Generated:** 2025-12-16T17:36:20.020391+00:00

## Verdict: ❌ NOT PARITY

**Primary Cause:** Policy config hash mismatch

## Run Information

- **FULLYEAR Run:** `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_FULLYEAR`
- **CANARY Run:** `gx1/wf_runs/CANARY_TEST_2025_Q1_SHORT`

## Config Parity

| Config Item | FULLYEAR | CANARY | Match |
|-------------|----------|--------|-------|
| Policy Hash | `a2b67a9860a0ffef...` | `c8a83be33ea788bc...` | ❌ |
| Entry Config Hash | `N/A` | `N/A` | ❌ |
| Exit Config Hash | `N/A` | `N/A` | ❌ |

## Price Source Audit (P0)

### Effective Price Source Contract

| Component | Price Source |
|-----------|--------------|
| Entry Features | bid/ask mid |
| Spread Calculation | bid/ask |
| Range Features | UNKNOWN |
| Exit Evaluation | UNKNOWN |
| Intratrade Metrics | UNKNOWN |
| Bar Close Semantics | last closed bar (iloc[-1] or close_vals[-1]) |

### Price Data Parity

| Run | Has Bid/Ask | Has OHLC | Price Data Hash |
|-----|-------------|----------|-----------------|
| FULLYEAR | ✅ | ✅ | `1a82ca81e4714de3...` |
| CANARY | ✅ | ✅ | `2152cf7dad106a6e...` |

### Price Data Sample Differences (OHLC close vs bid/ask mid)

| Timestamp | OHLC Close | Mid Close | Diff | Diff (bps) |
|-----------|------------|-----------|------|------------|
| 2025-01-01 23:00:00 | 2625.20000 | 2625.20000 | 0.00000 | 0.00 |
| 2025-01-01 23:05:00 | 2625.26000 | 2625.26000 | 0.00000 | 0.00 |
| 2025-01-01 23:10:00 | 2624.87000 | 2624.87000 | 0.00000 | 0.00 |
| 2025-01-01 23:15:00 | 2624.45500 | 2624.45500 | 0.00000 | 0.00 |
| 2025-01-01 23:20:00 | 2623.75000 | 2623.75000 | 0.00000 | 0.00 |

### Price Selection Logic Findings

| File | Line | Function | Description | Code Snippet |
|------|------|----------|-------------|--------------|
| `gx1/execution/exit_manager.py` | 60 | `unknown` | bid/ask OHLC usage | `current_bid = float(candles["bid_close"].iloc[-1])...` |
| `gx1/execution/exit_manager.py` | 61 | `unknown` | bid/ask OHLC usage | `current_ask = float(candles["ask_close"].iloc[-1])...` |
| `gx1/execution/exit_manager.py` | 81 | `unknown` | price trace collection | `self._collect_price_trace(candles, now_ts, open_trades_copy)...` |
| `gx1/execution/exit_manager.py` | 242 | `unknown` | entry/exit price assignment | `exit_price=exit_decision.exit_price,...` |
| `gx1/execution/exit_manager.py` | 323 | `unknown` | entry/exit price assignment | `exit_price=decision.exit_price,...` |
| `gx1/execution/exit_manager.py` | 464 | `unknown` | entry/exit price assignment | `exit_price=exit_decision.exit_price,...` |
| `gx1/execution/exit_manager.py` | 618 | `unknown` | entry/exit price assignment | `exit_price=mark_price,...` |
| `gx1/execution/exit_manager.py` | 735 | `_collect_price_trace` | price trace collection | `def _collect_price_trace(self, candles: pd.DataFrame, now_ts: pd.Timestamp, open...` |
| `gx1/execution/exit_manager.py` | 739 | `_collect_price_trace` | price trace collection | `Stores high/low/close per bar in trade.extra["_price_trace"]....` |
| `gx1/execution/exit_manager.py` | 747 | `_collect_price_trace` | bar close semantics | `last_bar = candles.iloc[-1]...` |
| `gx1/execution/exit_manager.py` | 754 | `_collect_price_trace` | bid/ask OHLC usage | `elif "bid_high" in candles.columns and "ask_high" in candles.columns:...` |
| `gx1/execution/exit_manager.py` | 755 | `unknown` | bid/ask OHLC usage | `bar_high = float((last_bar["bid_high"] + last_bar["ask_high"]) / 2.0)...` |
| `gx1/execution/exit_manager.py` | 756 | `unknown` | bid/ask OHLC usage | `bar_low = float((last_bar["bid_low"] + last_bar["ask_low"]) / 2.0)...` |
| `gx1/execution/exit_manager.py` | 757 | `unknown` | bid/ask OHLC usage | `bar_close = float((last_bar["bid_close"] + last_bar["ask_close"]) / 2.0)...` |
| `gx1/execution/exit_manager.py` | 762 | `unknown` | bid/ask OHLC usage | `elif "bid_close" in candles.columns and "ask_close" in candles.columns:...` |
| `gx1/execution/exit_manager.py` | 763 | `unknown` | bid/ask OHLC usage | `bar_close = float((last_bar["bid_close"] + last_bar["ask_close"]) / 2.0)...` |
| `gx1/execution/exit_manager.py` | 774 | `unknown` | price trace collection | `if "_price_trace" not in trade.extra:...` |
| `gx1/execution/exit_manager.py` | 775 | `unknown` | price trace collection | `trade.extra["_price_trace"] = []...` |
| `gx1/execution/exit_manager.py` | 784 | `unknown` | price trace collection | `trade.extra["_price_trace"].append(trace_point)...` |
| `gx1/execution/exit_manager.py` | 788 | `unknown` | price trace collection | `if len(trade.extra["_price_trace"]) > max_trace_size:...` |
| `gx1/execution/exit_manager.py` | 789 | `unknown` | price trace collection | `trade.extra["_price_trace"] = trade.extra["_price_trace"][-max_trace_size:]...` |
| `gx1/execution/exit_manager.py` | 823 | `unknown` | price trace collection | `# Remove _price_trace from trade.extra to prevent bloating trade logs...` |
| `gx1/execution/exit_manager.py` | 825 | `unknown` | price trace collection | `if "_price_trace" in trade.extra:...` |
| `gx1/execution/exit_manager.py` | 826 | `unknown` | price trace collection | `del trade.extra["_price_trace"]...` |
| `gx1/execution/exit_manager.py` | 832 | `unknown` | entry/exit price assignment | `exit_price=exit_price,...` |
| `gx1/execution/exit_manager.py` | 871 | `_compute_intratrade_metrics` | price trace collection | `trade: Trade object with entry_price, side, and _price_trace...` |
| `gx1/execution/exit_manager.py` | 878 | `_compute_intratrade_metrics` | price trace collection | `price_trace = trade.extra.get("_price_trace", [])...` |
| `gx1/execution/exit_manager.py` | 879 | `_compute_intratrade_metrics` | price trace collection | `if not price_trace:...` |
| `gx1/execution/exit_manager.py` | 886 | `unknown` | entry/exit price assignment | `entry_price = trade.entry_price...` |
| `gx1/execution/exit_manager.py` | 891 | `unknown` | price trace collection | `for point in price_trace:...` |

## Warmup/State Parity (P0)

| Setting | FULLYEAR | CANARY | Match |
|---------|----------|--------|-------|
| Warmup Bars | 288 | 288 | ✅ |
| N Workers | None | None | ✅ |
| Parallel Chunking | True | True | ✅ |

## Gate & Throttle Parity (P0)

| Gate/Throttle | FULLYEAR | CANARY | Match |
|---------------|----------|--------|-------|
| max_open_trades | None | None | ✅ |
| spread_filter_threshold | None | None | ✅ |
| min_time_between_trades_sec | None | None | ✅ |
| session_allowlist | None | None | ✅ |
| regime_allowlist | None | None | ✅ |

## Trades Per Day Analysis (P0)

### FULLYEAR
- Total Trades: 0
- Trades/Day: N/A
- ASIA Trades/Day: N/A

### CANARY
- Total Trades: 0
- Trades/Day: N/A

## Entry Model Parity

- Entry Model Version Match: ✅
  - FULLYEAR: None
  - CANARY: None

## Router Parity

- Router Version Match: ✅
  - FULLYEAR: HYBRID_ROUTER_V3
  - CANARY: HYBRID_ROUTER_V3

- Router Model Hash Match: ❌
  - FULLYEAR: `N/A`
  - CANARY: `N/A`

- Guardrail Cutoff Match: ❌
  - FULLYEAR: None
  - CANARY: 1.0

## Where Did '3 Trades/Day in Asia' Come From?

No historical runs found with ~3 trades/day. The claim may be:
- From a different run tag pattern
- From manual calculation
- From a different time period

---

*Report generated by `gx1/analysis/parity_audit.py`*