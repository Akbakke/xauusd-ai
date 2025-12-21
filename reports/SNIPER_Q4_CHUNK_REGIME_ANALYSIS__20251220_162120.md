# Q4 Chunk Regime Analysis

**Generated**: 2025-12-20 16:21:20
**Run Directory**: `SNIPER_OBS_Q4_2025_baseline_20251220_151442`
**Source**: `parallel_chunks/**/trade_journal/trades/*.json`

---

## Summary

- **Total Trades**: 4797
- **EV/Trade**: 29.52 bps
- **Winrate**: 53.0%
- **Payoff**: 1.44
- **P90 Loss**: -9.69 bps

### Overlay Coverage

- **Trades with entry_snapshot**: 4780 / 4797 (99.6%)
- **Trades with sniper_overlays**: 4780 / 4797 (99.6%)
- **Trades with Q4_C_CHOP_SESSION_SIZE overlay**: 4780 / 4797 (99.6%)
- **Trades with overlay_applied == True**: 1118 / 4797 (23.3%)

### Trade File Structure

- **Available top-level keys**: trade_id, run_tag, policy_sha256, router_sha256, manifest_sha256, entry_snapshot, feature_context, router_explainability, exit_configuration, exit_events, execution_events, exit_summary, chunk_id, trade_file
- **entry_snapshot type**: `dict`
- **feature_context type**: `dict`

⚠️ **CRITICAL ISSUE**: All trades have `entry_snapshot = None`.

This means:
- Overlay metadata (`sniper_overlays`) is not being written to trade journals.
- Regime classification must rely on `feature_context` or top-level fields (if available).
- Overlay trigger verification cannot be performed without `entry_snapshot`.

**Root Cause**: Trade journaling in `oanda_demo_runner.py` or `trade_journal.py` is not preserving `entry_snapshot`.

## Regime Breakdown

| Regime | Trades | EV/Trade | Winrate | Payoff | P90 Loss |
|--------|--------|----------|---------|--------|----------|
| A_TREND | 1250 | -8.40 | 47.4% | 0.98 | -15.84 |
| B_MIXED | 17 | 497.15 | 100.0% | 0.00 | 0.00 |
| C_CHOP | 3530 | 40.69 | 54.7% | 1.70 | -8.77 |

## Session Breakdown (Volume ≥ 200)

| Session | Trades | EV/Trade | Winrate | Payoff | P90 Loss |
|---------|--------|----------|---------|--------|----------|
| EU | 1711 | 16.28 | 51.2% | 1.25 | -10.38 |
| OVERLAP | 1217 | 31.51 | 55.0% | 1.40 | -18.26 |
| US | 1852 | 36.14 | 52.9% | 1.58 | -7.65 |

## Overlay Trigger Analysis

### Q4_C_CHOP_SESSION_SIZE Overlay

- **Trades with overlay metadata**: 4780
- **Trades with overlay_applied == True**: 1118

### Overlay Reasons (Top 15)

| Reason | Count |
|--------|-------|
| `not_c_chop:B_MIXED` | 3009 |
| `Q4_C_CHOP_session_gate` | 1118 |
| `multiplier_1.0` | 653 |

### Overlay by Session

| Session | Count |
|---------|-------|
| US | 3069 |
| EU | 1711 |

### Why overlay_applied == False

| Reason | Count |
|--------|-------|
| `not_c_chop:B_MIXED` | 3009 |
| `multiplier_1.0` | 653 |

### Sample Trade with overlay_applied == True

- **Trade ID**: `SIM-1766243688-000013`
- **File**: `SIM-1766243688-000013.json`
- **Chunk**: `chunk_0`
- **Quarter**: `Q4`
- **Regime**: `C_CHOP`
- **Session**: `US`
- **Multiplier**: `0.5`
- **Reason**: `Q4_C_CHOP_session_gate`

### Sample Trade with Non-Error Reason

- **Trade ID**: `SIM-1766243685-000001`
- **File**: `SIM-1766243685-000001.json`
- **Chunk**: `chunk_0`
- **Quarter**: `Q4`
- **Regime**: `B_MIXED`
- **Session**: `US`
- **Multiplier**: `0.5`
- **Reason**: `not_c_chop:B_MIXED`
