# Q4 Chunk Regime Analysis

**Generated**: 2025-12-20 10:36:44
**Run Directory**: `SNIPER_OBS_Q4_2025_baseline_20251219_132806`
**Source**: `parallel_chunks/**/trade_journal/trades/*.json`

---

## Summary

- **Total Trades**: 272
- **EV/Trade**: 60.60 bps
- **Winrate**: 76.8%
- **Payoff**: 22.35
- **P90 Loss**: -0.78 bps

### Overlay Coverage

- **Trades with entry_snapshot**: 0 / 272 (0.0%)
- **Trades with sniper_overlays**: 0 / 272 (0.0%)
- **Trades with Q4_C_CHOP_SESSION_SIZE overlay**: 0 / 272 (0.0%)
- **Trades with overlay_applied == True**: 0 / 272 (0.0%)

### Trade File Structure

- **Available top-level keys**: trade_id, run_tag, policy_sha256, router_sha256, manifest_sha256, entry_snapshot, feature_context, router_explainability, exit_configuration, exit_events, execution_events, exit_summary, chunk_id, trade_file
- **entry_snapshot type**: `NoneType`
- **feature_context type**: `NoneType`

⚠️ **CRITICAL ISSUE**: All trades have `entry_snapshot = None`.

This means:
- Overlay metadata (`sniper_overlays`) is not being written to trade journals.
- Regime classification must rely on `feature_context` or top-level fields (if available).
- Overlay trigger verification cannot be performed without `entry_snapshot`.

**Root Cause**: Trade journaling in `oanda_demo_runner.py` or `trade_journal.py` is not preserving `entry_snapshot`.

## Regime Breakdown

| Regime | Trades | EV/Trade | Winrate | Payoff | P90 Loss |
|--------|--------|----------|---------|--------|----------|
| B_MIXED | 272 | 60.60 | 76.8% | 22.35 | -0.78 |

## Overlay Trigger Analysis

### Q4_C_CHOP_SESSION_SIZE Overlay

- **Trades with overlay metadata**: 0
- **Trades with overlay_applied == True**: 0
