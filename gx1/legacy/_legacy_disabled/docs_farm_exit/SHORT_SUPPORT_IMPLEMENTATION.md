# ⚠️ EXPERIMENTAL / NOT USED IN PRODUCTION ⚠️
# This document is archived for experimental purposes only.
# Production pipelines use long-only strategies.
#
# Short Support Implementation Summary

## Tasks Completed

### Task A: Upgrade EXIT_A (RULE5) to support short ✅

**Changes in `gx1/policy/exit_farm_v2_rules.py`:**
- ✅ Removed `side != "long"` restriction in `reset_on_entry()`
- ✅ Fixed entry price logic: long uses ask, short uses bid
- ✅ Fixed exit price logic: long exits at bid, short exits at ask
- ✅ Trailing stop logic works for both long and short (no inversion needed - PnL is already signed correctly)
- ✅ MAE/MFE tracking works for both sides
- ✅ All exit rules (A, B, C, force_exit) handle shorts correctly
- ✅ Logging includes side information

**Key implementation details:**
- Entry price: `entry_ask` for long, `entry_bid` for short
- Exit price: `price_bid` for long (sell at bid), `price_ask` for short (buy back at ask)
- PnL calculation: Already correct in `compute_pnl_bps()` (handles both sides)
- Trailing stop: Works identically for both sides (PnL-based, not price-based)

**Tests added:** `gx1/tests/test_exit_farm_v2_rules_short.py`

### Task B: Add real min_prob_short support to ENTRY_V9 + FARM_V2B ✅

**Changes in `gx1/policy/entry_v9_policy_farm_v2b.py`:**
- ✅ Added `min_prob_short` parameter (defaults to 0.72 if not set)
- ✅ Side-aware filtering: signal passes if `p_long >= min_prob_long` OR `p_short >= min_prob_short`
- ✅ Added `_policy_side` column to indicate which side(s) passed threshold
- ✅ Policy score uses `max(p_long, p_short)` for side-aware scoring
- ✅ Logging includes side breakdown (long/short/both counts)

**Changes in `gx1/execution/entry_manager.py`:**
- ✅ Extracts `_policy_side` from policy result
- ✅ Uses policy-determined side when available (before calling `should_enter_trade_func`)
- ✅ Falls back to default logic if policy side not available

**Key implementation details:**
- If `allow_short=False`, only long threshold is checked
- If `allow_short=True`, both thresholds are checked independently
- If both thresholds pass, side is determined by max(prob_long, prob_short)
- Policy result includes `_policy_side` column: "long", "short", "both", or "none"

### Task C: Update SHORT_SNIPER config ✅

**Created files:**
- ✅ `gx1/configs/policies/active/ENTRY_V9_FARM_V2B_EXIT_A_SHORT_SNIPER_Q1_v2.yaml`
  - `min_prob_long: 1.10` (impossible → blocks all longs)
  - `min_prob_short: 0.75` (real short threshold)
  - `allow_short: true`
  - ASIA + LOW/MEDIUM regime

- ✅ `gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_SHORT_SNIPER_Q1_v2.yaml`
  - Uses v2 entry config
  - Uses EXIT_A (now supports shorts)
  - `max_open_trades: 2`
  - Output: `gx1/wf_runs/FARM_V2B_EXIT_A_SHORT_SNIPER_Q1_v2/`

### Task D: Run SHORT_SNIPER_Q1_v2 replay ⏳

**Command:**
```bash
export M5_DATA="data/raw/xauusd_m5_2025_bid_ask.parquet"
bash scripts/active/run_replay.sh \
  gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_SHORT_SNIPER_Q1_v2.yaml \
  2025-01-02 2025-03-31 6
```

**Expected output:**
- Trade log: `gx1/wf_runs/FARM_V2B_EXIT_A_SHORT_SNIPER_Q1_v2/trade_log.csv`
- Results: `gx1/wf_runs/FARM_V2B_EXIT_A_SHORT_SNIPER_Q1_v2/results.json`
- All trades should be shorts (side="short")
- All trades should have exit_time populated (no more exit errors)

## Verification Checklist

- [ ] Run replay and verify no exit errors
- [ ] Verify all trades are shorts (side="short")
- [ ] Verify all trades have exit_time populated
- [ ] Verify PnL calculations are correct for shorts
- [ ] Compare metrics to long-only baseline

## Next Steps

1. Run replay (Task D)
2. Analyze results and compare to long baseline
3. If successful, document findings and consider expanding short support to other exit profiles

