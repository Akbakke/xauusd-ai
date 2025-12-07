# FARM_V2B + EXIT_A Replay Analysis (Bid/Ask Active)

## Status: ⚠️ No Trades Found

**Run Location:** `gx1/wf_runs/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_FULL/`

**Issue:** The latest run generated **0 trades**.

### Why No Trades?

This was likely the short test run (2025-06-01 to 2025-06-05) that we ran earlier to verify the code works after fixing indentation issues. The run completed successfully but generated no trades, possibly because:

1. **Stage-0 filter blocked all entries**: The logs showed many `[STAGE_0] Skip entry consideration: trend=UNKNOWN vol=UNKNOWN session=ASIA` messages
2. **FARM_V2B policy too restrictive**: The policy may have been too strict for that specific period
3. **Short time window**: Only 4 days of data (1440 M5 bars)
4. **Big Brain V1 disabled**: Without Big Brain, regime detection defaults to UNKNOWN, which Stage-0 filters out

### What We Need

To perform a meaningful "realistic after-spread PnL assessment", we need:

1. **A longer replay run** (3-6 months minimum) with:
   - FARM_V2B entry policy active
   - EXIT_A (FARM_EXIT_V2_RULES_A) exit policy
   - Bid/ask prices enabled (already configured)
   - Sufficient data period to generate trades

2. **Or find an existing run** with trades from a previous hygiene test

### Next Steps

**Option 1: Run a new replay**
```bash
bash scripts/active/run_replay.sh \
  gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_FULL.yaml \
  2025-01-01 \
  2025-09-30 \
  7
```

**Option 2: Check for previous runs**
- Look for other `gx1/wf_runs/*FARM*V2B*` directories
- Check for trade logs in other locations
- Review previous hygiene test results

### Expected Analysis (Once We Have Trades)

Once we have a run with trades, the analysis will include:

1. **Basic Statistics**
   - Total trades (N)
   - Period (min/max timestamp)
   - Trades per day
   - EV/trade (bps)
   - EV/day (bps)
   - Winrate (%)
   - Median PnL (bps)
   - Std PnL
   - Min/Max PnL
   - Max drawdown (bps)

2. **Exit Breakdown**
   - Count and percentage per exit_reason:
     - FARM_EXIT_V2_RULES_A_PROFIT
     - FARM_EXIT_V2_RULES_A_TRAILING
     - FARM_EXIT_V2_RULES_A_TIMEOUT
     - Other exit reasons if present

3. **Regime Breakdown**
   - ASIA-LOW vs ASIA-MEDIUM:
     - EV/trade
     - Winrate
     - N trades

4. **Comparison with Previous Hygiene Test**
   - EV/trade before vs after bid/ask
   - Max DD before vs after
   - Winrate impact
   - Spread cost assessment

5. **Realistic After-Spread Assessment**
   - Is this still an attractive strategy?
   - Low gear, stable edge vs nearly flat after costs?

---

**Current Status:** Waiting for a replay run with trades to analyze.

---

## Root Cause Analysis

### Why No Trades Were Generated

From the replay log, we can see:

1. **Stage-0 filter blocked ALL entries:**
   - Every bar showed: `[STAGE_0] Skip entry consideration: trend=UNKNOWN vol=UNKNOWN session=ASIA risk=0.000`
   - This happened because Big Brain V1 is disabled (module not available)
   - Without Big Brain, regime detection defaults to UNKNOWN
   - Stage-0 filter (`should_consider_entry`) rejects UNKNOWN regimes

2. **The run did complete:**
   - Log shows: `[REPLAY PROGRESS] 855/1340 bars (63.8%) | Trades: 0 open, 1 total`
   - Wait, it says "1 total" but the trade log shows 0 trades
   - This might be a counting artifact or a trade that was opened but not logged

3. **Short time window:**
   - Only 4 days (2025-06-01 to 2025-06-05)
   - Even if Stage-0 allowed entries, FARM_V2B might be too restrictive for such a short period

### Solution

To get trades, we need to either:

1. **Disable Stage-0 filter for FARM_V2B** (since FARM_V2B has its own brutal guard)
2. **Enable Big Brain V1** (if available) to provide regime detection
3. **Run a longer period** (3-6 months) to increase chances of trades
4. **Check if Stage-0 should be disabled when Big Brain is unavailable**

The most likely fix is to disable Stage-0 when Big Brain is unavailable, since FARM_V2B has its own entry filtering via the brutal guard.

