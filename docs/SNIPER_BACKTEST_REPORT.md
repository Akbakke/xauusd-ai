# SNIPER Backtest Report Template

**Run Tag:** `SNIPER_OBS_<SPLIT>_<TIMESTAMP>`  
**Policy:** `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml`  
**Date Range:** `<START_DATE>` to `<END_DATE>`  
**Generated:** `<TIMESTAMP>`

---

## Executive Summary

**Total Trades:** `<N_TRADES>`  
**Trades/Day:** `<TRADES_PER_DAY>`  
**EV/Trade:** `<EV_PER_TRADE>` bps  
**Win Rate:** `<WIN_RATE>`%

**Session Distribution:**
- EU: `<N_EU>` trades (`<PCT_EU>`%)
- OVERLAP: `<N_OVERLAP>` trades (`<PCT_OVERLAP>`%)
- US: `<N_US>` trades (`<PCT_US>`%)

**Volatility Distribution:**
- LOW: `<N_LOW>` trades (`<PCT_LOW>`%)
- MEDIUM: `<N_MEDIUM>` trades (`<PCT_MEDIUM>`%)
- HIGH: `<N_HIGH>` trades (`<PCT_HIGH>`%)

---

## 1. Trades Per Session

| Session | Trades | % of Total | Trades/Day | EV/Trade (bps) | Win Rate |
|---------|--------|------------|------------|----------------|----------|
| EU | `<N_EU>` | `<PCT_EU>`% | `<EU_TRADES_PER_DAY>` | `<EU_EV>` | `<EU_WIN_RATE>`% |
| OVERLAP | `<N_OVERLAP>` | `<PCT_OVERLAP>`% | `<OVERLAP_TRADES_PER_DAY>` | `<OVERLAP_EV>` | `<OVERLAP_WIN_RATE>`% |
| US | `<N_US>` | `<PCT_US>`% | `<US_TRADES_PER_DAY>` | `<US_EV>` | `<US_WIN_RATE>`% |
| **TOTAL** | `<N_TOTAL>` | 100% | `<TOTAL_TRADES_PER_DAY>` | `<TOTAL_EV>` | `<TOTAL_WIN_RATE>`% |

**Key Findings:**
- `<SESSION>` session has highest trade rate
- `<SESSION>` session has highest EV/trade
- Overlap stress: `<OVERLAP_STRESS_METRIC>`

---

## 2. Trades Per Volatility Bucket

| Vol Regime | Trades | % of Total | Trades/Day | EV/Trade (bps) | Win Rate |
|------------|--------|------------|------------|----------------|----------|
| LOW | `<N_LOW>` | `<PCT_LOW>`% | `<LOW_TRADES_PER_DAY>` | `<LOW_EV>` | `<LOW_WIN_RATE>`% |
| MEDIUM | `<N_MEDIUM>` | `<PCT_MEDIUM>`% | `<MEDIUM_TRADES_PER_DAY>` | `<MEDIUM_EV>` | `<MEDIUM_WIN_RATE>`% |
| HIGH | `<N_HIGH>` | `<PCT_HIGH>`% | `<HIGH_TRADES_PER_DAY>` | `<HIGH_EV>` | `<HIGH_WIN_RATE>`% |
| **TOTAL** | `<N_TOTAL>` | 100% | `<TOTAL_TRADES_PER_DAY>` | `<TOTAL_EV>` | `<TOTAL_WIN_RATE>`% |

**Key Findings:**
- HIGH vol trades: `<N_HIGH>` (vs 0 for FARM)
- HIGH vol performance: `<HIGH_EV>` bps EV/trade
- Vol regime distribution: `<DISTRIBUTION_SUMMARY>`

---

## 3. p_long Distribution

**Overall:**
- Mean: `<P_LONG_MEAN>`
- Median: `<P_LONG_MEDIAN>`
- Min: `<P_LONG_MIN>`
- Max: `<P_LONG_MAX>`
- Std Dev: `<P_LONG_STD>`

**By Session:**
- EU: mean=`<EU_P_LONG_MEAN>`, median=`<EU_P_LONG_MEDIAN>`
- OVERLAP: mean=`<OVERLAP_P_LONG_MEAN>`, median=`<OVERLAP_P_LONG_MEDIAN>`
- US: mean=`<US_P_LONG_MEAN>`, median=`<US_P_LONG_MEDIAN>`

**By Vol Regime:**
- LOW: mean=`<LOW_P_LONG_MEAN>`, median=`<LOW_P_LONG_MEDIAN>`
- MEDIUM: mean=`<MEDIUM_P_LONG_MEAN>`, median=`<MEDIUM_P_LONG_MEDIAN>`
- HIGH: mean=`<HIGH_P_LONG_MEAN>`, median=`<HIGH_P_LONG_MEDIAN>`

**Threshold Analysis:**
- Trades with p_long >= 0.67: `<N_GE_067>` (`<PCT_GE_067>`%)
- Trades with p_long >= 0.68: `<N_GE_068>` (`<PCT_GE_068>`%)
- Trades with p_long >= 0.70: `<N_GE_070>` (`<PCT_GE_070>`%)

---

## 4. MFE/MAE Per Session

| Session | MFE p50 (bps) | MFE p90 (bps) | MAE p50 (bps) | MAE p90 (bps) | MFE/MAE Ratio |
|---------|---------------|---------------|---------------|---------------|---------------|
| EU | `<EU_MFE_P50>` | `<EU_MFE_P90>` | `<EU_MAE_P50>` | `<EU_MAE_P90>` | `<EU_MFE_MAE_RATIO>` |
| OVERLAP | `<OVERLAP_MFE_P50>` | `<OVERLAP_MFE_P90>` | `<OVERLAP_MAE_P50>` | `<OVERLAP_MAE_P90>` | `<OVERLAP_MFE_MAE_RATIO>` |
| US | `<US_MFE_P50>` | `<US_MFE_P90>` | `<US_MAE_P50>` | `<US_MAE_P90>` | `<US_MFE_MAE_RATIO>` |
| **TOTAL** | `<TOTAL_MFE_P50>` | `<TOTAL_MFE_P90>` | `<TOTAL_MAE_P50>` | `<TOTAL_MAE_P90>` | `<TOTAL_MFE_MAE_RATIO>` |

**Key Findings:**
- Best MFE/MAE ratio: `<SESSION>` session
- Highest MFE p90: `<SESSION>` session
- Highest MAE p90: `<SESSION>` session

---

## 5. Guard Blocks

**Session Mismatch Blocks:**
- ASIA session attempts: `<N_ASIA_BLOCKS>` (all blocked ✅)
- EU/OVERLAP/US attempts: `<N_EU_OVERLAP_US_ATTEMPTS>` (passed: `<N_PASSED>`)

**Vol Regime Mismatch Blocks:**
- EXTREME vol attempts: `<N_EXTREME_BLOCKS>` (all blocked ✅)
- LOW/MEDIUM/HIGH attempts: `<N_ALLOWED_VOL_ATTEMPTS>` (passed: `<N_PASSED>`)

**Trend Soft Gate (logged, not blocked):**
- TREND_UP: `<N_TREND_UP>` trades
- TREND_DOWN: `<N_TREND_DOWN>` trades
- RANGE: `<N_RANGE>` trades
- UNKNOWN: `<N_UNKNOWN>` trades

**Other Blocks:**
- Spread guard: `<N_SPREAD_BLOCKS>` blocks
- Margin guard: `<N_MARGIN_BLOCKS>` blocks
- Other: `<N_OTHER_BLOCKS>` blocks

---

## 6. Time-in-Market

**Overall:**
- Mean hold time: `<MEAN_HOLD_TIME>` minutes
- Median hold time: `<MEDIAN_HOLD_TIME>` minutes
- Min hold time: `<MIN_HOLD_TIME>` minutes
- Max hold time: `<MAX_HOLD_TIME>` minutes

**By Session:**
- EU: mean=`<EU_MEAN_HOLD>`, median=`<EU_MEDIAN_HOLD>`
- OVERLAP: mean=`<OVERLAP_MEAN_HOLD>`, median=`<OVERLAP_MEDIAN_HOLD>`
- US: mean=`<US_MEAN_HOLD>`, median=`<US_MEDIAN_HOLD>`

**Hold Time Distribution:**
- < 30 min: `<N_LT_30>` trades (`<PCT_LT_30>`%)
- 30-60 min: `<N_30_60>` trades (`<PCT_30_60>`%)
- 60-120 min: `<N_60_120>` trades (`<PCT_60_120>`%)
- > 120 min: `<N_GT_120>` trades (`<PCT_GT_120>`%)

---

## 7. Overlap Stress Analysis

**Overlap Entries:**
- Total overlap entries: `<N_OVERLAP_ENTRIES>`
- Overlap entries with MAE > 10 bps: `<N_OVERLAP_HIGH_MAE>` (`<PCT_OVERLAP_HIGH_MAE>`%)
- Overlap entries with MFE/MAE ratio < 1.0: `<N_OVERLAP_POOR_RATIO>` (`<PCT_OVERLAP_POOR_RATIO>`%)

**Overlap vs Non-Overlap:**
- Overlap EV/trade: `<OVERLAP_EV>` bps
- Non-overlap EV/trade: `<NON_OVERLAP_EV>` bps
- Difference: `<EV_DIFF>` bps

**Key Findings:**
- Overlap entries are `<BETTER/WORSE/SIMILAR>` compared to non-overlap
- Overlap stress metric: `<STRESS_METRIC>`

---

## 8. Exit Router Analysis

**RULE5 vs RULE6A Allocation:**
- RULE5: `<N_RULE5>` trades (`<PCT_RULE5>`%)
- RULE6A: `<N_RULE6A>` trades (`<PCT_RULE6A>`%)

**RULE5 Performance:**
- EV/trade: `<RULE5_EV>` bps
- Win rate: `<RULE5_WIN_RATE>`%

**RULE6A Performance:**
- EV/trade: `<RULE6A_EV>` bps
- Win rate: `<RULE6A_WIN_RATE>`%

**Router Guardrail:**
- RULE6A blocked by guardrail (range_edge_dist_atr >= 1.0): `<N_RULE6A_BLOCKED>` trades
- RULE6A allowed (range_edge_dist_atr < 1.0): `<N_RULE6A_ALLOWED>` trades

---

## 9. Comparison with FARM (Reference)

**FARM Baseline (for reference only - not a direct comparison):**
- Sessions: ASIA only
- Vol regimes: LOW, MEDIUM only
- Trades/day: ~3
- EV/trade: `<FARM_EV>` bps (from FARM runs)

**SNIPER (this run):**
- Sessions: EU, OVERLAP, US
- Vol regimes: LOW, MEDIUM, HIGH
- Trades/day: `<SNIPER_TRADES_PER_DAY>`
- EV/trade: `<SNIPER_EV>` bps

**Trade Rate Multiplier:** `<SNIPER_TRADES_PER_DAY>` / `<FARM_TRADES_PER_DAY>` = `<MULTIPLIER>`x

**Note:** This is not a direct comparison (different sessions, different vol regimes). Use for context only.

---

## 10. Recommendations

### Entry Threshold Tuning
- Current: `min_prob_long: 0.67`
- Recommendation: `<RECOMMENDATION>` (based on p_long distribution and trade rate)

### Session Policy
- Current: All EU/OVERLAP/US allowed
- Recommendation: `<RECOMMENDATION>` (based on session performance)

### Vol Policy
- Current: LOW/MEDIUM/HIGH allowed
- Recommendation: `<RECOMMENDATION>` (based on HIGH vol performance)

### Exit Profile Tuning
- Current: Same as FARM
- Recommendation: `<RECOMMENDATION>` (based on MFE/MAE analysis)

---

## Appendix: Data Sources

**Trade Journal:** `gx1/wf_runs/<RUN_TAG>/trade_journal/trades/*.json`  
**Results:** `gx1/wf_runs/<RUN_TAG>/results.json`  
**Run Header:** `gx1/wf_runs/<RUN_TAG>/run_header.json`  
**Logs:** `gx1/wf_runs/<RUN_TAG>/logs/`

---

*Report Template - Fill in values from actual backtest run*

