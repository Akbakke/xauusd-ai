# Phase-C.2 - Reduced Optimization Grid (Long-Only, Q2 Stress Test)

## Summary

Phase-C.2 tested 3 variants of the PROD baseline exit policy on Q2 2025 data (2025-04-01 to 2025-06-30).

## Baseline PROD (Q2 2025)

| Metric | Value |
|--------|-------|
| Trades | 150 |
| Trades/day | 0.44 |
| Win rate | 84.0% |
| EV/trade | 128.05 bps |
| EV/day | 56.49 bps |
| Max DD | -154.27 bps |

## Phase-C.2 Variants

### Variant D (max4)
**Parameter:** `max_open_trades = 4`

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Trades | 130 | -20 |
| Trades/day | 0.52 | +0.08 |
| Win rate | 79.2% | -4.8pp |
| EV/trade | 75.36 bps | -52.70 bps |
| EV/day | 39.03 bps | -17.46 bps |
| Max DD | -1721.81 bps | -1567.54 bps |

### Variant E (rule6)
**Parameter:** `rule_a_profit_min_bps = 6`

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Trades | 113 | -37 |
| Trades/day | 0.45 | +0.01 |
| Win rate | 77.9% | -6.1pp |
| EV/trade | 71.19 bps | -56.87 bps |
| EV/day | 31.92 bps | -24.57 bps |
| Max DD | -2157.57 bps | -2003.30 bps |

### Variant F (trail3)
**Parameter:** `rule_a_trailing_stop_bps = 3`

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Trades | 113 | -37 |
| Trades/day | 0.45 | +0.01 |
| Win rate | 75.2% | -8.8pp |
| EV/trade | 71.09 bps | -56.96 bps |
| EV/day | 31.88 bps | -24.62 bps |
| Max DD | -2157.57 bps | -2003.30 bps |

## Key Findings

1. **All variants underperform baseline:**
   - Lower EV/trade (-52 to -57 bps)
   - Lower EV/day (-17 to -25 bps)
   - Significantly higher max drawdown (-1567 to -2003 bps)

2. **Variant D (max4) performs best among variants:**
   - Highest EV/day (39.03 bps/day)
   - Lowest max DD (-1721.81 bps)
   - Highest EV/trade (75.36 bps)

3. **Variant E and F are nearly identical:**
   - Same max DD (-2157.57 bps)
   - Similar EV/trade (~71 bps)
   - Similar EV/day (~32 bps)

## Recommendations

- **None of the Phase-C.2 variants outperform baseline PROD**
- **Variant D (max4)** is the best candidate if forced to choose, but still significantly worse than baseline
- **Consider:** These variants may need different entry conditions or longer test periods
- **Next steps:** Focus on Phase-C.3 testing with different parameter combinations or entry policy adjustments

## Artifacts

- CSV: `gx1/tuning/exit_a_long_phaseC2_results.csv`
- JSON: `gx1/tuning/exit_a_long_phaseC2_results.json`
- Trade logs: `gx1/wf_runs/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_PHASEC2_*/trade_log_*merged.csv`
