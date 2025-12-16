# Phase-C Long-Only Tuning Results (Q2 2025)

**Test Period:** 2025-04-01 → 2025-06-30  
**Baseline:** PROD config (mp68_max5_rule5_cd0)  
**Dataset:** M5 bid/ask 2025, 6 workers

## Results Matrix

| Variant | Params | Trades | Trades/day | Win Rate | EV/trade | EV/day | Max DD | Missing EP |
|---------|--------|--------|------------|----------|----------|--------|--------|------------|
| **Baseline (PROD)** | min_prob_long=0.68, max_open_trades=5 | 73 | 0.96 | 69.9% | 42.37 | 40.69 | -2157.57 | 0 |
| **Variant A** | min_prob_long=0.67, max_open_trades=5 | 74 | 0.97 | 70.3% | 41.95 | 40.84 | -2157.57 | 0 |
| **Variant B** | min_prob_long=0.69, max_open_trades=5 | 73 | 0.96 | 69.9% | 42.27 | 40.59 | -2157.57 | 0 |
| **Variant C** | min_prob_long=0.68, max_open_trades=6 | 87 | 1.14 | 70.1% | 42.02 | 48.09 | -2597.28 | 0 |

## Key Observations

### Variant A (min_prob_long=0.67) vs Baseline
- **Trades/day:** +0.01 (0.97 vs 0.96) - minimal increase
- **Win rate:** +0.4% (70.3% vs 69.9%) - slight improvement
- **EV/day:** +0.15 bps (40.84 vs 40.69) - minimal improvement
- **Max DD:** No change (-2157.57 bps)
- **Interpretation:** Lowering threshold to 0.67 provides minimal benefit. Trade frequency barely increases, win rate improves slightly, but EV/day gain is negligible.

### Variant B (min_prob_long=0.69) vs Baseline
- **Trades/day:** No change (0.96 vs 0.96) - identical
- **Win rate:** No change (69.9% vs 69.9%) - identical
- **EV/day:** -0.10 bps (40.59 vs 40.69) - slight decrease
- **Max DD:** No change (-2157.57 bps)
- **Interpretation:** Raising threshold to 0.69 provides no benefit. Trade frequency and win rate unchanged, EV/day slightly lower.

### Variant C (max_open_trades=6) vs Baseline
- **Trades/day:** +0.18 (1.14 vs 0.96) - **+18.8% increase**
- **Win rate:** +0.3% (70.1% vs 69.9%) - slight improvement
- **EV/day:** +7.40 bps (48.09 vs 40.69) - **+18.2% increase**
- **Max DD:** -439.72 bps (-2597.28 vs -2157.57) - **worse drawdown**
- **Interpretation:** Increasing max_open_trades to 6 provides significant EV/day improvement (+18.2%) and higher trade frequency (+18.8%), but at the cost of worse drawdown (-439.72 bps). The max_open_trades limit appears to be binding in Q2, allowing more concurrent trades when increased.

## Recommendations

1. **Variant A (min_prob_long=0.67):** Not recommended - minimal benefit, no clear advantage over baseline.

2. **Variant B (min_prob_long=0.69):** Not recommended - no benefit, slight EV/day decrease.

3. **Variant C (max_open_trades=6):** **Consider for production** - significant EV/day improvement (+18.2%) and higher trade frequency, but requires careful risk management due to increased drawdown. The drawdown increase (-439.72 bps) is substantial and should be evaluated against risk tolerance.

**Best candidate:** Variant C shows the most promise with +18.2% EV/day improvement, but the drawdown increase needs careful consideration. If risk tolerance allows, Variant C could be a strong candidate for production adoption.

## Notes

- All variants maintained 100% exit_profile coverage (0 missing)
- All trades were longs (no shorts)
- Q2 appears to be a challenging quarter with high drawdown across all variants
- The max_open_trades limit appears to be frequently hit in Q2, making Variant C's increase effective

