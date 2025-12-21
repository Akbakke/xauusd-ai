# SNIPER 2025 Delta Report: Baseline vs Guarded

## Executive Summary

This report compares SNIPER performance between **baseline** and **guarded** variants
across all quarters to assess the impact of the SNIPER risk guard on tail risk and EV.

## Quarter-by-Quarter Comparison

| Quarter | Metric | Baseline | Guarded | Delta | % Change |
| --- | --- | --- | --- | --- | --- |
| Q2 | Trades | 6,102 | 6,226 | +124 | +2.0% |
| Q2 | EV/trade (bps) | 122.42 | 122.22 | -0.20 | -0.2% |
| Q2 | Win rate (%) | 69.7 | 70.9 | +1.2pp | +1.7% |
| Q2 | Avg loss (bps) | -111.94 | -111.39 | +0.54 | +0.5% |
| Q2 | P90 loss (bps) | 212.35 | 210.62 | -1.73 | -0.8% |
| Q2 | Payoff ratio | 2.00 | 1.96 | -0.05 | -2.3% |
| Q2 | Max concurrent (per-chunk max) | 981 | 1251 | +270 | - |
| Q2 | Max concurrent (global merged, FYI) | 1265 | 1265 | +0 | - |
|  |  |  |  |  |  |

## Guard Impact Summary

- **Average P90 loss reduction**: 0.8%
- **Average EV impact**: -0.2%

## Conclusion

❌ **Recommendation: Review guard configuration** – limited tail risk benefit or high EV cost.