# SNIPER 2025 Delta Report: Baseline vs Guarded

## Executive Summary

This report compares SNIPER performance between **baseline** and **guarded** variants
across all quarters to assess the impact of the SNIPER risk guard on tail risk and EV.

## Quarter-by-Quarter Comparison

| Quarter | Metric | Baseline | Guarded | Delta | % Change |
| --- | --- | --- | --- | --- | --- |
| Q1 | Status | missing | missing | - | - |
| Q3 | Trades | 6,689 | 6,576 | -113 | -1.7% |
| Q3 | EV/trade (bps) | 113.16 | 119.88 | +6.72 | +5.9% |
| Q3 | Win rate (%) | 72.1 | 73.6 | +1.5pp | +2.1% |
| Q3 | Avg loss (bps) | -44.92 | -43.70 | +1.22 | +2.7% |
| Q3 | P90 loss (bps) | 95.06 | 90.74 | -4.33 | -4.6% |
| Q3 | Payoff ratio | 3.88 | 4.08 | +0.20 | +5.2% |
| Q3 | Max concurrent (per-chunk max) | 1318 | 1297 | -21 | - |
| Q3 | Max concurrent (global merged, FYI) | 1339 | 1325 | -14 | - |
|  |  |  |  |  |  |
| Q4 | Trades | 4,482 | 4,388 | -94 | -2.1% |
| Q4 | EV/trade (bps) | 20.64 | 21.67 | +1.03 | +5.0% |
| Q4 | Win rate (%) | 52.6 | 52.1 | -0.5pp | -1.0% |
| Q4 | Avg loss (bps) | -103.48 | -102.14 | +1.34 | +1.3% |
| Q4 | P90 loss (bps) | 285.56 | 285.70 | +0.15 | +0.1% |
| Q4 | Payoff ratio | 1.28 | 1.33 | +0.05 | +3.7% |
| Q4 | Max concurrent (per-chunk max) | 815 | 768 | -47 | - |
| Q4 | Max concurrent (global merged, FYI) | 816 | 804 | -12 | - |
|  |  |  |  |  |  |

## Guard Impact Summary

- **Average P90 loss reduction**: 2.2%
- **Average EV impact**: +5.5%

## Conclusion

❌ **Recommendation: Review guard configuration** – limited tail risk benefit or high EV cost.