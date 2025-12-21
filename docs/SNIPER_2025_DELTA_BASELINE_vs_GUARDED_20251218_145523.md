# SNIPER 2025 Delta Report: Baseline vs Guarded

## Executive Summary

This report compares SNIPER performance between **baseline** and **guarded** variants
across all quarters to assess the impact of the SNIPER risk guard on tail risk and EV.

## Quarter-by-Quarter Comparison

| Quarter | Metric | Baseline | Guarded | Delta | % Change |
| --- | --- | --- | --- | --- | --- |
| Q1 | Trades | 6,935 | 7,208 | +273 | +3.9% |
| Q1 | EV/trade (bps) | 156.29 | 153.85 | -2.44 | -1.6% |
| Q1 | Win rate (%) | 84.6 | 84.8 | +0.2pp | +0.2% |
| Q1 | Avg loss (bps) | -61.11 | -61.37 | -0.25 | -0.4% |
| Q1 | P90 loss (bps) | 127.81 | 125.75 | -2.06 | -1.6% |
| Q1 | Payoff ratio | 3.20 | 3.13 | -0.07 | -2.1% |
| Q1 | Max concurrent (per-chunk max) | 1150 | 1200 | +50 | - |
| Q1 | Max concurrent (global merged, FYI) | 1150 | 1172 | +22 | - |
|  |  |  |  |  |  |
| Q2 | Trades | 6,102 | 6,226 | +124 | +2.0% |
| Q2 | EV/trade (bps) | 122.42 | 122.22 | -0.20 | -0.2% |
| Q2 | Win rate (%) | 69.7 | 70.9 | +1.2pp | +1.7% |
| Q2 | Avg loss (bps) | -111.94 | -111.39 | +0.54 | +0.5% |
| Q2 | P90 loss (bps) | 212.35 | 210.62 | -1.73 | -0.8% |
| Q2 | Payoff ratio | 2.00 | 1.96 | -0.05 | -2.3% |
| Q2 | Max concurrent (per-chunk max) | 981 | 1251 | +270 | - |
| Q2 | Max concurrent (global merged, FYI) | 1265 | 1265 | +0 | - |
|  |  |  |  |  |  |
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

- **Average P90 loss reduction**: 1.7%
- **Average EV impact**: +2.3%

## Conclusion

❌ **Recommendation: Review guard configuration** – limited tail risk benefit or high EV cost.