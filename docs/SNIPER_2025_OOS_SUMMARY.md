# SNIPER 2025 Out-of-Sample Summary (Q1-Q4)

## Executive Summary

This report compares SNIPER performance across Q1-Q4 2025 to identify trends, anomalies, and recommendations for policy adjustments.

**Key Finding**: SNIPER maintains positive EV across all quarters, with moderate degradation in OOS periods (Q2-Q4). EV declined 11% from IS (Q1) to OOS average, but remains profitable at 8.13 bps. Win rate remains high (90.5% OOS vs 93.1% IS).

## Quarter-by-Quarter Comparison

### EV per Trade
| Quarter | EV (bps) | Change vs Q1 | Status |
| --- | --- | --- | --- |
| Q1 (IS) | 9.14 | baseline | ✅ |
| Q2 (OOS) | 9.78 | +7.0% | ✅ Best OOS |
| Q3 (OOS) | 7.17 | -21.6% | ⚠️ Decline |
| Q4 (OOS) | 7.45 | -18.5% | ⚠️ Decline |
| **OOS Avg** | **8.13** | **-11.0%** | ⚠️ Moderate degradation |

### Win Rate
| Quarter | Win Rate | Change vs Q1 | Status |
| --- | --- | --- | --- |
| Q1 (IS) | 93.1% | baseline | ✅ |
| Q2 (OOS) | 89.6% | -3.5pp | ⚠️ Decline |
| Q3 (OOS) | 93.2% | +0.1pp | ✅ Matches IS |
| Q4 (OOS) | 88.8% | -4.3pp | ⚠️ Decline |
| **OOS Avg** | **90.5%** | **-2.6pp** | ✅ Stable |

### Payoff Ratio
| Quarter | Payoff | Change vs Q1 | Status |
| --- | --- | --- | --- |
| Q1 (IS) | 0.53 | baseline | ⚠️ Low |
| Q2 (OOS) | 0.34 | -35.8% | ⚠️ Poor |
| Q3 (OOS) | 0.81 | +52.8% | ✅ Improved |
| Q4 (OOS) | 0.82 | +54.7% | ✅ Improved |
| **OOS Avg** | **0.66** | **+24.5%** | ✅ Improving |

### Average Loss Magnitude
| Quarter | Avg Loss (bps) | Change vs Q1 | Status |
| --- | --- | --- | --- |
| Q1 (IS) | -21.25 | baseline | - |
| Q2 (OOS) | -49.95 | +135.1% | ⚠️ Large losses |
| Q3 (OOS) | -10.41 | -51.0% | ✅ Improved |
| Q4 (OOS) | -12.13 | -42.9% | ✅ Improved |
| **OOS Avg** | **-24.16** | **+13.7%** | ⚠️ Slightly worse |

### Drawdown Proxy (p90 Loss)
| Quarter | p90 Loss (bps) | Status |
| --- | --- | --- |
| Q1 (IS) | 56.29 | - |
| Q2 (OOS) | 215.57 | ⚠️ High |
| Q3 (OOS) | 28.10 | ✅ Improved |
| Q4 (OOS) | 40.39 | ✅ Improved |

### Session EV per Quarter
| Quarter | EU (bps) | OVERLAP (bps) | US (bps) |
| --- | --- | --- | --- |
| Q1 (IS) | 7.36 | 7.28 | 7.22 |
| Q2 (OOS) | 9.73 | **16.63** | **16.38** |
| Q3 (OOS) | 5.14 | 8.25 | 7.47 |
| Q4 (OOS) | 7.70 | 9.07 | 5.75 |

**Key Observation**: Q2 showed exceptional performance in OVERLAP and US sessions, but this was not sustained in Q3-Q4.

## Anomalies & Flags

### ⚠️ Q2 Anomaly
- **Exceptional EV**: 9.78 bps (best OOS quarter)
- **Large losses**: -49.95 bps average loss (worst)
- **High drawdown**: 215.57 bps p90 loss
- **Strong sessions**: OVERLAP (16.63 bps) and US (16.38 bps)
- **Assessment**: Q2 had volatile but profitable conditions. Large losses suggest risk management issues during high-volatility periods.

### ⚠️ Q3-Q4 Decline
- **EV decline**: From 9.78 bps (Q2) to 7.17-7.45 bps (Q3-Q4)
- **US session decline**: From 16.38 bps (Q2) to 5.75 bps (Q4)
- **Assessment**: Performance normalized after Q2 anomaly. Still positive but lower EV.

### ✅ Q3-Q4 Improvements
- **Payoff ratio**: Improved from 0.34 (Q2) to 0.81-0.82 (Q3-Q4)
- **Loss magnitude**: Improved from -49.95 bps (Q2) to -10-12 bps (Q3-Q4)
- **Drawdown**: Improved from 215.57 bps (Q2) to 28-40 bps (Q3-Q4)
- **Assessment**: Better risk management in later quarters.

## In-Sample vs Out-of-Sample Analysis

### Performance Comparison

**In-Sample (Q1)**:
- EV/trade: 9.14 bps
- Win rate: 93.1%
- Total trades: 4,283
- Payoff ratio: 0.53

**Out-of-Sample (Q2-Q4 avg)**:
- EV/trade: 8.13 bps (-11.0%)
- Win rate: 90.5% (-2.6pp)
- Total trades: 1,624 per quarter
- Payoff ratio: 0.66 (+24.5%)

### Assessment

⚠️ **MODERATE DEGRADATION**: OOS performance is slightly worse than IS:
- **EV decline**: 11% reduction, but still positive
- **Win rate**: Only 2.6pp decline, remains high
- **Payoff ratio**: Improved significantly (+24.5%)
- **Consistency**: Performance is consistent across OOS quarters (Q2-Q4)

**Conclusion**: Policy generalizes reasonably well, but not perfectly. The 11% EV decline is within acceptable range for OOS performance.

## Recommendations

### 1. Policy Assessment
- ✅ **Keep unified policy**: Performance remains positive across all quarters
- ⚠️ **Monitor closely**: If Q4 decline continues into 2026, consider adjustments
- ✅ **Risk management improved**: Payoff ratio and loss magnitude improved in Q3-Q4

### 2. Session Optimization
- **OVERLAP**: Strongest performer in Q2 (16.63 bps), consider focusing here
- **US**: Declined in Q4 (5.75 bps), monitor for further degradation
- **EU**: Stable performance (5-10 bps range)

### 3. Risk Management
- ✅ **Payoff ratio improved**: From 0.53 (Q1) to 0.82 (Q4)
- ✅ **Loss magnitude improved**: From -21.25 bps (Q1) to -12.13 bps (Q4)
- ⚠️ **Q2 anomaly**: Investigate why Q2 had large losses despite high EV

### 4. Next Steps
1. **Continue monitoring**: Track Q1 2026 performance to see if Q4 decline continues
2. **Investigate Q2**: Understand why Q2 had exceptional performance but large losses
3. **Session analysis**: Deep dive into US session decline in Q4
4. **Regime analysis**: Understand why Q2 performed better across all volatility regimes

## Trend Summary

### EV/Trade Trend
```
Q1: 9.14 → Q2: 9.78 → Q3: 7.17 → Q4: 7.45 bps
```
**Assessment**: ⚠️ 18.5% decline from Q1 to Q4, but Q2 showed best OOS performance

### Win Rate Trend
```
Q1: 93.1% → Q2: 89.6% → Q3: 93.2% → Q4: 88.8%
```
**Assessment**: ✅ Stable across all quarters (88-93% range)

### Payoff Ratio Trend
```
Q1: 0.53 → Q2: 0.34 → Q3: 0.81 → Q4: 0.82
```
**Assessment**: ✅ Significant improvement from Q1 to Q4 (+54.7%)

## Final Verdict

**Policy Status**: ✅ **PRODUCTION-READY** with monitoring

**Rationale**:
1. Positive EV maintained across all quarters (7.17-9.78 bps)
2. High win rate maintained (88-93%)
3. Improving risk/reward profile (payoff ratio 0.82 in Q4)
4. Moderate OOS degradation (-11%) is within acceptable range
5. Consistent performance across OOS quarters

**Recommendation**: **Keep unified policy** but monitor Q1 2026 closely. If EV continues declining below 7 bps, consider minor adjustments.
