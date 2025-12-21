# SNIPER 2025 Regime Split Report (20251218_142940)

This report analyzes existing SNIPER 2025 trade journals by coarse regime classes (A_TREND, B_MIXED, C_CHOP) without re-running any replay.

## Executive Summary

- **Best contributing regime (EV)**: `A_TREND` (overall mean EV ≈ 184.2 bps per trade).
- **Regime most associated with Q4 baseline weakness**: `B_MIXED` (Q4 baseline EV ≈ -6.4 bps).

## Per-quarter regime metrics

### Quarter Q1
#### Variant: `baseline`

| Regime | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) | Median dur (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_TREND | 2 | 248.6 | 100.0% | nan | 0.0 | 11305.0 |
| B_MIXED | 423 | 138.7 | 82.3% | 3.89 | 83.8 | 8590.0 |
| C_CHOP | 6510 | 157.4 | 84.8% | 3.16 | 128.2 | 9467.5 |

#### Variant: `guarded`

| Regime | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) | Median dur (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_TREND | 2 | 248.6 | 100.0% | nan | 0.0 | 11305.0 |
| B_MIXED | 446 | 143.3 | 84.1% | 3.96 | 84.4 | 8792.5 |
| C_CHOP | 6760 | 154.5 | 84.9% | 3.09 | 126.6 | 9785.0 |

### Quarter Q2
#### Variant: `baseline`

| Regime | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) | Median dur (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_TREND | 7 | 145.3 | 100.0% | nan | 0.0 | 17390.0 |
| B_MIXED | 1278 | 185.6 | 72.8% | 2.76 | 208.7 | 8875.0 |
| C_CHOP | 4817 | 105.6 | 68.9% | 1.81 | 213.7 | 9925.0 |

#### Variant: `guarded`

| Regime | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) | Median dur (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_TREND | 7 | 145.3 | 100.0% | nan | 0.0 | 17390.0 |
| B_MIXED | 1183 | 171.1 | 72.9% | 2.67 | 198.2 | 8895.0 |
| C_CHOP | 5036 | 110.7 | 70.4% | 1.81 | 213.8 | 9995.0 |

### Quarter Q3
#### Variant: `baseline`

| Regime | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) | Median dur (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_TREND | 3 | 386.2 | 100.0% | nan | 0.0 | 17695.0 |
| B_MIXED | 416 | 156.9 | 84.6% | 2.08 | 151.9 | 9912.5 |
| C_CHOP | 6270 | 110.1 | 71.2% | 4.00 | 90.1 | 9012.5 |

#### Variant: `guarded`

| Regime | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) | Median dur (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_TREND | 4 | 273.7 | 75.0% | 6.05 | 63.8 | 16592.5 |
| B_MIXED | 407 | 178.9 | 88.0% | 2.03 | 247.7 | 9910.0 |
| C_CHOP | 6165 | 115.9 | 72.7% | 4.19 | 87.3 | 8970.0 |

### Quarter Q4
#### Variant: `baseline`

| Regime | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) | Median dur (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_TREND | 7 | 13.1 | 71.4% | 0.49 | 320.0 | 4740.0 |
| B_MIXED | 1086 | -6.4 | 49.1% | 0.95 | 333.9 | 4465.0 |
| C_CHOP | 3389 | 29.3 | 53.7% | 1.48 | 278.7 | 3970.0 |

#### Variant: `guarded`

| Regime | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) | Median dur (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_TREND | 7 | 13.1 | 71.4% | 0.49 | 320.0 | 4740.0 |
| B_MIXED | 979 | -5.7 | 49.0% | 0.96 | 338.2 | 4675.0 |
| C_CHOP | 3402 | 29.6 | 53.0% | 1.52 | 277.2 | 4147.5 |

## Full-year regime metrics (Q1–Q4 combined)

### Variant: `baseline`

| Regime | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) | Median dur (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_TREND | 19 | 198.3 | 92.9% | 0.49 | 80.0 | 12782.5 |
| B_MIXED | 3203 | 118.7 | 72.2% | 2.42 | 194.6 | 7960.6 |
| C_CHOP | 20986 | 100.6 | 69.7% | 2.61 | 177.7 | 8093.8 |

### Variant: `guarded`

| Regime | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) | Median dur (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_TREND | 20 | 170.2 | 86.6% | 3.27 | 96.0 | 12506.9 |
| B_MIXED | 3015 | 121.9 | 73.5% | 2.41 | 217.1 | 8068.1 |
| C_CHOP | 21363 | 102.7 | 70.2% | 2.65 | 176.2 | 8224.4 |

## Sanity checks

- ✅ Sum of trades over A/B/C regimes matches total trades for each (quarter, variant).

- Regime classification coverage: 48606/48606 trades (100.0% with a regime_class).

## Next step (policy, no retrain)

- Consider using this regime split as a **gate** in policy space, without changing the learned models:
  - For example, if `C_CHOP` regimes consistently show low or negative EV and unfavorable payoff, you may choose to **turn SNIPER off or reduce size** in those regimes, while keeping full size in `A_TREND` and `B_MIXED`.
- Any such gating should be evaluated with out-of-sample backtests, but the logic itself can remain a light‑weight overlay on top of existing models.
