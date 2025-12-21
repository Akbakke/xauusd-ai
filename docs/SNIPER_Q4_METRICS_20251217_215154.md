# SNIPER Q4 2025 Replay Metrics

## Executive summary
- EV per trade: 7.45 bps
- Win rate: 88.8%
- Payoff ratio: 0.82

## Baseline & artifact fingerprint
- Policy path: `/Users/andrekildalbakke/Desktop/GX1 XAUUSD/gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml`
- Entry model sha: `n/a`
- Feature manifest sha: `n/a`
- Router sha: `n/a`
- Git commit: `b325f996f0b4c5daa674ae9f96234273a4bd4634`

## Global performance metrics
| Metric | Value |
| --- | --- |
| Total trades | 1,610 |
| Win rate | 88.8% |
| Avg win | 9.93 bps |
| Avg loss | -12.13 bps |
| Payoff ratio | 0.82 |
| EV per trade | 7.45 bps |
| PnL sum | 11,988.27 bps |
| Duration p50/p90/p95 | 30.0/330.0/900.0 min |
| Max duration | 5610.0 min |
| Max concurrent | 10 |

## Exit reasons
- RULE_A_PROFIT: 884 (54.9%)
- RULE_A_TRAILING: 666 (41.4%)
- REPLAY_EOF: 60 (3.7%)

## Session breakdown
| Session | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| EU | 617 | 89.0% | 7.70 |
| OVERLAP | 460 | 90.9% | 9.07 |
| US | 533 | 86.7% | 5.75 |

## Vol-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| LOW | 460 | 88.7% | 7.39 |
| MEDIUM | 494 | 89.5% | 7.48 |
| HIGH | 656 | 88.3% | 7.46 |

## Trend-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| TREND_UP | 285 | 87.0% | 7.67 |
| TREND_DOWN | 298 | 85.2% | 6.80 |
| TREND_NEUTRAL | 1,027 | 90.3% | 7.57 |

## Risk & duration
- Drawdown proxy (loss magnitude) p50/p90: 4.93/40.39 bps