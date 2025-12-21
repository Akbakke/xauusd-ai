# SNIPER Q3 2025 Replay Metrics

## Executive summary
- EV per trade: 7.17 bps
- Win rate: 93.2%
- Payoff ratio: 0.81

## Baseline & artifact fingerprint
- Policy path: `/Users/andrekildalbakke/Desktop/GX1 XAUUSD/gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml`
- Entry model sha: `n/a`
- Feature manifest sha: `n/a`
- Router sha: `n/a`
- Git commit: `b325f996f0b4c5daa674ae9f96234273a4bd4634`

## Global performance metrics
| Metric | Value |
| --- | --- |
| Total trades | 1,498 |
| Win rate | 93.2% |
| Avg win | 8.45 bps |
| Avg loss | -10.41 bps |
| Payoff ratio | 0.81 |
| EV per trade | 7.17 bps |
| PnL sum | 10,740.00 bps |
| Duration p50/p90/p95 | 45.0/1071.5/2398.0 min |
| Max duration | 7290.0 min |
| Max concurrent | 10 |

## Exit reasons
- RULE_A_PROFIT: 1,059 (70.7%)
- RULE_A_TRAILING: 385 (25.7%)
- REPLAY_EOF: 54 (3.6%)

## Session breakdown
| Session | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| EU | 399 | 93.0% | 5.14 |
| OVERLAP | 627 | 90.6% | 8.25 |
| US | 472 | 96.8% | 7.47 |

## Vol-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| LOW | 299 | 97.0% | 7.50 |
| MEDIUM | 482 | 95.0% | 7.69 |
| HIGH | 717 | 90.4% | 6.69 |

## Trend-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| TREND_UP | 230 | 90.4% | 6.97 |
| TREND_DOWN | 121 | 94.2% | 8.30 |
| TREND_NEUTRAL | 1,147 | 93.6% | 7.10 |

## Risk & duration
- Drawdown proxy (loss magnitude) p50/p90: 4.43/28.10 bps