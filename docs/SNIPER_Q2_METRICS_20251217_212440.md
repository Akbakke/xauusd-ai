# SNIPER Q2 2025 Replay Metrics

## Executive summary
- EV per trade: 9.78 bps
- Win rate: 89.6%
- Payoff ratio: 0.34

## Baseline & artifact fingerprint
- Policy path: `/Users/andrekildalbakke/Desktop/GX1 XAUUSD/gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml`
- Entry model sha: `n/a`
- Feature manifest sha: `n/a`
- Router sha: `n/a`
- Git commit: `b325f996f0b4c5daa674ae9f96234273a4bd4634`

## Global performance metrics
| Metric | Value |
| --- | --- |
| Total trades | 1,763 |
| Win rate | 89.6% |
| Avg win | 16.74 bps |
| Avg loss | -49.95 bps |
| Payoff ratio | 0.34 |
| EV per trade | 9.78 bps |
| PnL sum | 17,234.12 bps |
| Duration p50/p90/p95 | 35.0/695.0/1220.0 min |
| Max duration | 14145.0 min |
| Max concurrent | 23 |

## Exit reasons
- RULE_A_PROFIT: 1,058 (60.0%)
- RULE_A_TRAILING: 616 (34.9%)
- REPLAY_EOF: 89 (5.0%)

## Session breakdown
| Session | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| EU | 628 | 91.6% | 9.73 |
| OVERLAP | 565 | 88.5% | 16.63 |
| US | 548 | 92.0% | 16.38 |

## Vol-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| LOW | 471 | 94.7% | 18.10 |
| MEDIUM | 484 | 90.9% | 10.93 |
| HIGH | 786 | 88.2% | 13.56 |

## Trend-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| TREND_UP | 286 | 86.7% | 12.89 |
| TREND_DOWN | 390 | 88.2% | 17.26 |
| TREND_NEUTRAL | 1,065 | 92.7% | 13.20 |

## Risk & duration
- Drawdown proxy (loss magnitude) p50/p90: 5.16/215.57 bps