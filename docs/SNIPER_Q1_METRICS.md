# SNIPER Q1 Replay Metrics

## Executive summary
- Positive EV per trade (9.14 bps) and payoff ratio (0.53) with win rate 93.1%.

## Baseline & artifact fingerprint
- Policy path: `unknown`
- Entry model sha: `n/a`
- Feature manifest sha: `n/a`
- Exit config path: `n/a`
- Router sha: `n/a`
- Git commit: `unknown`

## Global performance metrics
| Metric | Value |
| --- | --- |
| Total trades | 4283 |
| Closed trades | 4283 |
| Win rate | 93.1% |
| Avg win | 11.37 bps |
| Avg loss | -21.25 bps |
| Payoff ratio | 0.53 |
| EV per trade | 9.14 bps |
| PnL sum | 39089.21 bps |
| Duration p50/p90/p95 | 45.0/1100.0/3005.0 min |
| Exit reasons | RULE_A_PROFIT: 3077, RULE_A_TRAILING: 1053, REPLAY_EOF: 153 |

## Session breakdown
| Session | Trades | Win rate | EV/trade (bps) | Payoff | Avg duration (min) |
| EU | 1215 | 96.5% | 7.36 | 1.74 | 221.45 |
| OVERLAP | 1472 | 91.4% | 7.28 | 1.28 | 278.83 |
| US | 1518 | 95.1% | 7.22 | 2.75 | 804.76 |

## Vol-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) | Payoff | Avg duration (min) |
| LOW | 38 | 57.9% | 1.94 | 2.01 | 55.00 |
| MEDIUM | 44 | 100.0% | 8.93 | nan | 46.36 |
| HIGH | 26 | 100.0% | 14.21 | nan | 82.31 |

## Risk & duration
- Max concurrent open trades observed: 43
- Time-in-market sum: 1901125.0 min | mean: 452.11 min
- Drawdown proxy (loss magnitude) p50/p90: 3.08/56.29 bps

## EOF impact
- 153 trades were closed via `REPLAY_EOF` (3.13% of dataset) to ensure replay completeness when natural exits were pending.

## Konklusjon
- ✅ Klar for Q2-replay (positive EV, full exit coverage, left_open == 0).