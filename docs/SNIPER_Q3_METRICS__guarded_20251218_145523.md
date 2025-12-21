# SNIPER Q3 2025 Replay Metrics (guarded)

**Variant**: guarded
**Run tag**: replay_20251218_110644
**Quarter**: Q3

## Policy & Guard configuration
- Policy path (variant): `/private/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_pallkebf/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY__guarded.yaml`
- Base policy path: `unknown`
- Guard config path: `unknown`
- Guard enabled: `unknown`

## Executive summary
- EV per trade: 119.88 bps
- Win rate: 73.6%
- Payoff ratio: 4.08

## Baseline & artifact fingerprint
- Policy path: `/private/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_pallkebf/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY__guarded.yaml`
- Entry model sha: `n/a`
- Feature manifest sha: `n/a`
- Router sha: `n/a`
- Git commit: `b325f996f0b4c5daa674ae9f96234273a4bd4634`

## Global performance metrics
| Metric | Value |
| --- | --- |
| Total trades | 6,576 |
| Win rate | 73.6% |
| Avg win | 178.48 bps |
| Avg loss | -43.70 bps |
| Payoff ratio | 4.08 |
| EV per trade | 119.88 bps |
| PnL sum | 788,342.98 bps |
| Duration p50/p90/p95 | 8985.0/16199.0/17345.0 min |
| Max duration | 18580.0 min |
| Max concurrent | 1325 |

## Exit reasons
- REPLAY_EOF: 6,445 (98.0%)
- RULE_A_PROFIT: 93 (1.4%)
- RULE_A_TRAILING: 38 (0.6%)

## Session breakdown
| Session | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| EU | 2,167 | 70.4% | 119.50 |
| OVERLAP | 1,953 | 74.6% | 122.03 |
| US | 2,452 | 75.7% | 116.68 |

## Vol-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| LOW | 2,229 | 75.3% | 119.72 |
| MEDIUM | 2,000 | 72.0% | 116.73 |
| HIGH | 2,343 | 73.4% | 120.81 |

## Trend-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| TREND_UP | 479 | 86.6% | 140.02 |
| TREND_DOWN | 295 | 72.5% | 131.17 |
| TREND_NEUTRAL | 5,798 | 72.6% | 116.87 |

## Risk & duration
- Drawdown proxy (loss magnitude) p50/p90: 29.05/90.74 bps
- Max concurrent (global merged): 1325
- Max concurrent (per-chunk max): 1297
- Max concurrent (per-chunk p90): 579.4
- Chunks: 7