# SNIPER Q4 2025 Replay Metrics (guarded)

**Variant**: guarded
**Run tag**: replay_20251218_112755
**Quarter**: Q4

## Policy & Guard configuration
- Policy path (variant): `/private/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_pallkebf/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY__guarded.yaml`
- Base policy path: `unknown`
- Guard config path: `unknown`
- Guard enabled: `unknown`

## Executive summary
- EV per trade: 21.67 bps
- Win rate: 52.1%
- Payoff ratio: 1.33

## Baseline & artifact fingerprint
- Policy path: `/private/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_pallkebf/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY__guarded.yaml`
- Entry model sha: `n/a`
- Feature manifest sha: `n/a`
- Router sha: `n/a`
- Git commit: `b325f996f0b4c5daa674ae9f96234273a4bd4634`

## Global performance metrics
| Metric | Value |
| --- | --- |
| Total trades | 4,388 |
| Win rate | 52.1% |
| Avg win | 135.38 bps |
| Avg loss | -102.14 bps |
| Payoff ratio | 1.33 |
| EV per trade | 21.67 bps |
| PnL sum | 95,105.71 bps |
| Duration p50/p90/p95 | 4332.5/10915.0/11290.0 min |
| Max duration | 15380.0 min |
| Max concurrent | 804 |

## Exit reasons
- REPLAY_EOF: 4,211 (96.0%)
- RULE_A_PROFIT: 98 (2.2%)
- RULE_A_TRAILING: 79 (1.8%)

## Session breakdown
| Session | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| EU | 1,586 | 52.5% | 19.23 |
| OVERLAP | 1,141 | 54.6% | 30.13 |
| US | 1,661 | 50.1% | 18.20 |

## Vol-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| LOW | 1,785 | 50.4% | 16.58 |
| MEDIUM | 1,275 | 53.6% | 26.29 |
| HIGH | 1,328 | 53.0% | 24.09 |

## Trend-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| TREND_UP | 651 | 38.9% | -37.72 |
| TREND_DOWN | 453 | 57.8% | 26.44 |
| TREND_NEUTRAL | 3,284 | 54.0% | 32.79 |

## Risk & duration
- Drawdown proxy (loss magnitude) p50/p90: 54.50/285.70 bps
- Max concurrent (global merged): 804
- Max concurrent (per-chunk max): 768
- Max concurrent (per-chunk p90): 411.5
- Chunks: 6