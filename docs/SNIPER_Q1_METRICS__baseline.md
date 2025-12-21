# SNIPER Q1 2025 Replay Metrics (baseline)

**Variant**: baseline
**Run tag**: replay_20251218_124411
**Quarter**: Q1

## Policy & Guard configuration
- Policy path (variant): `/private/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_klsjywcp/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY__baseline.yaml`
- Base policy path: `unknown`
- Guard config path: `unknown`
- Guard enabled: `unknown`

## Executive summary
- EV per trade: 156.29 bps
- Win rate: 84.6%
- Payoff ratio: 3.20

## Baseline & artifact fingerprint
- Policy path: `/private/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_klsjywcp/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY__baseline.yaml`
- Entry model sha: `n/a`
- Feature manifest sha: `n/a`
- Router sha: `n/a`
- Git commit: `b325f996f0b4c5daa674ae9f96234273a4bd4634`

## Global performance metrics
| Metric | Value |
| --- | --- |
| Total trades | 6,935 |
| Win rate | 84.6% |
| Avg win | 195.75 bps |
| Avg loss | -61.11 bps |
| Payoff ratio | 3.20 |
| EV per trade | 156.29 bps |
| PnL sum | 1,083,878.82 bps |
| Duration p50/p90/p95 | 9290.0/16123.0/17075.0 min |
| Max duration | 18395.0 min |
| Max concurrent | 1150 |

## Exit reasons
- REPLAY_EOF: 6,781 (97.8%)
- RULE_A_PROFIT: 116 (1.7%)
- RULE_A_TRAILING: 38 (0.5%)

## Session breakdown
| Session | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| EU | 2,152 | 85.1% | 166.19 |
| OVERLAP | 1,822 | 82.6% | 150.86 |
| US | 2,954 | 85.5% | 151.73 |

## Vol-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| LOW | 2,368 | 85.7% | 165.61 |
| MEDIUM | 2,067 | 85.9% | 149.23 |
| HIGH | 2,493 | 82.6% | 152.47 |

## Trend-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| TREND_UP | 401 | 83.5% | 137.07 |
| TREND_DOWN | 384 | 82.6% | 159.86 |
| TREND_NEUTRAL | 6,143 | 84.8% | 156.99 |

## Risk & duration
- Drawdown proxy (loss magnitude) p50/p90: 61.63/127.81 bps
- Max concurrent (global merged): 1150
- Max concurrent (per-chunk max): 1150
- Max concurrent (per-chunk p90): 1150.0
- Chunks: 1