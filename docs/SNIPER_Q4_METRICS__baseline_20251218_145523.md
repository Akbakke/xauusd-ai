# SNIPER Q4 2025 Replay Metrics (baseline)

**Variant**: baseline
**Run tag**: replay_20251218_111934
**Quarter**: Q4

## Policy & Guard configuration
- Policy path (variant): `/private/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_pallkebf/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY__baseline.yaml`
- Base policy path: `unknown`
- Guard config path: `unknown`
- Guard enabled: `unknown`

## Executive summary
- EV per trade: 20.64 bps
- Win rate: 52.6%
- Payoff ratio: 1.28

## Baseline & artifact fingerprint
- Policy path: `/private/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_pallkebf/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY__baseline.yaml`
- Entry model sha: `n/a`
- Feature manifest sha: `n/a`
- Router sha: `n/a`
- Git commit: `b325f996f0b4c5daa674ae9f96234273a4bd4634`

## Global performance metrics
| Metric | Value |
| --- | --- |
| Total trades | 4,482 |
| Win rate | 52.6% |
| Avg win | 132.31 bps |
| Avg loss | -103.48 bps |
| Payoff ratio | 1.28 |
| EV per trade | 20.64 bps |
| PnL sum | 92,519.57 bps |
| Duration p50/p90/p95 | 4220.0/10920.0/11315.0 min |
| Max duration | 15380.0 min |
| Max concurrent | 816 |

## Exit reasons
- REPLAY_EOF: 4,307 (96.1%)
- RULE_A_PROFIT: 96 (2.1%)
- RULE_A_TRAILING: 79 (1.8%)

## Session breakdown
| Session | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| EU | 1,638 | 50.9% | 14.45 |
| OVERLAP | 1,157 | 56.9% | 33.31 |
| US | 1,684 | 51.3% | 16.62 |

## Vol-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| LOW | 1,798 | 50.6% | 15.25 |
| MEDIUM | 1,267 | 52.8% | 23.38 |
| HIGH | 1,414 | 55.0% | 23.45 |

## Trend-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| TREND_UP | 663 | 38.8% | -38.45 |
| TREND_DOWN | 540 | 57.0% | 17.98 |
| TREND_NEUTRAL | 3,276 | 54.7% | 32.35 |

## Risk & duration
- Drawdown proxy (loss magnitude) p50/p90: 55.73/285.56 bps
- Max concurrent (global merged): 816
- Max concurrent (per-chunk max): 815
- Max concurrent (per-chunk p90): 506.6
- Chunks: 5