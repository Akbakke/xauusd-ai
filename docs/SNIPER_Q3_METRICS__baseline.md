# SNIPER Q3 2025 Replay Metrics (baseline)

**Variant**: baseline
**Run tag**: replay_20251218_105343
**Quarter**: Q3

## Policy & Guard configuration
- Policy path (variant): `/private/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_pallkebf/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY__baseline.yaml`
- Base policy path: `unknown`
- Guard config path: `/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_pallkebf/SNIPER_RISK_GUARD_V1__baseline.yaml`
- Guard enabled: `False`

## Executive summary
- EV per trade: 113.16 bps
- Win rate: 72.1%
- Payoff ratio: 3.88

## Baseline & artifact fingerprint
- Policy path: `/private/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_pallkebf/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY__baseline.yaml`
- Entry model sha: `n/a`
- Feature manifest sha: `n/a`
- Router sha: `n/a`
- Git commit: `b325f996f0b4c5daa674ae9f96234273a4bd4634`

## Global performance metrics
| Metric | Value |
| --- | --- |
| Total trades | 6,689 |
| Win rate | 72.1% |
| Avg win | 174.34 bps |
| Avg loss | -44.92 bps |
| Payoff ratio | 3.88 |
| EV per trade | 113.16 bps |
| PnL sum | 756,954.25 bps |
| Duration p50/p90/p95 | 9030.0/16220.0/17215.0 min |
| Max duration | 18640.0 min |
| Max concurrent | 1339 |

## Exit reasons
- REPLAY_EOF: 6,555 (98.0%)
- RULE_A_PROFIT: 97 (1.5%)
- RULE_A_TRAILING: 37 (0.6%)

## Session breakdown
| Session | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| EU | 2,243 | 67.8% | 113.20 |
| OVERLAP | 1,965 | 74.4% | 118.49 |
| US | 2,477 | 74.1% | 107.15 |

## Vol-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| LOW | 2,253 | 74.6% | 113.56 |
| MEDIUM | 2,031 | 69.9% | 110.71 |
| HIGH | 2,401 | 71.6% | 113.05 |

## Trend-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| TREND_UP | 480 | 82.7% | 118.94 |
| TREND_DOWN | 308 | 72.7% | 126.48 |
| TREND_NEUTRAL | 5,897 | 71.2% | 111.26 |

## Risk & duration
- Drawdown proxy (loss magnitude) p50/p90: 29.42/95.06 bps
- Max concurrent (global merged): 1339
- Max concurrent (per-chunk max): 1318
- Max concurrent (per-chunk p90): 566.2
- Chunks: 7