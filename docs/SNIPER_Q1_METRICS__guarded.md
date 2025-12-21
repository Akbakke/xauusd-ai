# SNIPER Q1 2025 Replay Metrics (guarded)

**Variant**: guarded
**Run tag**: replay_20251218_115832
**Quarter**: Q1

## Policy & Guard configuration
- Policy path (variant): `/private/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_xfqbv3bm/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY__guarded.yaml`
- Base policy path: `unknown`
- Guard config path: `/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_xfqbv3bm/SNIPER_RISK_GUARD_V1__guarded.yaml`
- Guard enabled: `True`

## Executive summary
- EV per trade: 153.85 bps
- Win rate: 84.8%
- Payoff ratio: 3.13

## Baseline & artifact fingerprint
- Policy path: `/private/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_xfqbv3bm/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY__guarded.yaml`
- Entry model sha: `n/a`
- Feature manifest sha: `n/a`
- Router sha: `n/a`
- Git commit: `b325f996f0b4c5daa674ae9f96234273a4bd4634`

## Global performance metrics
| Metric | Value |
| --- | --- |
| Total trades | 7,208 |
| Win rate | 84.8% |
| Avg win | 192.33 bps |
| Avg loss | -61.37 bps |
| Payoff ratio | 3.13 |
| EV per trade | 153.85 bps |
| PnL sum | 1,108,947.28 bps |
| Duration p50/p90/p95 | 9735.0/16250.0/17185.0 min |
| Max duration | 18345.0 min |
| Max concurrent | 1172 |

## Exit reasons
- REPLAY_EOF: 7,040 (97.7%)
- RULE_A_PROFIT: 125 (1.7%)
- RULE_A_TRAILING: 43 (0.6%)

## Session breakdown
| Session | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| EU | 2,289 | 84.4% | 160.83 |
| OVERLAP | 1,846 | 83.9% | 152.49 |
| US | 3,067 | 85.7% | 148.40 |

## Vol-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| LOW | 2,490 | 85.7% | 160.75 |
| MEDIUM | 2,141 | 85.8% | 147.11 |
| HIGH | 2,571 | 83.2% | 151.52 |

## Trend-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| TREND_UP | 416 | 84.1% | 135.26 |
| TREND_DOWN | 420 | 81.9% | 155.85 |
| TREND_NEUTRAL | 6,366 | 85.0% | 154.43 |

## Risk & duration
- Drawdown proxy (loss magnitude) p50/p90: 62.55/125.75 bps
- Max concurrent (global merged): 1172
- Max concurrent (per-chunk max): 1200
- Max concurrent (per-chunk p90): 520.8
- Chunks: 7