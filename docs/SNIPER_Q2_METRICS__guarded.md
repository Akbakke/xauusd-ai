# SNIPER Q2 2025 Replay Metrics (guarded)

**Variant**: guarded
**Run tag**: replay_20251218_100626
**Quarter**: Q2

## Policy & Guard configuration
- Policy path (variant): `/private/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_svh9nnn5/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY__guarded.yaml`
- Base policy path: `unknown`
- Guard config path: `/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_svh9nnn5/SNIPER_RISK_GUARD_V1__guarded.yaml`
- Guard enabled: `True`

## Executive summary
- EV per trade: 122.22 bps
- Win rate: 70.9%
- Payoff ratio: 1.96

## Baseline & artifact fingerprint
- Policy path: `/private/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_svh9nnn5/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY__guarded.yaml`
- Entry model sha: `n/a`
- Feature manifest sha: `n/a`
- Router sha: `n/a`
- Git commit: `b325f996f0b4c5daa674ae9f96234273a4bd4634`

## Global performance metrics
| Metric | Value |
| --- | --- |
| Total trades | 6,226 |
| Win rate | 70.9% |
| Avg win | 217.97 bps |
| Avg loss | -111.39 bps |
| Payoff ratio | 1.96 |
| EV per trade | 122.22 bps |
| PnL sum | 760,938.33 bps |
| Duration p50/p90/p95 | 9845.0/16445.0/17365.0 min |
| Max duration | 19675.0 min |
| Max concurrent | 1265 |

## Exit reasons
- REPLAY_EOF: 6,014 (96.6%)
- RULE_A_PROFIT: 118 (1.9%)
- RULE_A_TRAILING: 94 (1.5%)

## Session breakdown
| Session | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| EU | 2,173 | 72.1% | 126.93 |
| OVERLAP | 1,709 | 70.0% | 108.08 |
| US | 2,344 | 70.5% | 128.17 |

## Vol-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| LOW | 2,282 | 68.7% | 123.50 |
| MEDIUM | 1,818 | 73.9% | 139.17 |
| HIGH | 2,126 | 70.7% | 106.35 |

## Trend-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| TREND_UP | 714 | 72.7% | 126.38 |
| TREND_DOWN | 780 | 74.9% | 182.79 |
| TREND_NEUTRAL | 4,732 | 70.0% | 111.61 |

## Risk & duration
- Drawdown proxy (loss magnitude) p50/p90: 93.98/210.62 bps
- Max concurrent (global merged): 1265
- Max concurrent (per-chunk max): 1251
- Max concurrent (per-chunk p90): 553.2
- Chunks: 7