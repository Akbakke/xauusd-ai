# SNIPER Q2 2025 Replay Metrics (baseline)

**Variant**: baseline
**Run tag**: replay_20251218_095617
**Quarter**: Q2

## Policy & Guard configuration
- Policy path (variant): `/private/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_svh9nnn5/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY__baseline.yaml`
- Base policy path: `unknown`
- Guard config path: `unknown`
- Guard enabled: `unknown`

## Executive summary
- EV per trade: 122.42 bps
- Win rate: 69.7%
- Payoff ratio: 2.00

## Baseline & artifact fingerprint
- Policy path: `/private/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_svh9nnn5/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY__baseline.yaml`
- Entry model sha: `n/a`
- Feature manifest sha: `n/a`
- Router sha: `n/a`
- Git commit: `b325f996f0b4c5daa674ae9f96234273a4bd4634`

## Global performance metrics
| Metric | Value |
| --- | --- |
| Total trades | 6,102 |
| Win rate | 69.7% |
| Avg win | 224.14 bps |
| Avg loss | -111.94 bps |
| Payoff ratio | 2.00 |
| EV per trade | 122.42 bps |
| PnL sum | 746,981.57 bps |
| Duration p50/p90/p95 | 9735.0/16525.0/17400.0 min |
| Max duration | 18830.0 min |
| Max concurrent | 1265 |

## Exit reasons
- REPLAY_EOF: 5,891 (96.5%)
- RULE_A_PROFIT: 122 (2.0%)
- RULE_A_TRAILING: 89 (1.5%)

## Session breakdown
| Session | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| EU | 2,096 | 70.8% | 125.28 |
| OVERLAP | 1,747 | 69.0% | 115.41 |
| US | 2,258 | 69.3% | 125.21 |

## Vol-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| LOW | 2,141 | 67.1% | 119.20 |
| MEDIUM | 1,768 | 72.5% | 135.12 |
| HIGH | 2,192 | 70.1% | 115.34 |

## Trend-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| TREND_UP | 718 | 73.3% | 135.88 |
| TREND_DOWN | 858 | 74.2% | 196.43 |
| TREND_NEUTRAL | 4,525 | 68.3% | 106.26 |

## Risk & duration
- Drawdown proxy (loss magnitude) p50/p90: 93.82/212.35 bps
- Max concurrent (global merged): 1265
- Max concurrent (per-chunk max): 981
- Max concurrent (per-chunk p90): 556.8
- Chunks: 7