# SNIPER Q2 2025 Replay Metrics (baseline)

**Variant**: baseline
**Run tag**: replay_20251221_182115
**Quarter**: Q2

## Policy & Guard configuration
- Policy path (variant): `/private/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_nx47vze9/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY__baseline.yaml`
- Base policy path: `unknown`
- Guard config path: `/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_nx47vze9/SNIPER_RISK_GUARD_V1__baseline.yaml`
- Guard enabled: `False`

## Executive summary
- EV per trade: 126.79 bps
- Win rate: 70.9%
- Payoff ratio: 2.02

## Baseline & artifact fingerprint
- Policy path: `/private/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_nx47vze9/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY__baseline.yaml`
- Entry model sha: `n/a`
- Feature manifest sha: `n/a`
- Router sha: `n/a`
- Git commit: `53e395abeadcc80d0d6247a668dfa2efb3aea7c3`

## Global performance metrics
| Metric | Value |
| --- | --- |
| Total trades | 6,389 |
| Win rate | 70.9% |
| Avg win | 224.82 bps |
| Avg loss | -111.56 bps |
| Payoff ratio | 2.02 |
| EV per trade | 126.79 bps |
| PnL sum | 810,033.75 bps |
| Duration p50/p90/p95 | 9895.0/16600.0/17345.0 min |
| Max duration | 19690.0 min |
| Max concurrent | 1265 |

## Exit reasons
- REPLAY_EOF: 6,170 (96.6%)
- RULE_A_PROFIT: 126 (2.0%)
- RULE_A_TRAILING: 93 (1.5%)

## Session breakdown
| Session | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| EU | 2,205 | 72.0% | 130.24 |
| OVERLAP | 0 | n/a | n/a |
| US | 4,184 | 70.3% | 124.96 |

## Vol-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| LOW | 2,268 | 68.7% | 126.00 |
| MEDIUM | 1,842 | 73.2% | 138.44 |
| HIGH | 2,279 | 71.2% | 118.15 |

## Trend-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| TREND_UP | 716 | 72.6% | 134.45 |
| TREND_DOWN | 892 | 75.1% | 191.65 |
| TREND_NEUTRAL | 4,781 | 69.8% | 113.53 |

## Risk & duration
- Drawdown proxy (loss magnitude) p50/p90: 93.02/212.17 bps
- Max concurrent (global merged): 1265
- Max concurrent (per-chunk max): 1188
- Max concurrent (per-chunk p90): 528.0
- Chunks: 7