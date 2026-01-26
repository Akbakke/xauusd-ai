# SNIPER Q1 2025 Replay Metrics (baseline)

**Variant**: baseline
**Run tag**: replay_20251221_180957
**Quarter**: Q1

## Policy & Guard configuration
- Policy path (variant): `/private/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_nx47vze9/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY__baseline.yaml`
- Base policy path: `unknown`
- Guard config path: `/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_nx47vze9/SNIPER_RISK_GUARD_V1__baseline.yaml`
- Guard enabled: `False`

## Executive summary
- EV per trade: 156.63 bps
- Win rate: 84.9%
- Payoff ratio: 3.19

## Baseline & artifact fingerprint
- Policy path: `/private/var/folders/hz/y88m0lpx5sg5v_l83qb8ww2h0000gn/T/sniper_variants_nx47vze9/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY__baseline.yaml`
- Entry model sha: `n/a`
- Feature manifest sha: `n/a`
- Router sha: `n/a`
- Git commit: `53e395abeadcc80d0d6247a668dfa2efb3aea7c3`

## Global performance metrics
| Metric | Value |
| --- | --- |
| Total trades | 6,925 |
| Win rate | 84.9% |
| Avg win | 195.23 bps |
| Avg loss | -61.14 bps |
| Payoff ratio | 3.19 |
| EV per trade | 156.63 bps |
| PnL sum | 1,084,676.49 bps |
| Duration p50/p90/p95 | 9255.0/16130.0/17095.0 min |
| Max duration | 18350.0 min |
| Max concurrent | 1132 |

## Exit reasons
- REPLAY_EOF: 6,771 (97.8%)
- RULE_A_PROFIT: 114 (1.6%)
- RULE_A_TRAILING: 40 (0.6%)

## Session breakdown
| Session | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| EU | 2,204 | 84.3% | 162.33 |
| OVERLAP | 0 | n/a | n/a |
| US | 4,708 | 85.2% | 153.65 |

## Vol-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| LOW | 2,342 | 86.6% | 169.19 |
| MEDIUM | 2,060 | 86.1% | 150.40 |
| HIGH | 2,510 | 82.4% | 149.44 |

## Trend-regime breakdown
| Regime | Trades | Win rate | EV/trade (bps) |
| --- | --- | --- | --- |
| TREND_UP | 412 | 84.0% | 135.89 |
| TREND_DOWN | 424 | 80.4% | 152.88 |
| TREND_NEUTRAL | 6,076 | 85.3% | 158.06 |

## Risk & duration
- Drawdown proxy (loss magnitude) p50/p90: 60.50/127.89 bps
- Max concurrent (global merged): 1132
- Max concurrent (per-chunk max): 1127
- Max concurrent (per-chunk p90): 604.5
- Chunks: 6