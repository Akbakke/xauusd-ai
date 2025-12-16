# EXIT_A Q2 Phase-A micro-sweep

Variants share FARM_V2B mp68_max5_cd0 entry. Sweep adjusts `rule_a_profit_min_bps` only (4/5/6).

| Variant | Label | rule_a_min | Trades | Trades/day | Win% | Avg bps | EV/day | Missing exit_profile | Max DD (bps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_rule4 | mp68_max5_rule4_cd0 (baseline) | 4.0 | 73 | 0.96 | 69.86% | 42.37 | 40.69 | 0 | 2157.57 |
| rule5 | rule_a_min=5 | 5.0 | 239 | 3.14 | 78.24% | 92.26 | 290.08 | 0 | 2157.57 |
| rule6 | rule_a_min=6 | 6.0 | 73 | 0.96 | 69.86% | 42.19 | 40.51 | 0 | 2157.57 |

Top variant meeting win_rate ≥ 75% and EV/day ≥ 35 bps: **rule_a_min=5 (rule5)**