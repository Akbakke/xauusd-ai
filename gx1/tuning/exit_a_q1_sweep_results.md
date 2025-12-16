# EXIT_A Q1 Sweep (2025-12-08T18:21:31.523838Z)

| Variant | min_prob_long | max_open | rule_a_min | cooldown_s | trades | closed | miss_exit_profile | trades/day | win_rate% | EV/trade (bps) | EV/day (bps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.72 | 3 | 6.0 | 60 | 36 | 36 | 0 | 0.41 | 91.67 | 119.65 | 48.95 |
| mp70_max4_rule5_cd60 | 0.70 | 4 | 5.0 | 60 | 51 | 51 | 0 | 0.58 | 92.16 | 123.71 | 71.70 |
| mp70_max4_rule4_cd0 | 0.70 | 4 | 4.0 | 0 | 51 | 51 | 0 | 0.58 | 92.16 | 123.71 | 71.70 |
| mp68_max5_rule5_cd60 | 0.68 | 5 | 5.0 | 60 | 64 | 64 | 0 | 0.73 | 92.19 | 125.02 | 90.92 |
| mp70_max5_rule4_cd0 | 0.70 | 5 | 4.0 | 0 | 64 | 64 | 0 | 0.73 | 92.19 | 124.81 | 90.77 |
| mp68_max5_rule5_cd0 | 0.68 | 5 | 5.0 | 0 | 64 | 64 | 0 | 0.73 | 92.19 | 125.02 | 90.92 |
| mp68_max5_rule4_cd60 | 0.68 | 5 | 4.0 | 60 | 64 | 64 | 0 | 0.73 | 92.19 | 125.02 | 90.92 |
| mp68_max4_rule4_cd0 | 0.68 | 4 | 4.0 | 0 | 51 | 51 | 0 | 0.58 | 92.16 | 124.10 | 71.92 |
| mp68_max5_rule4_cd0 | 0.68 | 5 | 4.0 | 0 | 94 | 88 | 0 | 1.00 | 86.36 | 104.99 | 104.99 |

## Top Variants (win_rate ≥ 80%, trades/day ≥ 0.8, sorted by EV/day)

- **mp68_max5_rule4_cd0** → EV/day 104.99 bps, trades/day 1.00, win_rate 86.36%, min_prob_long 0.68, max_open 5, rule_a_min 4.0, cooldown 0s