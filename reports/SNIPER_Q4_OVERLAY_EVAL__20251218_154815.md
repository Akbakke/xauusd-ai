# SNIPER Q4 Overlay Evaluation (20251218_154815)

This report compares Q4 2025 SNIPER performance with and without the Q4 × B_MIXED size overlay (no replay; existing run_dirs only).

## Variant base: `baseline`


- Normal run: `gx1/wf_runs/SNIPER_OBS_Q4_2025_baseline_20251218_151357` (source: trades/*.json (CSV missing/empty))
- Overlay run: `gx1/wf_runs/SNIPER_OBS_Q4_2025_baseline_overlay_20251218_152108` (source: trades/*.json (CSV missing/empty))

### Data source & coverage

- normal: source=trades/*.json (CSV missing/empty), pnl_coverage=100.0%, regime_coverage=100.0%
- overlay: source=trades/*.json (CSV missing/empty), pnl_coverage=100.0%, regime_coverage=100.0%

### Q4 total metrics

| Run | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal | 4443 | 16.04 | 51.7% | 1.24 | 285.4 |
| overlay | 4605 | 23.50 | 52.6% | 1.33 | 284.5 |

### Q4 B_MIXED metrics only

| Run | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal | 3 | 815.34 | 100.0% | nan | 0.0 |
| overlay | 8 | 862.41 | 100.0% | nan | 0.0 |

### Overlay sanity

- overlay_present (JSON with `sniper_overlay`): 4597
- overlay_applied (`overlay_applied=True`): 0
- By construction, overlay_applied trades are only Q4 × B_MIXED and have consistent size_before/after and multiplier.

## Variant base: `guarded`


- Normal run: `gx1/wf_runs/SNIPER_OBS_Q4_2025_guarded_20251218_152918` (source: trades/*.json (CSV missing/empty))
- Overlay run: `gx1/wf_runs/SNIPER_OBS_Q4_2025_guarded_overlay_20251218_153815` (source: missing (no trade_journal_index.csv, no trades/*.json))

### Data source & coverage

- normal: source=trades/*.json (CSV missing/empty), pnl_coverage=100.0%, regime_coverage=100.0%
- overlay: source=missing (no trade_journal_index.csv, no trades/*.json), pnl_coverage=0.0%, regime_coverage=0.0%

### Q4 total metrics

| Run | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal | 4359 | 25.18 | 53.0% | 1.34 | 286.4 |
| overlay | 0 | 0.00 | 0.0% | nan | 0.0 |

### Q4 B_MIXED metrics only

| Run | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal | 6 | 797.42 | 100.0% | nan | 0.0 |
| overlay | 0 | 0.00 | 0.0% | nan | 0.0 |

### Overlay sanity

- overlay_present (JSON with `sniper_overlay`): 0
- overlay_applied (`overlay_applied=True`): 0
- By construction, overlay_applied trades are only Q4 × B_MIXED and have consistent size_before/after and multiplier.
