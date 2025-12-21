# SNIPER Q4 Overlay Evaluation (20251218_150643)

This report compares Q4 2025 SNIPER performance with and without the Q4 × B_MIXED size overlay (no replay; existing run_dirs only).

## Variant base: `baseline`

- Normal run: `gx1/wf_runs/SNIPER_OBS_Q4_2025_baseline_overlay_20251218_150613`
- Overlay run: `gx1/wf_runs/SNIPER_OBS_Q4_2025_baseline_overlay_20251218_150613`

### Q4 total metrics

| Run | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal | 0 | 0.00 | 0.0% | nan | 0.0 |
| overlay | 0 | 0.00 | 0.0% | nan | 0.0 |

### Q4 B_MIXED metrics only

| Run | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal | 0 | 0.00 | 0.0% | nan | 0.0 |
| overlay | 0 | 0.00 | 0.0% | nan | 0.0 |

### Overlay sanity

- overlay_present (JSON with `sniper_overlay`): 0
- overlay_applied (`overlay_applied=True`): 0
- By construction, overlay_applied trades are only Q4 × B_MIXED and have consistent size_before/after and multiplier.

## Variant base: `guarded`

- Normal run: `gx1/wf_runs/SNIPER_OBS_Q4_2025_guarded_overlay_20251218_150633`
- Overlay run: `gx1/wf_runs/SNIPER_OBS_Q4_2025_guarded_overlay_20251218_150633`

### Q4 total metrics

| Run | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal | 0 | 0.00 | 0.0% | nan | 0.0 |
| overlay | 0 | 0.00 | 0.0% | nan | 0.0 |

### Q4 B_MIXED metrics only

| Run | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal | 0 | 0.00 | 0.0% | nan | 0.0 |
| overlay | 0 | 0.00 | 0.0% | nan | 0.0 |

### Overlay sanity

- overlay_present (JSON with `sniper_overlay`): 0
- overlay_applied (`overlay_applied=True`): 0
- By construction, overlay_applied trades are only Q4 × B_MIXED and have consistent size_before/after and multiplier.
