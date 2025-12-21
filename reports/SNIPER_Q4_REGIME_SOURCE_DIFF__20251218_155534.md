# SNIPER Q4 Regime Source Comparison (20251218_155534)

This report compares regime inputs/classification between:
- INDEX source: trade_journal_index.csv
- JSON source: trade_journal/trades/*.json

## Run pair: `baseline`

- Baseline run_dir: `gx1/wf_runs/SNIPER_OBS_Q4_2025_baseline_overlay_20251218_152108`
- Baseline_overlay run_dir: `gx1/wf_runs/SNIPER_OBS_Q4_2025_baseline_overlay_20251218_152108`

- INDEX source available: False
- JSON  source available: True

- Joined rows (index∩json): 0
- Join rate (vs json rows): 0.0%

### A) Field coverage (index vs json)

| Field | index non-null% | json non-null% |
| --- | ---: | ---: |
| trend_regime | 0.0% | 99.8% |
| vol_regime | 0.0% | 99.8% |
| atr_bps | 0.0% | 0.0% |
| spread_bps | 0.0% | 0.0% |
| session | 0.0% | 99.8% |

- atr_bps index: min=nan max=nan
- atr_bps json : min=nan max=nan
- spread_bps index: min=nan max=nan
- spread_bps json : min=nan max=nan

### B) Regime distribution (A/B/C)

| Regime | count_index | count_json |
| --- | ---: | ---: |
| A_TREND | 0 | 1233 |
| B_MIXED | 0 | 8 |
| C_CHOP | 0 | 3364 |

### C) Regime agreement matrix (index vs json)

_No joined rows; cannot compute agreement matrix._

### D) B_MIXED disappearance drilldown

_No joined rows; cannot analyze B_MIXED drift._

### E) Auto hypothesis about source mismatch

- Index B_MIXED=0, json B_MIXED=8. Counts are closer; mismatch may be due to per-trade differences rather than gross drift.
