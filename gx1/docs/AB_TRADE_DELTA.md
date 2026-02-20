# A/B Trade Outcomes Delta (Exit ML)

TRUTH-grade analyzer that compares **trade_outcomes_*_MERGED.parquet** from Run A (baseline) and Run B (ML-frozen). Deterministic, no network, no legacy replay imports.

## Command

```bash
/home/andre2/venvs/gx1/bin/python -m gx1.scripts.ab_trade_outcomes_delta \
  --run-a /home/andre2/GX1_DATA/reports/truth_e2e_sanity/A_BASELINE_AB_EXIT_ML_20260219_152051 \
  --run-b /home/andre2/GX1_DATA/reports/truth_e2e_sanity/B_ML_FROZEN_AB_EXIT_ML_20260219_152051 \
  --ab-run-id AB_EXIT_ML_20260219_152051
```

## Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--run-a` | yes | — | Run A (baseline) root dir |
| `--run-b` | yes | — | Run B (ML-frozen) root dir |
| `--out-dir` | no | `/home/andre2/GX1_DATA/reports/ab_fullyear_2025_exit_ml` | Output root |
| `--ab-run-id` | no | `AB_TRADE_DELTA_<utc_ts>` | Subdir name under out-dir |
| `--max-rows` | no | 50 | Max rows in top/worst tables in MD |
| `--strict` | no | off | Hard-fail if key columns missing (default: best-effort) |

## Outputs

All written under **`<out-dir>/<ab-run-id>/`** (e.g. `/home/andre2/GX1_DATA/reports/ab_fullyear_2025_exit_ml/AB_EXIT_ML_20260219_152051/`):

| File | Description |
|------|-------------|
| **AB_TRADE_DELTA.md** | Summary: join strategy, coverage, delta stats, exit transitions, top winners/losers |
| **AB_TRADE_DELTA.csv** | Full joined table with A_, B_, delta_* columns |
| **AB_TRADE_DELTA_META.json** | Metadata: run dirs, join strategy, column mappings, counts |

Writes are atomic (tmp then rename).

## Smoke test

```bash
/home/andre2/venvs/gx1/bin/python -m gx1.scripts.smoke_ab_trade_delta \
  --run-a /home/andre2/GX1_DATA/reports/truth_e2e_sanity/A_BASELINE_AB_EXIT_ML_20260219_152051 \
  --run-b /home/andre2/GX1_DATA/reports/truth_e2e_sanity/B_ML_FROZEN_AB_EXIT_ML_20260219_152051
```

Output goes to `/home/andre2/GX1_DATA/reports/_smoke/smoke_ab_trade_delta/`. Exit 0 if MD+CSV exist and matched > 0.
