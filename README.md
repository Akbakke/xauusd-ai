# GX1 / FARM – Active Runtime

This repository now contains only the components that power the current GX1 FARM pipeline:

- **Entry:** `ENTRY_V9_FARM_V2B` (with optional V2 fallback) – transformer based p_long policy that runs through the new `EntryManager`.
- **Exit:** `FARM_EXIT_V2_RULES_A` plus the fixed/random sanity exits, orchestrated via `ExitManager`.
- **Exit Router:** `HYBRID_ROUTER_V3_RANGE` with guardrail – ML decision tree with range features. RULE6A er en edge-spesialisert exit, kun aktivert når range_edge_dist_atr < 1.0. I øvrige regimer gir den ingen målbar forbedring i PnL eller risiko og er derfor deaktivert.
- **Runtime:** `gx1/execution/oanda_demo_runner.py` with the broker, entry, exit, and replay helpers.
- **Features:** `gx1/features/basic_v1.py`, `gx1/features/runtime_v9.py`, and `gx1/seq/sequence_features.py`.
- **Policies & Configs:** `gx1/configs/policies/active/*.yaml` (only the three active bundles) and the matching exits in `gx1/configs/exits/`.
- **Docs:** `gx1/docs/FARM_V2B_EXIT_A_audit.md` and `gx1/docs/GX1_ACTIVE_PIPELINE.md`.
- **Scripts:** `scripts/run_replay.sh` plus the minimal helpers inside `scripts/active/`.
- **Models:** `gx1/models/entry_v9/nextgen_2020_2025_clean/` (with scalers + meta) and the light-weight meta / seq artefacts required for FARM logging.

Everything else (legacy entries/exits, historical reports, experiments, wf runs, cached data, etc.) has been removed to avoid accidental coupling to deprecated systems.

## Quick start

Create a Python 3.11+ environment, install your dependencies, and set the required environment variables (`OANDA_API_KEY`, `OANDA_ACCOUNT_ID`, `M5_DATA`, etc.).

Run a short replay:

```bash
bash scripts/run_replay.sh \
  gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_FULL.yaml \
  2025-06-01 \
  2025-06-05 \
  1
```

`M5_DATA` must point to the local XAUUSD M5 dataset (parquet or CSV). Results are written under `gx1/wf_runs/<policy-name>/`.

For random/fixed-bar sanity checks, swap the policy path for the `_RANDOM_EXIT` or `_FIXED_EXIT` bundles.

## Repository layout

```
gx1/
  configs/        ← only active policy + exit configs
  docs/           ← FARM_V2B + pipeline overview
  execution/      ← oanda_demo_runner + managers
  features/       ← runtime feature builders
  models/         ← entry_v9 weights + FARM meta helpers
  policy/         ← FARM entry/exit policies + guards
  seq/            ← sequence feature helpers used by runtime_v9
  utils/          ← env + pnl utilities
scripts/
  active/         ← replay harness + helpers
  run_replay.sh   ← single entry point for batch replays
```

Anything that is not part of the list above should live outside this repository (private archives, experiments, wf outputs, large datasets, etc.).
