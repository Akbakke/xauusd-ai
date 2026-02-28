# Active GX1 / FARM policies

This directory is intentionally small and only carries the configurations we run in the FARM_V2B pipeline.

## Entry configs
- `ENTRY_V9_FARM_V2B.yaml` – primary p_long-driven entry (no meta-filter, ASIA + LOW/MEDIUM).
- `ENTRY_V9_FARM_V2.yaml` – retained as the only fallback entry while we migrate everything to V2B.

## Policy bundles
- `GX1_V11_OANDA_DEMO_V9_FARM_V2B.yaml` – hygiene/coverage replay (V2B + FARM_EXIT_V2_AGGRO).
- `GX1_V11_OANDA_DEMO_V9_FARM_V2B_RULES_A.yaml` – default runtime stack (V2B + FARM_EXIT_V2_RULES_A).
- `GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_FULL.yaml` – long-window replay for audits.
- `GX1_V11_OANDA_DEMO_V9_FARM_V2B_RANDOM_EXIT.yaml` – sanity harness that reuses the entry but forces a random/fixed-bar exit to measure entry-only EV.
- `GX1_V11_OANDA_DEMO_V9_FARM_V2B_FIXED_EXIT.yaml` – deterministic fixed-bar sanity once we wire it up (shares the same exit template as the random variant).

If a policy is no longer part of these flows it must be moved to `gx1/archive/configs/policies/` with a note explaining its historical purpose.
