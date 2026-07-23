# GX1 paths

## Canonical local roots

- repository: `/home/andre2/src/GX1_ENGINE`
- repository Python: `/home/andre2/src/GX1_ENGINE/.venv/bin/python`
- large data/artifacts: `/home/andre2/GX1_DATA`
- current Entry launch state:
  `/home/andre2/src/GX1_ENGINE/PROJECT_STATE_xau_direction_launch.json`
- retained artifact registry:
  `/home/andre2/src/GX1_ENGINE/PROJECT_STATE_artifacts.json`
- current audited Entry dataset event:
  `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v24_6yr_rebuild_20260722_seq513_model_native_v24`
- current audited Entry dataset:
  `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v24_6yr_rebuild_20260722_seq513_model_native_v24/dataset`
- current dataset terminal evidence:
  `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v24_6yr_rebuild_20260722_seq513_model_native_v24/CHAIN_TERMINAL_20260722T130501752412Z_GREEN.json`
- current immutable smoke recipe evidence:
  `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v24_6yr_rebuild_20260722_seq513_model_native_v24/train_recipe_20260723T114200Z/ENTRY_MODEL_NATIVE_SEQ513_TRAIN_RECIPE_AUDIT_20260723T114138588679Z.json`
- declared smoke output (currently absent; never infer existence/authority):
  `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v24_6yr_rebuild_20260722_seq513_model_native_v24/v11_entry_model_native_seq513_smoke_20260723T114200Z`
- rejected-split cleanup evidence:
  `/home/andre2/GX1_DATA/cleanup_events/XAU_FAILED_SPLIT_CLEANUP_20260722_V4`

Code, small contracts and tests belong in the repository. Market tapes,
feature datasets, model bundles, checkpoints and large evidence outputs belong
under `GX1_DATA` at explicit immutable paths.

Do not infer a decision artifact from a directory default, glob, mtime or
`latest` link. Required paths must be explicit, absolute, non-symlinked,
hash-bound and accepted by the relevant contract.

Environment variables may configure non-authoritative tooling, but they may
not override immutable Entry artifact identity or introduce a fallback bundle.

The V24 data and V6 recipe paths above are admitted only to exact capped smoke
execution. Recipe v2 binds training/output V6 separately from launch-derived
dataset V24. V6 preserves signed MFE/path-quality target semantics and binds
repository commit `87b0cec2`; its executable bindings retain the exact
`f05b3390` canonical MTF batch-target repair bytes. It
changes only the explicit horizon to eight epochs/patience six after V5's
one-epoch learned evidence failed checkpoint admission; no gate is relaxed. The
recipe/dry-run is not a bundle, model, direction or launch
authority; `PROJECT_STATE_xau_direction_launch.json` remains `BLOCK`.

Never delete or move `GX1_DATA` content merely to reduce repository search
cost. Repository source cleanup and external artifact cleanup are separate:
the latter requires an explicit inventory and cleanup decision.
