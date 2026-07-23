# GX1 paths

## Canonical local roots

- repository: `/home/andre2/src/GX1_ENGINE`
- repository Python: `/home/andre2/src/GX1_ENGINE/.venv/bin/python`
- large data/artifacts: `/home/andre2/GX1_DATA`
- current Entry launch state:
  `/home/andre2/src/GX1_ENGINE/PROJECT_STATE_xau_direction_launch.json`
- retained artifact registry:
  `/home/andre2/src/GX1_ENGINE/PROJECT_STATE_artifacts.json`
- current immutable failed Entry dataset/training event:
  `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v24_6yr_rebuild_20260722_seq513_model_native_v24`
- rejected V24 Entry dataset retained as failure evidence:
  `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v24_6yr_rebuild_20260722_seq513_model_native_v24/dataset`
- current dataset terminal evidence:
  `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v24_6yr_rebuild_20260722_seq513_model_native_v24/CHAIN_TERMINAL_20260722T130501752412Z_GREEN.json`
- terminal failed V7 smoke recipe evidence:
  `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v24_6yr_rebuild_20260722_seq513_model_native_v24/train_recipe_20260723T124100Z/ENTRY_MODEL_NATIVE_SEQ513_TRAIN_RECIPE_AUDIT_20260723T124040048490Z.json`
- terminal V7 output path (absent because no checkpoint was admitted):
  `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v24_6yr_rebuild_20260722_seq513_model_native_v24/v12_entry_model_native_seq513_smoke_20260723T124100Z`
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

The V24 data and V7 recipe paths above are historical, immutable failure
evidence. V7 completed six epochs, failed hard-red with
`TRAIN_FAIL_NO_BEST_STATE`, wrote no bundle and cleaned its temporary memmap.
The post-run audit found signed dip-MFE target corruption and active training
objective mismatches; neither V24 nor V7 may be reused for another run.
`PIPELINE_AUDIT_XAU_20260723.md` is the detailed repair boundary.
Those findings plus normalization, context routing, MTF component identity,
bundle/event publication, active-Exit byte identity and runtime launch
fail-close are now source-repaired only. No new data/model artifacts were
written. The transactional launch finalizer and canonical/live December-2024
tape parity remain open.
`PROJECT_STATE_xau_direction_launch.json` remains `BLOCK`.

Never delete or move `GX1_DATA` content merely to reduce repository search
cost. Repository source cleanup and external artifact cleanup are separate:
the latter requires an explicit inventory and cleanup decision.
