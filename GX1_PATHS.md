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

The V24 paths above identify data admitted only to the next smoke
implementation review. They are not bundle, model, direction or launch
authority; `PROJECT_STATE_xau_direction_launch.json` remains `BLOCK`.

Never delete or move `GX1_DATA` content merely to reduce repository search
cost. Repository source cleanup and external artifact cleanup are separate:
the latter requires an explicit inventory and cleanup decision.
