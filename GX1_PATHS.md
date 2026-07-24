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
The first-wave findings plus normalization, context routing, MTF component
identity, bundle/event publication, active-Exit byte identity, the
identity-bound transactional launch finalizer and runtime launch fail-close
are source-repaired only. The second audit additionally repairs exact
T+5/closed-M5 Exit timing, V3 window coverage, transactional TradeState and
production-only Exit loading. The current repair adds full-history native-M5
state, atomic canonical/BASE generations, strict source/closure ownership,
causal spread-only semantics, XGB-bound V3 lineage and the exact V3 dataset
writer/event in the existing owner. Current data have not been migrated or
pair-bootstrapped; a fresh V3 dataset and the Exit rebuild/rescore/retrain
remain artifact gates. The
public launch route owns canonical repository
registry/state targets and canonical
`/home/andre2/GX1_DATA/reports/entry_model_native_launch_authority` evidence
roots; callers may not substitute alternate roots. No new data/model or launch
artifacts were written. The existing historical OANDA owner now has immutable
`model-native-native-m1-source` and `model-native-native-m5-source`
publication routes with one strict v3 schema and fixed 3-day/15-day chunk
policy, but neither has been executed. A complete initial
native→canonical-v3/raw-BASE28 build and immutable TRAIN-only rank reference
remain missing.
Caller-supplied joint Exit replay/trace parquets have
zero launch authority; the retained Exit bundle is research-only and
non-production. The canonical full-TEST producer is code-complete in the
existing sizing owner and is routed through
`model-native-canonical-active-exit-replay`, but it has not run on an accepted
fresh chain. Causal Exit rebuild, native-M1/M5 execution/pair bootstrap and canonical/live
December-2024 tape parity remain open.
`PROJECT_STATE_xau_direction_launch.json` remains `BLOCK`.

Never delete or move `GX1_DATA` content merely to reduce repository search
cost. Repository source cleanup and external artifact cleanup are separate:
the latter requires an explicit inventory and cleanup decision.
