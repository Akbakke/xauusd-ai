# GX1 path authority

Updated 2026-08-03.

Paths are evidence identities, never “latest file” hints. Consumers receive
explicit canonical paths plus immutable hashes. Globs, mtimes, lexical version
selection and inferred siblings have no authority.

## Repository owners

- repository: `/home/andre2/src/GX1_ENGINE`
- Python: `/home/andre2/src/GX1_ENGINE/.venv/bin/python`
- control surface: `/home/andre2/src/GX1_ENGINE/scripts/entry_next_edge_control.sh`
- handover: `/home/andre2/src/GX1_ENGINE/scripts/gx1_handover.sh`
- launch state: `/home/andre2/src/GX1_ENGINE/PROJECT_STATE_xau_direction_launch.json`

## Current offline dataset

Dataset directory:

`/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_BASE28_OFFLINE_20260801_FINAL_DATASET_V8`

Exact split hashes:

- TRAIN: `71a652fa18f10d274fb457f34f96bc68fc96625361a29e95021c0733e7d5321b`
- VAL: `c92ddfb9e053f39e04ef4a79b0850e8a218eea99f529996ad16ca4ab7a461bff`
- TEST: `6111a9a5aa1d04f975ed8b8eb1f664820ef78b3a671cf89cebfa2c49e1bb9171`

Rows are 369,303/5,904/6,071. This is current offline evidence, not a
launch-admitted dataset or model.

## Current MTF V4 evidence

Manifest:

`/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_BASE28_OFFLINE_20260801_MTF_V4/manifest.json`

- manifest schema: `htf_v4_disk_cache_manifest_v3`
- manifest SHA-256: `c01703e7be1e4271aabb91d3d4994355ccd3f97c177d1c22586b10f79e6ede6e`
- cache identity: `68568bf9431b1c770876a05e5051eefc252c6eccbf145ca024a9350688ca31b4`
- full-input liveness decision: `PASS`

The old V26 schema-v2 cache is historical launch-checkpoint evidence only. It
must not replace or invalidate this current offline schema-v3 binding.

## Current shared Entry/Exit evidence

Shared M1 featurebase:

`/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_BASE28_OFFLINE_20260801_M1_FEATURE_BASE_FINAL_V4_ALIGNED.parquet`

Unified Exit lifecycle manifest:

`/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_BASE28_OFFLINE_20260801_FINAL_EXIT_LIFECYCLE_V13/UNIFIED_EXIT_LIFECYCLE_MANIFEST.json`

- manifest SHA-256: `db00fa2318ff31e49f2b43c694db6f7cac586713d50b2b233e7f07533d38a05b`
- decision: `PASS`
- Entry: M5, 96 bars
- Exit: M1, 480 bars
- shared ordered signal width: 513
- shared specialist families: 8

## Current smoke recipe

Recipe audit:

`/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_BASE28_OFFLINE_20260801_FINAL_TRAIN_RECIPE_AUDIT_V18_20260803T184234Z/ENTRY_MODEL_NATIVE_SEQ513_TRAIN_RECIPE_AUDIT_20260803T184244728764Z.json`

- SHA-256: `818d8202bd0ab56a29fd43eea46e05bc2a9bfef285d811cd38a1e0909ca18285`
- decision: `PASS`
- source commit: `98ea1c62f49e1d100f64a4cf7ad3a8a591d287ad`
- output bundle path: absent, as required before execution

The executable handover revalidates this recipe through the existing
train-launch contract, including current source bindings and immutable
artifact stat/hash identities. A path existing by itself is not enough.

## Model authority

There is no accepted model or bundle path. The launch registry remains
`BLOCK`, and historical model names or directories cannot be selected as a
fallback.

## Cleanup

Large-data cleanup has one owner:
`gx1/scripts/cleanup_gx1_evidence_v1.py`. It requires an immutable exact-target
plan, separate approval, quarantine verification and terminal evidence.
Never delete a parent directory by exclusion pattern or direct recursive
command.
