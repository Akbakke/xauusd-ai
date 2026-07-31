# GX1 path authority

Updated 2026-07-31.

Paths are evidence identities, not “latest file” hints. Consumers must receive
explicit canonical paths and SHA-256 bindings. Directory globs, mtimes,
lexical-latest selection and inferred sibling filenames have no authority.

## Source repository

- repository: `/home/andre2/src/GX1_ENGINE`
- Python: `/home/andre2/src/GX1_ENGINE/.venv/bin/python`
- operator control: `/home/andre2/src/GX1_ENGINE/scripts/entry_next_edge_control.sh`
- handover: `/home/andre2/src/GX1_ENGINE/scripts/gx1_handover.sh`
- launch state:
  `/home/andre2/src/GX1_ENGINE/PROJECT_STATE_xau_direction_launch.json`

## Current XAU source/input evidence

- V26 event root:
  `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v26_6yr_rebuild_20260725_seq513_model_native_v26`
- canonical-v3 M5 source:
  `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v26_6yr_rebuild_20260725_seq513_model_native_v26/cv3/xauusd_m5_CANONICAL_V3_2020_2026.parquet`
- source SHA-256:
  `eca51c97ac5a1097ff1b2baae5aea8c38ca162466103d5c2f3c1c18d135848ac`
- frozen historical schema-v2 V4 cache:
  `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v26_6yr_rebuild_20260725_seq513_model_native_v26/MULTI_TF_V4_CACHE_20260729`
- V4 manifest:
  `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v26_6yr_rebuild_20260725_seq513_model_native_v26/MULTI_TF_V4_CACHE_20260729/manifest.json`
- cache identity:
  `ff9cac78cdf6d5d4338f4d07b77df822c95efb568ed80a1e864600580a2b361a`
- embedded liveness identity:
  `42b2b9a4af1870796cf9b22c9257550cb004515095e5e4d2fa31fb22fe4a4b18`

The dated cache path is intentional. It proves the code/source lineage used
for this historical input checkpoint, but active schema v3 rejects it. A fresh
rebuild event must publish its own
event-local `MULTI_TF_V4_CACHE`; do not rename or copy this directory into
authority.

## Current dataset evidence

There is no admitted dataset path. The stale V19/V26 split, audit, smoke
manifest and rejected trainability-bundle artifacts were retired from launch
state and deleted through exact cleanup evidence on 2026-07-29. A fresh
schema-v3 V4 build must publish a new event-local path; no historical path may
be inferred or reused.

## Model/bundle authority

There is no accepted V4 model or bundle path. The former V18 artifact is
absent; V21C survives only as a documented diagnostic result. Do not create a
path here until the immutable producer publishes and audits it.

## Native/canonical pair

The accepted 2026-07-24 native M1/M5 roots and frozen pair generation
`077e5419…` remain source evidence for future unified lifecycle training and
replay. They are not a current live-tail authority. Resolve their exact paths through
their immutable manifests and registry bindings; do not copy their pointer
name into a new artifact.

## Live-tail authority

There is no admitted live-tail event path. The existing snapshot/pair and
live-tail contract owners can publish exact immutable native schema-v4
successors, canonical successor/publication events and two-event admission
artifacts through the public control surface, but no real chain has executed.
Native successors require exact parent root plus parent `MANIFEST.json`
SHA-256 and fetch only bounded overlap plus tail. Launch state stores the exact
admission/event roots, pair pointer/generation root, producer and anchor;
runtime selects the newest immutable admission and requires exact equality
with the pair used for Entry inference. The collector does not by itself
create an advancing canonical pair, and the retired incremental daemon has no
authority. Do not invent a `latest` path; record exact event paths and SHA-256
only after their owners publish them.

## Cleanup

Large-data cleanup must use
`gx1/scripts/cleanup_gx1_evidence_v1.py` with an immutable plan, separate
approval, quarantine validation and terminal evidence. Never delete a parent
directory by exclusion pattern.
